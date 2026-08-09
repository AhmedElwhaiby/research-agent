"""
Runs the full graph against every topic in test_topics.py and reports:
- citation coverage % (from citation_checker.py — deterministic, no LLM)
- source consistency (orphaned citations / unused sources)
- iterations used and whether critique_pass was ever true
- wall-clock time per topic

Deliberately isolates failures per-topic: a single Groq rate-limit error
or a single bad topic shouldn't kill the whole eval run and lose every
result gathered so far. Each topic gets its own try/except, and a fixed
delay between topics to stay under the account's TPM cap rather than
firing requests back-to-back and expecting them all to clear.

This does NOT check factual accuracy of citations (whether [3] really
says what the draft claims) — that requires human spot-checking. This
script only catches what is mechanically verifiable: citation marker
presence, marker-to-source resolution, and coverage consistency across
different topic types.
"""

import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

from graph.build_graph import build_graph
from eval.test_topics import TEST_TOPICS
from eval.citation_checker import run_full_check

SECONDS_BETWEEN_TOPICS = 20  # pacing to stay under the 8000 TPM account cap


def run_eval():
    load_dotenv()
    app = build_graph()
    results = []

    for i, topic in enumerate(TEST_TOPICS, start=1):
        print(f"\n[{i}/{len(TEST_TOPICS)}] Running: {topic}")
        start = time.time()

        try:
            state = app.invoke({"topic": topic, "iteration": 0})
            final_report = state.get("final_report", "")
            check = run_full_check(final_report)

            result = {
                "topic": topic,
                "status": "ok",
                "critique_pass": state.get("critique_pass"),
                "critique_issues": state.get("critique_issues", []),
                "iterations": state.get("iteration"),
                "duration_sec": round(time.time() - start, 1),
                **check,
            }
            print(
                f"  pass={result['critique_pass']} "
                f"iterations={result['iterations']} "
                f"coverage={result['coverage_pct']}% "
                f"orphaned={len(result['orphaned_citations'])} "
                f"unused={len(result['unused_sources'])} "
                f"critic_issues={result['critique_issues']}"
            )

        except Exception as e:
            result = {
                "topic": topic,
                "status": "error",
                "error": str(e),
            }
            print(f"  FAILED: {e}")
            traceback.print_exc()

        results.append(result)

        if i < len(TEST_TOPICS):
            time.sleep(SECONDS_BETWEEN_TOPICS)

    _print_summary(results)
    _save_results(results)
    return results


def _print_summary(results: list[dict]):
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]

    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY — {len(ok)}/{len(results)} topics completed without error")
    print("=" * 60)

    if ok:
        avg_coverage = sum(r["coverage_pct"] for r in ok) / len(ok)
        avg_iterations = sum(r["iterations"] for r in ok) / len(ok)
        passed = sum(1 for r in ok if r["critique_pass"])
        total_orphaned = sum(len(r["orphaned_citations"]) for r in ok)
        total_unused = sum(len(r["unused_sources"]) for r in ok)

        print(f"Avg citation coverage: {avg_coverage:.1f}%")
        print(f"Avg iterations to finish: {avg_iterations:.2f}")
        print(f"Passed critique (not just capped out): {passed}/{len(ok)}")
        print(f"Total orphaned citations across all runs: {total_orphaned}")
        print(f"Total unused sources across all runs: {total_unused}")

    if failed:
        print(f"\nFailed topics ({len(failed)}):")
        for r in failed:
            print(f"  - {r['topic']}: {r['error']}")


def _save_results(results: list[dict]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"eval/results_{timestamp}.md"

    lines = [f"# Eval Results — {timestamp}\n"]
    for r in results:
        lines.append(f"## {r['topic']}")
        if r["status"] == "error":
            lines.append(f"- **FAILED**: {r['error']}\n")
            continue
        lines.append(f"- critique_pass: {r['critique_pass']}")
        lines.append(f"- iterations: {r['iterations']}")
        lines.append(f"- duration: {r['duration_sec']}s")
        lines.append(f"- coverage: {r['coverage_pct']}% ({r['cited_sentences']}/{r['total_sentences']} sentences)")
        lines.append(f"- orphaned citations: {r['orphaned_citations']}")
        lines.append(f"- unused sources: {r['unused_sources']}")
        if r.get("critique_issues"):
            lines.append("- critique issues:")
            for issue in r["critique_issues"]:
                lines.append(f"  - {issue}")
        if r["uncited_sentences"]:
            lines.append("- uncited sentences (sample):")
            for s in r["uncited_sentences"][:3]:
                lines.append(f"  - {s}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nFull results written to {path}")


if __name__ == "__main__":
    run_eval()