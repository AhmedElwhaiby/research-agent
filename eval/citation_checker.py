"""
Deterministic citation checker — no LLM involved on purpose.

The critic node already checks "does a citation exist near this claim"
as part of an LLM judgment call, which showed in manual testing that it
can pass a draft on citation *presence* without anyone verifying citation
*accuracy*. This module doesn't try to fix that (verifying accuracy needs
either a human or a separate fact-checking pass against real source
content) — it answers a narrower, fully mechanical question instead:
does every sentence carry a citation marker, and does every citation
marker resolve to a real entry in the Sources list? A regex can answer
that exactly, with no ambiguity and no per-run cost, so there's no reason
to delegate it to an LLM.
"""

import re
from typing import TypedDict, List, Dict

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SOURCE_LINE_PATTERN = re.compile(r"^\s*(\d+)\.\s+(\S+)", re.MULTILINE)

# Lines that are structural, not claims — don't expect a citation on these.
SKIP_LINE_PATTERN = re.compile(
    r"^\s*(#{1,6}\s|>\s*\*\*Note|\|.*\||-{3,}\s*$)"
)


class CoverageResult(TypedDict):
    total_sentences: int
    cited_sentences: int
    coverage_pct: float
    uncited_sentences: List[str]


class ConsistencyResult(TypedDict):
    orphaned_citations: List[str]   # cited in body, no matching Sources entry
    unused_sources: List[str]       # in Sources list, never cited in body


def split_body_and_sources(report_text: str) -> tuple[str, str]:
    """
    Splits a final report into (body, sources_section) on the '## Sources'
    heading. If the heading isn't found, sources_section is empty — callers
    should treat that as a hard failure (a report with no Sources section
    at all), not silently skip the check.
    """
    marker = "## Sources"
    idx = report_text.find(marker)
    if idx == -1:
        return report_text, ""
    return report_text[:idx], report_text[idx + len(marker):]


def parse_source_numbers(sources_section: str) -> Dict[str, str]:
    """Returns {citation_number: url} from the numbered Sources list."""
    return {num: url for num, url in SOURCE_LINE_PATTERN.findall(sources_section)}


def check_citation_coverage(body: str) -> CoverageResult:
    """
    Splits the body into sentences (naive split on '.', '!', '?' followed by
    whitespace — good enough for prose, not meant to handle abbreviations
    perfectly) and checks each non-structural sentence for at least one
    [n] citation marker.
    """
    uncited: List[str] = []
    total = 0
    cited = 0

    for line in body.splitlines():
        if not line.strip() or SKIP_LINE_PATTERN.match(line):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            total += 1
            if CITATION_PATTERN.search(sentence):
                cited += 1
            else:
                uncited.append(sentence)

    coverage_pct = (cited / total * 100) if total else 0.0
    return CoverageResult(
        total_sentences=total,
        cited_sentences=cited,
        coverage_pct=round(coverage_pct, 1),
        uncited_sentences=uncited,
    )


def check_source_consistency(body: str, sources_section: str) -> ConsistencyResult:
    """
    Cross-checks citation numbers used in the body against the numbered
    Sources list: flags any [n] with no matching entry (orphaned), and any
    Sources entry never referenced in the body (unused — usually means the
    synthesizer padded the reference list, or renumbered inconsistently).
    """
    cited_numbers = set(CITATION_PATTERN.findall(body))
    source_map = parse_source_numbers(sources_section)
    source_numbers = set(source_map.keys())

    orphaned = sorted(cited_numbers - source_numbers, key=int)
    unused = sorted(source_numbers - cited_numbers, key=int)

    return ConsistencyResult(orphaned_citations=orphaned, unused_sources=unused)


def run_full_check(report_text: str) -> dict:
    """Convenience wrapper: runs both checks and returns a single combined dict."""
    body, sources_section = split_body_and_sources(report_text)
    has_sources_section = bool(sources_section.strip())

    coverage = check_citation_coverage(body)
    consistency = check_source_consistency(body, sources_section)

    return {
        "has_sources_section": has_sources_section,
        **coverage,
        **consistency,
    }