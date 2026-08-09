"""
Node functions: planner -> researcher -> synthesizer -> critic -> (reviser -> critic)*.

MAX_ITERATIONS controls the maximum number of critique/revise cycles.
build_graph.py's conditional edge reads critique_pass and iteration to decide
whether to route back to reviser or forward to finalize.
"""

from graph.state import ResearchState, SubQuestionResearch
from tools.search import search_web
from tools.llm import call_llm, call_llm_json

MAX_SUB_QUESTIONS = 4
RESULTS_PER_SUB_QUESTION = 2
MAX_ITERATIONS = 2  # maximum critique/revise cycles before finalize forces exit


def planner_node(state: ResearchState) -> dict:
    """Split the topic into a small set of concrete, searchable sub-questions."""
    topic = state["topic"]

    system = (
        "You are a research planner. Given a topic, break it into "
        f"{MAX_SUB_QUESTIONS} distinct, concrete sub-questions that together "
        "give thorough coverage of the topic. Each sub-question should be "
        "specific enough to search for directly (not vague or overlapping). "
        "Return ONLY a JSON array of strings, nothing else."
    )
    user = f"Topic: {topic}"

    sub_questions = call_llm_json(system, user)
    if not isinstance(sub_questions, list) or not sub_questions:
        # fall back to a single sub-question so the pipeline never dead-ends
        sub_questions = [topic]

    return {
        "sub_questions": sub_questions[:MAX_SUB_QUESTIONS],
        "status": f"Planned {len(sub_questions[:MAX_SUB_QUESTIONS])} sub-questions",
    }


def research_node(state: ResearchState) -> dict:
    """
    For every sub-question: search the web, then have the LLM condense each
    result into a short grounded summary. Runs all sub-questions sequentially
    in one node call rather than as separate LangGraph nodes per question —
    keeps the graph shape simple; can be split into a fan-out/fan-in subgraph
    later if per-question retries are needed.
    """
    research: dict[str, SubQuestionResearch] = {}

    for i, question in enumerate(state["sub_questions"], start=1):
        notes = search_web(question, max_results=RESULTS_PER_SUB_QUESTION)

        for note in notes:
            if not note["snippet"]:
                continue
            summary = call_llm(
                system=(
                    "Summarize the following source in 2-3 sentences, focused "
                    "only on content relevant to the research question. Be "
                    "factual — do not add claims not present in the source."
                ),
                user=f"Research question: {question}\n\nSource content:\n{note['snippet']}",
            )
            note["summary"] = summary.strip()

        research[question] = SubQuestionResearch(question=question, notes=notes)

    return {
        "research": research,
        "status": f"Researched {len(state['sub_questions'])}/{len(state['sub_questions'])} sub-questions",
    }


def synthesizer_node(state: ResearchState) -> dict:
    """Draft a cited markdown report from the collected research notes."""
    topic = state["topic"]
    research = state["research"]

    # Build a flat, numbered source list so the LLM can cite by URL directly
    # rather than inventing citation markers.
    context_blocks = []
    for question, entry in research.items():
        block = [f"### Sub-question: {question}"]
        for note in entry["notes"]:
            if note["summary"]:
                block.append(f"- {note['summary']} (Source: {note['url']})")
        context_blocks.append("\n".join(block))
    context = "\n\n".join(context_blocks)

    system = (
        "You are a research writer. Using ONLY the provided research notes, "
        "write a well-organized markdown report on the topic.\n\n"
        "CITATION FORMAT — follow exactly:\n"
        "- Cite every factual claim with a plain numbered reference in "
        "square brackets, e.g. 'Teens report higher rates of X [3].'\n"
        "- Use ASCII brackets only: [ and ]. Never use 【 】 or any other "
        "bracket style, and never write the word 'Source' inside the "
        "brackets — just the number.\n"
        "- At the end, include a '## Sources' section as a numbered list "
        "mapping each number to its URL, e.g. '1. https://example.com'\n"
        "- Reuse the same number if you cite the same URL again; don't "
        "create duplicate numbers for one source.\n\n"
        "Keep the report focused: cover each sub-question in one section, "
        "no more than 2-3 sentences per point. Do not add a recommendations "
        "section or extra material beyond what the research notes support. "
        "Do not invent sources or facts not present in the notes."
    )
    user = f"Topic: {topic}\n\nResearch notes:\n{context}"

    draft = call_llm(system, user, temperature=0.4, max_tokens=2500)
    draft = _normalize_citation_brackets(draft)

    return {
        "draft": draft,
        "status": "Draft synthesized",
    }


def _normalize_citation_brackets(text: str) -> str:
    """
    Defensive cleanup: if the model still slips in full-width brackets
    (observed even after explicit prompt instructions not to), rewrite them
    to plain ASCII brackets rather than trusting the prompt alone to
    prevent it. Doesn't fix deeper structural issues (e.g. a citation with
    no closing bracket at all) — those should still get caught by the critic.
    """
    return text.replace("【", "[").replace("】", "]")



def critic_node(state: ResearchState) -> dict:
    """
    Check the current draft against a fixed rubric and return a structured
    pass/fail plus a list of concrete issues. This is the node the conditional
    edge in build_graph.py reads to decide whether to route to END or to
    reviser_node.

    Rubric is deliberately narrow (citation format, uncited claims, coverage,
    unsupported specifics) rather than "is this a good report?" in general —
    a vague rubric produces vague issues the reviser can't act on.
    """
    draft = state["draft"]
    topic = state["topic"]
    sub_questions = state.get("sub_questions", [])

    system = (
        "You are a strict editor reviewing a research report against a fixed "
        "rubric. Check for exactly these things:\n"
        "1. CITATION FORMAT: every citation must be a plain numbered "
        "reference in ASCII square brackets, e.g. '[3]' — flag any use of "
        "【】, any citation containing the word 'Source', any malformed or "
        "unclosed bracket, or any claim with no citation at all.\n"
        "2. SOURCES LIST: flag if the '## Sources' section is missing, or if "
        "any number cited in the body doesn't appear in that list, or vice "
        "versa.\n"
        "3. UNCITED CLAIMS: flag any specific factual claim (a statistic, a "
        "named study, a named finding) that has no citation immediately "
        "following it.\n"
        "4. COVERAGE: flag if any of the given sub-questions is not "
        "substantively addressed anywhere in the draft.\n"
        "Do not comment on writing style, tone, or organization — only these "
        "four things. Return ONLY a JSON object of the form: "
        '{"pass": true or false, "issues": ["specific issue 1", "specific issue 2"]}. '
        'If there are no issues, return {"pass": true, "issues": []}.'
    )
    user = (
        f"Topic: {topic}\n"
        f"Sub-questions the report should cover: {sub_questions}\n\n"
        f"Draft report:\n{draft}"
    )

    result = call_llm_json(system, user, temperature=0.1)

    # Defensive parsing: don't let a malformed critic response silently pass
    # a bad draft — if the shape is wrong, treat it as a fail with a generic
    # issue rather than crashing or (worse) defaulting to pass=True.
    if not isinstance(result, dict) or "pass" not in result:
        return {
            "critique_pass": False,
            "critique_issues": ["Critic returned an unparseable response; treating as fail."],
            "status": "Critique failed to parse",
        }

    return {
        "critique_pass": bool(result.get("pass", False)),
        "critique_issues": result.get("issues", []) or [],
        "status": f"Critique: {'pass' if result.get('pass') else 'fail'}",
    }


def reviser_node(state: ResearchState) -> dict:
    """
    Rewrite the draft to address the critic's issues, using the same
    research notes as the synthesizer (so the reviser can't introduce new,
    ungrounded claims to fix a gap — it can only better use what's already
    there, or drop a claim it can't support).
    """
    topic = state["topic"]
    draft = state["draft"]
    issues = state.get("critique_issues", [])
    research = state["research"]

    context_blocks = []
    for question, entry in research.items():
        block = [f"### Sub-question: {question}"]
        for note in entry["notes"]:
            if note["summary"]:
                block.append(f"- {note['summary']} (Source: {note['url']})")
        context_blocks.append("\n".join(block))
    context = "\n\n".join(context_blocks)

    system = (
        "You are revising a research report to fix specific issues an editor "
        "found. Using ONLY the original research notes provided, rewrite the "
        "report to address every issue listed. Do not invent new facts or "
        "sources to patch a gap — if a claim can't be supported by the notes, "
        "cut it rather than leave it uncited. Citation format: plain numbered "
        "references in ASCII brackets, e.g. '[3]', never 【】 and never the "
        "word 'Source' inside brackets. Keep the '## Sources' section as a "
        "numbered list matching every number used in the body."
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Original research notes:\n{context}\n\n"
        f"Current draft:\n{draft}\n\n"
        f"Issues to fix:\n" + "\n".join(f"- {issue}" for issue in issues)
    )

    revised = call_llm(system, user, temperature=0.3, max_tokens=2500)
    revised = _normalize_citation_brackets(revised)

    return {
        "draft": revised,
        "iteration": state.get("iteration", 0) + 1,
        "status": f"Revised (iteration {state.get('iteration', 0) + 1})",
    }


def finalize_node(state: ResearchState) -> dict:
    """
    Copies the current draft into final_report, the moment the loop exits
    (whether by passing critique or by hitting MAX_ITERATIONS). Keeping this
    as its own node — rather than setting final_report inside critic_node —
    means it's set exactly once, at the actual exit point, and it can flag
    the case where the loop ended without ever passing, which callers (the
    Gradio UI, the eval script) need to distinguish from a clean pass.
    """
    passed = bool(state.get("critique_pass"))
    note = (
        ""
        if passed
        else "\n\n> **Note:** this report exhausted its revision attempts "
        "without fully passing the citation/coverage check. Remaining known "
        f"issues: {state.get('critique_issues', [])}"
    )
    return {
        "final_report": state["draft"] + note,
        "status": "Finalized (passed)" if passed else "Finalized (hit iteration cap)",
    }