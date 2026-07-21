"""
Shared state schema for the research agent graph.

Every node reads from and writes to this TypedDict. Keeping it in one place
(rather than scattering ad-hoc dicts across nodes) is what lets the critique
loop (Phase 2) bolt on cleanly later: critic/reviser just add fields, they
don't change the shape of what already exists.
"""

from typing import TypedDict, List, Dict, Optional


class SourceNote(TypedDict):
    """One piece of research grounding for a sub-question."""
    url: str
    title: str
    snippet: str          # the raw content pulled from the search result
    summary: str          # LLM-condensed takeaway used when drafting


class SubQuestionResearch(TypedDict):
    """All research gathered for a single sub-question."""
    question: str
    notes: List[SourceNote]


class ResearchState(TypedDict, total=False):
    # --- input ---
    topic: str

    # --- planner output ---
    sub_questions: List[str]

    # --- research loop output ---
    # keyed by sub-question so nodes can look up / update a single entry
    # without re-scanning a list
    research: Dict[str, SubQuestionResearch]

    # --- synthesizer output ---
    draft: str

    # --- critic output (Phase 2 — present now so the schema doesn't change
    # shape when the critique loop is added, only these fields start getting used) ---
    critique_pass: Optional[bool]
    critique_issues: Optional[List[str]]
    iteration: int  # how many critique/revise cycles have run, capped at 2

    # --- final output ---
    final_report: Optional[str]

    # --- status, for the Gradio UI to display live progress ---
    status: str
