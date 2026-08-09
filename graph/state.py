"""
Shared state schema for the research agent graph.

Every node reads from and writes to this TypedDict. Keeping the schema
in one place — rather than scattering ad-hoc dicts across nodes — means
all state fields are declared explicitly and changes propagate cleanly.
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

    # --- critic output ---
    critique_pass: Optional[bool]
    critique_issues: Optional[List[str]]
    iteration: int  # number of critique/revise cycles completed

    # --- final output ---
    final_report: Optional[str]

    # --- status, for the Gradio UI to display live progress ---
    status: str
