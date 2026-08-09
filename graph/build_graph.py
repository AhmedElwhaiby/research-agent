"""
Wires nodes into a LangGraph StateGraph:

    planner -> research -> synthesizer -> critic --pass--> finalize -> END
                                              |
                                            fail
                                              v
                                           reviser --> critic (loop, capped at MAX_ITERATIONS)

The loop's stopping condition lives in _route_after_critic: pass, or hitting
MAX_ITERATIONS, both route to finalize. Failing before the cap routes back to
reviser.
"""

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from graph.nodes import (
    planner_node,
    research_node,
    synthesizer_node,
    critic_node,
    reviser_node,
    finalize_node,
    MAX_ITERATIONS,
)


def _route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge out of critic_node.

    Checked in this order: pass first (stop as soon as it's good), then the
    iteration cap (stop even on failure once we've spent the retry budget —
    otherwise a persistently failing draft loops forever). Only fails BOTH
    checks does it go back to reviser.
    """
    if state.get("critique_pass"):
        return "end"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "end"
    return "revise"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "end": "finalize",
            "revise": "reviser",
        },
    )
    graph.add_edge("reviser", "critic")
    graph.add_edge("finalize", END)

    return graph.compile()


if __name__ == "__main__":
    # quick manual smoke test:
    #   python -m graph.build_graph ["optional topic"]
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    app = build_graph()
    topic = sys.argv[1] if len(sys.argv) > 1 else "The current state of small modular nuclear reactors"
    result = app.invoke({"topic": topic, "iteration": 0})

    print(f"--- critique_pass: {result.get('critique_pass')} | iterations used: {result.get('iteration')} ---\n")
    print(result.get("final_report", "No final report produced."))