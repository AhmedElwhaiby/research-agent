"""
Wires Phase 1 nodes into a LangGraph StateGraph: a straight line,
topic -> planner -> research -> synthesizer -> END.

No conditional edges yet. Phase 2 adds critic_node and reviser_node plus a
conditional edge out of critic_node (pass -> END, fail -> reviser, capped
at 2 iterations) — that's the only structural change expected later;
this file is deliberately written so that change is additive, not a rewrite.
"""

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from graph.nodes import planner_node, research_node, synthesizer_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


if __name__ == "__main__":
    # quick manual smoke test:
    #   GROQ_API_KEY=... TAVILY_API_KEY=... python -m graph.build_graph
    import sys

    app = build_graph()
    topic = sys.argv[1] if len(sys.argv) > 1 else "The current state of small modular nuclear reactors"
    result = app.invoke({"topic": topic, "iteration": 0})
    print(result.get("draft", "No draft produced."))
