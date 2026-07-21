"""
Thin wrapper around the Tavily search API.

Kept isolated from the graph nodes so the search backend can be swapped
(or mocked in tests) without touching node logic.
"""

import os
from typing import List
from tavily import TavilyClient

from graph.state import SourceNote

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Add it to your .env or environment."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def search_web(query: str, max_results: int = 5) -> List[SourceNote]:
    """
    Run a Tavily search and return results shaped as SourceNote entries
    (summary left empty — the researcher node fills that in via the LLM).
    """
    client = _get_client()
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
    )

    notes: List[SourceNote] = []
    for result in response.get("results", []):
        notes.append(
            SourceNote(
                url=result.get("url", ""),
                title=result.get("title", ""),
                snippet=(result.get("content", "") or "")[:2000],
                summary="",  # filled in by researcher node
            )
        )
    return notes
