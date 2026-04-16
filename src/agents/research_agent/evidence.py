"""Evidence data model for research-agent scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Evidence:
    """A single piece of evidence extracted from a fetched web page."""

    source_url: str
    passage: str
    relevance_note: str
    search_query: str
    fetch_timestamp: float
