"""MCP server exposing PubMed search as a tool — the agent's doorway to PubMed.

This wraps the existing Entrez fetch as a standard MCP tool. ADK's McpToolset (or the
project's own MCP client) launches this over stdio and calls `pubmed_search`. This is
the project's MCP-Server concept artifact: MCP gives reach to an external system (PubMed).
"""

import os
import sys

# Allow `import feline_monitor...` when launched as a standalone subprocess.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Use the OS trust store so Entrez TLS works behind inspecting proxies (no-op otherwise).
try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

from mcp.server.fastmcp import FastMCP

from feline_monitor.pubmed import fetch

mcp = FastMCP("pubmed")


@mcp.tool()
def pubmed_search(query: str, max_results: int = 50, since: str | None = None) -> list[dict]:
    """Search PubMed Central for `query` and return up to `max_results` full-text papers.

    `since` ('YYYY-MM-DD') floors results by publication date (backfill anchor).
    Each paper is a dict with keys: pmid, pmcid, title, abstract, full_text,
    has_full_text, article_type, pub_date, url.
    """
    pmc_ids = fetch.fetch_pmc_ids(query, max_results, since)
    return fetch.fetch_papers(pmc_ids)


@mcp.tool()
def pubmed_search_page(
    query: str, retstart: int = 0, retmax: int = 200, since: str | None = None
) -> dict:
    """Return one page of PMC ids (newest first) + total count — the backfill primitive.

    Cheap (ids only, no full text). Page through the whole `since` window by stepping
    `retstart` until the page is empty or `retstart` >= count. Shape: {"ids": [...], "count": N}.
    """
    ids, count = fetch.fetch_pmc_ids_page(query, retstart, retmax, since)
    return {"ids": ids, "count": count}


@mcp.tool()
def pubmed_fetch(pmc_ids: list[str]) -> list[dict]:
    """Fetch and parse PMC full text for the given PMC ids (after dedup selection)."""
    return fetch.fetch_papers(pmc_ids)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
