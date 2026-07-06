"""Synchronous MCP client that calls our PubMed MCP server over stdio.

This makes the agent system genuinely reach PubMed through MCP (client -> server),
not just expose a server. run.py uses it for retrieval, with a direct-fetch
fallback if the MCP round-trip fails.
"""

import asyncio
import json
import os
import sys

_SERVER = os.path.join(os.path.dirname(__file__), "mcp_server.py")


def _extract(result) -> list[dict]:
    """Pull the list[Paper] out of an MCP tool result across SDK shapes."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and isinstance(structured.get("result"), list):
        return structured["result"]
    for chunk in getattr(result, "content", []) or []:
        text = getattr(chunk, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("result"), list):
            return data["result"]
    return []


def _extract_page(result) -> dict:
    """Pull the {'ids': [...], 'count': N} page dict out of an MCP tool result."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and "ids" in structured:
        return structured
    for chunk in getattr(result, "content", []) or []:
        text = getattr(chunk, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "ids" in data:
            return data
    return {"ids": [], "count": 0}


async def _search(query: str, max_results: int, since: str | None) -> list[dict]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    args = {"query": query, "max_results": max_results}
    if since:
        args["since"] = since
    params = StdioServerParameters(command=sys.executable, args=[_SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("pubmed_search", args)
            return _extract(result)


def pubmed_search(query: str, max_results: int = 50, since: str | None = None) -> list[dict]:
    """Search PubMed via the MCP server (launches it as a subprocess)."""
    return asyncio.run(_search(query, max_results, since))


async def _call(tool: str, args: dict):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[_SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


def pubmed_search_page(
    query: str, retstart: int = 0, retmax: int = 200, since: str | None = None
) -> dict:
    """One page of PMC ids + total count via MCP. Shape: {'ids': [...], 'count': N}."""
    args = {"query": query, "retstart": retstart, "retmax": retmax}
    if since:
        args["since"] = since
    return _extract_page(asyncio.run(_call("pubmed_search_page", args)))


def pubmed_fetch(pmc_ids: list[str]) -> list[dict]:
    """Fetch full-text papers for the selected PMC ids via MCP."""
    if not pmc_ids:
        return []
    return _extract(asyncio.run(_call("pubmed_fetch", {"pmc_ids": pmc_ids})))
