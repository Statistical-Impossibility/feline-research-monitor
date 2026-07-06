"""Render a deterministic Markdown digest from research items (no LLM, pure formatting)."""

import os
from datetime import date, datetime


def render_markdown(items: list[dict]) -> str:
    """Build a Markdown digest string from a list of digest items.

    A digest item is a dict with the keys: ``title``, ``url``, ``pmid``,
    ``summary`` and the optional keys ``study_type``, ``priority`` and
    ``interventions`` (which may be absent or None).

    This is a pure function: it performs no I/O and depends only on its input
    and today's date.
    """
    today = date.today().isoformat()
    heading = f"# Feline Research Digest — {today}"

    if not items:
        return f"{heading}\n\nNo new papers.\n"

    lines = [heading, "", f"{len(items)} new paper(s).", ""]

    for item in items:
        lines.append(f"## [{item['title']}]({item['url']})")

        # Metadata line: include only the parts that are present.
        meta_parts = []
        if item.get("priority"):
            meta_parts.append(f"Priority: {item['priority']}")
        if item.get("study_type"):
            meta_parts.append(f"Study: {item['study_type']}")
        if meta_parts:
            lines.append(f"*{' · '.join(meta_parts)}*")

        interventions = item.get("interventions")
        if interventions:
            lines.append(f"🆕 New interventions: {', '.join(interventions)}")

        lines.append(item["summary"])

        # Treatment Radar (v2) comes AFTER the v1 summary: v1 = verdict + summary, then
        # v2 appends the first-seen-treatment note for this paper.
        radar = item.get("radar")
        if radar:
            lines.append("")
            lines.append("**Treatment Radar**")
            for r in radar:
                cat = f" ({r['category']})" if r.get("category") else ""
                lines.append(f"- **{r['entity']}**{cat}: {r['note']}")

        lines.append("")

    return "\n".join(lines) + "\n"


def write_digest(md: str, dir_path: str) -> str:
    """Write the digest to ``<dir_path>/<YYYY-MM-DD_HHMMSS>.md`` and return its path.

    The timestamp (not just the date) means a second run on the same day creates a NEW
    file instead of overwriting the first — each run's digest is preserved.
    """
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{datetime.now():%Y-%m-%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
