"""ADK agents and the Agent-Skills loader."""

from pathlib import Path


def read_skills(names: list[str], base: str = "skills") -> str:
    """Concatenate one or more Agent-Skill files into a single instruction string.

    Each skill lives at `<base>/<name>/SKILL.md`. Joining several lets one agent
    carry multiple skills (e.g. the screener loads relevance-screening + study-triage).
    """
    parts = [(Path(base) / n / "SKILL.md").read_text(encoding="utf-8") for n in names]
    return "\n\n---\n\n".join(parts)
