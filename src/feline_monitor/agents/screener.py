"""Screening agent: judges relevance and triages a paper (relevance + study-triage skills).

The ADK import is lazy so the pure output-parser can be unit-tested without the
framework installed.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel

from feline_monitor.agents import read_skills

_STUDY_TYPES = {"trial", "case_report", "review", "lab", "other"}
_PRIORITIES = {"high", "medium", "low"}


class ScreeningVerdict(BaseModel):
    """Enforced verdict shape when structured output is enabled (ADK output_schema).

    Field names/values mirror what parse_screening_json expects, so the tolerant
    parser handles both structured and prompt-instructed output identically.
    """

    relevant: bool
    reason: str
    study_type: Literal["trial", "case_report", "review", "lab", "other"]
    priority: Literal["high", "medium", "low"]


def parse_screening_json(text: str) -> dict:
    """Extract the screener's JSON verdict from raw model text, with safe defaults.

    Tolerates surrounding prose / code fences. Unknown or missing fields fall back
    to conservative defaults (not relevant, lowest priority, 'other').
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    parsed = False
    raw: dict = {}
    if match:
        try:
            raw = json.loads(match.group(0))
            parsed = isinstance(raw, dict)
        except json.JSONDecodeError:
            parsed = False

    study = str(raw.get("study_type", "other")).lower()
    if study not in _STUDY_TYPES:
        study = "other"
    priority = str(raw.get("priority", "low")).lower()
    if priority not in _PRIORITIES:
        priority = "low"
    return {
        # parsed=False means the model returned no valid JSON verdict (not a real
        # "not relevant" — the caller should skip + retry rather than trust defaults).
        "parsed": parsed,
        "relevant": bool(raw.get("relevant", False)),
        "reason": str(raw.get("reason", "")).strip(),
        "study_type": study,
        "priority": priority,
    }


def build_screening_agent(model, skills_dir: str = "skills", structured: bool = False):
    """Construct the ADK screening agent (loads relevance-screening + study-triage).

    When `structured` is True, attach `output_schema=ScreeningVerdict` so the model's
    structured-output support enforces a valid JSON verdict. The screener has no tools,
    so the output_schema restriction (no tool/transfer use) costs us nothing. A model
    that doesn't support schemas raises → run_with_fallback fails over as usual, and
    parse_screening_json still guards the result regardless.
    """
    from google.adk.agents import LlmAgent

    from feline_monitor.tracing import after_model_callback, before_model_callback

    instruction = read_skills(["relevance-screening", "study-triage"], skills_dir)
    extra = {"output_schema": ScreeningVerdict} if structured else {}
    return LlmAgent(
        name="screening_agent",
        model=model,
        description="Screens PubMed papers for feline relevance and triages them.",
        instruction=instruction,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        **extra,
    )
