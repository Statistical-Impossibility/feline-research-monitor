"""Treatment Radar agent: confirms first-seen interventions and explains them.

The ADK import is lazy so `parse_radar_output` is unit-testable without the framework.
"""

import json
import re

from feline_monitor.agents import read_skills


def parse_radar_output(text: str) -> list[dict]:
    """Extract the radar's JSON list of confirmed interventions, tolerating prose/fences."""
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("entity") and d.get("note"):
                out.append(
                    {
                        "entity": str(d["entity"]).strip(),
                        "category": str(d.get("category", "")).strip(),
                        "note": str(d["note"]).strip(),
                    }
                )
    return out


def build_radar_agent(model, skills_dir: str = "skills"):
    """Construct the ADK radar agent (loads the treatment-radar skill)."""
    from google.adk.agents import LlmAgent

    from feline_monitor.tracing import after_model_callback, before_model_callback

    instruction = read_skills(["treatment-radar"], skills_dir)
    return LlmAgent(
        name="radar_agent",
        model=model,
        description="Confirms genuinely new feline interventions and explains why they matter.",
        instruction=instruction,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
    )
