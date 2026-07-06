"""Summarizer agent: writes a scholarly, plain-language summary of one relevant paper."""

from feline_monitor.agents import read_skills

_SUMMARY_MARKER = "===SUMMARY==="


def strip_reasoning(text: str) -> str:
    """Return the final summary, dropping any reasoning a model printed before the marker.

    Reasoning/instruct models sometimes leak their scratchpad ("We need to produce a
    summary...") into the output. The finding-summary skill tells the model to put the
    final summary after a `===SUMMARY===` marker; we keep only what follows the LAST marker.
    If the marker is absent (a model that ignored the instruction) or nothing usable
    follows it, return the text unchanged — clean models (e.g. Gemini) are never harmed and
    we never drop a real summary. Pure/stdlib so it is unit-testable without the framework.
    """
    if not text:
        return text
    idx = text.rfind(_SUMMARY_MARKER)
    if idx == -1:
        return text.strip()
    tail = text[idx + len(_SUMMARY_MARKER):].strip()
    return tail if len(tail) >= 40 else text.strip()


def build_summarizer_agent(model, skills_dir: str = "skills"):
    """Construct the ADK summarizer agent (loads the finding-summary skill)."""
    from google.adk.agents import LlmAgent

    from feline_monitor.tracing import after_model_callback, before_model_callback

    instruction = read_skills(["finding-summary"], skills_dir)
    return LlmAgent(
        name="summarizer_agent",
        model=model,
        description="Writes a scholarly plain-language summary of a relevant feline paper.",
        instruction=instruction,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
    )
