"""Model construction and a synchronous agent runner (validated in the Phase-1 spike).

No model id is hardcoded: the provider + model id come from config, the endpoint/key
from the environment. ADK is model-agnostic via LiteLLM.
"""

import asyncio
import inspect
import logging
import os
import uuid

log = logging.getLogger("feline_monitor")


class MissingApiKey(RuntimeError):
    """A required provider API key is absent from the environment (.env).

    Raised early (before any network call) so the user gets one clean, actionable
    line ("set X in app/.env") instead of a wall of 401 AuthenticationError traces.
    """


def model_specs(model_cfg) -> list[tuple[str, str]]:
    """(provider, model_id) list to try in order — the chain if set, else the single model."""
    if getattr(model_cfg, "chain", None):
        return [(c["provider"], c["model_id"]) for c in model_cfg.chain]
    return [(model_cfg.provider, model_cfg.model_id)]


def build_model(provider: str, model_id: str):
    """Return an ADK model object/id for the configured provider. Never hardcodes an id.

    Per-provider API keys come from the environment (.env):
      openrouter -> OPENROUTER_API_KEY   gemini -> GEMINI_API_KEY (or GOOGLE_API_KEY)
      lmstudio / ollama -> none (local).
    """
    provider = provider.lower()
    if provider == "gemini":
        # ADK-native: just pass the model id string. google-genai reads GEMINI_API_KEY (or
        # GOOGLE_API_KEY) straight from the environment itself, so we do NOT copy one to the
        # other. (The old GEMINI->GOOGLE copy is what made both names appear set, triggering
        # google-genai's noisy "Both ... are set, using GOOGLE_API_KEY" warning.)
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise MissingApiKey(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to app/.env."
            )
        return model_id

    from google.adk.models.lite_llm import LiteLlm

    if provider == "lmstudio":
        return LiteLlm(
            model=f"openai/{model_id}",
            api_base=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        )
    if provider == "ollama":
        return LiteLlm(
            model=f"ollama/{model_id}",
            api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise MissingApiKey("OPENROUTER_API_KEY is not set. Add it to app/.env.")
        return LiteLlm(model=f"openrouter/{model_id}", api_key=key)
    raise ValueError(f"Unknown model provider: {provider!r}")


def unload_lmstudio_model(model_id: str) -> None:
    """Best-effort: ask LM Studio to unload a DROPPED model so its RAM frees for the next one.

    Fixes the run-long RAM climb when the chain fails over between local models: LM Studio keeps
    every model it has loaded resident (e.g. gemma-4-31b ~20GB + gpt-oss-20b ~12GB both held),
    so a machine with plenty of RAM still overflows. When we disable a model for the rest of the
    run we no longer need it loaded, so we tell LM Studio to eject it.

    LM Studio 0.4.0+ REST: ``POST {host}/api/v1/models/unload`` with ``{"instance_id": model_id}``.
    Non-fatal by design: an older LM Studio (no v1 API), a wrong id, or a server that's down just
    logs one DEBUG line and the run continues. Only the model_id + HTTP status are ever logged —
    never the URL/key (CLAUDE.md rule 0).
    """
    base = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")  # OpenAI-compat path -> host root for the native REST API
    url = f"{base}/api/v1/models/unload"
    key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    try:
        import httpx

        resp = httpx.post(
            url, json={"instance_id": model_id},
            headers={"Authorization": f"Bearer {key}"}, timeout=10.0,
        )
        if resp.status_code < 300:
            log.info("  LM Studio: unloaded %s (freed RAM)", model_id)
        else:
            log.debug("LM Studio unload %s -> HTTP %s (ignored)", model_id, resp.status_code)
    except Exception as exc:  # noqa: BLE001 - best-effort; never break the run, never log secrets
        log.debug("LM Studio unload %s failed (%s) — ignored", model_id, type(exc).__name__)


def _maybe_unload(label: str) -> None:
    """If a just-disabled chain label is an LM Studio model, eject it to reclaim its RAM."""
    provider, _, model_id = (label or "").partition("/")
    if provider == "lmstudio" and model_id:
        unload_lmstudio_model(model_id)


async def _run_async(agent, user_text: str) -> str:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    app, user, sess = "feline_monitor", "user", uuid.uuid4().hex
    svc = InMemorySessionService()
    maybe = svc.create_session(app_name=app, user_id=user, session_id=sess)
    if inspect.isawaitable(maybe):
        await maybe
    runner = Runner(agent=agent, app_name=app, session_service=svc)
    content = types.Content(role="user", parts=[types.Part(text=user_text)])

    final = ""
    async for event in runner.run_async(user_id=user, session_id=sess, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final = " ".join(texts)
    return final


def run_agent(agent, user_text: str, timeout_s: float | None = None) -> str:
    """Run one agent turn and return its final text (synchronous wrapper).

    `timeout_s` bounds the whole call: a stalled backend (e.g. a queued free-tier
    model) raises TimeoutError instead of hanging the run forever. The caller
    (`run_with_fallback`) treats that like any transport failure — disable + fail over.
    """
    async def _driver():
        coro = _run_async(agent, user_text)
        if timeout_s and timeout_s > 0:
            return await asyncio.wait_for(coro, timeout_s)
        return await coro

    try:
        return asyncio.run(_driver())
    except asyncio.TimeoutError as exc:  # 3.10: asyncio.TimeoutError != builtin; normalise
        raise TimeoutError(f"model call exceeded {timeout_s}s") from exc


def run_with_fallback(
    agents: list,
    user_text: str,
    labels: list[str] | None = None,
    validate=None,
    disabled: set | None = None,
    timeout_s: float | None = None,
    unload_local: bool = False,
) -> tuple[str, str]:
    """Run `agents` in order, skipping any already-`disabled`; return (text, label).

    Falls through to the next agent when one RAISES (transport error: crash /
    HTTP / connection / `timeout_s` exceeded) OR — if `validate` is given — when its returned text fails
    `validate(text)` (e.g. a weak free model that returns 200 with no valid JSON).

    A model that fails EITHER way is added to `disabled` (a set of chain indices) so
    later calls in the SAME run skip it entirely — no point re-hitting a rate-limited
    or consistently-junk free model on every remaining paper. Pass one shared set for
    the whole run (screeners + summarizers share it, since indices are chain positions).

    Returns (text, label) for the first agent that answers usefully. If some returned
    text but none validated, returns the LAST (text, label) best-effort (caller's own
    parse check catches it). If every tried agent raised, re-raise the last error. If
    all agents were already disabled, raise RuntimeError (caller's guard skips + retries).
    """
    disabled = disabled if disabled is not None else set()
    last_exc: BaseException | None = None
    last: tuple[str, str] | None = None
    tried_any = False
    for i, agent in enumerate(agents):
        if i in disabled:
            continue
        tried_any = True
        label = labels[i] if labels and i < len(labels) else f"model[{i}]"
        log.info("  trying %s ...", label)  # attempt line: if the next log is silence, this model hung
        try:
            text = run_agent(agent, user_text, timeout_s)
        except Exception as exc:  # noqa: BLE001 - transport/timeout failure → disable + try next
            last_exc = exc
            disabled.add(i)
            log.warning("model %s failed (%s) — disabled for this run", label, type(exc).__name__)
            if unload_local:
                _maybe_unload(label)  # free its RAM if it's a local LM Studio model
            continue
        last = (text, label)
        if validate is None or validate(text):
            return text, label
        disabled.add(i)
        log.warning("model %s returned unusable output — disabled for this run", label)
        if unload_local:
            _maybe_unload(label)  # free its RAM if it's a local LM Studio model
    if last is not None:
        return last
    if last_exc is not None:
        raise last_exc
    if not tried_any:
        raise RuntimeError("run_with_fallback: all models disabled this run")
    raise RuntimeError("run_with_fallback: no agents provided")
