"""ADK model-callback tracing: make each agent's model call visible in the log.

The framework's own trace is silenced to CRITICAL (it dumps request/response bodies
that can carry provider `user_id`s and API keys — a secret-leak vector, CLAUDE.md rule 0).
So instead of trusting ADK's logger, we attach our OWN before/after model callbacks that
log a SHORT, redacted line at each call boundary: which agent, which model, that the call
started, its finish reason, and token counts. Never the prompt or the response text
(that would put article content and secrets back in the log). This is the agent tracing
used to debug the pipeline — you can see exactly where a run is (and where it stalls).

Callbacks return None so the model call proceeds normally. They never raise: a broken
trace line must not break the run.
"""

import logging

from feline_monitor.logging_setup import redact

log = logging.getLogger("feline_monitor")


def before_model_callback(callback_context, llm_request):
    """Log 'agent -> calling <model>' just before the model runs (a hang stops right here)."""
    try:
        model = getattr(llm_request, "model", None) or "?"
        n_msgs = len(getattr(llm_request, "contents", None) or [])
        log.info("  trace: %s -> calling %s (%d msg)", callback_context.agent_name, model, n_msgs)
    except Exception:  # noqa: BLE001 - tracing must never break the run
        pass
    return None


def after_model_callback(callback_context, llm_response):
    """Log the model's finish reason + token usage (or a redacted error) after it returns."""
    try:
        err = getattr(llm_response, "error_message", None)
        if err:
            log.warning("  trace: %s <- error: %s", callback_context.agent_name, redact(str(err)))
            return None
        finish = getattr(llm_response, "finish_reason", None)
        usage = getattr(llm_response, "usage_metadata", None)
        toks = ""
        if usage is not None:
            prompt_t = getattr(usage, "prompt_token_count", None)
            out_t = getattr(usage, "candidates_token_count", None)
            toks = f" tokens={prompt_t}->{out_t}"
        log.info("  trace: %s <- finish=%s%s", callback_context.agent_name, finish, toks)
    except Exception:  # noqa: BLE001 - tracing must never break the run
        pass
    return None
