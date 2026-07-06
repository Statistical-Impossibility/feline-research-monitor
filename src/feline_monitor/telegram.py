"""Deliver digest messages to Telegram (token-safe: never logs URL or token)."""

import logging
import os

import httpx

log = logging.getLogger(__name__)


def send_message(text: str) -> bool:
    """Send a Markdown message via the Telegram Bot API.

    Returns True on success, False if not configured or on HTTP error.
    The bot token lives inside the request URL, so the URL, token, and
    raw response body are never logged.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("telegram not configured")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # First attempt with Markdown formatting. Telegram's legacy Markdown parser rejects the
    # whole message (HTTP 400) if the free text carries an unbalanced *, _, `, [ or ] — which
    # LLM-written summaries and paper titles routinely do. On that failure we resend the exact
    # same text as plain (no parse_mode): delivery beats formatting.
    if _post(url, chat_id, text, "Markdown"):
        return True
    return _post(url, chat_id, text, None)


def _post(url: str, chat_id: str, text: str, parse_mode: str | None) -> bool:
    """POST one sendMessage. Returns True on success. Never logs URL/token/body."""
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = httpx.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Telegram's own `description` is a human-readable reason (e.g. "can't parse entities")
        # and is NOT a secret — safe to log. The token lives in the URL, never in the body.
        description = _telegram_description(getattr(exc, "response", None))
        log.error(
            "telegram send failed: %s status=%s mode=%s reason=%s",
            type(exc).__name__, status, parse_mode or "plain", description,
        )
        return False
    return True


def _telegram_description(response) -> str | None:
    """Extract Telegram's `description` field from an error response, if any."""
    if response is None:
        return None
    try:
        return response.json().get("description")
    except Exception:
        return None
