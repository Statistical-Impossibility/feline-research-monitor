"""Tests for the token-safe Telegram delivery module."""

import logging

import httpx
import pytest

from feline_monitor import telegram


class _FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def raise_for_status(self) -> None:
        pass


def test_send_message_happy_path(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummytoken")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    assert telegram.send_message("hello") is True
    assert "sendMessage" in captured["url"]
    assert captured["json"]["chat_id"] == "12345"


def test_send_message_missing_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert telegram.send_message("hello") is False


def test_send_message_error_is_secret_safe(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET123:TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    def fake_post(url, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError(
            "Bad Request", request=request, response=response
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with caplog.at_level(logging.DEBUG):
        assert telegram.send_message("hello") is False

    assert "SECRET123" not in caplog.text
    assert "api.telegram.org/botSECRET123" not in caplog.text


def test_send_message_falls_back_to_plain_on_markdown_400(monkeypatch):
    # A 400 from the Markdown attempt (unbalanced entity) must trigger a plain-text resend.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummytoken")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json.get("parse_mode"))
        if json.get("parse_mode") == "Markdown":
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400, request=request,
                json={"ok": False, "description": "Bad Request: can't parse entities"},
            )
            raise httpx.HTTPStatusError("Bad Request", request=request, response=response)
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    assert telegram.send_message("stray * star") is True
    assert calls == ["Markdown", None]  # tried Markdown, then plain


def test_send_message_returns_false_when_plain_also_fails(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummytoken")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    def fake_post(url, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError("Bad Request", request=request, response=response)

    monkeypatch.setattr(httpx, "post", fake_post)
    assert telegram.send_message("anything") is False


def test_send_message_logs_telegram_description(monkeypatch, caplog):
    # Telegram's own `description` is not a secret and is the key diagnostic — it must be logged.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET123:TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    def fake_post(url, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            400, request=request,
            json={"ok": False, "description": "Bad Request: can't parse entities"},
        )
        raise httpx.HTTPStatusError("Bad Request", request=request, response=response)

    monkeypatch.setattr(httpx, "post", fake_post)
    with caplog.at_level(logging.DEBUG):
        telegram.send_message("bad * text")

    assert "can't parse entities" in caplog.text  # reason surfaced
    assert "SECRET123" not in caplog.text  # token still never logged
