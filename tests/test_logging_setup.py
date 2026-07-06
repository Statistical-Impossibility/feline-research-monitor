"""Tests for logging setup and clean error reporting (Bug #2)."""

import logging

from feline_monitor.logging_setup import log_fatal, redact, setup_logging, short_error

# Fake sensitive values built at runtime so no secret-shaped literal sits in the file.
_UID = "user_" + "".join(["2ab3", "leak", "123"])
_KEY = "sk-" + "A" * 20
_BEARER = "B" * 14
_BOT = "bot" + "1234567890" + ":" + "C" * 30


def test_redact_user_id():
    out = redact('{"error":{"code":429},"user_id":"' + _UID + '"}')
    assert _UID not in out
    assert "[REDACTED]" in out


def test_redact_keys_and_tokens():
    assert _KEY not in redact("Authorization: " + _KEY)
    assert "[REDACTED]" in redact("Bearer " + _BEARER)
    assert redact(_BOT).startswith("bot[REDACTED]")


def test_short_error_redacts_user_id():
    exc = RuntimeError('OpenrouterException {"user_id":"' + _UID + '"}')
    assert _UID not in short_error(exc)


def test_log_fatal_redacts_traceback(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")
    try:
        raise RuntimeError('rate limited {"user_id":"' + _UID + '"}')
    except RuntimeError as exc:
        log_fatal(exc, log_path)
    assert _UID not in log_path.read_text(encoding="utf-8")


def test_setup_logging_creates_log_file(tmp_path):
    log_path = setup_logging(str(tmp_path / "logs"))
    assert log_path.exists()
    assert log_path.suffix == ".log"


def test_setup_logging_quiets_noisy_loggers(tmp_path):
    setup_logging(str(tmp_path / "logs"))
    # Traceback-dumping frameworks are silenced hard (they embed user_id/secrets).
    assert logging.getLogger("LiteLLM").level == logging.CRITICAL
    assert logging.getLogger("google_adk").level == logging.CRITICAL
    # Merely noisy libraries are quieted to WARNING.
    assert logging.getLogger("httpx").level == logging.WARNING


def test_short_error_is_one_line_and_typed():
    out = short_error(ConnectionError("cannot connect\nsecond line"))
    assert out == "ConnectionError: cannot connect"
    assert "\n" not in out


def test_short_error_truncates():
    out = short_error(ValueError("x" * 500))
    assert len(out) <= 200


def test_log_fatal_writes_traceback_and_returns_short_line(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")
    try:
        raise ConnectionError("refused")
    except ConnectionError as exc:
        line = log_fatal(exc, log_path)
    assert line == "ConnectionError: refused"
    contents = log_path.read_text(encoding="utf-8")
    assert "Traceback" in contents
    assert "ConnectionError: refused" in contents
