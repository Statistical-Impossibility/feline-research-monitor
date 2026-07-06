"""Logging setup: short console + per-run logs/<ts>.log, with hard secret redaction.

Goals:
- Normal run prints short, readable step lines (no third-party spam, no article text).
- NOTHING sensitive ever reaches a log sink: API keys, bearer tokens, Telegram bot
  tokens, and provider `user_id`s are scrubbed from EVERY record by a redaction filter
  (CLAUDE.md rule 0). Frameworks that dump full tracebacks containing such data
  (google-adk, litellm) are silenced to CRITICAL; our own code reports errors cleanly.
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Frameworks that log full tracebacks at ERROR — those tracebacks embed provider
# error bodies (with user_id) and request detail. Silence to CRITICAL; our per-paper
# guard already surfaces failures as one clean line.
_SILENCED = ("google_adk", "google.adk", "litellm", "LiteLLM")
# Merely noisy libraries — INFO/retry chatter and (at DEBUG) request payloads.
# google_genai logs "AFC is enabled with max remote calls: 10" at INFO on every call —
# pure clutter for us (we don't use automatic function calling), so raise it to WARNING.
_NOISY = ("httpx", "httpcore", "openai", "openai._base_client", "google_genai", "google.genai")

# Redaction patterns — applied to every log record and to error strings/tracebacks.
_REDACTIONS = [
    (re.compile(r'("?user_id"?\s*[:=]\s*"?)[^"\',}\s]+', re.I), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}"), "[REDACTED_KEY]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\bbot\d{5,}:[A-Za-z0-9_-]{20,}", re.I), "bot[REDACTED]"),
    (re.compile(r'("?api[_-]?key"?\s*[:=]\s*"?)[A-Za-z0-9._\-]{6,}', re.I), r"\1[REDACTED]"),
]


def redact(text: str) -> str:
    """Scrub keys / bearer tokens / bot tokens / user_id from any string."""
    if not text:
        return text
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class _RedactionFilter(logging.Filter):
    """Redacts every log record's message before it reaches any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - never let logging itself crash
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def _quiet_windows_asyncio_teardown() -> None:
    """Silence the benign SSL/Proactor teardown noise seen at process exit on Windows.

    We call ``asyncio.run()`` once per model call, so each call opens and closes its own
    event loop. litellm's async httpx client isn't awaited-closed, so when a later loop
    is gone its transport ``__del__`` fires "Event loop is closed" / "'NoneType' has no
    attribute 'send'" AND asyncio logs "Fatal error on SSL transport". Both happen AFTER
    the digest is written and change nothing — but "fatal error" alarms users (and juries).
    We can't switch off the Proactor loop (the MCP client needs subprocess support), so we
    swallow just these two known-benign shutdown exceptions and mute the asyncio logger.
    """
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    previous = sys.unraisablehook

    def _hook(unraisable):  # pragma: no cover - only fires during interpreter teardown
        exc = unraisable.exc_value
        message = str(exc) if exc else ""
        if isinstance(exc, (RuntimeError, AttributeError)) and (
            "Event loop is closed" in message or "has no attribute 'send'" in message
        ):
            return  # benign asyncio/SSL teardown on Windows — drop it
        previous(unraisable)

    sys.unraisablehook = _hook


def setup_logging(log_dir: str = "logs", console_level: int = logging.INFO) -> Path:
    """Configure root logging (INFO file + console, both redacted). Returns the log path."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    redaction = _RedactionFilter()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):  # replace any basicConfig handlers
        root.removeHandler(handler)

    # File = verbose DEBUG detail (stage-by-stage, per-chunk NER, etc.) so a failed run can be
    # reloaded and read "where it stumbled". Console stays short (INFO). This is safe because
    # only OUR package logs at DEBUG (set below); third-party payload loggers (openai/httpx)
    # are pinned at WARNING, so raising the file level cannot leak article text or secrets.
    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    file_h.addFilter(redaction)
    root.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setLevel(console_level)
    console_h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    console_h.addFilter(redaction)
    root.addHandler(console_h)

    for name in _SILENCED:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    # OUR package logs DEBUG -> reaches the DEBUG file handler but not the INFO console.
    # (Third-party loggers above stay WARNING+, so nothing sensitive is emitted at DEBUG.)
    logging.getLogger("feline_monitor").setLevel(logging.DEBUG)
    # Native-crash safety net: a segfault / OOM abort (e.g. torch running out of RAM) kills the
    # interpreter below Python, so no traceback and no top-level handler fire — the log just
    # stops. faulthandler dumps a C-level stack into the SAME log file when that happens, so the
    # next silent death is diagnosable instead of a blank cutoff.
    try:  # pragma: no cover - defensive
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable(file=file_h.stream, all_threads=True)
    except Exception:  # pragma: no cover
        pass
    # NER libs (v2): quiet HF Hub warnings + the "Token indices ... > 512" tokenizer notice
    # and the model-loading progress bar. Set env before transformers is imported.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("huggingface_hub", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:  # pragma: no cover - optional dependency
        import litellm

        litellm.suppress_debug_info = True
    except Exception:  # pragma: no cover
        pass

    _quiet_windows_asyncio_teardown()
    return log_path


def blank_line() -> None:
    """Write a true blank line to console + the run's file log (no timestamp/level prefix).

    Used to separate each paper's block in the console/log stream, which otherwise reads
    as one unbroken wall of text across multiple papers.
    """
    for handler in logging.getLogger().handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None:
            stream.write("\n")
            stream.flush()


def short_error(exc: BaseException) -> str:
    """One-line, redacted summary of an exception (type + first message line)."""
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    return redact(f"{type(exc).__name__}: {first_line}").strip()[:200]


def log_fatal(exc: BaseException, log_path: Path) -> str:
    """Append the REDACTED full traceback to the log file; return the short console line."""
    import traceback

    tb = "".join(traceback.format_exception(exc))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + redact(tb))
    return short_error(exc)
