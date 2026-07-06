"""CLI entrypoint: run one monitoring cycle.

Usage:  python run.py
Reads config.yaml in the current directory and secrets from the environment (.env).
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main() -> None:
    from feline_monitor.logging_setup import log_fatal, setup_logging

    log_path = setup_logging()

    # Load secrets from .env (ENTREZ_EMAIL, OPENROUTER_API_KEY, GEMINI_API_KEY, Telegram)
    # into the environment BEFORE any model is built. We load by ABSOLUTE PATH next to this
    # file so it works from any working directory. (litellm bundles python-dotenv but only
    # auto-loads .env in -c/REPL mode — from `python run.py` its search starts in
    # site-packages and never finds app/.env, so we must load it ourselves.) Values go into
    # os.environ only — never logged: the redaction filter scrubs any that slip toward a sink.
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except Exception:  # pragma: no cover - dotenv ships with litellm; MissingApiKey guides if absent
        pass

    # Use the OS trust store (helps behind TLS-inspecting proxies; no-op otherwise).
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    from feline_monitor.run import run_once
    from feline_monitor.llm import MissingApiKey

    try:
        run_once()
    except MissingApiKey as exc:  # config problem, not a bug — one clean line, no traceback
        logging.getLogger("feline_monitor").error("%s", exc)
        print(f"ERROR: {exc}")
        print("Add the missing key to app/.env and re-run.")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - top-level guard: one clean line, detail to file
        line = log_fatal(exc, log_path)
        print(f"ERROR: {line}")
        print(f"Run failed. Full details: {log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
