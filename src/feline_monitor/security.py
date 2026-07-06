"""Security guardrails: network allowlist and filesystem sandboxing.

The agent must only reach a small set of known hosts and only write inside its
own output directory. These helpers are deliberately simple and dependency-free.
"""

from pathlib import Path
from urllib.parse import urlparse

# Fixed hosts the system is ever allowed to contact. The LLM endpoint host is
# dynamic (local LM Studio / OpenRouter / Gemini), so it is passed as `extra`.
ALLOWED_HOSTS = {
    "eutils.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "api.telegram.org",
}


def allowed_host(url: str, extra: set[str] | None = None) -> bool:
    """True only if the URL's host is in the allowlist (plus any `extra` hosts)."""
    host = urlparse(url).hostname or ""
    return host in (ALLOWED_HOSTS | (extra or set()))


def safe_path(path: str, base: str) -> Path:
    """Resolve `path` under `base`, refusing anything that escapes `base`."""
    base_p = Path(base).resolve()
    target = (base_p / path).resolve()
    if target != base_p and base_p not in target.parents:
        raise ValueError("path escapes the allowed base directory")
    return target
