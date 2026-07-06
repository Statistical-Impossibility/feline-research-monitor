import pytest

from feline_monitor.security import allowed_host, safe_path


def test_allowed_host_accepts_known_and_rejects_unknown():
    assert allowed_host("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    assert allowed_host("https://api.telegram.org/botX/sendMessage")
    assert not allowed_host("https://evil.example.com/steal")


def test_allowed_host_extra_for_model_endpoint():
    assert allowed_host("http://localhost:1234/v1", extra={"localhost"})
    assert not allowed_host("http://localhost:1234/v1")


def test_safe_path_allows_within_base(tmp_path):
    p = safe_path("digests/today.md", str(tmp_path))
    assert str(tmp_path) in str(p)


def test_safe_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_path("../../etc/passwd", str(tmp_path))
