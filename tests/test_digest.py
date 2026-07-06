"""Tests for the deterministic Markdown digest renderer."""

from datetime import date

from feline_monitor.digest import render_markdown, write_digest

FULL_ITEM = {
    "title": "Feline FIP treatment outcomes",
    "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
    "pmid": "12345",
    "summary": "A randomized trial of GS-441524 in cats with FIP.",
    "study_type": "trial",
    "priority": "high",
    "interventions": ["GS-441524", "remdesivir"],
}


def test_render_full_item():
    md = render_markdown([FULL_ITEM])
    assert FULL_ITEM["title"] in md
    assert FULL_ITEM["url"] in md
    assert FULL_ITEM["summary"] in md
    assert "Priority: high" in md
    assert f"# Feline Research Digest — {date.today().isoformat()}" in md


def test_render_item_missing_optional_fields():
    item = {
        "title": "Minimal paper",
        "url": "https://example.com/1",
        "pmid": "1",
        "summary": "Just a summary.",
    }
    md = render_markdown([item])
    assert item["title"] in md
    assert item["summary"] in md


def test_render_empty():
    md = render_markdown([])
    assert "No new papers" in md


def test_write_digest(tmp_path):
    md = render_markdown([FULL_ITEM])
    path = write_digest(md, str(tmp_path))
    import os

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == md


from feline_monitor.digest import render_markdown


def test_render_includes_radar_block():
    items = [{
        "title": "T", "url": "u", "pmid": "1", "summary": "S",
        "radar": [{"entity": "saquinavir", "category": "MEDICATION", "note": "antiviral, ↓ viral load"}],
    }]
    md = render_markdown(items)
    assert "**Treatment Radar**" in md
    assert "saquinavir" in md
    assert "antiviral, ↓ viral load" in md


def test_render_omits_radar_block_when_absent():
    items = [{"title": "T", "url": "u", "pmid": "1", "summary": "S"}]
    assert "Treatment Radar" not in render_markdown(items)
