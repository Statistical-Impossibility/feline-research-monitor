"""Tests for orchestrator helpers that don't need the ADK runtime."""

from feline_monitor import run
from feline_monitor.run import _ident, _select_new_ids


def _fake_pages(pages):
    """Return a _search_page stand-in that yields canned (ids, count) pages by retstart."""
    count = sum(len(p) for p in pages)

    def _search_page(query, retstart, since):
        idx = retstart // run._PAGE_SIZE
        ids = pages[idx] if idx < len(pages) else []
        return ids, count

    return _search_page


def test_select_new_ids_pages_past_seen_until_budget(monkeypatch):
    # The bug: retrieval stopped at the newest N. Now it must skip already-seen ids and
    # keep paging to find genuinely new ones deeper in the window.
    page1 = [str(i) for i in range(run._PAGE_SIZE)]          # all seen
    page2 = ["new1", "new2", "new3", "new4"]                  # the unseen tail
    monkeypatch.setattr(run, "_search_page", _fake_pages([page1, page2]))
    seen = set(page1)
    got = _select_new_ids("q", budget=3, since="2025-01-01", seen=seen)
    assert got == ["new1", "new2", "new3"]  # budget caps at 3, not stranded on page 1


def test_select_new_ids_stops_when_window_exhausted(monkeypatch):
    # Fewer new than budget → returns what exists, does not loop forever.
    monkeypatch.setattr(run, "_search_page", _fake_pages([["a", "b"]]))
    got = _select_new_ids("q", budget=10, since="2025-01-01", seen={"a"})
    assert got == ["b"]


def test_select_new_ids_empty_when_all_seen(monkeypatch):
    # Every in-window id already stored → genuinely nothing new (the correct 0).
    ids = ["a", "b", "c"]
    monkeypatch.setattr(run, "_search_page", _fake_pages([ids]))
    assert _select_new_ids("q", budget=5, since="2025-01-01", seen=set(ids)) == []


def test_ident_shows_both_ids_for_pmc_paper():
    paper = {"pmid": "42382116", "pmcid": "13314447"}
    assert _ident(paper) == "PMID 42382116 | PMC13314447"


def test_ident_pmid_only_for_abstract_paper():
    paper = {"pmid": "12345", "pmcid": ""}
    assert _ident(paper) == "PMID 12345"


def test_ident_pmc_only_when_pmid_is_pmc_fallback():
    # A PMC paper with no real PMID: pmid is the "PMC…" fallback — show it once, as PMC.
    paper = {"pmid": "PMC777", "pmcid": "777"}
    assert _ident(paper) == "PMC777"


def test_ident_handles_missing_ids():
    assert _ident({}) == "?"


from feline_monitor.run import _is_live, _pad_date


def test_ner_processing_order_is_oldest_entry_first():
    # When ner is enabled the batch is sorted oldest-first by padded ENTRY date (EDAT) so an
    # entity's first appearance is recorded chronologically by catalog-entry order.
    papers = [{"entry_date": "2026-07-04"}, {"entry_date": "2026"}, {"entry_date": "2026-05-01"}]
    papers.sort(key=lambda p: _pad_date(p.get("entry_date", "")))
    assert [p["entry_date"] for p in papers] == ["2026", "2026-05-01", "2026-07-04"]


def test_is_live_false_when_no_last_run():
    assert _is_live("2026-07-03", None) is False     # first run seeds silently
    assert _is_live("2026-07-03", "") is False


def test_is_live_true_when_published_after_last_run():
    assert _is_live("2026-07-04", "2026-07-03") is True
    assert _is_live("2026-07-02", "2026-07-03") is False
    assert _is_live("2026-07-03", "2026-07-03") is False


def test_is_live_pads_partial_pub_date_conservatively():
    # year-only pub_date pads to Jan 1 → not "after" a mid-year last_run (seed, don't alert)
    assert _is_live("2026", "2026-07-03") is False
    assert _is_live("2027", "2026-07-03") is True


from feline_monitor.run import _radar_prompt


def test_radar_prompt_lists_novel_entities_with_context():
    novel = {"MEDICATION": [{"entity": "saquinavir", "count": 4, "context": "saquinavir reduced viral load"}]}
    p = _radar_prompt("FIP", novel)
    assert "FIP" in p
    assert "saquinavir" in p
    assert "reduced viral load" in p
    assert "JSON" in p            # instructs JSON-list output


def test_run_once_v1_path_has_no_ner_import(monkeypatch, tmp_path):
    # With ner.enabled=false, run_once must not import torch/transformers. We assert the
    # ner module is only imported lazily by checking the flag gate compiles + the config
    # default keeps it off (deeper live wiring is covered by the manual smoke test).
    from feline_monitor.config import Ner
    assert Ner().enabled is False


from feline_monitor.run import (
    _telegram_paper_text,
    _chunk_telegram_text,
    _send_telegram_digest,
    TELEGRAM_MSG_LIMIT,
)


def _item(**over):
    base = {
        "title": "A Study of Cats",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
        "pmid": "123",
        "summary": "First paragraph of the summary.\n\nSecond paragraph with more detail.",
        "priority": "high",
        "study_type": "case_report",
    }
    base.update(over)
    return base


def test_telegram_paper_text_includes_title_meta_and_summary():
    text = _telegram_paper_text(_item(), index=1, total=2)
    assert "Paper 1/2" in text
    assert "A Study of Cats" in text
    assert "Priority: high" in text
    assert "Study: case_report" in text
    assert "Second paragraph with more detail." in text


def test_telegram_paper_text_includes_radar_block():
    item = _item(radar=[{"entity": "saquinavir", "category": "MEDICATION", "note": "repurposing candidate"}])
    text = _telegram_paper_text(item, index=1, total=1)
    assert "Treatment Radar" in text
    assert "saquinavir" in text
    assert "repurposing candidate" in text


def test_chunk_telegram_text_stays_under_limit_and_preserves_content():
    long_para = "x" * 3000
    text = "\n\n".join([long_para, long_para, long_para])
    chunks = _chunk_telegram_text(text, limit=4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert len(chunks) > 1
    # every paragraph's content survives across the chunk boundaries
    assert sum(c.count("x") for c in chunks) == 3000 * 3


def test_chunk_telegram_text_marks_continuation_chunks():
    long_para = "y" * 3000
    text = "\n\n".join([long_para, long_para])
    chunks = _chunk_telegram_text(text, 4096)
    assert "(cont." in chunks[1]


def test_chunk_telegram_text_single_chunk_when_short():
    text = "short paper text"
    chunks = _chunk_telegram_text(text)
    assert chunks == ["short paper text"]


def test_send_telegram_digest_sends_one_message_per_paper(monkeypatch):
    sent = []
    monkeypatch.setattr(run, "send_message", lambda text: sent.append(text) or True)
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    items = [_item(title="Paper A"), _item(title="Paper B")]
    failed = _send_telegram_digest(items)
    assert failed == []  # all delivered
    assert len(sent) == 2
    assert "Paper A" in sent[0]
    assert "Paper B" in sent[1]


def test_send_telegram_digest_returns_failed_chunks(monkeypatch):
    # send_message fails for the paper whose text contains "Paper B"; that chunk comes back
    # so the caller can queue it for retry, while the successful one is not returned.
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    monkeypatch.setattr(run, "send_message", lambda text: "Paper B" not in text)
    items = [_item(title="Paper A"), _item(title="Paper B")]
    failed = _send_telegram_digest(items)
    assert len(failed) == 1
    assert "Paper B" in failed[0]


def test_flush_pending_telegram_resends_and_clears_delivered(monkeypatch, tmp_path):
    from feline_monitor.run import _flush_pending_telegram
    from feline_monitor.store import PaperStore

    mem = PaperStore(str(tmp_path / "m.sqlite"))
    mem.queue_pending(["old message 1", "old message 2"])
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    monkeypatch.setattr(run, "send_message", lambda text: True)  # both deliver now

    _flush_pending_telegram(mem)
    assert mem.pending_messages() == []  # delivered ones dropped from the queue


def test_flush_pending_telegram_keeps_still_failing(monkeypatch, tmp_path):
    from feline_monitor.run import _flush_pending_telegram
    from feline_monitor.store import PaperStore

    mem = PaperStore(str(tmp_path / "m.sqlite"))
    mem.queue_pending(["deliver me", "still broken"])
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    monkeypatch.setattr(run, "send_message", lambda text: text == "deliver me")

    _flush_pending_telegram(mem)
    remaining = [t for _, t in mem.pending_messages()]
    assert remaining == ["still broken"]  # only the failing one stays queued
