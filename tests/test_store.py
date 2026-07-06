"""Tests for the SQLite paper store."""

from feline_monitor.store import PaperStore


def test_known_pmcids_returns_stored_pmc_ids(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    m.mark_seen([
        {"pmid": "1", "pmcid": "111", "title": "a", "abstract": "x", "pub_date": "2026", "url": "u"},
        {"pmid": "2", "pmcid": "222", "title": "b", "abstract": "y", "pub_date": "2026", "url": "u"},
        {"pmid": "3", "pmcid": "", "title": "c", "abstract": "z", "pub_date": "2026", "url": "u"},
    ])
    assert m.known_pmcids() == {"111", "222"}  # blank pmcid (abstract-only) excluded


def test_mark_seen_tolerates_missing_pmcid(tmp_path):
    # Papers from the abstract-only path carry no pmcid — must still store, keyed by pmid.
    m = PaperStore(str(tmp_path / "m.sqlite"))
    m.mark_seen([{"pmid": "9", "title": "t", "abstract": "a", "pub_date": "2026", "url": "u"}])
    assert m.new_pmids(["9"]) == []
    assert m.known_pmcids() == set()


def test_mark_seen_records_date_added(tmp_path):
    # date_added is stamped at store time (distinct from pub_date) so we can see when a
    # paper entered the DB. Auto-set even when the paper dict doesn't carry it.
    m = PaperStore(str(tmp_path / "m.sqlite"))
    m.mark_seen([{"pmid": "1", "pmcid": "1", "title": "t", "abstract": "a", "pub_date": "2026-01-02", "url": "u"}])
    pub, added = m._conn.execute("SELECT pub_date, date_added FROM papers WHERE pmid='1'").fetchone()
    assert pub == "2026-01-02"
    assert added and added.startswith("20")  # ISO timestamp was written


def test_pending_telegram_roundtrip(tmp_path):
    # Failed Telegram messages are queued verbatim, read back oldest-first, and dropped
    # once delivered — the retry-on-next-run safety net.
    m = PaperStore(str(tmp_path / "m.sqlite"))
    assert m.pending_messages() == []
    m.queue_pending(["msg one", "msg two"])
    pending = m.pending_messages()
    assert [t for _, t in pending] == ["msg one", "msg two"]
    m.delete_pending([pending[0][0]])  # delete only the first
    assert [t for _, t in m.pending_messages()] == ["msg two"]


def test_new_pmids_filters_seen(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    m.mark_seen([{"pmid": "1", "title": "a", "abstract": "x", "pub_date": "2026", "url": "u"}])
    assert m.new_pmids(["1", "2", "3"]) == ["2", "3"]


def test_mark_seen_is_idempotent(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    paper = {"pmid": "1", "title": "a", "abstract": "x", "pub_date": "2026", "url": "u"}
    m.mark_seen([paper])
    m.mark_seen([paper])
    assert m.new_pmids(["1"]) == []


def test_entities_stored_and_known_by_category(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    hits = {
        "MEDICATION": [{"entity": "gs-441524", "count": 5, "context": "c"}],
        "DISEASE": [{"entity": "fip", "count": 9, "context": "c"}],
    }
    m.add_entities("FIP", "1", hits)
    assert m.known_entities("FIP", "MEDICATION") == {"gs-441524"}
    assert m.known_entities("FIP", "DISEASE") == {"fip"}
    assert m.known_entities("FIP", "PROCEDURE") == set()
    assert m.known_entities("CKD", "MEDICATION") == set()   # scoped by profile name


def test_add_entities_is_idempotent(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    hits = {"MEDICATION": [{"entity": "gs-441524", "count": 5, "context": "c"}]}
    m.add_entities("FIP", "1", hits)
    m.add_entities("FIP", "1", hits)
    assert m.known_entities("FIP", "MEDICATION") == {"gs-441524"}


def test_record_search_warns_only_on_changed_search_under_a_name(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    assert m.record_search("FIP", "fpA", "[]", "[]") is False   # first ever → no warn
    assert m.record_search("FIP", "fpA", "[]", "[]") is False   # same search again → no warn
    assert m.record_search("FIP", "fpB", "[]", "[]") is True    # changed search → WARN
    assert m.record_search("FIP", "fpA", "[]", "[]") is False   # earlier search recurs → no re-warn


def test_last_run_roundtrip(tmp_path):
    m = PaperStore(str(tmp_path / "m.sqlite"))
    assert m.get_last_run("FIP") is None
    m.set_last_run("FIP", "2026-07-03")
    assert m.get_last_run("FIP") == "2026-07-03"
    m.set_last_run("FIP", "2026-07-04")            # upsert
    assert m.get_last_run("FIP") == "2026-07-04"
