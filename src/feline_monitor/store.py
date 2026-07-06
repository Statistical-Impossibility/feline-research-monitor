"""SQLite-backed paper store: track seen papers and known interventions.

Lets the monitor skip papers already processed and detect the first-ever
mention of an intervention for a given condition (de-duplication across runs).

Named `PaperStore` (not "Memory") to avoid confusion with LLM/agent memory —
this is plain persistence, not conversational state.
"""

import sqlite3
from datetime import datetime


class PaperStore:
    """Persistent store of seen papers and recorded interventions."""

    def __init__(self, path: str) -> None:
        """Open (or create) the SQLite database and ensure tables exist."""
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS papers ("
            "pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT, "
            "pub_date TEXT, url TEXT, pmcid TEXT, date_added TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entities ("
            "name TEXT, pmid TEXT, category TEXT, entity TEXT, count INTEGER, "
            "PRIMARY KEY (name, pmid, category, entity))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS profile_searches ("
            "name TEXT, fingerprint TEXT, must TEXT, mesh TEXT, "
            "first_seen TEXT, last_seen TEXT, PRIMARY KEY (name, fingerprint))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS profile_state (name TEXT PRIMARY KEY, last_run TEXT)"
        )
        # Telegram messages that failed to send this run. Held verbatim so the next run can
        # resend them first (the built text, not the paper row — no LLM regeneration needed).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_telegram ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created TEXT NOT NULL)"
        )
        # `date_added` was added after some DBs already existed — add it in place so an
        # existing store keeps its rows instead of being wiped.
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(papers)")}
        if "date_added" not in cols:
            self._conn.execute("ALTER TABLE papers ADD COLUMN date_added TEXT")
        # `entry_date` = the paper's Entrez/index date (EDAT) — when it entered the catalog.
        # This is the seeding-vs-live clock (NOT pub_date). Added in place for existing DBs.
        if "entry_date" not in cols:
            self._conn.execute("ALTER TABLE papers ADD COLUMN entry_date TEXT")
        self._conn.commit()

    def new_pmids(self, pmids: list[str]) -> list[str]:
        """Return the subset of pmids not yet stored, preserving input order."""
        rows = self._conn.execute("SELECT pmid FROM papers").fetchall()
        seen = {row[0] for row in rows}
        return [pmid for pmid in pmids if pmid not in seen]

    def known_pmcids(self) -> set[str]:
        """Return every PMC id already stored — the pre-fetch dedup key for backfill.

        Lets the backfill sweep skip already-seen papers using cheap esearch ids,
        without efetching the full XML just to discover it is a duplicate.
        """
        rows = self._conn.execute(
            "SELECT pmcid FROM papers WHERE pmcid IS NOT NULL AND pmcid != ''"
        ).fetchall()
        return {row[0] for row in rows}

    def mark_seen(self, papers: list[dict]) -> None:
        """Insert Paper dicts into the papers table (ignoring duplicates).

        `date_added` records when this run first stored the paper (local ISO timestamp) —
        distinct from `pub_date` (when the paper was published).
        """
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            {
                "pmid": p["pmid"],
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "pub_date": p.get("pub_date", ""),
                "url": p.get("url", ""),
                "pmcid": p.get("pmcid", ""),
                "date_added": now,
                "entry_date": p.get("entry_date", ""),
            }
            for p in papers
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO papers "
            "(pmid, title, abstract, pub_date, url, pmcid, date_added, entry_date) "
            "VALUES (:pmid, :title, :abstract, :pub_date, :url, :pmcid, :date_added, :entry_date)",
            rows,
        )
        self._conn.commit()

    def queue_pending(self, texts: list[str]) -> None:
        """Store Telegram message texts that failed to send, for retry on the next run."""
        now = datetime.now().isoformat(timespec="seconds")
        self._conn.executemany(
            "INSERT INTO pending_telegram (text, created) VALUES (?, ?)",
            [(t, now) for t in texts],
        )
        self._conn.commit()

    def pending_messages(self) -> list[tuple[int, str]]:
        """Every undelivered Telegram message, oldest first, as (id, text)."""
        return self._conn.execute(
            "SELECT id, text FROM pending_telegram ORDER BY id"
        ).fetchall()

    def delete_pending(self, ids: list[int]) -> None:
        """Drop the given pending messages once they have been delivered."""
        self._conn.executemany(
            "DELETE FROM pending_telegram WHERE id = ?", [(i,) for i in ids]
        )
        self._conn.commit()

    def known_entities(self, name: str, category: str) -> set[str]:
        """Every entity seen for this profile name + category (the novelty basket)."""
        rows = self._conn.execute(
            "SELECT DISTINCT entity FROM entities WHERE name = ? AND category = ?",
            (name, category),
        ).fetchall()
        return {row[0] for row in rows}

    def add_entities(self, name: str, pmid: str, hits: dict) -> None:
        """Store a paper's entity profile. `hits` = {category: [{entity, count, ...}]}."""
        rows = [
            (name, pmid, category, h["entity"], int(h.get("count", 1)))
            for category, items in hits.items()
            for h in items
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO entities (name, pmid, category, entity, count) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def record_search(self, name: str, fingerprint: str, must: str, mesh: str) -> bool:
        """Log this run's search under the profile name. Return True iff it should warn.

        Append-only: each distinct search is one row. Warn only when a NEW search variant
        appears under a name that already had other variants (never re-warn on a recurrence).
        """
        now = datetime.now().isoformat(timespec="seconds")
        existing = {
            row[0]
            for row in self._conn.execute(
                "SELECT fingerprint FROM profile_searches WHERE name = ?", (name,)
            )
        }
        if fingerprint in existing:
            self._conn.execute(
                "UPDATE profile_searches SET last_seen = ? WHERE name = ? AND fingerprint = ?",
                (now, name, fingerprint),
            )
            self._conn.commit()
            return False
        self._conn.execute(
            "INSERT INTO profile_searches (name, fingerprint, must, mesh, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, fingerprint, must, mesh, now, now),
        )
        self._conn.commit()
        return len(existing) > 0

    def get_last_run(self, name: str) -> str | None:
        """The date of the last completed run for this profile, or None if never run."""
        row = self._conn.execute(
            "SELECT last_run FROM profile_state WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_last_run(self, name: str, day: str) -> None:
        """Upsert the profile's last-run date (used as the seeding-vs-live cutoff)."""
        self._conn.execute(
            "INSERT INTO profile_state (name, last_run) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET last_run = excluded.last_run",
            (name, day),
        )
        self._conn.commit()
