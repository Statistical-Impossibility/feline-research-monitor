"""One full monitoring cycle: retrieve -> dedup -> screen -> summarize -> deliver.

Emits one log line per step (lightweight tracing, no enterprise harness).
"""

import gc
import logging
import time

from feline_monitor.config import load_config
from feline_monitor.pubmed.query import build_query, groups_from_profile, query_fingerprint
from feline_monitor.pubmed import fetch
from feline_monitor.store import PaperStore
from feline_monitor import llm
from feline_monitor.logging_setup import short_error, blank_line
from feline_monitor.agents.screener import build_screening_agent, parse_screening_json
from feline_monitor.agents.summarizer import build_summarizer_agent, strip_reasoning
from feline_monitor.digest import render_markdown, write_digest
from feline_monitor.telegram import send_message

log = logging.getLogger("feline_monitor")

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# esearch page size for the backfill sweep (ids only — cheap). One page usually covers
# the whole window; larger windows just take more pages.
_PAGE_SIZE = 200


def _pad_date(d: str) -> str:
    """Normalize a possibly-partial pub date ('2026' / '2026-06') to ISO 'YYYY-MM-DD'.

    Missing month/day pad to '01' (conservative: a year-only paper sorts to Jan 1, so it
    won't count as 'new' against a mid-year cutoff).
    """
    parts = (d or "").strip().split("-")
    while len(parts) < 3:
        parts.append("01")
    return "-".join(parts[:3])


def _is_live(entry_date: str, last_run: str | None) -> bool:
    """True if the paper ENTERED the catalog after the profile's last run (→ radar evaluates).

    Uses the Entrez/index date (EDAT), not publication date: "new" means "appeared since we last
    looked", so a paper published long ago but indexed today is correctly treated as new. No
    last_run (first run) → False: seed silently, no alerts.
    """
    if not last_run:
        return False
    return _pad_date(entry_date) > _pad_date(last_run)


def _search_page(query: str, retstart: int, since: str | None) -> tuple[list[str], int]:
    """One page of PMC ids + total count, via MCP with a direct-fetch fallback."""
    try:
        from feline_monitor.pubmed.mcp_client import pubmed_search_page

        page = pubmed_search_page(query, retstart, _PAGE_SIZE, since)
        return page.get("ids", []), int(page.get("count", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - resilience: never let MCP kill the run
        log.warning("MCP page failed (%s); using direct fetch", type(exc).__name__)
        return fetch.fetch_pmc_ids_page(query, retstart, _PAGE_SIZE, since)


def _fetch_selected(pmc_ids: list[str]) -> list[dict]:
    """Fetch full text for the chosen (already deduped) PMC ids, via MCP or direct."""
    try:
        from feline_monitor.pubmed.mcp_client import pubmed_fetch

        return pubmed_fetch(pmc_ids)
    except Exception as exc:  # noqa: BLE001
        log.warning("MCP fetch failed (%s); using direct fetch", type(exc).__name__)
        return fetch.fetch_papers(pmc_ids)


def _select_new_ids(query: str, budget: int, since: str | None, seen: set[str]) -> list[str]:
    """Page the whole `since` window newest-first, returning up to `budget` unseen PMC ids.

    Fixes the backfill bug: retrieval no longer stops at the newest `max_per_run` ids (which,
    once stored, made every re-run report "nothing new" while older in-window papers were never
    pulled). We sweep page by page, skipping already-seen ids, until the budget is filled OR the
    window is exhausted. `budget` is now a per-run PROCESSING cap, not a fetch ceiling.
    """
    selected: list[str] = []
    retstart = 0
    swept = 0
    while len(selected) < budget:
        ids, count = _search_page(query, retstart, since)
        if not ids:
            break
        swept += len(ids)
        for pmc_id in ids:
            if pmc_id not in seen:
                selected.append(pmc_id)
                if len(selected) >= budget:
                    break
        retstart += _PAGE_SIZE
        if retstart >= count:
            break
    log.info("backfill: swept %d id(s), %d new, budget %d", swept, len(selected), budget)
    return selected


def _retrieve(
    query: str, budget: int, include_abstracts: bool, since: str | None, seen: set[str]
) -> list[dict]:
    """Select unseen PMC ids across the window, fetch only those, + optional abstracts."""
    if since:
        new_ids = _select_new_ids(query, budget, since, seen)
        papers = _fetch_selected(new_ids)
        log.info("retrieval: %d new PMC paper(s) fetched", len(papers))
    else:
        # No date floor = "latest N sample" (no backfill: the window would be all of PubMed).
        papers = _fetch_selected(fetch.fetch_pmc_ids(query, budget, since))
        log.info("retrieval: %d latest PMC candidate(s)", len(papers))
    if include_abstracts:
        have = {p["pmid"] for p in papers}
        extra_pmids = [x for x in fetch.fetch_pmids(query, budget, since) if x not in have]
        extra = fetch.fetch_abstract_papers(extra_pmids)
        log.info("retrieval: +%d abstract-only (paywalled) papers", len(extra))
        papers += extra
    return papers


def _ident(paper: dict) -> str:
    """Verifiable ids for logs — whichever are present, e.g. 'PMID 42382116 | PMC13314447'.

    PMID -> pubmed.ncbi.nlm.nih.gov/<pmid>/ ; PMC -> pmc.ncbi.nlm.nih.gov/articles/PMC<pmcid>/.
    A PMC full-text paper carries both; a paywalled abstract-only hit has PMID only.
    (When a PMC paper has no PMID, `pmid` is the 'PMC...' fallback — shown once, as PMC.)
    ASCII-only separator: a non-ASCII char can crash a Windows cp1252 console handler.
    """
    pmid = str(paper.get("pmid", "") or "")
    pmcid = str(paper.get("pmcid", "") or "")
    parts = []
    if pmid and not pmid.upper().startswith("PMC"):
        parts.append(f"PMID {pmid}")
    if pmcid:
        parts.append(f"PMC{pmcid}")
    return " | ".join(parts) if parts else (pmid or "?")


def _access_note(paper: dict) -> str:
    return "full text" if paper.get("has_full_text") else "ABSTRACT ONLY (full text not accessible)"


def _screen_prompt(profile_name: str, keywords: list[str], paper: dict, text: str) -> str:
    return (
        f"Condition profile: {profile_name}\n"
        f"Keywords: {', '.join(keywords or [])}\n\n"
        f"Paper (source: {_access_note(paper)}):\n"
        f"Title: {paper['title']}\n"
        f"Article type: {paper.get('article_type') or 'unknown'}\n\n"
        f"{text}\n\n"
        "Return ONLY the JSON verdict."
    )


def _summary_prompt(profile_name: str, paper: dict, text: str) -> str:
    return (
        f"Condition being monitored: {profile_name}\n\n"
        f"Source: {_access_note(paper)}\n"
        f"Title: {paper['title']}\nLink: {paper['url']}\n\n"
        f"{text}\n\n"
        "Write the summary."
    )


def _radar_prompt(condition: str, novel: dict) -> str:
    """Prompt the radar agent with the first-seen candidates + their context sentences."""
    lines = [
        f"Condition: {condition}",
        "These entities appear for the FIRST time in this condition's monitored literature.",
        "Confirm which are genuine, central interventions studied/used in this paper, and "
        "for each real one give one sentence on why it matters.",
        'Return ONLY a JSON list: [{"entity":..,"category":..,"note":..}].',
        "",
    ]
    for category, hits in novel.items():
        for h in hits:
            lines.append(f"- [{category}] {h['entity']} — context: {h.get('context', '')}")
    return "\n".join(lines)


TELEGRAM_MSG_LIMIT = 4096  # Telegram Bot API hard cap per sendMessage call
TELEGRAM_INTER_MSG_DELAY = 1.0  # seconds between messages, avoids Telegram flood limit


def _telegram_paper_text(item: dict, index: int, total: int) -> str:
    """Full digest content for one paper, as a single Telegram message (pre-chunking)."""
    header = [f"*Paper {index}/{total}*", f"[{item['title']}]({item['url']})"]
    meta_parts = []
    if item.get("priority"):
        meta_parts.append(f"Priority: {item['priority']}")
    if item.get("study_type"):
        meta_parts.append(f"Study: {item['study_type']}")
    if meta_parts:
        header.append(" · ".join(meta_parts))
    blocks = ["\n".join(header), item["summary"]]

    interventions = item.get("interventions")
    if interventions:
        blocks.append(f"New interventions: {', '.join(interventions)}")

    radar = item.get("radar")
    if radar:
        radar_lines = ["*Treatment Radar*"]
        for r in radar:
            cat = f" ({r['category']})" if r.get("category") else ""
            radar_lines.append(f"- {r['entity']}{cat}: {r['note']}")
        blocks.append("\n".join(radar_lines))

    return "\n\n".join(blocks)


def _chunk_telegram_text(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Split ``text`` into Telegram-sized chunks on paragraph (blank-line) boundaries.

    A single paragraph longer than ``limit`` is hard-split as a last resort.
    Continuation chunks get a leading marker so a multi-message paper reads clearly.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush():
        if current:
            chunks.append("\n\n".join(current))

    for para in paragraphs:
        if len(para) > limit:
            _flush()
            current, current_len = [], 0
            for i in range(0, len(para), limit):
                chunks.append(para[i : i + limit])
            continue

        added_len = len(para) + (2 if current else 0)
        if current and current_len + added_len > limit:
            _flush()
            current, current_len = [], 0

        current.append(para)
        current_len += len(para) + (2 if len(current) > 1 else 0)

    _flush()

    if len(chunks) > 1:
        chunks = [chunks[0]] + [f"*(cont.)*\n\n{c}" for c in chunks[1:]]
    return chunks


def _send_telegram_digest(items: list[dict]) -> list[str]:
    """Send the digest to Telegram, one message per paper (chunked if too long).

    Returns the list of message chunks that failed to send, so the caller can queue
    them for retry on the next run. An empty list means everything was delivered.
    """
    total = len(items)
    failed: list[str] = []
    for i, item in enumerate(items, start=1):
        text = _telegram_paper_text(item, i, total)
        for chunk in _chunk_telegram_text(text):
            if not send_message(chunk):
                failed.append(chunk)
            time.sleep(TELEGRAM_INTER_MSG_DELAY)
    return failed


def _flush_pending_telegram(mem: PaperStore) -> None:
    """Resend Telegram messages that a previous run failed to deliver, before the new run.

    Delivered ones are dropped from the queue; ones that fail again stay queued and are
    reported. These are not counted as new papers — they are last run's undelivered digest.
    """
    pending = mem.pending_messages()
    if not pending:
        return
    log.info("telegram: %d message(s) from a previous run were never delivered — retrying", len(pending))
    delivered: list[int] = []
    for pid, text in pending:
        if send_message(text):
            delivered.append(pid)
        time.sleep(TELEGRAM_INTER_MSG_DELAY)
    mem.delete_pending(delivered)
    if delivered:
        log.info("telegram: %d previously-unsent message(s) now delivered", len(delivered))
    still = len(pending) - len(delivered)
    if still:
        log.error("telegram: %d previously-unsent message(s) STILL failing", still)


def run_once(config_path: str = "config.yaml", db_path: str = "frm.sqlite") -> None:
    cfg = load_config(config_path)
    psrc = cfg.sources.pubmed
    cap = psrc.get("full_text_char_cap", 50000)
    since = psrc.get("since")  # 'YYYY-MM-DD' date floor, or None
    budget = psrc["max_per_run"]  # per-run PROCESSING budget (new papers pulled this run)
    must_groups, mesh_terms = groups_from_profile(cfg.profile)
    query = build_query(must_groups, mesh_terms)
    if since:
        log.info("profile=%r — backfill since %s, up to %d new/run", cfg.profile.name, since, budget)
    else:
        log.info("profile=%r — latest %d", cfg.profile.name, budget)

    mem = PaperStore(db_path)
    if cfg.delivery.telegram:
        _flush_pending_telegram(mem)
    candidates = _retrieve(
        query, budget, psrc.get("include_paywalled_abstracts", False), since, mem.known_pmcids()
    )
    # Attach each paper's Entrez date (EDAT) — the catalog-entry date that drives seeding-vs-live
    # and the within-batch order (NOT pub_date). Fall back to pub_date if EDAT is unavailable.
    edat = fetch.entrez_dates([c["pmid"] for c in candidates])
    for c in candidates:
        c["entry_date"] = edat.get(c["pmid"]) or _pad_date(c.get("pub_date", ""))
    # Backfill already deduped by PMC id pre-fetch; new_pmids stays as the final safety net
    # (covers the abstract-only PubMed path, which has no PMC id).
    new_ids = set(mem.new_pmids([c["pmid"] for c in candidates]))
    new_papers = [c for c in candidates if c["pmid"] in new_ids]
    log.info("fetched=%d new=%d", len(candidates), len(new_papers))
    if not new_papers:
        log.info("nothing new — done")
        return

    specs = llm.model_specs(cfg.model)
    if not any(m for _, m in specs):
        raise ValueError(
            "No model configured. Set model.chain (list of {provider, model_id}) "
            "or model.provider + model.model_id in config.yaml."
        )
    labels = [f"{p}/{m}" for p, m in specs]
    structured = getattr(cfg.model, "structured_screening", False)
    screeners = [build_screening_agent(llm.build_model(p, m), structured=structured) for p, m in specs]
    summarizers = [build_summarizer_agent(llm.build_model(p, m)) for p, m in specs]
    if structured:
        log.info("screener: structured output ENABLED (JSON schema enforced)")
    delay = cfg.model.request_delay_s
    timeout = cfg.model.request_timeout_s
    log.info("model chain: %s", " → ".join(labels))
    log.info(
        "screening %d new paper(s) (delay %.1fs/call, timeout %.0fs/call)",
        len(new_papers), delay, timeout,
    )

    # --- v2 Treatment Radar setup (only when enabled; keeps v1 torch-free) ---
    ner_cfg = cfg.ner
    radar_agents: list = []
    last_run: str | None = None
    ner_pipeline = None
    ner_nlp = None
    if ner_cfg.enabled:
        import json as _json

        from feline_monitor.agents.radar import build_radar_agent, parse_radar_output
        from feline_monitor import ner as _ner

        fp = query_fingerprint(must_groups, mesh_terms)
        if mem.record_search(cfg.profile.name, fp, _json.dumps(must_groups), _json.dumps(mesh_terms)):
            log.warning(
                "profile %r search changed since last run — novelty now mixes searches; "
                "rename the profile for a clean basket", cfg.profile.name,
            )
        last_run = mem.get_last_run(cfg.profile.name)
        radar_agents = [build_radar_agent(llm.build_model(p, m)) for p, m in specs]
        # Load the NER model + spaCy ONCE for the whole run and reuse across papers. Building
        # them per paper (the old bug) reloaded ~400MB each time without freeing → RAM crept up
        # run-long until the box OOM'd. One load, reused, freed in `finally` below.
        log.info("loading NER model (once for the run): %s", ner_cfg.model_id)
        ner_pipeline = _ner._load_pipeline(ner_cfg.model_id)
        ner_nlp = _ner._load_spacy()
        log.info("Treatment Radar ENABLED (alert on %s)", ", ".join(ner_cfg.alert_categories))

    # Models that rate-limit / crash / time out / emit junk get added here and skipped
    # for the rest of THIS run (shared by screeners + summarizers — same chain indices).
    disabled: set[int] = set()
    unload_local = getattr(cfg.model, "lmstudio_unload_on_drop", False)

    def call(agents: list, prompt: str, validate=None) -> tuple[str, str]:
        time.sleep(delay)  # rate-limit hygiene before every model request
        return llm.run_with_fallback(agents, prompt, labels, validate, disabled, timeout, unload_local)

    def _valid_verdict(text: str) -> bool:
        return parse_screening_json(text)["parsed"]

    if ner_cfg.enabled:
        # Process oldest-ENTRY-date first so novelty attribution is chronological by when papers
        # entered the catalog: an entity first seen in an earlier-indexed paper is recorded before
        # a later-indexed one is judged, so it is not falsely flagged as "new". Aligns with the
        # is_live gate (same EDAT clock). v1 (ner off) keeps retrieval order.
        new_papers.sort(key=lambda p: _pad_date(p.get("entry_date", "")))

    items: list[dict] = []
    decided: list[dict] = []  # only papers fully processed get marked seen (errors retry next run)
    for paper in new_papers:
        if len(disabled) >= len(specs):
            log.warning("all %d models disabled this run — stopping; remaining papers retry next run", len(specs))
            break
        ident = _ident(paper)  # PMID/PMCID — verify at pubmed.ncbi.nlm.nih.gov/<PMID>/ or the PMC link
        blank_line()  # visually separate each paper's block in console + log
        log.info("Processing %s ...", ident)  # human-readable: which paper is being handled now
        if fetch.is_excluded_type(paper.get("article_type", "")):
            log.info("skip  %s — non-research article_type=%s", ident, paper["article_type"])
            decided.append(paper)
            continue

        raw = paper["full_text"] if paper.get("has_full_text") and paper.get("full_text") else paper.get("abstract", "")
        text = fetch.cap_text(raw, cap)
        if len(raw) > len(text):
            log.info("%s: text truncated %d→%d chars", ident, len(raw), len(text))

        # Screen (relevance + triage). A model crash here = transient: skip, retry next run.
        try:
            raw_verdict, model_label = call(
                screeners,
                _screen_prompt(cfg.profile.name, [t for g in must_groups for t in g] + mesh_terms, paper, text),
                validate=_valid_verdict,
            )
        except Exception as exc:  # noqa: BLE001 - one paper's model failure must not kill the run
            log.warning("screen %s FAILED (%s) — skipping, will retry next run", ident, short_error(exc))
            continue
        verdict = parse_screening_json(raw_verdict)

        if not verdict["parsed"]:
            # Model answered but with no valid JSON verdict — do NOT treat as "not relevant".
            log.warning("screen %s: no valid JSON from model — skipping, will retry next run", ident)
            continue

        if not verdict["relevant"]:
            log.info(
                "screen %s → not relevant (via %s): %s",
                ident, model_label, verdict["reason"][:80] or "(no reason given)",
            )
            decided.append(paper)
            continue
        log.info(
            "screen %s → relevant [%s/%s] (via %s)",
            ident, verdict["study_type"], verdict["priority"], model_label,
        )

        try:
            summary_text, sum_label = call(summarizers, _summary_prompt(cfg.profile.name, paper, text))
            summary = strip_reasoning(summary_text)  # drop any reasoning scratchpad before the marker
        except Exception as exc:  # noqa: BLE001 - same resilience for the summary call
            log.warning("summary %s FAILED (%s) — skipping, will retry next run", ident, short_error(exc))
            continue

        items.append(
            {
                "title": paper["title"],
                "url": paper["url"],
                "pmid": paper["pmid"],
                "summary": summary,
                "study_type": verdict["study_type"],
                "priority": verdict["priority"],
            }
        )
        decided.append(paper)
        log.info("summary %s → done (%d chars, via %s)", ident, len(summary), sum_label)

        if ner_cfg.enabled:
            log.info("  trace: Feline-NER -> reading %s for entities ...", ident)
            try:
                # Reuse the run-wide pipeline/spaCy (loaded once above) — no per-paper reload.
                # 0 = no additional cap: NER sees exactly the same (already full_text_char_cap-
                # capped) text the Screener/Summarizer saw, never less.
                hits = _ner.extract_entities(
                    text, ner_cfg.model_id, 0, pipeline=ner_pipeline, nlp=ner_nlp
                )
            except Exception as exc:  # noqa: BLE001 - NER failure must not kill the run
                log.warning("NER %s FAILED (%s) — skipping radar", ident, short_error(exc))
            else:
                counts = ", ".join(f"{cat} {len(v)}" for cat, v in hits.items()) or "none"
                log.info("NER %s → %d entities (%s)", ident, sum(len(v) for v in hits.values()), counts)
                if _is_live(paper.get("entry_date", ""), last_run):
                    novel = {}
                    for cat in ner_cfg.alert_categories:
                        known = mem.known_entities(cfg.profile.name, cat)
                        cand = [h for h in hits.get(cat, []) if h["entity"] not in known]
                        if cand:
                            novel[cat] = cand
                    if novel:
                        try:
                            raw_radar, radar_label = call(radar_agents, _radar_prompt(cfg.profile.name, novel))
                            confirmed = parse_radar_output(raw_radar)
                            if confirmed:
                                items[-1]["radar"] = confirmed
                            log.info("Treatment Radar %s → %d new treatment(s) (via %s)", ident, len(confirmed), radar_label)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("Treatment Radar %s FAILED (%s)", ident, short_error(exc))
                mem.add_entities(cfg.profile.name, paper["pmid"], hits)  # store AFTER novelty

        gc.collect()  # reclaim per-paper transients so the run's memory stays flat

    # Free the run-wide NER model now the loop is done (helps if run_once is ever called in a
    # long-lived process; a one-shot CLI would reclaim on exit anyway).
    ner_pipeline = ner_nlp = None
    gc.collect()

    # Mark only fully-decided papers seen; ones that errored stay unseen so they retry.
    mem.mark_seen(decided)
    if ner_cfg.enabled:
        from datetime import date as _date

        mem.set_last_run(cfg.profile.name, _date.today().isoformat())

    if not items:
        log.info("no relevant papers after screening — done")
        return

    items.sort(key=lambda x: _PRIORITY_ORDER.get(x["priority"], 3))
    path = write_digest(render_markdown(items), cfg.delivery.markdown_dir)
    blank_line()  # separate the digest-written line from the last paper's block
    log.info("digest written: %s", path)
    if cfg.delivery.telegram:
        failed = _send_telegram_digest(items)
        if failed:
            mem.queue_pending(failed)
            log.error(
                "telegram: %d message(s) failed and were queued for retry next run (%d paper(s))",
                len(failed), len(items),
            )
        else:
            log.info("telegram sent OK (%d paper(s))", len(items))
