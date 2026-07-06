"""Fetch and parse PubMed Central (PMC) full text via NCBI Entrez.

Primary path is PMC-only: search ``db="pmc"`` (the free full-text subset),
``efetch`` each hit as full JATS XML, and extract title + abstract + body as plain
text. Papers whose XML has no ``<body>`` (metadata stubs) are kept but flagged
``has_full_text=False`` so the LLM can mark them abstract-only.

An optional path (``fetch_abstract_papers``) pulls ``db="pubmed"`` abstracts for the
``include_paywalled_abstracts`` flag — papers that are in PubMed but not in PMC.

XML is parsed with defusedxml because Entrez responses are external input
(XXE / billion-laughs hardening). Identical parsing logic to the author's original
Feline-Project collector; only the import is hardened.
"""

import os
import time

from defusedxml import ElementTree as ET
from Bio import Entrez

# JATS ``article-type`` values that are not primary research and waste an LLM call.
EXCLUDED_ARTICLE_TYPES = {
    "letter", "editorial", "correction", "retraction", "reply",
    "news", "book-review", "product-review", "discussion", "in-brief",
}


def _configure() -> None:
    """Set Entrez credentials from the environment (email required by NCBI)."""
    Entrez.email = os.getenv("ENTREZ_EMAIL", "you@example.com")
    key = os.getenv("ENTREZ_API_KEY")
    if key:
        Entrez.api_key = key


def _localname(tag) -> str:
    """Return the namespace-stripped local tag name (safe against non-str tags)."""
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _date_params(since: str | None) -> dict:
    """esearch date-floor kwargs for a `since` date ('YYYY-MM-DD'), or {} if unset."""
    if not since:
        return {}
    return {"datetype": "pdat", "mindate": since.replace("-", "/"), "maxdate": "3000"}


def _efetch(db: str, ids: list[str], rettype: str, attempts: int = 3) -> str:
    """efetch the given ids as XML, with exponential backoff. Returns raw text."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            handle = Entrez.efetch(
                db=db, id=",".join(ids), rettype=rettype, retmode="xml"
            )
            text = handle.read()
            handle.close()
            if isinstance(text, bytes):
                text = text.decode("utf-8", "ignore")
            return text
        except Exception as err:  # noqa: BLE001 - retried below, re-raised on exhaustion
            last_err = err
            if attempt < attempts - 1:
                time.sleep(min(30.0, (2 ** (attempt + 1)) * 0.5))
            else:
                raise last_err
    raise last_err  # pragma: no cover - loop always returns or raises


def entrez_dates(pmids: list[str]) -> dict[str, str]:
    """Map each PMID to its Entrez date (EDAT) as 'YYYY-MM-DD' — the date the record entered
    the catalog (what esearch `most+recent` sorts on, and our true "new since last run" clock).

    Publication date is when the paper was PUBLISHED; EDAT is when it was ADDED to PubMed/PMC.
    A paper published long ago but indexed today has a recent EDAT, so it is correctly "new" to
    us. Fetched from db=pubmed (the PMC XML does not carry EDAT). Missing/failed → omitted, and
    the caller falls back to pub_date for that paper.
    """
    pmids = [p for p in pmids if p and not str(p).upper().startswith("PMC")]
    if not pmids:
        return {}
    _configure()
    try:
        root = ET.fromstring(_efetch("pubmed", pmids, "null"))
    except Exception:  # noqa: BLE001 - resilience: EDAT is best-effort, never kill the run
        return {}
    out: dict[str, str] = {}
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        if not pmid:
            continue
        for ppd in art.iter("PubMedPubDate"):
            if ppd.attrib.get("PubStatus") != "entrez":
                continue
            y = ppd.findtext("Year") or ""
            m = ppd.findtext("Month") or ""
            d = ppd.findtext("Day") or ""
            if y:
                date = y
                if m.isdigit():
                    date += f"-{int(m):02d}"
                    if d.isdigit():
                        date += f"-{int(d):02d}"
                out[pmid] = date
            break
    return out


# --- helpers used by the orchestrator -------------------------------------

def is_excluded_type(article_type: str) -> bool:
    """True if the JATS article-type is non-research (drop before screening)."""
    return (article_type or "").strip().lower() in EXCLUDED_ARTICLE_TYPES


def cap_text(text: str, cap: int) -> str:
    """Truncate text to `cap` characters (a guard, not routine). cap<=0 disables."""
    if cap and len(text) > cap:
        return text[:cap]
    return text


def select_text(paper: dict, cap: int) -> str:
    """Text to feed the LLM: full text when available, else the abstract (capped)."""
    if paper.get("has_full_text") and paper.get("full_text"):
        return cap_text(paper["full_text"], cap)
    return cap_text(paper.get("abstract", ""), cap)


# --- PMC full-text parsing -------------------------------------------------

def _find_text(elem, localname: str) -> str:
    """Return the joined itertext of the first descendant with this local name."""
    for el in elem.iter():
        if _localname(el.tag) == localname:
            return "".join(el.itertext()).strip()
    return ""


def _article_id(article, id_type: str) -> str:
    """Return the <article-id pub-id-type=id_type> text, or '' if absent."""
    for el in article.iter():
        if _localname(el.tag) == "article-id" and el.attrib.get("pub-id-type") == id_type:
            return (el.text or "").strip()
    return ""


def _pmcid(article) -> str:
    """Bare PMC accession number (e.g. '13314447'), from whichever id tag carries it.

    Real PMC efetch(full) tags it ``pub-id-type="pmcid"`` with a ``PMC``-prefixed value
    (``PMC13314447``); other/older shapes use ``pmc`` or ``pmcaid`` with the bare number.
    We normalise to the bare number so ``PMC{pmcid}`` builds the canonical URL.
    """
    raw = (
        _article_id(article, "pmcid")
        or _article_id(article, "pmc")
        or _article_id(article, "pmcaid")
    )
    return raw[3:] if raw[:3].upper() == "PMC" else raw


def _abstract_text(article) -> str:
    """Join all <abstract> paragraph text."""
    parts: list[str] = []
    for el in article.iter():
        if _localname(el.tag) == "abstract":
            for p in el.iter():
                if _localname(p.tag) == "p":
                    txt = " ".join(p.itertext()).split()
                    if txt:
                        parts.append(" ".join(txt))
    return "\n".join(parts).strip()


def _body_text(article) -> str:
    """Extract <body> as plain text: section titles + paragraphs."""
    body = None
    for el in article.iter():
        if _localname(el.tag) == "body":
            body = el
            break
    if body is None:
        return ""
    out: list[str] = []
    secs = [el for el in body.iter() if _localname(el.tag) == "sec"]
    if secs:
        for sec in secs:
            for child in sec:
                name = _localname(child.tag)
                if name == "title":
                    t = "".join(child.itertext()).strip()
                    if t:
                        out.append(t)
                elif name == "p":
                    t = " ".join(" ".join(child.itertext()).split())
                    if t:
                        out.append(t)
    else:  # body holds bare <p> with no sections
        for p in body.iter():
            if _localname(p.tag) == "p":
                t = " ".join(" ".join(p.itertext()).split())
                if t:
                    out.append(t)
    return "\n\n".join(out).strip()


def _pub_date(article) -> str:
    """Full publication date as 'YYYY-MM-DD' (or 'YYYY-MM' / 'YYYY' when parts are missing).

    JATS carries several <pub-date> elements (epub, ppub, collection); we keep the most
    complete one. Non-numeric months (rare in PMC) are skipped rather than guessed.
    """
    best = ""
    for el in article.iter():
        if _localname(el.tag) != "pub-date":
            continue
        year = month = day = ""
        for child in el:
            name = _localname(child.tag)
            text = (child.text or "").strip()
            if name == "year":
                year = text
            elif name == "month":
                month = text
            elif name == "day":
                day = text
        if not year:
            continue
        date = year
        if month.isdigit():
            date += f"-{int(month):02d}"
            if day.isdigit():
                date += f"-{int(day):02d}"
        if len(date) > len(best):  # prefer the most complete date available
            best = date
    return best


def _parse_one_article(article) -> dict:
    pmcid = _pmcid(article)
    pmid = _article_id(article, "pmid") or f"PMC{pmcid}"
    title = " ".join(_find_text(article, "article-title").split())  # collapse newlines
    abstract = _abstract_text(article)
    body = _body_text(article)
    pub_date = _pub_date(article) or _find_text(article, "year")
    url = (
        f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
        if pmcid
        else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    )
    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title,
        "abstract": abstract,
        "full_text": body,
        "has_full_text": bool(body),
        "article_type": (article.attrib.get("article-type") or "").strip(),
        "pub_date": pub_date,
        "url": url,
    }


def parse_pmc_articleset(xml_text: str) -> list[dict]:
    """Parse a PMC efetch (full) payload into a list of Paper dicts."""
    root = ET.fromstring(xml_text)
    articles = [el for el in root.iter() if _localname(el.tag) == "article"]
    if not articles and _localname(root.tag) == "article":
        articles = [root]
    return [_parse_one_article(a) for a in articles]


# --- live network calls ----------------------------------------------------

def fetch_pmc_ids(query: str, max_results: int, since: str | None = None) -> list[str]:
    """Return PMC ids matching the query, most recent first (live network call).

    `since` ('YYYY-MM-DD') floors results by publication date — the backfill anchor.
    """
    _configure()
    handle = Entrez.esearch(
        db="pmc", term=query, retmax=max_results, sort="most+recent",
        **_date_params(since),
    )
    rec = Entrez.read(handle)
    handle.close()
    return [str(x) for x in rec.get("IdList", [])]


def fetch_pmc_ids_page(
    query: str, retstart: int, retmax: int, since: str | None = None
) -> tuple[list[str], int]:
    """One page of PMC ids (newest first) plus the total match count.

    Backfill walks the whole `since` window by stepping `retstart` until the page is
    empty or `retstart` reaches the returned count. esearch is cheap (ids only), so the
    caller can dedup against the store BEFORE efetching any full text.
    """
    _configure()
    handle = Entrez.esearch(
        db="pmc", term=query, retstart=retstart, retmax=retmax, sort="most+recent",
        **_date_params(since),
    )
    rec = Entrez.read(handle)
    handle.close()
    ids = [str(x) for x in rec.get("IdList", [])]
    count = int(rec.get("Count", 0) or 0)
    return ids, count


def fetch_papers(pmc_ids: list[str]) -> list[dict]:
    """Fetch and parse PMC full text for the given PMC ids (live network call)."""
    if not pmc_ids:
        return []
    _configure()
    return parse_pmc_articleset(_efetch("pmc", pmc_ids, "full"))


# --- optional abstract-only path (db=pubmed) -------------------------------

def parse_efetch_xml(xml_text: str) -> list[dict]:
    """Parse a db=pubmed efetch payload into a list of Paper dicts (abstract-only)."""
    root = ET.fromstring(xml_text)
    papers: list[dict] = []
    for cit in root.iter("MedlineCitation"):
        pmid_el = cit.find("PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text
        title = (cit.findtext("Article/ArticleTitle") or "").strip()
        abstract = " ".join(
            (t.text or "") for t in cit.iter("AbstractText")
        ).strip()
        year = cit.findtext("Article/Journal/JournalIssue/PubDate/Year") or ""
        papers.append(
            {
                "pmid": pmid,
                "pmcid": "",
                "title": title,
                "abstract": abstract,
                "full_text": "",
                "has_full_text": False,
                "article_type": "",
                "pub_date": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return papers


def fetch_pmids(query: str, max_results: int, since: str | None = None) -> list[str]:
    """Return PubMed PMIDs matching the query, most recent first (live network call)."""
    _configure()
    handle = Entrez.esearch(
        db="pubmed", term=query, retmax=max_results, sort="most+recent",
        **_date_params(since),
    )
    rec = Entrez.read(handle)
    handle.close()
    return [str(x) for x in rec.get("IdList", [])]


def fetch_abstract_papers(pmids: list[str]) -> list[dict]:
    """Fetch abstract-only papers from db=pubmed (for the paywalled-abstracts flag)."""
    if not pmids:
        return []
    _configure()
    return parse_efetch_xml(_efetch("pubmed", pmids, "abstract"))
