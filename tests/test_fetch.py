from feline_monitor.pubmed import fetch


def test_entrez_dates_parses_edat(monkeypatch):
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42375312</PMID></MedlineCitation>
      <PubmedData><History>
        <PubMedPubDate PubStatus="received"><Year>2025</Year><Month>11</Month><Day>4</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="entrez"><Year>2026</Year><Month>6</Month><Day>30</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="pubmed"><Year>2026</Year><Month>6</Month><Day>30</Day></PubMedPubDate>
      </History></PubmedData></PubmedArticle></PubmedArticleSet>"""
    monkeypatch.setattr(fetch, "_configure", lambda: None)
    monkeypatch.setattr(fetch, "_efetch", lambda *a, **k: xml)
    assert fetch.entrez_dates(["42375312"]) == {"42375312": "2026-06-30"}


def test_entrez_dates_skips_pmc_ids_and_empty():
    assert fetch.entrez_dates([]) == {}
    assert fetch.entrez_dates(["PMC13314179"]) == {}


def test_parse_efetch_xml_to_papers():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>12345</PMID>
      <Article><ArticleTitle>FIP study</ArticleTitle>
      <Abstract><AbstractText>GS-441524 helps.</AbstractText></Abstract>
      <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    papers = fetch.parse_efetch_xml(xml)
    assert len(papers) == 1
    assert papers[0]["pmid"] == "12345"
    assert papers[0]["title"] == "FIP study"
    assert "GS-441524" in papers[0]["abstract"]
    assert papers[0]["pub_date"] == "2026"
    assert papers[0]["url"].endswith("/12345/")


def test_parse_efetch_xml_handles_multipart_abstract():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>9</PMID>
      <Article><ArticleTitle>T</ArticleTitle>
      <Abstract>
        <AbstractText Label="BACKGROUND">Part one.</AbstractText>
        <AbstractText Label="RESULTS">Part two.</AbstractText>
      </Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    papers = fetch.parse_efetch_xml(xml)
    assert papers[0]["abstract"] == "Part one. Part two."


def test_parse_efetch_xml_empty_set():
    assert fetch.parse_efetch_xml("<PubmedArticleSet></PubmedArticleSet>") == []


# --- PMC full-text parsing (Bug #1) ---

_PMC_FULL = """<pmc-articleset><article article-type="research-article">
  <front><article-meta>
    <article-id pub-id-type="pmc">11011152</article-id>
    <article-id pub-id-type="pmid">38512345</article-id>
    <title-group><article-title>GS-441524 for FIP</article-title></title-group>
    <pub-date pub-type="epub"><year>2026</year></pub-date>
    <abstract><p>We treated cats with the antiviral.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Methods</title><p>Twelve cats received the drug.</p></sec>
    <sec><title>Results</title><p>All recovered fully.</p></sec>
  </body>
</article></pmc-articleset>"""


def test_parse_pmc_full_article():
    papers = fetch.parse_pmc_articleset(_PMC_FULL)
    assert len(papers) == 1
    p = papers[0]
    assert p["pmid"] == "38512345"
    assert p["pmcid"] == "11011152"
    assert p["title"] == "GS-441524 for FIP"
    assert "treated cats" in p["abstract"]
    assert p["has_full_text"] is True
    assert "Twelve cats received the drug." in p["full_text"]
    assert "All recovered fully." in p["full_text"]
    assert p["article_type"] == "research-article"
    assert p["pub_date"] == "2026"
    assert "PMC11011152" in p["url"]


def test_parse_pmc_real_pmcid_tag_shape():
    # Real PMC efetch(full) tags the accession pub-id-type="pmcid" with a PMC-prefixed
    # value (not pub-id-type="pmc" with a bare number). Regression guard: pmcid must
    # still resolve to the bare number and the URL must be the PMC full-text link.
    xml = """<pmc-articleset><article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmcid">PMC13314447</article-id>
        <article-id pub-id-type="pmcid-ver">PMC13314447.1</article-id>
        <article-id pub-id-type="pmcaid">13314447</article-id>
        <article-id pub-id-type="pmid">42382116</article-id>
        <title-group><article-title>Real shape</article-title></title-group>
      </article-meta></front>
      <body><p>Body.</p></body>
    </article></pmc-articleset>"""
    p = fetch.parse_pmc_articleset(xml)[0]
    assert p["pmcid"] == "13314447"
    assert p["pmid"] == "42382116"
    assert p["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC13314447/"


def test_parse_pmc_full_publication_date():
    # The DB must store the full Y-M-D date, not just the year. Prefer the most complete
    # <pub-date> (here the epub with day+month) over a year-only collection date.
    xml = """<pmc-articleset><article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmcid">PMC500</article-id>
        <title-group><article-title>Dated</article-title></title-group>
        <pub-date pub-type="collection"><year>2026</year></pub-date>
        <pub-date pub-type="epub"><day>7</day><month>3</month><year>2026</year></pub-date>
      </article-meta></front>
      <body><p>Body.</p></body>
    </article></pmc-articleset>"""
    p = fetch.parse_pmc_articleset(xml)[0]
    assert p["pub_date"] == "2026-03-07"  # zero-padded, most complete date wins


def test_parse_pmc_year_only_date_falls_back():
    # No month/day → keep whatever exists (year), not an empty string.
    xml = """<pmc-articleset><article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmcid">PMC501</article-id>
        <title-group><article-title>YearOnly</article-title></title-group>
        <pub-date pub-type="epub"><year>2025</year></pub-date>
      </article-meta></front>
      <body><p>Body.</p></body>
    </article></pmc-articleset>"""
    assert fetch.parse_pmc_articleset(xml)[0]["pub_date"] == "2025"


def test_parse_pmc_metadata_stub_has_no_full_text():
    xml = """<pmc-articleset><article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmc">999</article-id>
        <article-id pub-id-type="pmid">111</article-id>
        <title-group><article-title>Stub</article-title></title-group>
        <abstract><p>Abstract only.</p></abstract>
      </article-meta></front>
    </article></pmc-articleset>"""
    p = fetch.parse_pmc_articleset(xml)[0]
    assert p["has_full_text"] is False
    assert p["full_text"] == ""
    assert "Abstract only." in p["abstract"]


def test_parse_pmc_pmid_falls_back_to_pmcid():
    xml = """<pmc-articleset><article article-type="case-report">
      <front><article-meta>
        <article-id pub-id-type="pmc">777</article-id>
        <title-group><article-title>No PMID</article-title></title-group>
      </article-meta></front>
      <body><p>Body text here.</p></body>
    </article></pmc-articleset>"""
    p = fetch.parse_pmc_articleset(xml)[0]
    assert p["pmid"] == "PMC777"


def test_parse_pmc_multiple_articles():
    xml = f"<pmc-articleset>{_PMC_FULL.split('>', 1)[1].rsplit('</pmc-articleset>', 1)[0]}" \
          f"{_PMC_FULL.split('<pmc-articleset>', 1)[1]}"
    papers = fetch.parse_pmc_articleset(xml)
    assert len(papers) == 2


def test_is_excluded_type():
    assert fetch.is_excluded_type("letter") is True
    assert fetch.is_excluded_type("Editorial") is True
    assert fetch.is_excluded_type("correction") is True
    assert fetch.is_excluded_type("research-article") is False
    assert fetch.is_excluded_type("case-report") is False
    assert fetch.is_excluded_type("") is False


def test_cap_text():
    assert fetch.cap_text("abcdef", 3) == "abc"
    assert fetch.cap_text("ab", 10) == "ab"
    assert fetch.cap_text("ab", 0) == "ab"


def test_select_text_prefers_full_text_when_present():
    paper = {"has_full_text": True, "full_text": "FULL BODY", "abstract": "abs"}
    assert fetch.select_text(paper, 1000) == "FULL BODY"


def test_select_text_falls_back_to_abstract():
    paper = {"has_full_text": False, "full_text": "", "abstract": "just the abstract"}
    assert fetch.select_text(paper, 1000) == "just the abstract"


def test_select_text_applies_cap():
    paper = {"has_full_text": True, "full_text": "x" * 50, "abstract": ""}
    assert fetch.select_text(paper, 10) == "x" * 10


def test_fetch_pmc_ids_page_returns_ids_and_count(monkeypatch):
    # esearch is mocked: verify we surface both the id page and the total Count, and
    # forward the paging window (retstart/retmax) to Entrez.
    captured = {}

    class _Handle:
        def close(self):
            pass

    def fake_esearch(**kwargs):
        captured.update(kwargs)
        return _Handle()

    monkeypatch.setattr(fetch, "_configure", lambda: None)
    monkeypatch.setattr(fetch.Entrez, "esearch", fake_esearch)
    monkeypatch.setattr(fetch.Entrez, "read", lambda h: {"IdList": ["10", "11"], "Count": "143"})

    ids, count = fetch.fetch_pmc_ids_page("q", retstart=40, retmax=20, since="2025-01-01")
    assert ids == ["10", "11"]
    assert count == 143
    assert captured["retstart"] == 40
    assert captured["retmax"] == 20
    assert captured["mindate"] == "2025/01/01"


def test_date_params():
    assert fetch._date_params(None) == {}
    assert fetch._date_params("") == {}
    assert fetch._date_params("2024-01-01") == {
        "datetype": "pdat",
        "mindate": "2024/01/01",
        "maxdate": "3000",
    }
