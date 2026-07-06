from types import SimpleNamespace

from feline_monitor import ner


def test_clean_text_normalizes_unicode_html_and_linebreaks():
    assert ner.clean_text("GS‐441524") == "GS-441524"      # NFKC unicode hyphen -> ascii
    assert ner.clean_text("<b>fip</b>  cat") == "fip cat"        # html strip + ws collapse
    assert ner.clean_text("pred-\nnisolone") == "prednisolone"   # de-hyphenate a line break


def test_normalize_entity_lowercases_strips_and_unifies_hyphen():
    assert ner.normalize_entity("  GS-441524. ") == "gs-441524"
    assert ner.normalize_entity("(Prednisolone)") == "prednisolone"
    assert ner.normalize_entity("GS ‐ 441524") == "gs-441524"


def test_expand_to_word_boundaries_grows_to_whole_word():
    text = "gave prednisolone daily"
    s, e = ner.expand_to_word_boundaries(text, 8, 12)  # a fragment inside "prednisolone"
    assert text[s:e] == "prednisolone"


def test_is_valid_entity_rejects_garbage():
    assert ner.is_valid_entity("prednisolone") is True
    assert ner.is_valid_entity("f") is False        # too short
    assert ner.is_valid_entity("5") is False        # no letters
    assert ner.is_valid_entity("##cn") is False     # subword marker


def test_chunk_sentences_respects_token_budget():
    sents = [{"text": f"s{i}", "start": i * 3, "end": i * 3 + 2} for i in range(5)]
    chunks = ner._chunk_sentences(sents, lambda t: list(t), max_tokens=5)  # 1 token/char
    assert chunks[0]["text"] == "s0 s1"   # "s0 s1" = 5 tokens; adding "s2" would exceed
    assert chunks[0]["offset"] == 0


def test_aggregate_dedups_counts_and_drops_hapax():
    text = "gs-441524 helped. gs-441524 again. remdesivir once."
    spans = [
        {"category": "MEDICATION", "surface": "gs-441524", "start": 0},
        {"category": "MEDICATION", "surface": "GS-441524", "start": 18},   # dedups w/ above
        {"category": "MEDICATION", "surface": "remdesivir", "start": 37},  # hapax -> dropped
    ]
    agg = ner.aggregate(spans, text)
    assert [h["entity"] for h in agg["MEDICATION"]] == ["gs-441524"]
    assert agg["MEDICATION"][0]["count"] == 2
    assert agg["MEDICATION"][0]["context"]  # a context sentence was captured


def _fake_nlp_factory(seen=None):
    def fake_nlp(t):
        if seen is not None:
            seen["len"] = len(t)
        return SimpleNamespace(sents=[SimpleNamespace(text=t, start_char=0, end_char=len(t))])
    return fake_nlp


class _FakeTokenizer:
    def tokenize(self, t):
        return t.split()


def test_extract_entities_end_to_end_with_fakes():
    text = "Cats got GS-441524. GS-441524 worked well."

    class FakePipe:
        tokenizer = _FakeTokenizer()

        def __call__(self, chunk):
            out, idx = [], chunk.find("GS-441524")
            while idx != -1:
                out.append({"entity_group": "MEDICATION", "start": idx, "end": idx + 9, "score": 0.99})
                idx = chunk.find("GS-441524", idx + 1)
            return out

    out = ner.extract_entities(text, "x", char_cap=0, pipeline=FakePipe(), nlp=_fake_nlp_factory())
    assert out["MEDICATION"][0]["entity"] == "gs-441524"  # sliced from original, boundary-clean
    assert out["MEDICATION"][0]["count"] == 2


def test_extract_entities_drops_low_confidence():
    text = "Cats got GS-441524 GS-441524 here."

    class FakePipe:
        tokenizer = _FakeTokenizer()

        def __call__(self, chunk):
            idx = chunk.find("GS-441524")
            return [{"entity_group": "MEDICATION", "start": idx, "end": idx + 9, "score": 0.10}]

    out = ner.extract_entities(text, "x", char_cap=0, pipeline=FakePipe(), nlp=_fake_nlp_factory())
    assert out == {}  # score 0.10 < 0.50 threshold


def test_extract_entities_caps_text_before_ner():
    seen = {}

    class FakePipe:
        tokenizer = _FakeTokenizer()

        def __call__(self, chunk):
            return []

    ner.extract_entities("x" * 5000, "x", char_cap=100, pipeline=FakePipe(), nlp=_fake_nlp_factory(seen))
    assert seen["len"] == 100  # text was cleaned + capped before NER
