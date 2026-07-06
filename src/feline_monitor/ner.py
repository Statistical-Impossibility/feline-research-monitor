"""Feline-NER wrapper: extract 5-category entities from a paper (v2 radar input).

Ported from the author's proven HF demo (Feline-Project 04_5_HF_demo_app):
- NFKC cleaning (unifies unicode dashes/quotes at the source, strips HTML, de-hyphenates
  line breaks);
- spaCy sentence splitting, then chunking measured by the REAL tokenizer (<=450 tokens,
  a safe margin under the model's 512 limit) — the correct fix for the size-mismatch crash;
- CRITICAL: entities are read by char-offset from the ORIGINAL text with word-boundary
  expansion, never from the tokenizer's subword ``word`` field. That is what keeps output
  clean ("prednisolone", not "##cn"/"prednisolo") — the demo's key insight.
- a confidence threshold (score > 0.50) drops low-quality predictions.

torch / transformers / spaCy import lazily so importing this module (or running v1) needs
none of them. The pure helpers are unit-testable with injected fakes.
"""

import re
import unicodedata
from collections import defaultdict

_WS = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")
_LINEBREAK_HYPHEN = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_HYPHEN_SPACES = re.compile(r"\s*-\s*")
_DASHES = re.compile(r"[‐-―−]")  # unicode hyphen/dashes/minus (NFKC leaves these)
_STRIP = " .,:;()[]{}\"'"

_MAX_TOKENS = 450          # per-chunk token budget (safe margin under the 512 model limit)
_SCORE_MIN = 0.50          # drop low-confidence predictions (demo threshold)
_OVERLAP_SENTS = 2         # sentence overlap between consecutive chunks


def clean_text(text: str) -> str:
    """Normalize paste / PDF / HTML artifacts (ported from the demo's clean_text).

    NFKC folds unicode dash/quote variants to ASCII, so 'gs ‐ 441524' can never split the
    same drug into two entities downstream.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _DASHES.sub("-", text)  # 1:1 length-preserving so char offsets stay valid
    text = _HTML.sub("", text)
    text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)   # join words hyphenated across a line break
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _WS.sub(" ", text)
    return text.strip()


def normalize_entity(s: str) -> str:
    """Canonical surface form for dedup: NFKC, lowercase, trim punctuation, unify hyphens."""
    s = unicodedata.normalize("NFKC", s or "").lower().strip().strip(_STRIP)
    s = _DASHES.sub("-", s)
    s = _WS.sub(" ", s).strip()
    return _HYPHEN_SPACES.sub("-", s)


def expand_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a span outward to whole-word edges so we never keep a subword fragment."""
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ("-", "'")):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] in ("-", "'")):
        end += 1
    return start, end


def is_valid_entity(surface: str) -> bool:
    """Reject garbage: too short, no letters, or a leftover subword marker."""
    s = surface.strip()
    if len(s) < 2:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return not s.startswith("##")


def _sentence_for(text: str, pos: int) -> str:
    """A short context window (the sentence) around a char position."""
    lo = text.rfind(". ", 0, pos)
    lo = 0 if lo == -1 else lo + 2
    hi = text.find(". ", pos)
    hi = len(text) if hi == -1 else hi + 1
    return text[lo:hi].strip()[:300]


def _sentences(nlp, text: str) -> list[dict]:
    """Split text into sentences with exact char offsets (spaCy sentencizer)."""
    return [
        {"text": s.text, "start": s.start_char, "end": s.end_char}
        for s in nlp(text).sents
    ]


def _chunk_sentences(sentences: list[dict], tokenize, max_tokens: int = _MAX_TOKENS) -> list[dict]:
    """Group sentences into chunks of <= max_tokens (measured by the real tokenizer).

    Consecutive chunks overlap by ``_OVERLAP_SENTS`` sentences so entities near a boundary
    are not lost. Each chunk carries the char offset of its first sentence for global remap.
    """
    chunks: list[dict] = []
    i = 0
    while i < len(sentences):
        chunk_sents: list[dict] = []
        chunk_text = ""
        for j in range(i, len(sentences)):
            candidate = f"{chunk_text} {sentences[j]['text']}" if chunk_text else sentences[j]["text"]
            if len(tokenize(candidate)) > max_tokens and chunk_sents:
                break
            chunk_sents.append(sentences[j])
            chunk_text = candidate
        if chunk_sents:
            chunks.append({"text": chunk_text, "offset": chunk_sents[0]["start"]})
        i += max(1, len(chunk_sents) - _OVERLAP_SENTS)
    return chunks


def _extract_spans(text: str, chunks: list[dict], pl) -> list[dict]:
    """Run the NER pipeline per chunk; return clean {category, surface, start} spans.

    Offsets are remapped to the global text, expanded to word boundaries, and the surface
    is sliced from the ORIGINAL text (never the tokenizer word), then validated.
    """
    spans: list[dict] = []
    for chunk in chunks:
        for r in pl(chunk["text"]):
            if float(r.get("score", 1.0)) < _SCORE_MIN:
                continue
            start = r["start"] + chunk["offset"]
            end = r["end"] + chunk["offset"]
            start, end = expand_to_word_boundaries(text, start, end)
            surface = text[start:end]
            if is_valid_entity(surface):
                spans.append({"category": r["entity_group"], "surface": surface, "start": start})
    return spans


def aggregate(spans: list[dict], text: str) -> dict:
    """Fold clean spans into {category: [{entity, count, context}]}.

    Dedup by normalized surface, count mentions, drop single-mention (hapax) noise, attach a
    context sentence. Categories with no survivors are omitted.
    """
    buckets: dict[str, dict[str, dict]] = defaultdict(dict)
    for sp in spans:
        category = sp.get("category")
        norm = normalize_entity(sp.get("surface", ""))
        if not category or not norm:
            continue
        rec = buckets[category].get(norm)
        if rec:
            rec["count"] += 1
        else:
            buckets[category][norm] = {
                "entity": norm,
                "count": 1,
                "context": _sentence_for(text, sp.get("start", 0)),
            }
    out: dict[str, list[dict]] = {}
    for category, recs in buckets.items():
        kept = [r for r in recs.values() if r["count"] >= 2]  # hapax denoise
        if kept:
            out[category] = kept
    return out


def _load_pipeline(model_id: str):
    """Lazily build the HF token-classification pipeline (heavy optional deps).

    The tokenizer is capped at model_max_length=512 as a hard guard for the rare single
    sentence that alone exceeds the chunk budget; normal chunks stay well under it.
    """
    import logging as _logging  # noqa: PLC0415

    from transformers import AutoTokenizer, pipeline as hf_pipeline  # noqa: PLC0415 - lazy
    from transformers.utils import logging as hf_logging  # noqa: PLC0415 - lazy

    hf_logging.set_verbosity_error()   # drop the "Token indices ... > 512" tokenizer warning
    hf_logging.disable_progress_bar()  # drop the "Loading weights" bar (ugly on video)
    # HF Hub sets its own logger level on import, so quiet it here (after import) to drop the
    # "unauthenticated requests to the HF Hub" warning.
    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    return hf_pipeline(
        "token-classification", model=model_id, tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def _load_spacy():
    """Lazily load a lightweight spaCy sentencizer (downloads the model once if missing)."""
    import spacy  # noqa: PLC0415 - lazy on purpose

    disable = ["ner", "tagger", "lemmatizer", "parser", "attribute_ruler"]
    try:
        nlp = spacy.load("en_core_web_sm", disable=disable)
    except OSError:  # pragma: no cover - first-run model download
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=disable)
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def extract_entities(text: str, model_id: str, char_cap: int, pipeline=None, nlp=None) -> dict:
    """Clean -> sentence-split -> token-budgeted chunks -> NER -> {category: [{entity,count,context}]}.

    `pipeline` and `nlp` are injectable for tests; in production they are loaded from `model_id`.
    """
    text = clean_text(text)
    if char_cap and len(text) > char_cap:
        text = text[:char_cap]
    pl = pipeline or _load_pipeline(model_id)
    nlp = nlp or _load_spacy()
    sentences = _sentences(nlp, text)
    if not sentences:
        return {}
    chunks = _chunk_sentences(sentences, pl.tokenizer.tokenize)
    spans = _extract_spans(text, chunks, pl)
    return aggregate(spans, text)
