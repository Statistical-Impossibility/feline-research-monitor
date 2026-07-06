# Feline Veterinary Research Monitor

> Research monitoring only. This project does **not** provide clinical, diagnostic, or
> treatment advice. It surfaces and summarizes scientific literature for awareness.

## The problem

Veterinary research is vast, and hard to track by hand. Many feline diseases are still
considered incurable — but new findings publish daily, scattered across journals, and no
one can monitor all of it manually. Emerging treatments, including old medications that
turn out to work on a new disease, get buried in the noise.

## The solution

An AI **multi-agent** system that watches PubMed for new papers about a configurable
feline condition, reasons about which are genuinely relevant, summarizes the keepers, and
delivers a deduplicated digest to Telegram — built on **Google ADK**.

This demo tracks **FIP** (feline infectious peritonitis) as one concrete example — the
pipeline generalizes to any condition by editing a single config file. The bigger idea:
surface connections across papers, including known medications that could be repurposed
for a disease from another field, or from human medicine.

## Architecture

```
PubMed (own MCP server)
        │
        ▼
Dedup (SQLite — drop PMIDs already seen)
        │  new papers only
        ▼
Screener Agent (ADK LlmAgent)  — relevance + study-type + priority
        │  relevant papers
        ▼
Summarizer Agent (ADK LlmAgent)  — grounded, human-readable summary
        │
        ▼
[optional mode]  Feline-NER (local model, no LLM call)  — extract entities
        │  compared against everything seen before
        ▼
[optional mode]  Novelty gate (deterministic)  — anti-join vs. everything seen before
        │  genuinely new entity found
        ▼
Treatment Radar Agent (ADK LlmAgent)
        │  judges whether a genuinely new entity is really relevant
        ▼
Digest (Markdown, timestamped)  +  Telegram (opt-in)
```

Three agents are called in a fixed sequence — there's no orchestrator, no inter-agent
autonomy, and the Treatment Radar step only fires when the deterministic novelty gate
says something is actually new. PubMed retrieval, dedup, and NER extraction are
deterministic code, not agents; Screener, Summarizer, and Treatment Radar are the three
LLM-driven steps, each a single independent call with its own prompt.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -e .                 # installs google-adk, biopython, mcp, etc.
pip install -e ".[ner]"          # optional: only if you want the Treatment Radar mode
```

Copy `config.example.yaml` to `config.yaml` and edit — every field is commented there,
and the full definition is in `src/feline_monitor/config.py` if you need more detail.

Create `.env` (git-ignored — never commit it):

| Variable | Purpose |
|---|---|
| `ENTREZ_EMAIL` | Required by NCBI for PubMed access (any valid email). |
| `ENTREZ_API_KEY` | Optional; raises your PubMed rate limit. |
| `LMSTUDIO_BASE_URL` | If a chain entry uses `provider: lmstudio` — default `http://localhost:1234/v1`. |
| `OLLAMA_BASE_URL` | If a chain entry uses `provider: ollama` — default `http://localhost:11434`. |
| `GEMINI_API_KEY` | If a chain entry uses `provider: gemini`. |
| `OPENROUTER_API_KEY` | If a chain entry uses `provider: openrouter`. |
| `TELEGRAM_BOT_TOKEN` | Only if `delivery.telegram: true` (kept secret; never logged). |
| `TELEGRAM_CHAT_ID` | Only if `delivery.telegram: true`. |

The model provider + id are chosen entirely in `config.yaml` (`model.chain`) — **no model
is ever hardcoded**. ADK is model-agnostic via LiteLLM: local LM Studio / Ollama,
OpenRouter, or Gemini all work, tried in order with automatic failover.

## Run

```bash
python run.py
```

One cycle: retrieve → dedup → screen → summarize → (optional) NER → novelty gate →
Treatment Radar → write `digests/<date>.md` → (optional) Telegram.

For periodic/unattended monitoring, call `python run.py` from `cron` (Linux/macOS) or
Task Scheduler (Windows) on whatever interval fits your PubMed polling needs — no
dedicated scheduler is bundled, `run.py` is a single idempotent cycle by design.

## Tests

```bash
venv/Scripts/python.exe -m pytest -q
```

## Notes

- NER is a candidate signal (~64% micro-F1), never a clinical claim.
