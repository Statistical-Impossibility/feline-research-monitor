"""Load and validate the project configuration (config.yaml)."""

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Profile:
    name: str
    keywords: list[str] | None = None
    mesh_terms: list[str] | None = None
    must: list[list[str]] | None = None   # concept groups: synonyms OR'd inside, groups AND'd
    mesh: list[str] | None = None          # MeSH terms OR'd into the first group


@dataclass
class Ner:
    enabled: bool = False
    model_id: str = "Statistical-Impossibility/Feline-NER"
    alert_categories: list[str] | None = None

    def __post_init__(self) -> None:
        if self.alert_categories is None:
            self.alert_categories = ["MEDICATION", "PROCEDURE"]


@dataclass
class Sources:
    pubmed: dict[str, Any]


@dataclass
class Model:
    # Ordered fallback chain [{provider, model_id}, ...] is the primary form; each
    # model is tried in order, falling through only on a transport error
    # (crash/HTTP/connection), not on a bad-but-returned answer. provider/model_id
    # are the single-model fallback when no chain is given (either form works).
    provider: str = ""
    model_id: str = ""
    request_delay_s: float = 2.0  # pause before each model call (rate-limit hygiene)
    request_timeout_s: float = 60.0  # hard cap per model call; a stall raises → fail over
    #   (60s: a small screening/summary reply should be fast; a dead free-tier model shouldn't
    #    hold the run for minutes. Raise in config.yaml if a slow local model needs longer.)
    chain: list[dict] | None = None
    # Opt-in: enforce a JSON schema on the SCREENER via the model's structured-output
    # support (ADK output_schema). The tolerant regex parser stays as the fallback, so a
    # model that doesn't support schemas still works (it just fails over like any error).
    structured_screening: bool = False
    # Opt-in: when a local LM Studio model is dropped mid-run, POST an unload so its RAM frees
    # for the next model. Off by default — LM Studio 0.4.0+ manages memory well on its own, and
    # keeping it off avoids any interference with model swapping during a demo.
    lmstudio_unload_on_drop: bool = False


@dataclass
class Delivery:
    telegram: bool
    markdown_dir: str


@dataclass
class Config:
    profile: Profile
    sources: Sources
    model: Model
    delivery: Delivery
    ner: Ner = None  # populated by load_config; never None after construction


def load_config(path: str) -> Config:
    """Read a YAML config file into a typed Config object."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        profile=Profile(**raw["profile"]),
        sources=Sources(**raw["sources"]),
        model=Model(**raw["model"]),
        delivery=Delivery(**raw["delivery"]),
        ner=Ner(**raw.get("ner", {})),
    )
