from feline_monitor.config import load_config


def test_load_config_reads_profile(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "profile:\n"
        "  name: FIP\n"
        '  keywords: ["feline infectious peritonitis", "FIP"]\n'
        '  mesh_terms: ["Feline Infectious Peritonitis"]\n'
        "sources:\n  pubmed: {active: true, backfill: true, max_per_run: 50}\n"
        "model:\n  provider: openrouter\n  model_id: SET_ME\n"
        "delivery:\n  telegram: true\n  markdown_dir: ./digests\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.profile.keywords == ["feline infectious peritonitis", "FIP"]
    assert cfg.sources.pubmed["max_per_run"] == 50
    assert cfg.model.model_id == "SET_ME"
    assert cfg.delivery.markdown_dir == "./digests"


def test_structured_screening_defaults_false_and_loads(tmp_path):
    base = (
        "profile:\n  name: FIP\n  keywords: [FIP]\n  mesh_terms: [x]\n"
        "sources:\n  pubmed: {active: true, backfill: true, max_per_run: 1}\n"
        "delivery:\n  telegram: false\n  markdown_dir: ./digests\n"
    )
    p1 = tmp_path / "a.yaml"
    p1.write_text(base + "model:\n  provider: gemini\n  model_id: x\n", encoding="utf-8")
    assert load_config(str(p1)).model.structured_screening is False  # opt-in, off by default
    p2 = tmp_path / "b.yaml"
    p2.write_text(
        base + "model:\n  provider: gemini\n  model_id: x\n  structured_screening: true\n",
        encoding="utf-8",
    )
    assert load_config(str(p2)).model.structured_screening is True


def test_ner_defaults_when_absent(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "profile:\n  name: FIP\n  keywords: [FIP]\n  mesh_terms: [x]\n"
        "sources:\n  pubmed: {active: true, backfill: true, max_per_run: 1}\n"
        "model:\n  provider: gemini\n  model_id: x\n"
        "delivery:\n  telegram: false\n  markdown_dir: ./digests\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.ner.enabled is False
    assert cfg.ner.alert_categories == ["MEDICATION", "PROCEDURE"]
    assert cfg.ner.model_id == "Statistical-Impossibility/Feline-NER"


def test_ner_and_concept_groups_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "profile:\n  name: FIP\n"
        "  must:\n    - [FIP, feline infectious peritonitis]\n    - [vomiting, emesis]\n"
        "  mesh: [Feline Infectious Peritonitis]\n"
        "sources:\n  pubmed: {active: true, backfill: true, max_per_run: 1}\n"
        "model:\n  provider: gemini\n  model_id: x\n"
        "ner:\n  enabled: true\n  alert_categories: [MEDICATION]\n"
        "delivery:\n  telegram: false\n  markdown_dir: ./digests\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.ner.enabled is True
    assert cfg.ner.alert_categories == ["MEDICATION"]
    assert cfg.profile.must == [["FIP", "feline infectious peritonitis"], ["vomiting", "emesis"]]
    assert cfg.profile.mesh == ["Feline Infectious Peritonitis"]
