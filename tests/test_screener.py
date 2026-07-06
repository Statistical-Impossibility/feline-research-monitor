from feline_monitor.agents.screener import parse_screening_json


def test_parse_clean_json():
    out = parse_screening_json(
        '{"relevant": true, "reason": "feline FIP trial", "study_type": "trial", "priority": "high"}'
    )
    assert out == {
        "parsed": True,
        "relevant": True,
        "reason": "feline FIP trial",
        "study_type": "trial",
        "priority": "high",
    }


def test_parse_marks_unparseable():
    # No JSON at all -> parsed=False (caller must not treat this as "not relevant").
    assert parse_screening_json("upstream 429, service unavailable")["parsed"] is False
    assert parse_screening_json("")["parsed"] is False
    # Valid JSON -> parsed=True.
    assert parse_screening_json('{"relevant": false}')["parsed"] is True


def test_parse_with_surrounding_prose_and_fences():
    text = 'Sure!\n```json\n{"relevant": false, "reason": "human medicine", "study_type": "review", "priority": "low"}\n```'
    out = parse_screening_json(text)
    assert out["relevant"] is False
    assert out["study_type"] == "review"


def test_parse_garbage_defaults_safe():
    out = parse_screening_json("not json at all")
    assert out["parsed"] is False
    assert out["relevant"] is False
    assert out["study_type"] == "other"
    assert out["priority"] == "low"


def test_parse_coerces_invalid_enum_values():
    out = parse_screening_json('{"relevant": true, "study_type": "meta", "priority": "urgent"}')
    assert out["study_type"] == "other"
    assert out["priority"] == "low"


def test_structured_verdict_roundtrips_through_parser():
    # The structured-output schema must serialize to JSON the tolerant parser accepts,
    # so structured and prompt-instructed screening are handled identically downstream.
    from feline_monitor.agents.screener import ScreeningVerdict

    v = ScreeningVerdict(relevant=True, reason="feline FIP trial", study_type="trial", priority="high")
    out = parse_screening_json(v.model_dump_json())
    assert out["parsed"] is True
    assert out["relevant"] is True
    assert out["study_type"] == "trial"
    assert out["priority"] == "high"
