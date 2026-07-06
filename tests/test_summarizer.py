from feline_monitor.agents.summarizer import strip_reasoning


def test_strips_reasoning_before_marker():
    text = (
        "We need to produce a scholarly summary, 1-2 paragraphs, covering what the paper did.\n"
        "===SUMMARY===\n"
        "This study investigated FCoV in 145 cats and found a 59% positivity rate across the cohort."
    )
    out = strip_reasoning(text)
    assert out.startswith("This study investigated FCoV")
    assert "We need to" not in out


def test_clean_output_with_marker_returns_prose():
    text = "===SUMMARY===\nThis prospective trial evaluated remdesivir in 45 cats over 60 days."
    assert strip_reasoning(text) == "This prospective trial evaluated remdesivir in 45 cats over 60 days."


def test_no_marker_returns_unchanged():
    # A model that ignored the marker instruction must not have its text mangled.
    text = "This study investigated an oral GS-441524 suspension and its stability over twelve weeks."
    assert strip_reasoning(text) == text


def test_empty_after_marker_falls_back_to_original():
    # Marker present but no usable summary after it -> keep the original (never drop a body).
    text = "Some reasoning here that is long enough to be a fallback body.\n===SUMMARY===\n   "
    assert strip_reasoning(text) == text.strip()


def test_empty_input():
    assert strip_reasoning("") == ""
