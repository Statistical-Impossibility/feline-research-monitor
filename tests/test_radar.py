from feline_monitor.agents.radar import parse_radar_output


def test_parse_radar_output_extracts_list():
    text = 'noise [{"entity":"saquinavir","category":"MEDICATION","note":"antiviral"}] tail'
    out = parse_radar_output(text)
    assert out == [{"entity": "saquinavir", "category": "MEDICATION", "note": "antiviral"}]


def test_parse_radar_output_drops_incomplete_and_handles_junk():
    assert parse_radar_output("not json") == []
    assert parse_radar_output('[{"entity":"x"}]') == []          # no note → dropped
    assert parse_radar_output("[]") == []
