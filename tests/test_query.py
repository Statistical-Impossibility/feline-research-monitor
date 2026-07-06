from feline_monitor.pubmed.query import (
    build_query, groups_from_profile, query_fingerprint,
)
from feline_monitor.config import Profile


def test_build_query_single_group_backcompat():
    q = build_query([["FIP", "GS-441524"]], ["Feline Infectious Peritonitis"])
    assert '"FIP"[Title/Abstract]' in q
    assert '"GS-441524"[Title/Abstract]' in q
    assert '"Feline Infectious Peritonitis"[MeSH Terms]' in q
    assert '"cats"[MeSH Terms]' in q          # feline guard always present
    assert q.count(" AND ") == 1              # only the guard AND (one topic group)


def test_build_query_multiple_groups_are_anded():
    q = build_query([["FIP"], ["vomiting", "emesis"]], [])
    assert '("FIP"[Title/Abstract]) AND ("vomiting"[Title/Abstract] OR "emesis"[Title/Abstract])' in q
    assert q.count(" AND ") == 2              # between the two groups + the guard


def test_groups_from_profile_prefers_must():
    prof = Profile(name="x", must=[["FIP"], ["vomiting"]], mesh=["M"])
    assert groups_from_profile(prof) == ([["FIP"], ["vomiting"]], ["M"])


def test_groups_from_profile_falls_back_to_keywords():
    prof = Profile(name="x", keywords=["FIP", "GS"], mesh_terms=["M"])
    assert groups_from_profile(prof) == ([["FIP", "GS"]], ["M"])


def test_query_fingerprint_stable_and_sensitive():
    a = query_fingerprint([["FIP"]], ["M"])
    assert a == query_fingerprint([["FIP"]], ["M"])     # stable
    assert a != query_fingerprint([["FIP"], ["vomiting"]], ["M"])  # changes with search
