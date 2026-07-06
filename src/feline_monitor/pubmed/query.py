"""Build a PubMed search string from a condition profile (deterministic, no LLM)."""

import hashlib
import json

# Always-applied guard so results stay about cats, never human medicine.
FELINE_GUARD = (
    '("cats"[MeSH Terms] OR "feline"[Title/Abstract] OR "felis catus"[Title/Abstract])'
)


def build_query(must_groups: list[list[str]], mesh: list[str]) -> str:
    """Build a PubMed query: synonyms OR'd inside each group, groups AND'd together.

    MeSH terms are OR'd into the first group (the primary concept). The feline guard is
    always AND'd. A single group reproduces the flat v1 behaviour exactly.
    """
    clauses: list[str] = []
    for i, group in enumerate(must_groups):
        ors = [f'"{t}"[Title/Abstract]' for t in group]
        if i == 0:
            ors += [f'"{m}"[MeSH Terms]' for m in mesh]
        clauses.append("(" + " OR ".join(ors) + ")")
    topic = " AND ".join(clauses)
    return f"({topic}) AND {FELINE_GUARD}"


def groups_from_profile(profile) -> tuple[list[list[str]], list[str]]:
    """Derive (must_groups, mesh) from a Profile, supporting both config shapes.

    `must`/`mesh` (concept groups) take precedence; otherwise flat `keywords` become one
    OR-group and `mesh_terms` the MeSH list (v1 back-compat).
    """
    if profile.must:
        return profile.must, (profile.mesh or [])
    return [profile.keywords or []], (profile.mesh_terms or [])


def query_fingerprint(must_groups: list[list[str]], mesh: list[str]) -> str:
    """Stable short hash of the search definition — the key for the change-warning."""
    payload = json.dumps(
        {"must": must_groups, "mesh": mesh, "guard": FELINE_GUARD}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
