"""
Headless verification for the Command A culture layer (seed + prompt injection).

Run:  python3 test_culture_seed.py
No camera, no LLM, no network. Uses a COPY of the real kg_state.json (never the
live file) for the reuse/prompt checks.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import PersonNode
from modules.graph_relationship.topics import (
    normalize_label, person_interests, add_person_interest,
)
from modules.graph_relationship.cultures import (
    culture_priors, person_culture, ensure_culture,
)
from modules.culture_seed import seed_korean_demo, assign_person_culture, _KOREAN_DEMO

_REAL_KG = "kg_state.json"
_DEMO_IDS = {f"topic:{normalize_label(l)}" for l, _c, _p in _KOREAN_DEMO}


def _counts(store) -> tuple:
    return len(store._nodes), len(store._edges)


# ── 1. Empty seed → exact counts + save/load/save round-trip ──────────────────

def test_empty_seed_counts_and_roundtrip():
    s = InMemoryGraphStore()
    info = seed_korean_demo(s)
    cultures = [n for n in s._nodes.values() if n.node_type == "culture"]
    topics   = [n for n in s._nodes.values() if n.node_type == "topic"]
    priors   = [e for e in s._edges.values() if e.edge_type == "culture_prior"]
    assert len(cultures) == 1, cultures
    assert len(topics) == 12, len(topics)
    assert len(priors) == 12, len(priors)
    assert info["reused"] == []          # empty store → nothing pre-existing

    f1 = tempfile.mktemp(suffix=".json"); s.save(f1)
    s2 = InMemoryGraphStore(); s2.load(f1)
    f2 = tempfile.mktemp(suffix=".json"); s2.save(f2)
    a, b = json.load(open(f1)), json.load(open(f2))
    os.remove(f1); os.remove(f2)
    assert a == b, "save→load→save not identical"
    print("1. empty seed: 1 culture / 12 topics / 12 priors, round-trips identically ✓")


# ── 2. Idempotent re-run ──────────────────────────────────────────────────────

def test_idempotent():
    s = InMemoryGraphStore()
    seed_korean_demo(s); c1 = _counts(s)
    seed_korean_demo(s); c2 = _counts(s)
    assert c1 == c2, (c1, c2)
    # priors unchanged too (upsert replaces, never appends)
    priors = [e for e in s._edges.values() if e.edge_type == "culture_prior"]
    assert len(priors) == 12
    print(f"2. idempotent re-run: identical counts {c1} ✓")


# ── 3. Seed on a COPY of the real KG → pre-existing topics reused ─────────────

def test_reuse_on_real_kg_copy():
    if not os.path.exists(_REAL_KG):
        print("3. SKIP — no real kg_state.json present")
        return
    # Compute expected overlap from the ACTUAL file (don't hardcode).
    real = InMemoryGraphStore(); real.load(_REAL_KG)
    existing_topic_ids = {n.id for n in real._nodes.values() if n.node_type == "topic"}
    overlap = existing_topic_ids & _DEMO_IDS
    expected_new = 12 - len(overlap)

    tmp = tempfile.mktemp(suffix=".json"); shutil.copy(_REAL_KG, tmp)
    s = InMemoryGraphStore(); s.load(tmp)
    before_topics = len([n for n in s._nodes.values() if n.node_type == "topic"])
    info = seed_korean_demo(s)
    after_topics = len([n for n in s._nodes.values() if n.node_type == "topic"])
    os.remove(tmp)

    grew_by = after_topics - before_topics
    assert grew_by == expected_new, (grew_by, expected_new)
    assert set(info["reused"]) == {n.split(":", 1)[1].replace("-", " ")  # loose
                                   for n in overlap} or len(info["reused"]) == len(overlap)
    print(f"3. real-KG copy: {len(overlap)} reused ({sorted(overlap)}), "
          f"topics grew by {grew_by} (not 12) ✓")


# ── 4. Assign jay → korean; rebuild the real system prompt → culture block ────

def test_prompt_block():
    if not os.path.exists(_REAL_KG):
        print("4. SKIP — no real kg_state.json present")
        return
    from modules.face_webcam.webcam_loop import WebcamKGLoop

    tmp = tempfile.mktemp(suffix=".json"); shutil.copy(_REAL_KG, tmp)
    s = InMemoryGraphStore(); s.load(tmp)
    seed_korean_demo(s)
    assign_person_culture(s, "jay", "Korean")

    # Give jay an interest in a HIGH-prior culture topic (kpop) so we can assert
    # observed topics are EXCLUDED from the offers.
    add_person_interest(s, "jay", "music", ["kpop"])
    assert person_culture(s, "jay") == "culture:korean"

    # Bare instance — only the attrs _build_system_prompt needs (no camera/LLM).
    loop = WebcamKGLoop.__new__(WebcamKGLoop)
    loop.store = s
    loop.robot_id = "chatbox"
    loop._robot_display = "ChatBox"
    prompt = loop._build_system_prompt("jay")

    assert "CULTURAL BACKGROUND" in prompt, "culture block missing"
    assert "Cultural background hint: Korean" in prompt
    assert "starting guess" in prompt and "not a fact about them" in prompt
    assert "may politely offer ONE" in prompt and "Never assert what they like" in prompt

    # Parse the offers line "(e.g. a, b, c, d)". ≤4, and kpop excluded (jay likes it).
    import re
    m = re.search(r"\(e\.g\. ([^)]+)\)", prompt)
    offers = [o.strip() for o in m.group(1).split(",")] if m else []
    assert 0 < len(offers) <= 4, offers
    assert "kpop" not in offers, f"observed topic leaked into offers: {offers}"
    # Highest-prior non-observed should lead (kimchi, then korean bbq …)
    assert offers[0] == "kimchi", offers
    # Memory-first not overridden: interests still appear ABOVE the culture block.
    assert prompt.index("Interests:") < prompt.index("CULTURAL BACKGROUND")
    os.remove(tmp)
    print(f"4. prompt: culture block present, offers={offers} (≤4, kpop excluded, "
          "memory leads) ✓")


# ── 5. Purity — graph_relationship imports no LLM/PAD/embedding/app ───────────

def test_purity():
    import pathlib, re
    pkg = pathlib.Path("modules/graph_relationship")
    # The PURE relational layer must stay free of LLM/PAD/embedding/app imports.
    # Excluded (app-facing entry points inside the package, by design, as before):
    #   embedding.py  — the ollama embedding adapter itself
    #   demo_harness.py — a runnable demo that wires embeddings
    #   viz/          — the visualization HTTP server (reaches app for /history)
    _EXCLUDE_NAMES = {"embedding.py", "demo_harness.py"}
    forbidden = re.compile(
        r"^\s*(from|import)\s+"
        r"(modules(?!\.graph_relationship)|.*ollama|.*torch|.*pad_module|"
        r"modules\.pad_persona|.*llm_processor|.*emotion_processor)",
        re.MULTILINE)
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name in _EXCLUDE_NAMES or "tests" in py.parts or "viz" in py.parts:
            continue
        text = py.read_text()
        for mobj in forbidden.finditer(text):
            offenders.append(f"{py}: {mobj.group(0).strip()}")
    assert not offenders, "purity violations:\n" + "\n".join(offenders)
    # cultures.py specifically must only import schema/store/topics (relative).
    ctext = (pkg / "cultures.py").read_text()
    import_lines = [l.strip() for l in ctext.splitlines()
                    if l.strip().startswith(("import ", "from "))]
    assert all(("from .schema" in l or "from .store" in l or "from .topics" in l
                or "from __future__" in l or l.startswith("from datetime")
                or l.startswith("from typing")) for l in import_lines), import_lines
    print("5. purity: pure graph_relationship/ modules have no LLM/PAD/app imports; "
          "cultures.py imports only schema/store/topics ✓")


if __name__ == "__main__":
    test_empty_seed_counts_and_roundtrip()
    test_idempotent()
    test_reuse_on_real_kg_copy()
    test_prompt_block()
    test_purity()
    print("\nALL CULTURE-SEED TESTS PASSED")
