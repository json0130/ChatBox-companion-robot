"""
Unit tests for kg_bridge.py.

Three critical guarantees (A/B/C) plus standard behaviour tests.

  A. Same-turn blend isolation — post_turn cannot corrupt an already-returned BridgeInput
  B. D-axis purity — SLOW edges affect only structured_memory; v/a/tier are unchanged;
                     post_turn never writes a D (dominance) edge
  C. Cold-start / null-person safety — pre_turn and post_turn handle None or unknown
     person_id without crashing
"""

import pytest
from datetime import datetime, timezone

from .schema import (
    Embodiment,
    PersonNode,
    Provenance,
    RapportEdge,
    RobotNode,
    TopicNode,
    TraitEdge,
    PreferenceEdge,
    TrustEdge,
    InteractionCountEdge,
    MoodEdge,
)
from .store import InMemoryGraphStore
from .kg_bridge import (
    BridgeInput,
    KGBridge,
    derive_tier,
    emotion_label_to_va,
    format_slow_edges,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _prov(source: str = "chatbox") -> Provenance:
    return Provenance(source=source, confidence=0.9, timestamp=datetime.now(timezone.utc))


def _make_store_with_robot_and_person(robot_id: str = "chatbox"):
    """Return (store, robot_node, person_node) with robot node id == robot_id."""
    store = InMemoryGraphStore()
    robot = store.upsert_node(RobotNode(
        id=robot_id, name=robot_id,
        embodiment=Embodiment.CAT if robot_id == "chatbox" else Embodiment.ELEPHANT,
    ))
    person = store.upsert_node(PersonNode())
    return store, robot, person


# ---------------------------------------------------------------------------
# derive_tier — threshold boundaries
# ---------------------------------------------------------------------------

def test_derive_tier_close():
    edges = [
        RapportEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.8),
        TrustEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.75),
    ]
    assert derive_tier(edges) == "close"


def test_derive_tier_close_boundary():
    # score = (0.71 + 0.71) / 2 = 0.71 > 0.70
    edges = [
        RapportEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.71),
        TrustEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.71),
    ]
    assert derive_tier(edges) == "close"


def test_derive_tier_known_via_score():
    # score = (0.5 + 0.42) / 2 = 0.46 > 0.45
    edges = [
        RapportEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.5),
        TrustEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.42),
    ]
    assert derive_tier(edges) == "known"


def test_derive_tier_known_via_count():
    # No rapport/trust; count=6 > 5 → "known"
    edges = [
        InteractionCountEdge(source_id="r", target_id="p", provenance=_prov(), count=6),
    ]
    assert derive_tier(edges) == "known"


def test_derive_tier_visitor():
    # count=3, score=0 → "visitor"
    edges = [
        InteractionCountEdge(source_id="r", target_id="p", provenance=_prov(), count=3),
    ]
    assert derive_tier(edges) == "visitor"


def test_derive_tier_unknown():
    assert derive_tier([]) == "unknown"
    # score exactly 0.45 does NOT reach "known" (requires strictly >)
    edges = [
        RapportEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.45),
        TrustEdge(source_id="r", target_id="p", provenance=_prov(), weight=0.45),
    ]
    assert derive_tier(edges) == "unknown"


# ---------------------------------------------------------------------------
# post_turn — exactly three upserts, correct types, correct provenance
# ---------------------------------------------------------------------------

def test_post_turn_exactly_three_upserts():
    store, robot, person = _make_store_with_robot_and_person()
    bridge = KGBridge(store)

    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.5, 0.3, 0.2)})

    ctx = store.get_person_context(person.id)
    all_edges = ctx.person_attribute_edges + ctx.relationship_edges
    assert len(all_edges) == 3
    assert {e.edge_type for e in all_edges} == {"mood", "attention", "interaction_count"}


def test_post_turn_provenance_source_is_robot_id():
    store, robot, person = _make_store_with_robot_and_person("ellebot")
    bridge = KGBridge(store)

    bridge.post_turn(person.id, "ellebot", {"pad_state": (0.4, -0.2, 0.1)})

    ctx = store.get_person_context(person.id)
    for edge in ctx.person_attribute_edges + ctx.relationship_edges:
        assert edge.provenance.source == "ellebot", (
            f"Expected provenance source 'ellebot', got '{edge.provenance.source}' "
            f"on edge type '{edge.edge_type}'"
        )


def test_interaction_count_increments():
    store, robot, person = _make_store_with_robot_and_person()
    bridge = KGBridge(store)

    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.3, 0.1, 0.0)})
    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.4, 0.2, 0.0)})
    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.5, 0.3, 0.0)})

    ctx = store.get_person_context(person.id)
    count_edge = next(e for e in ctx.relationship_edges if e.edge_type == "interaction_count")
    assert count_edge.count == 3  # accumulated, not overwritten


# ---------------------------------------------------------------------------
# A. Same-turn blend isolation
# ---------------------------------------------------------------------------

def test_A_same_turn_blend_isolation():
    """
    The BridgeInput returned by pre_turn must reflect graph state AT THAT MOMENT.
    A subsequent post_turn write must NOT alter that already-returned snapshot.
    The second pre_turn must see the updated mood from post_turn.
    """
    store, robot, person = _make_store_with_robot_and_person()
    bridge = KGBridge(store)

    # Seed the graph with mood = 0.5
    store.upsert_edge(MoodEdge(
        source_id=person.id, target_id=person.id,
        provenance=_prov(), value=0.5,
    ))

    # Turn 1 pre: camera "neutral" → VA=(0,0); graph_mood=0.5
    # Expected blended_v = 0.7 * 0.0 + 0.3 * 0.5 = 0.15
    inp1 = bridge.pre_turn(person.id, "chatbox", "neutral")
    assert abs(inp1.valence - 0.15) < 1e-9

    # Turn 1 post: PAD P = 0.8 → writes MoodEdge(value=0.8)
    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.8, 0.3, 0.2)})

    # inp1 must still be 0.15 — frozen dataclass, no aliasing into the store
    assert abs(inp1.valence - 0.15) < 1e-9, "post_turn must not retroactively mutate inp1"

    # Turn 2 pre: graph_mood is now 0.8
    # Expected blended_v = 0.7 * 0.0 + 0.3 * 0.8 = 0.24
    inp2 = bridge.pre_turn(person.id, "chatbox", "neutral")
    assert abs(inp2.valence - 0.24) < 1e-9, "Second pre_turn must see the updated mood"


# ---------------------------------------------------------------------------
# B. D-axis purity
# ---------------------------------------------------------------------------

def test_B_slow_edges_affect_only_structured_memory():
    """
    Adding/changing SLOW edges must change structured_memory but must NOT
    change blended_v, arousal, or tier.
    """
    store, robot, person = _make_store_with_robot_and_person()

    # Set up identical relationship state for both pre_turn calls
    store.upsert_edge(RapportEdge(
        source_id=robot.id, target_id=person.id, provenance=_prov(), weight=0.6
    ))

    bridge = KGBridge(store)

    # First read — no SLOW edges yet
    inp1 = bridge.pre_turn(person.id, "chatbox", "happy")
    assert inp1.structured_memory == ""

    # Add SLOW edges
    topic = store.upsert_node(TopicNode(label="space"))
    store.upsert_edge(TraitEdge(
        source_id=person.id, target_id=person.id, provenance=_prov(), value="curious"
    ))
    store.upsert_edge(PreferenceEdge(
        source_id=person.id, target_id=topic.id, provenance=_prov(), weight=0.8
    ))

    # Second read — structured_memory must change
    inp2 = bridge.pre_turn(person.id, "chatbox", "happy")
    assert "curious" in inp2.structured_memory
    assert inp2.structured_memory != ""

    # v / a / tier must be IDENTICAL to inp1 — SLOW edges must not touch numerics
    assert inp1.valence == inp2.valence
    assert inp1.arousal == inp2.arousal
    assert inp1.tier == inp2.tier


def test_B_post_turn_never_writes_dominance():
    """
    D (Dominance) is NEVER persisted. post_turn must write exactly three edges
    and none may carry a "dominance" edge_type (which doesn't exist in the schema,
    but we assert it explicitly to document the contract).
    """
    store, robot, person = _make_store_with_robot_and_person()
    bridge = KGBridge(store)

    # Use a high D value to ensure it's not accidentally smuggled into an edge
    bridge.post_turn(person.id, "chatbox", {"pad_state": (0.5, 0.3, 0.99)})

    ctx = store.get_person_context(person.id)
    all_edges = ctx.person_attribute_edges + ctx.relationship_edges
    for edge in all_edges:
        assert edge.edge_type != "dominance", "D must never be written to the graph"

    # Verify it's exactly the three expected types and nothing extra
    assert {e.edge_type for e in all_edges} == {"mood", "attention", "interaction_count"}


# ---------------------------------------------------------------------------
# C. Cold-start / null-person safety
# ---------------------------------------------------------------------------

def test_C_pre_turn_none_person_id_does_not_crash():
    store = InMemoryGraphStore()
    bridge = KGBridge(store)

    v_happy, a_happy = emotion_label_to_va("happy")
    inp = bridge.pre_turn(None, "chatbox", "happy")

    assert inp.tier == "unknown"
    assert inp.structured_memory == ""
    # Unblended: camera_va only
    assert abs(inp.valence - v_happy) < 1e-9
    assert abs(inp.arousal - a_happy) < 1e-9


def test_C_pre_turn_unknown_person_id_does_not_crash():
    store = InMemoryGraphStore()
    bridge = KGBridge(store)

    v, a = emotion_label_to_va("sad")
    inp = bridge.pre_turn("nonexistent-person", "chatbox", "sad")

    assert inp.tier == "unknown"
    assert inp.structured_memory == ""
    assert abs(inp.valence - v) < 1e-9
    assert abs(inp.arousal - a) < 1e-9


def test_C_post_turn_creates_person_node_and_three_edges():
    """post_turn on a brand-new person_id must create the person node and write edges."""
    store = InMemoryGraphStore()
    bridge = KGBridge(store)

    new_person_id = "fresh-child-id"
    assert store.get_node(new_person_id) is None  # not in store yet

    bridge.post_turn(new_person_id, "chatbox", {"pad_state": (0.3, 0.2, 0.0)})

    # Person node must now exist
    assert store.get_node(new_person_id) is not None

    # Exactly three edges must have been written
    ctx = store.get_person_context(new_person_id)
    all_edges = ctx.person_attribute_edges + ctx.relationship_edges
    assert len(all_edges) == 3
    assert {e.edge_type for e in all_edges} == {"mood", "attention", "interaction_count"}


def test_C_post_turn_none_person_id_is_silent_noop():
    """post_turn(None, ...) must return without writing anything."""
    store = InMemoryGraphStore()
    bridge = KGBridge(store)
    initial_node_count = len(store._nodes)

    bridge.post_turn(None, "chatbox", {"pad_state": (0.5, 0.5, 0.5)})

    assert len(store._nodes) == initial_node_count  # nothing created
