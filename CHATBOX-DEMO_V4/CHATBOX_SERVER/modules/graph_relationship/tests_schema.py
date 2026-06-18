"""Unit tests for the graph_relationship schema module."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from .schema import (
    Embodiment,
    GraphSchema,
    PersonNode,
    Provenance,
    RobotNode,
    RapportEdge,
    MoodEdge,
    PreferenceEdge,
    TopicNode,
    Timescale,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prov(confidence: float = 0.9) -> Provenance:
    return Provenance(source="robot:cat", confidence=confidence)


def _make_graph_with_robot_and_person():
    g = GraphSchema()
    robot = g.add_node(RobotNode(name="CHATBOX", embodiment=Embodiment.CAT))
    person = g.add_node(PersonNode(display_name="Alex"))
    return g, robot, person


# ---------------------------------------------------------------------------
# Test 1 — valid edge creation and retrieval
# ---------------------------------------------------------------------------

def test_valid_rapport_edge_round_trip():
    g, robot, person = _make_graph_with_robot_and_person()

    edge = g.add_edge(
        RapportEdge(
            source_id=robot.id,
            target_id=person.id,
            provenance=_prov(0.8),
            weight=0.6,
        )
    )

    fetched = g.get_edge(edge.id)
    assert fetched is not None
    assert fetched.edge_type == "rapport"
    assert fetched.weight == 0.6
    assert fetched.provenance.source == "robot:cat"
    assert fetched.provenance.confidence == 0.8


# ---------------------------------------------------------------------------
# Test 2 — edge rejected when source_id is missing from graph
# ---------------------------------------------------------------------------

def test_edge_rejected_missing_source_node():
    g = GraphSchema()
    person = g.add_node(PersonNode(display_name="Sam"))

    with pytest.raises(ValueError, match="source_id"):
        g.add_edge(
            RapportEdge(
                source_id="nonexistent-robot-id",
                target_id=person.id,
                provenance=_prov(),
                weight=0.5,
            )
        )


# ---------------------------------------------------------------------------
# Test 3 — edge rejected when confidence is out of [0, 1]
# ---------------------------------------------------------------------------

def test_edge_rejected_bad_confidence_range():
    g, robot, person = _make_graph_with_robot_and_person()

    with pytest.raises(ValidationError):
        # confidence=1.5 violates ge=0.0 le=1.0
        g.add_edge(
            RapportEdge(
                source_id=robot.id,
                target_id=person.id,
                provenance=Provenance(source="robot:cat", confidence=1.5),
                weight=0.5,
            )
        )

    with pytest.raises(ValidationError):
        # confidence=-0.1 violates ge=0.0
        g.add_edge(
            RapportEdge(
                source_id=robot.id,
                target_id=person.id,
                provenance=Provenance(source="robot:cat", confidence=-0.1),
                weight=0.5,
            )
        )


# ---------------------------------------------------------------------------
# Test 4 — FAST / SLOW timescale flags survive serialisation round-trip
# ---------------------------------------------------------------------------

def test_timescale_flag_round_trip():
    g, robot, person = _make_graph_with_robot_and_person()
    topic = g.add_node(TopicNode(label="school stress"))

    mood_edge = g.add_edge(
        MoodEdge(
            source_id=person.id,
            target_id=person.id,
            provenance=_prov(),
            value=0.3,
        )
    )
    pref_edge = g.add_edge(
        PreferenceEdge(
            source_id=person.id,
            target_id=topic.id,
            provenance=_prov(),
            weight=0.7,
        )
    )

    assert mood_edge.timescale == Timescale.FAST
    assert pref_edge.timescale == Timescale.SLOW

    # Serialise and restore
    restored = GraphSchema.from_dict(g.to_dict())

    restored_mood = restored.get_edge(mood_edge.id)
    restored_pref = restored.get_edge(pref_edge.id)

    assert restored_mood.timescale == Timescale.FAST
    assert restored_pref.timescale == Timescale.SLOW


# ---------------------------------------------------------------------------
# Bonus — weight out-of-range rejected at construction
# ---------------------------------------------------------------------------

def test_weight_out_of_range_rejected():
    with pytest.raises(ValidationError):
        RapportEdge(
            source_id="a",
            target_id="b",
            provenance=_prov(),
            weight=1.5,   # must be ≤ 1.0
        )
