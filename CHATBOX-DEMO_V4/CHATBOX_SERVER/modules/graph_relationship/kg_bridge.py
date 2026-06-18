"""
PAD ↔ KG bridge — reads graph state before each turn, writes PAD output after.

Dependency direction
--------------------
  kg_bridge → graph_relationship (store + schema)
  kg_bridge reads from pad_result dict (string contract only; no import of pad_persona)
  pad_persona     does NOT import kg_bridge
  graph_relationship  does NOT import kg_bridge

This is the ONLY module that couples the two subsystems.

D-axis invariant
----------------
D (Dominance) is derived fresh from relationship_edges each turn via derive_tier()
and is NEVER written back to the graph.  V and A come from the face model only
(blended with the graph's FAST MoodEdge to soften frame spikes).  Long-term
state enters the PAD adapter exclusively as text via BridgeInput.structured_memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .schema import (
    AnyEdge,
    AttentionEdge,
    Embodiment,
    InteractionCountEdge,
    MoodEdge,
    PersonNode,
    Provenance,
    RobotNode,
)
from .store import GraphStore

# ---------------------------------------------------------------------------
# Emotion → (valence, arousal) — Russell (1980) circumplex model
# Mirrors pad_persona.pipeline_adapter.EMOTION_VA; kept separate so this
# module has no runtime dependency on pad_persona.
# ---------------------------------------------------------------------------
_EMOTION_VA: dict[str, tuple[float, float]] = {
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "angry":    (-0.6,  0.7),
    "fear":     (-0.5,  0.8),
    "disgust":  (-0.6,  0.3),
    "surprise": ( 0.1,  0.8),
}

# Used when auto-creating a robot node on first post_turn.
# Robot node id is set to robot_id (not a UUID) for stable cross-session lookup.
_ROBOT_EMBODIMENT: dict[str, Embodiment] = {
    "chatbox": Embodiment.CAT,
    "ellebot": Embodiment.ELEPHANT,
}


# ---------------------------------------------------------------------------
# Public result type — frozen so callers can never alias into the store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BridgeInput:
    """
    Snapshot produced by pre_turn and consumed by PADPipelineAdapter.process_turn.

    Frozen so that a subsequent post_turn write cannot mutate a caller's reference —
    this is the structural guarantee for same-turn blend isolation.
    """
    valence: float
    arousal: float
    tier: str
    structured_memory: str


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def emotion_label_to_va(label: Optional[str]) -> tuple[float, float]:
    """Return (valence, arousal) for a camera emotion label. Unknown/None → (0, 0)."""
    if label is None:
        return (0.0, 0.0)
    return _EMOTION_VA.get(label.lower(), (0.0, 0.0))


def derive_tier(relationship_edges: List[AnyEdge]) -> str:
    """
    Map relationship edges to a PAD tier string.

    score = (rapport.weight + trust.weight) / 2
      score > 0.70  → "close"
      score > 0.45  → "known"
      count > 5     → "known"   (even with low score — seen enough turns)
      count > 0     → "visitor"
      else          → "unknown"
    """
    rapport = 0.0
    trust = 0.0
    count = 0
    for edge in relationship_edges:
        if edge.edge_type == "rapport":
            rapport = edge.weight
        elif edge.edge_type == "trust":
            trust = edge.weight
        elif edge.edge_type == "interaction_count":
            count = edge.count
    score = (rapport + trust) / 2.0
    if score > 0.70:
        return "close"
    if score > 0.45:
        return "known"
    if count > 5:
        return "known"
    if count > 0:
        return "visitor"
    return "unknown"


def format_slow_edges(attribute_edges: List[AnyEdge]) -> str:
    """
    Render SLOW (trait / preference) edges as prompt-ready text.

    Output: "[trait: shy] [prefers: <topic_id>]"
    Only SLOW-timescale edges are included; FAST edges (mood, attention,
    current_topic) are intentionally excluded — they must never enter the
    prompt text path that could influence tier derivation or V/A blending.
    Topic label resolution (target_id → TopicNode.label) is deferred to the
    server wiring step.
    """
    parts: list[str] = []
    for edge in attribute_edges:
        if edge.edge_type == "trait":
            parts.append(f"[trait: {edge.value}]")
        elif edge.edge_type == "preference":
            parts.append(f"[prefers: {edge.target_id}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prov(source: str) -> Provenance:
    return Provenance(source=source, confidence=1.0, timestamp=datetime.now(timezone.utc))


def _ensure_robot_node(store: GraphStore, robot_id: str) -> str:
    """Return robot_id after ensuring a robot node with id=robot_id exists."""
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(
            id=robot_id,
            name=robot_id,
            embodiment=_ROBOT_EMBODIMENT.get(robot_id.lower(), Embodiment.CAT),
        ))
    return robot_id


def _ensure_person_node(store: GraphStore, person_id: str) -> None:
    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id))


# ---------------------------------------------------------------------------
# KGBridge
# ---------------------------------------------------------------------------

class KGBridge:
    """
    Thin stateful bridge: reads the KG before each PAD turn, writes back after.

    One instance per server session.  Thread-safety is the caller's responsibility.
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def pre_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        camera_emotion: Optional[str],
    ) -> BridgeInput:
        """
        Collect graph context for this turn and return frozen PAD inputs.

        Cold-start / null-person path: does not touch the store; returns camera
        VA unblended, tier "unknown", empty structured_memory.
        """
        if person_id is None or self._store.get_node(person_id) is None:
            v, a = emotion_label_to_va(camera_emotion)
            return BridgeInput(valence=v, arousal=a, tier="unknown", structured_memory="")

        ctx = self._store.get_person_context(person_id)

        # Tier: relationship edges only. D is never read from the graph here.
        tier = derive_tier(ctx.relationship_edges)

        # Valence blend: camera is primary (0.7); graph MoodEdge softens spikes (0.3).
        # Arousal is NOT blended — only the camera frame contributes A.
        camera_v, camera_a = emotion_label_to_va(camera_emotion)
        graph_mood = next(
            (e.value for e in ctx.person_attribute_edges if e.edge_type == "mood"),
            None,
        )
        blended_v = (0.7 * camera_v + 0.3 * graph_mood) if graph_mood is not None else camera_v

        # Structured memory: SLOW edges only — text path, never numeric.
        structured_memory = format_slow_edges(ctx.person_attribute_edges)

        return BridgeInput(
            valence=blended_v,
            arousal=camera_a,
            tier=tier,
            structured_memory=structured_memory,
        )

    def post_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        pad_result: dict,
    ) -> None:
        """
        Write PAD turn output to the KG via exactly three edge upserts.

        D (Dominance) is NOT written — it is re-derived from relationship edges
        each turn and must never be persisted as its own edge.
        """
        if person_id is None:
            return

        _ensure_person_node(self._store, person_id)
        robot_node_id = _ensure_robot_node(self._store, robot_id)

        p, a, _d = pad_result["pad_state"]
        prov = _prov(robot_id)

        # AttentionEdge expects [0, 1]; rescale PAD arousal from [-1, 1]
        attention_value = max(0.0, min(1.0, (a + 1.0) / 2.0))

        # INVARIANT: exactly three writes, no graph scan (apply_delta contract)
        self._store.apply_delta(
            edges=[
                MoodEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=max(-1.0, min(1.0, p)),
                ),
                AttentionEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=attention_value,
                ),
                InteractionCountEdge(
                    source_id=robot_node_id,
                    target_id=person_id,
                    provenance=prov,
                    count=1,
                ),
            ]
        )
