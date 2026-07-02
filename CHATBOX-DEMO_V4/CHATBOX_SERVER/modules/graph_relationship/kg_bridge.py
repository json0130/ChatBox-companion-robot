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
D (Dominance) is derived fresh from KG edges each turn via derive_tier()
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
    RapportEdge,
    RobotNode,
    TrustEdge,
)
from .store import GraphStore, InMemoryGraphStore
from .events import (
    append_turn,
    count_person_sessions,
    count_person_turns,
    start_session_event,
)

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
    rapport: float = 0.0
    trust: float = 0.0
    interaction_count: int = 0


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def emotion_label_to_va(label: Optional[str]) -> tuple[float, float]:
    """Return (valence, arousal) for a camera emotion label. Unknown/None → (0, 0)."""
    if label is None:
        return (0.0, 0.0)
    return _EMOTION_VA.get(label.lower(), (0.0, 0.0))


def _tier_from_edges(relationship_edges: List[AnyEdge]) -> str:
    """
    Core tier logic over a pre-fetched edge list.
    Used by derive_tier() and by tests that build edge lists directly.

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


def derive_tier(person_id: str, robot_id: str, store: GraphStore) -> str:
    """
    Derive the relationship tier from the KG store.

    Scoring is unchanged (_tier_from_edges); only the interaction *count* is now
    rerouted through Event nodes rather than a direct InteractionCountEdge:
      RapportEdge:  person → robot          (read directly)
      TrustEdge:    person → robot          (read directly)
      interaction count = number of the person's Event nodes (count_person_events)

    A synthetic InteractionCountEdge is fed to _tier_from_edges so the pure
    threshold logic — and its tests — stay exactly as before.
    """
    rapport_edge = store.get_edge(person_id, robot_id, "rapport")
    trust_edge   = store.get_edge(person_id, robot_id, "trust")
    edges = [e for e in (rapport_edge, trust_edge) if e is not None]
    count = count_person_turns(store, person_id)
    if count > 0:
        edges.append(InteractionCountEdge(
            source_id=robot_id, target_id=person_id,
            provenance=_prov(robot_id), count=count,
        ))
    return _tier_from_edges(edges)


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
        # One session (meetup) per person for the lifetime of this bridge.
        # A fresh bridge = a fresh run = a new session, so next-meetup turns
        # land on a new Event node. person_id -> current session event id.
        self._session_event: dict[str, str] = {}

    def pre_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        camera_emotion: Optional[str],
        camera_va: Optional[tuple] = None,
    ) -> BridgeInput:
        """
        Collect graph context for this turn and return frozen PAD inputs.

        camera_va: optional (valence, arousal) from a weighted softmax blend;
                   when provided, the emotion-label lookup table is bypassed.

        Cold-start / null-person path: does not touch the store; returns camera
        VA unblended, tier "unknown", empty structured_memory.
        """
        if camera_va is not None:
            camera_v, camera_a = camera_va
        else:
            camera_v, camera_a = emotion_label_to_va(camera_emotion)

        if person_id is None or self._store.get_node(person_id) is None:
            return BridgeInput(valence=camera_v, arousal=camera_a,
                               tier="unknown", structured_memory="")

        ctx = self._store.get_person_context(person_id)

        # Tier: derived fresh from the three KG relationship edges each turn.
        # D is never read from the graph here.
        tier = derive_tier(person_id, robot_id, self._store)

        # Read rapport / trust for prompt injection. Interaction count is
        # rerouted through Event nodes (count_person_events), not a direct edge.
        rapport = 0.0
        trust = 0.0
        for edge in ctx.relationship_edges:
            if edge.edge_type == "rapport":
                rapport = edge.weight
            elif edge.edge_type == "trust":
                trust = edge.weight
        count = count_person_turns(self._store, person_id)

        # Valence blend: camera is primary (0.7); graph MoodEdge softens spikes (0.3).
        # Arousal is NOT blended — only the camera frame contributes A.
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
            rapport=rapport,
            trust=trust,
            interaction_count=count,
        )

    def post_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        pad_result: dict,
        *,
        emotion: Optional[str] = None,
        child_message: Optional[str] = None,
        reply: Optional[str] = None,
    ) -> None:
        """
        Write PAD turn output to the KG.

        Writes two self-attribute edges (MoodEdge, AttentionEdge) and appends
        this turn to the current SESSION Event node linking the person and robot
        (creating that Event on the first turn of the session). The Event REPLACES
        the old direct InteractionCountEdge: interaction count is now the total
        turns across a person's sessions (see derive_tier).

        The optional emotion / child_message / reply are stored on the Event's
        turn list, so the graph holds the conversation (no separate transcript).

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

        # Self-attribute edges (mood + attention). No graph scan (apply_delta contract).
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
            ]
        )

        # Reroute interaction through a per-meetup session Event. Start one on
        # the session's first turn; append every turn to it thereafter.
        event_id = self._session_event.get(person_id)
        if event_id is None:
            label = f"session {count_person_sessions(self._store, person_id) + 1}"
            event = start_session_event(
                self._store, person_id=person_id, robot_id=robot_node_id,
                label=label, source=robot_id,
            )
            event_id = event.id
            self._session_event[person_id] = event_id
        append_turn(
            self._store, event_id=event_id,
            emotion=emotion, child_message=child_message, reply=reply,
        )


# ---------------------------------------------------------------------------
# KG — scripting / REPL façade over InMemoryGraphStore
# ---------------------------------------------------------------------------

class KG:
    """
    Thin convenience wrapper for scripts and notebooks.

    Methods use SET semantics (not accumulate): calling set_interaction_count(r, p, 6)
    after set_interaction_count(r, p, 1) yields count=6, not 7.

    Edge directions match the production KGBridge contract:
      RapportEdge / TrustEdge:   person → robot
      InteractionCountEdge:      robot  → person
    """

    def __init__(self) -> None:
        self._store = InMemoryGraphStore()

    def _prov(self) -> Provenance:
        return Provenance(
            source="kg-facade", confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )

    def _ensure(self, person_id: str, robot_id: str) -> None:
        if self._store.get_node(person_id) is None:
            self._store.upsert_node(PersonNode(id=person_id))
        if self._store.get_node(robot_id) is None:
            self._store.upsert_node(RobotNode(
                id=robot_id, name=robot_id,
                embodiment=_ROBOT_EMBODIMENT.get(robot_id.lower(), Embodiment.CAT),
            ))

    def set_rapport(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        self._store.upsert_edge(RapportEdge(
            source_id=person_id, target_id=robot_id,
            provenance=self._prov(), weight=weight,
        ))

    def set_trust(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        self._store.upsert_edge(TrustEdge(
            source_id=person_id, target_id=robot_id,
            provenance=self._prov(), weight=weight,
        ))

    def set_interaction_count(self, robot_id: str, person_id: str, count: int) -> None:
        """Set the interaction count to an exact value (delete-then-insert)."""
        self._ensure(person_id, robot_id)
        self._store.delete_edge(robot_id, person_id, "interaction_count")
        if count > 0:
            self._store.upsert_edge(InteractionCountEdge(
                source_id=robot_id, target_id=person_id,
                provenance=self._prov(), count=count,
            ))
