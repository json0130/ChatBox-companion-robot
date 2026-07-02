"""
Session-event helpers: represent a meetup between a Person and a Robot as one
Event node that both link to, and that accumulates the meetup's turns. This is
how the two subgraphs connect — THROUGH an event — instead of by a direct
Person↔Robot edge. A later meetup creates a new Event node.

Design contract
---------------
* Imports ONLY schema.py and store.py — no PAD, no kg_bridge, no adapters.
  Keeps graph_relationship/ copy-pasteable.
* The Event node itself holds the conversation (its `turns` list), so the graph
  (kg_state.json) is the single source of truth — no separate transcript file.
* A person's interaction "count" is the total number of turns across all their
  session events (count_person_turns), which preserves the tier thresholds.
* It does NOT compute rapport/trust or relationship-metric accumulation — that
  is deferred to a later step.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from .schema import EventNode, ParticipatedEdge, Provenance
from .store import GraphStore


def _prov(source: str) -> Provenance:
    return Provenance(source=source, confidence=1.0,
                      timestamp=datetime.now(timezone.utc))


def _person_events(store: GraphStore, person_id: str) -> List[EventNode]:
    """All Event (session) nodes this person participated in (O(neighbours))."""
    return [
        n for _edge, n in store.query_neighbors(person_id, "participated_in")
        if n.node_type == "event"
    ]


def count_person_sessions(store: GraphStore, person_id: str) -> int:
    """Number of distinct meetups (session Event nodes) for this person."""
    return len(_person_events(store, person_id))


def count_person_turns(store: GraphStore, person_id: str) -> int:
    """Total turns across all of a person's sessions (the interaction count)."""
    return sum(e.turn_count for e in _person_events(store, person_id))


def start_session_event(
    store: GraphStore,
    *,
    person_id: str,
    robot_id: str,
    label: Optional[str] = None,
    source: Optional[str] = None,
) -> EventNode:
    """Create a new session Event and link both participants to it.

    Writes 1 EventNode + 2 ParticipatedEdges (person→event, robot→event). The
    person and robot nodes must already exist. Returns the created EventNode.
    """
    event = EventNode(label=label or "session", turn_count=0, turns=[])
    prov = _prov(source or robot_id)
    # apply_delta upserts nodes before edges, so the event exists for the edges.
    store.apply_delta(
        nodes=[event],
        edges=[
            ParticipatedEdge(source_id=person_id, target_id=event.id, provenance=prov),
            ParticipatedEdge(source_id=robot_id, target_id=event.id, provenance=prov),
        ],
    )
    return event


def append_turn(
    store: GraphStore,
    *,
    event_id: str,
    emotion: Optional[str] = None,
    child_message: Optional[str] = None,
    reply: Optional[str] = None,
) -> Optional[EventNode]:
    """Append one turn to a session Event's transcript and bump its turn_count.

    Returns the updated EventNode, or None if event_id is missing/not an event.
    """
    event = store.get_node(event_id)
    if event is None or event.node_type != "event":
        return None
    turns = list(event.turns)
    turns.append({
        "turn": event.turn_count + 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "emotion": emotion,
        "child": child_message,
        "reply": reply,
    })
    updated = event.model_copy(update={"turns": turns, "turn_count": event.turn_count + 1})
    store.upsert_node(updated)
    return updated
