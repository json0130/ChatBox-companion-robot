"""
Shared-TopicNode helpers: a single TopicNode is reached from both sides.

  robot  --knows-------------------------> Topic("jazz")
  person --has_interest--> Interest("music") --about--> Topic("jazz")

resolve_topic() gives every topic a deterministic id ("topic:" + slug), so the
robot's "jazz" and a human interest's "jazz" resolve to ONE node. Read helpers
traverse index-based (O(neighbours)); no full-graph scans.

Design contract: imports ONLY schema.py + store.py — no PAD, no kg_bridge.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schema import (
    AboutEdge, HasInterestEdge, InterestNode, Provenance, TopicNode,
)
from .store import GraphStore


def _prov(source: Optional[str]) -> Provenance:
    return Provenance(source=source or "topics", confidence=1.0,
                      timestamp=datetime.now(timezone.utc))


def normalize_label(label: str) -> str:
    """Stable slug: lowercase alnum, dashes for the rest."""
    return re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower()).strip("-")


def topic_id(label: str) -> str:
    return f"topic:{normalize_label(label)}"


def interest_id(person_id: str, label: str) -> str:
    return f"interest:{person_id}:{normalize_label(label)}"


def add_person_interest(
    store: GraphStore, person_id: str, interest_label: str,
    topic_labels: Optional[List[str]] = None, *, summary: Optional[str] = None,
    source: Optional[str] = None,
) -> Optional[InterestNode]:
    """Upsert a person's Interest (deterministic id) with has_interest, and an
    about-edge to each shared Topic. Idempotent — re-adding does not duplicate.

    If `summary` is given, it is attached as a per-person note on each of the
    interest's Topic nodes. The person node must already exist.
    """
    label = str(interest_label).strip()
    if not label:
        return None
    inode = InterestNode(id=interest_id(person_id, label), label=label)
    store.upsert_node(inode)
    store.upsert_edge(HasInterestEdge(
        source_id=person_id, target_id=inode.id, provenance=_prov(source)))
    for t in topic_labels or []:
        t = str(t).strip()
        if not t:
            continue
        topic = resolve_topic(store, t)
        store.upsert_edge(AboutEdge(
            source_id=inode.id, target_id=topic.id, provenance=_prov(source)))
        if summary:
            add_topic_note(store, topic, person_id, summary)
    return inode


def resolve_topic(store: GraphStore, label: str) -> TopicNode:
    """Get-or-create the shared TopicNode for `label` (deterministic id).

    Re-resolving the same label returns the SAME existing node — so both the
    robot and a human interest land on ONE Topic node, and existing topic notes
    are preserved (not overwritten).
    """
    tid = topic_id(label)
    existing = store.get_node(tid)
    if existing is not None and existing.node_type == "topic":
        return existing
    node = TopicNode(id=tid, label=str(label))
    store.upsert_node(node)
    return node


def add_topic_note(
    store: GraphStore, topic_ref, person_id: str, text: str,
) -> Optional[TopicNode]:
    """Append a per-person conversation summary to a Topic's `notes` list.

    Idempotent: an identical (person, text) note is not appended twice.
    `topic_ref` may be a TopicNode or a topic label.
    """
    text = str(text).strip()
    if not text:
        return None
    tid = topic_ref.id if hasattr(topic_ref, "id") else topic_id(topic_ref)
    node = store.get_node(tid)
    if node is None or node.node_type != "topic":
        return None
    notes = list(node.notes)
    if any(n.get("person") == person_id and n.get("text") == text for n in notes):
        return node
    notes.append({
        "person": person_id, "text": text,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    node = node.model_copy(update={"notes": notes})
    store.upsert_node(node)
    return node


def _neighbors_of_type(store: GraphStore, node_id: str, edge_type: str, node_type: str):
    return [
        n for _e, n in store.query_neighbors(node_id, edge_type)
        if n.node_type == node_type
    ]


def person_interests(store: GraphStore, person_id: str) -> List[Tuple[InterestNode, List[TopicNode]]]:
    """[(InterestNode, [TopicNode it is about]), ...] for one person (index-based)."""
    out: List[Tuple[InterestNode, List[TopicNode]]] = []
    for interest in _neighbors_of_type(store, person_id, "has_interest", "interest"):
        topics = _neighbors_of_type(store, interest.id, "about", "topic")
        out.append((interest, topics))
    return out


def _person_topic_ids(store: GraphStore, person_id: str) -> set:
    """Topic ids the person reaches via any interest -> about -> topic path."""
    ids = set()
    for _interest, topics in person_interests(store, person_id):
        ids.update(t.id for t in topics)
    return ids


def robot_topics(store: GraphStore, robot_id: str) -> List[TopicNode]:
    """Topics the robot reaches via
    has_capability -> capability(hub) -> has_skill -> skill -> about -> topic."""
    seen: dict = {}
    for capability in _neighbors_of_type(store, robot_id, "has_capability", "capability"):
        for skill in _neighbors_of_type(store, capability.id, "has_skill", "skill"):
            for topic in _neighbors_of_type(store, skill.id, "about", "topic"):
                seen[topic.id] = topic
    return list(seen.values())


def shared_topics(store: GraphStore, person_id: str, robot_id: str) -> List[str]:
    """Labels of topics BOTH sides reach — the robot via its capability and the
    person via an interest — i.e. the shared TopicNodes.

    Traverses robot -> capability -> about -> topic intersected with
    person -> interest -> about -> topic. Index-based; O(neighbours), no scan.
    """
    robot_by_id = {t.id: t for t in robot_topics(store, robot_id)}
    person_ids = _person_topic_ids(store, person_id)
    shared_ids = person_ids & set(robot_by_id)
    return sorted(robot_by_id[tid].label for tid in shared_ids)
