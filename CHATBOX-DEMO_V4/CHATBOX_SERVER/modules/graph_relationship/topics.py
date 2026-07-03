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
    topic_labels: Optional[List[str]] = None, *, source: Optional[str] = None,
) -> Optional[InterestNode]:
    """Upsert a person's Interest (deterministic id) with has_interest, and an
    about-edge to each shared Topic. Idempotent — re-adding does not duplicate.

    The person node must already exist.
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
    return inode


def resolve_topic(store: GraphStore, label: str) -> TopicNode:
    """Upsert and return the shared TopicNode for `label` (deterministic id).

    Re-resolving the same label returns the same node — so both the robot and a
    human interest land on ONE Topic node.
    """
    node = TopicNode(id=topic_id(label), label=str(label))
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
    """Topics the robot reaches via has_capability -> capability -> about -> topic."""
    seen: dict = {}
    for capability in _neighbors_of_type(store, robot_id, "has_capability", "capability"):
        for topic in _neighbors_of_type(store, capability.id, "about", "topic"):
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
