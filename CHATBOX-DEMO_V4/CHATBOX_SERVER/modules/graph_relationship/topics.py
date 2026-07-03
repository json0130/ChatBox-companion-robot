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
from typing import List, Tuple

from .schema import InterestNode, TopicNode
from .store import GraphStore


def normalize_label(label: str) -> str:
    """Stable slug: lowercase alnum, dashes for the rest."""
    return re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower()).strip("-")


def topic_id(label: str) -> str:
    return f"topic:{normalize_label(label)}"


def interest_id(person_id: str, label: str) -> str:
    return f"interest:{person_id}:{normalize_label(label)}"


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


def shared_topics(store: GraphStore, person_id: str, robot_id: str) -> List[str]:
    """Labels of topics the robot KNOWS that the person reaches via an interest.

    Traverses person -> interest -> topic (one extra hop) intersected with
    robot -> knows -> topic. Index-based; O(neighbours), no full scan.
    """
    robot_topics = {t.id: t for t in _neighbors_of_type(store, robot_id, "knows", "topic")}
    person_ids = _person_topic_ids(store, person_id)
    shared_ids = person_ids & set(robot_topics)
    return sorted(robot_topics[tid].label for tid in shared_ids)
