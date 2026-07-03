"""
Tests for the shared-TopicNode / Interest layer (topics.py + seeding).
"""

from datetime import timezone

from .schema import (
    AboutEdge, CapabilityNode, Embodiment, HasCapabilityEdge, HasInterestEdge,
    InterestNode, PersonNode, Provenance, RobotNode,
)
from .store import InMemoryGraphStore
from .topics import (
    interest_id, person_interests, resolve_topic, shared_topics, topic_id,
)


def _prov():
    return Provenance(source="test", confidence=1.0)


def _seed_shared(store):
    """robot reaches jazz via its capability; person via interest 'music'."""
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id="jay", display_name="Jay"))
    # robot: has_capability -> capability -> about -> topic
    cap = CapabilityNode(id="chatbox:capability", items=["tells stories", "knows jazz"])
    store.upsert_node(cap)
    store.upsert_edge(HasCapabilityEdge(source_id="chatbox", target_id=cap.id, provenance=_prov()))
    t1 = resolve_topic(store, "jazz")
    store.upsert_edge(AboutEdge(source_id=cap.id, target_id=t1.id, provenance=_prov()))
    # person: has_interest -> interest -> about -> topic
    inode = InterestNode(id=interest_id("jay", "music"), label="music")
    store.upsert_node(inode)
    store.upsert_edge(HasInterestEdge(source_id="jay", target_id=inode.id, provenance=_prov()))
    t2 = resolve_topic(store, "jazz")
    store.upsert_edge(AboutEdge(source_id=inode.id, target_id=t2.id, provenance=_prov()))
    return t1, t2


_CAP_ID = "chatbox:capability"


def test_resolve_topic_is_one_shared_node():
    store = InMemoryGraphStore()
    t1, t2 = _seed_shared(store)
    assert t1.id == t2.id == topic_id("jazz") == "topic:jazz"
    topics = [n for n in store._nodes.values() if n.node_type == "topic"]
    assert len(topics) == 1  # robot 'jazz' and interest-about 'jazz' collapse to one


def test_shared_topics_traversal():
    store = InMemoryGraphStore()
    _seed_shared(store)
    assert shared_topics(store, "jay", "chatbox") == ["jazz"]


def test_shared_topics_empty_when_human_interest_removed():
    store = InMemoryGraphStore()
    _seed_shared(store)
    store.delete_edge(interest_id("jay", "music"), "topic:jazz", "about")
    assert shared_topics(store, "jay", "chatbox") == []


def test_shared_topics_empty_when_robot_capability_topic_removed():
    store = InMemoryGraphStore()
    _seed_shared(store)
    store.delete_edge(_CAP_ID, "topic:jazz", "about")   # robot no longer reaches jazz
    assert shared_topics(store, "jay", "chatbox") == []


def test_person_interests():
    store = InMemoryGraphStore()
    _seed_shared(store)
    result = person_interests(store, "jay")
    assert len(result) == 1
    interest, topics = result[0]
    assert interest.label == "music"
    assert [t.label for t in topics] == ["jazz"]


def test_interest_id_deterministic():
    assert interest_id("jay", "Music") == interest_id("jay", "music")
    assert interest_id("jay", "music") == "interest:jay:music"
