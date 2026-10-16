"""Role/persona layer: attach/detach keeps one role, knowledge stays with its role,
OCEAN maps to words + style, junk rejected, old graphs still load."""

import pytest

from .roles import (
    apply_assignment, attach_role, attached_persona, attached_role, ensure_role,
    ensure_role_topic, library_roles, ocean_style, ocean_words, role_knowledge,
)
from .schema import Embodiment, HasRoleEdge, Provenance, RobotNode, RoleNode
from .store import InMemoryGraphStore

SHY_TUTOR = {"O": 0.0, "C": 0.8, "E": -0.9, "A": 0.6, "N": 0.1}


@pytest.fixture
def store():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    # what the spec seed leaves behind: an anchor-scoped role
    s.upsert_node(RoleNode(id="chatbox:role:companion", descriptor="companion"))
    s.upsert_edge(HasRoleEdge(source_id="chatbox", target_id="chatbox:role:companion",
                              provenance=Provenance(source="spec", confidence=1.0)))
    tutor = ensure_role(s, "tutor", "help with maths", "give hints, not answers")
    ensure_role_topic(s, tutor.id, "fractions", ["One half is two quarters."])
    ensure_role(s, "storyteller")
    return s


def test_attach_detach_moves_one_edge(store):
    attach_role(store, "chatbox", "role:tutor")
    assert attached_role(store, "chatbox").id == "role:tutor"
    attach_role(store, "chatbox", "role:storyteller")
    assert [n.id for _e, n in store.query_neighbors("chatbox", "has_role")] == ["role:storyteller"]
    # detached role keeps its knowledge subgraph, ready to re-attach
    assert role_knowledge(store, "role:tutor") == [("fractions", ["One half is two quarters."])]
    assert store.get_node("chatbox:role:companion") is not None
    assert [r.id for r in library_roles(store)] == ["role:storyteller", "role:tutor"]


def test_persona_words_and_style(store):
    _role, persona = apply_assignment(store, "chatbox", "role:tutor", SHY_TUTOR)
    assert persona.descriptor == "organised, quiet, warm"
    assert attached_persona(store, "chatbox").ocean == SHY_TUTOR
    style = ocean_style(SHY_TUTOR)
    assert len(style) == 3 and "short and low-key" in style[0]     # E is strongest
    assert ocean_words({k: 0.0 for k in "OCEAN"}) == "even-tempered"
    apply_assignment(store, "chatbox", "role:tutor", {**SHY_TUTOR, "E": 0.9})
    assert len(store.query_neighbors("chatbox", "has_persona")) == 1


@pytest.mark.parametrize("rid,ocean", [("role:nope", SHY_TUTOR),
                                       ("role:tutor", {"O": 1}),
                                       ("role:tutor", {**SHY_TUTOR, "E": "loud"})])
def test_rejects_bad_input(store, rid, ocean):
    with pytest.raises(ValueError):
        apply_assignment(store, "chatbox", rid, ocean)


def test_round_trip(store, tmp_path):
    apply_assignment(store, "chatbox", "role:tutor", SHY_TUTOR)
    p = str(tmp_path / "kg.json")
    store.save(p)
    s2 = InMemoryGraphStore()
    s2.load(p)
    assert attached_role(s2, "chatbox").brief == "help with maths"
    assert role_knowledge(s2, "role:tutor")[0][0] == "fractions"
