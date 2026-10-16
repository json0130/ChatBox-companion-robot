"""
Role + persona layer — what the robot is FOR, how it talks, and what it knows for it.

  robot --has_role-----> Role("tutor", brief, style_hint)       (attached role)
  Role  --role_knows---> RoleTopic("fractions", facts)          (role knowledge)
  robot --has_persona--> Persona(descriptor, ocean)              (assigned persona)

Library roles (`role:<slug>`) all live in the graph; a robot PLAYS the one its
has_role edge points at. Attach / detach = moving that edge, the same way a culture
lens attaches to whoever is in front of the camera: the other roles and their
knowledge stay in the graph, unattached, and contribute nothing to the prompt.

Persona is Big-Five (OCEAN, each in [-1, 1], 0 = population average) mapped
straight to interaction-style instructions — no PAD, no affect dynamics.

Design contract (same as cultures.py): imports ONLY schema.py + store.py (+ the pure
topics slug). No LLM, no PAD, nothing in modules/. Stateless functions only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .schema import (
    HasPersonaEdge, HasRoleEdge, PersonaNode, Provenance, RoleKnowsEdge, RoleNode,
    RoleTopicNode,
)
from .store import GraphStore
from .topics import normalize_label

TRAITS = ("O", "C", "E", "A", "N")


def _prov(source: Optional[str]) -> Provenance:
    return Provenance(source=source or "roles", confidence=1.0,
                      timestamp=datetime.now(timezone.utc))


# ── ids ───────────────────────────────────────────────────────────────────────

def role_id(label: str) -> str:
    return f"role:{normalize_label(label)}"


def role_topic_id(rid: str, label: str) -> str:
    return f"rk:{rid.split(':', 1)[-1]}:{normalize_label(label)}"


def persona_id(robot_id: str) -> str:
    return f"{robot_id}:persona:assigned"


# ── library (seeded) ──────────────────────────────────────────────────────────

def ensure_role(store: GraphStore, label: str, brief: str = "",
                style_hint: str = "") -> RoleNode:
    """Upsert a library role. Re-seeding refreshes brief/style_hint."""
    node = RoleNode(id=role_id(label), descriptor=str(label).strip(),
                    brief=brief.strip(), style_hint=style_hint.strip())
    store.upsert_node(node)
    return node


def ensure_role_topic(store: GraphStore, rid: str, label: str,
                      facts: Optional[List[str]] = None) -> RoleTopicNode:
    """Upsert a knowledge topic under a role and link it (role_knows)."""
    node = RoleTopicNode(id=role_topic_id(rid, label), label=str(label).strip(),
                         facts=[str(f).strip() for f in (facts or []) if str(f).strip()])
    store.upsert_node(node)
    store.upsert_edge(RoleKnowsEdge(source_id=rid, target_id=node.id,
                                    provenance=_prov("role-seed")))
    return node


def library_roles(store: GraphStore) -> List[RoleNode]:
    return sorted((n for n in list(store._nodes.values())   # noqa: SLF001
                   if n.node_type == "role" and n.id.startswith("role:")),
                  key=lambda n: n.id)


def role_knowledge(store: GraphStore, rid: str) -> List[Tuple[str, List[str]]]:
    """[(topic label, facts), ...] for one role, sorted by label."""
    return sorted(((n.label, list(n.facts))
                   for _e, n in store.query_neighbors(rid, "role_knows")
                   if n.node_type == "role_topic"), key=lambda x: x[0])


# ── attach / detach ───────────────────────────────────────────────────────────

def _detach(store: GraphStore, robot_id: str, edge_type: str) -> None:
    for _e, n in store.query_neighbors(robot_id, edge_type):
        store.delete_edge(robot_id, n.id, edge_type)


def attached_role(store: GraphStore, robot_id: str) -> Optional[RoleNode]:
    roles = sorted((n for _e, n in store.query_neighbors(robot_id, "has_role")
                    if n.node_type == "role"), key=lambda n: n.id)
    return roles[0] if roles else None


def attach_role(store: GraphStore, robot_id: str, rid: str,
                *, source: Optional[str] = None) -> RoleNode:
    """Detach whatever role the robot plays and attach `rid` (a library role)."""
    node = store.get_node(rid)
    if node is None or node.node_type != "role":
        raise ValueError(f"unknown role {rid!r}")
    _detach(store, robot_id, "has_role")
    store.upsert_edge(HasRoleEdge(source_id=robot_id, target_id=rid,
                                  provenance=_prov(source)))
    return node


# ── persona: OCEAN -> words + interaction style ───────────────────────────────

# A trait within ±EDGE of the average says nothing distinctive: no word, no rule.
EDGE = 0.25

# (word below average, word above average)
_WORDS = {
    "O": ("practical", "curious"),
    "C": ("easygoing", "organised"),
    "E": ("quiet", "outgoing"),
    "A": ("frank", "warm"),
    "N": ("calm", "sensitive"),
}

# Written as things to DO, not adjectives to be: models apply a trait label loosely,
# but a behaviour ("ask one question at a time") is followed and can be checked
# against a transcript.
_STYLE = {
    "O": ("Stick to concrete, everyday examples; don't wander into hypotheticals.",
          "Be imaginative: ask 'what if…' questions and make playful connections."),
    "C": ("Keep it loose and go with the flow; it's fine to follow a tangent.",
          "Be structured: one point or step at a time, and finish what you started."),
    "E": ("Keep replies short and low-key, let them lead, and don't push new topics.",
          "Bring energy: sound enthusiastic, share your own ideas, suggest what to do next."),
    "A": ("Be direct and honest; you may politely disagree or point out a mistake.",
          "Be encouraging: acknowledge what they said before adding anything, and praise effort."),
    "N": ("Stay steady and reassuring, even when something goes wrong.",
          "Show gentle caution: check they're okay and that things make sense before moving on."),
}


def clean_ocean(ocean: Dict) -> Dict[str, float]:
    """Validate an OCEAN dict → each trait a float clamped to [-1, 1]."""
    try:
        return {k: max(-1.0, min(1.0, float(ocean[k]))) for k in TRAITS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ocean needs numeric {'/'.join(TRAITS)}") from exc


def ocean_words(ocean: Dict[str, float]) -> str:
    words = [_WORDS[k][ocean[k] > 0] for k in TRAITS if abs(ocean[k]) >= EDGE]
    return ", ".join(words) or "even-tempered"


def ocean_style(ocean: Dict[str, float]) -> List[str]:
    """Interaction-style instructions for the prompt, strongest trait first."""
    ranked = sorted((k for k in TRAITS if abs(ocean[k]) >= EDGE),
                    key=lambda k: -abs(ocean[k]))
    return [_STYLE[k][ocean[k] > 0] for k in ranked]


def assign_persona(store: GraphStore, robot_id: str, ocean: Dict,
                   *, source: Optional[str] = None) -> PersonaNode:
    """Detach the robot's persona and attach one built from `ocean`."""
    ocean = clean_ocean(ocean)
    node = PersonaNode(id=persona_id(robot_id), descriptor=ocean_words(ocean),
                       ocean=ocean)
    _detach(store, robot_id, "has_persona")
    store.upsert_node(node)
    store.upsert_edge(HasPersonaEdge(source_id=robot_id, target_id=node.id,
                                     provenance=_prov(source)))
    return node


def attached_persona(store: GraphStore, robot_id: str) -> Optional[PersonaNode]:
    ps = sorted((n for _e, n in store.query_neighbors(robot_id, "has_persona")
                 if n.node_type == "persona"), key=lambda n: n.id)
    return ps[0] if ps else None


def apply_assignment(store: GraphStore, robot_id: str, rid: str, ocean: Dict,
                     *, source: Optional[str] = None) -> Tuple[RoleNode, PersonaNode]:
    """Attach role `rid` and the OCEAN persona to the robot (one assignment)."""
    return (attach_role(store, robot_id, rid, source=source),
            assign_persona(store, robot_id, ocean, source=source))
