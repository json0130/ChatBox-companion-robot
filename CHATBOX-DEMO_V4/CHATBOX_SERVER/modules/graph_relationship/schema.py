"""
Dual-cluster relational knowledge graph schema for the CHATBOX multi-robot system.

Architecture overview
---------------------
The graph holds two entity clusters:

  Robot cluster  — one node per robot (CHATBOX cat, elephant).
                   Robots own *Relationship* edges that track the evolving bond
                   with each child (rapport, trust, disclosure depth, etc.).

  Person cluster — one node per child user plus Topic/Event nodes that
                   describe the child's current mental state and history.
                   *PersonAttribute* edges attach state (mood, attention, …)
                   and stable traits/preferences to the person.

Both robots share ONE graph instance so they read the same person/relationship
data. Every edge carries a Provenance record (source + timestamp + confidence)
so attribution and later reliability-weighting are always available.

Edge types are tagged FAST or SLOW (Timescale enum).  The update-policy module
(built separately) branches on this flag to decide decay cadence and promotion
thresholds.  Nothing in this module performs updates — schema only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Embodiment(str, Enum):
    CAT = "CAT"
    ELEPHANT = "ELEPHANT"


class Timescale(str, Enum):
    """Decay cadence hint consumed by the update-policy module."""
    FAST = "FAST"   # mood, attention, current_topic — decay within a session
    SLOW = "SLOW"   # traits, preferences — stable across sessions


# ---------------------------------------------------------------------------
# Provenance — required on every edge
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Who wrote this edge, when, and how confident."""
    source: str = Field(
        ...,
        description="Writer identifier, e.g. 'robot:cat', 'robot:elephant', 'sensor:emotion'",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class RobotNode(BaseModel):
    """One node per robot in the system (CAT / ELEPHANT)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    embodiment: Embodiment
    persona_traits: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    node_type: Literal["robot"] = "robot"


class PersonNode(BaseModel):
    """One node per child user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    display_name: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_type: Literal["person"] = "person"


class TopicNode(BaseModel):
    """A concept or subject that a person engages with."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    node_type: Literal["topic"] = "topic"


class EventNode(BaseModel):
    """A discrete interaction event."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_type: Literal["event"] = "event"


# Union alias used by GraphSchema
AnyNode = Union[RobotNode, PersonNode, TopicNode, EventNode]


# ---------------------------------------------------------------------------
# Edge base — provenance required on all edges
# ---------------------------------------------------------------------------

class EdgeBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    provenance: Provenance


# ---------------------------------------------------------------------------
# Relationship edges  (Robot → Person)
# ---------------------------------------------------------------------------

class RapportEdge(EdgeBase):
    """Perceived warmth / positive affect between robot and child."""
    edge_type: Literal["rapport"] = "rapport"
    weight: float = Field(..., ge=0.0, le=1.0)


class TrustEdge(EdgeBase):
    """Child's willingness to rely on the robot."""
    edge_type: Literal["trust"] = "trust"
    weight: float = Field(..., ge=0.0, le=1.0)


class DisclosureDepthEdge(EdgeBase):
    """Depth of personal information the child shares."""
    edge_type: Literal["disclosure_depth"] = "disclosure_depth"
    weight: float = Field(..., ge=0.0, le=1.0)


class InteractionCountEdge(EdgeBase):
    """Running count of completed turns / exchanges."""
    edge_type: Literal["interaction_count"] = "interaction_count"
    count: int = Field(..., ge=0)


RelationshipEdge = Union[
    RapportEdge, TrustEdge, DisclosureDepthEdge, InteractionCountEdge
]


# ---------------------------------------------------------------------------
# Person-attribute edges  (Person → Topic | scalar state)
# ---------------------------------------------------------------------------

class MoodEdge(EdgeBase):
    """Current affective state — FAST, decays between sessions."""
    edge_type: Literal["mood"] = "mood"
    value: float = Field(..., ge=-1.0, le=1.0)  # valence: -1 sad … +1 happy
    timescale: Timescale = Timescale.FAST


class AttentionEdge(EdgeBase):
    """Estimated engagement level — FAST."""
    edge_type: Literal["attention"] = "attention"
    value: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.FAST


class CurrentTopicEdge(EdgeBase):
    """Person → Topic for the active conversational subject — FAST."""
    edge_type: Literal["current_topic"] = "current_topic"
    timescale: Timescale = Timescale.FAST


class TraitEdge(EdgeBase):
    """Stable personality/character attribute — SLOW."""
    edge_type: Literal["trait"] = "trait"
    value: Any
    timescale: Timescale = Timescale.SLOW


class PreferenceEdge(EdgeBase):
    """Person → Topic affinity that persists across sessions — SLOW."""
    edge_type: Literal["preference"] = "preference"
    weight: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.SLOW


PersonAttributeEdge = Union[
    MoodEdge, AttentionEdge, CurrentTopicEdge, TraitEdge, PreferenceEdge
]

AnyEdge = Union[RelationshipEdge, PersonAttributeEdge]


# ---------------------------------------------------------------------------
# GraphSchema — in-memory container
# ---------------------------------------------------------------------------

class GraphSchema(BaseModel):
    """
    In-memory dual-cluster graph.

    Robot-cluster nodes and person-cluster nodes are stored in the same dict
    keyed by node id so cross-cluster relationship edges resolve in O(1).
    Persistence is out of scope here — call to_dict() / from_dict() to hand off
    to a storage backend without modifying this class.
    """

    nodes: Dict[str, AnyNode] = Field(default_factory=dict)
    edges: Dict[str, AnyEdge] = Field(default_factory=dict)

    # -- node operations -------------------------------------------------- #

    def add_node(self, node: AnyNode) -> AnyNode:
        """Register a node; silently replaces an existing node with the same id."""
        self.nodes[node.id] = node
        return node

    def get_node(self, node_id: str) -> Optional[AnyNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[AnyNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    # -- edge operations -------------------------------------------------- #

    def add_edge(self, edge: AnyEdge) -> AnyEdge:
        """
        Register an edge.

        Raises ValueError if:
        - source_id or target_id do not exist in the graph
        - provenance is absent (enforced by Pydantic, but double-checked here)
        """
        if edge.source_id not in self.nodes:
            raise ValueError(
                f"source_id '{edge.source_id}' does not exist in the graph"
            )
        if edge.target_id not in self.nodes:
            raise ValueError(
                f"target_id '{edge.target_id}' does not exist in the graph"
            )
        # provenance is a required field on EdgeBase; Pydantic rejects edges
        # without it at construction time, but we re-assert for clarity.
        if edge.provenance is None:
            raise ValueError("Every edge must carry a Provenance record")
        self.edges[edge.id] = edge
        return edge

    def get_edge(self, edge_id: str) -> Optional[AnyEdge]:
        return self.edges.get(edge_id)

    def get_edges_by_type(self, edge_type: str) -> List[AnyEdge]:
        return [e for e in self.edges.values() if e.edge_type == edge_type]

    def get_edges_between(self, source_id: str, target_id: str) -> List[AnyEdge]:
        return [
            e for e in self.edges.values()
            if e.source_id == source_id and e.target_id == target_id
        ]

    # -- serialisation ---------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict; a storage backend can persist this."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphSchema":
        """Reconstruct from a plain dict produced by to_dict()."""
        return cls.model_validate(data)
