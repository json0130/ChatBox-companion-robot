"""
Storage abstraction layer for the dual-cluster relational knowledge graph.

Call-site code (response loop, learning loop) depends only on GraphStore —
never on a concrete backend.  Two implementations are provided:

  InMemoryGraphStore   — dict-backed, fully working, for dev and tests.
  PluggableGraphStore  — thin delegation layer over an injected BackendAdapter;
                         swap in SQLite, an embedded graph DB, or a cloud store
                         by subclassing BackendAdapter.

Indexing contract
-----------------
Every method that reads edges is O(neighbors of the queried node), not
O(whole graph).  InMemoryGraphStore maintains a _node_edge_index that maps
each node_id to the set of edge_ids touching it (source or target).  Any
BackendAdapter implementation must provide an equivalent indexed read path.
"""

from __future__ import annotations

import abc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import Field, TypeAdapter
from typing import Annotated

from .schema import (
    AnyEdge,
    AnyNode,
    InteractionCountEdge,
)

# ---------------------------------------------------------------------------
# Discriminated TypeAdapters for serialisation round-trips
# ---------------------------------------------------------------------------
# Pydantic v2 flattens nested Union aliases so AnyEdge/AnyNode work directly
# with an explicit discriminator annotation.

_AnyNodeDisc = Annotated[AnyNode, Field(discriminator="node_type")]
_AnyEdgeDisc = Annotated[AnyEdge, Field(discriminator="edge_type")]
_node_adapter: TypeAdapter = TypeAdapter(_AnyNodeDisc)
_edge_adapter: TypeAdapter = TypeAdapter(_AnyEdgeDisc)

# ---------------------------------------------------------------------------
# Edge-type classification (consumed by get_person_context)
# ---------------------------------------------------------------------------

_PERSON_ATTRIBUTE_TYPES: frozenset = frozenset(
    {"mood", "attention", "current_topic", "trait", "preference"}
)
_RELATIONSHIP_TYPES: frozenset = frozenset(
    {"rapport", "trust", "disclosure_depth", "interaction_count"}
)


# ---------------------------------------------------------------------------
# PersonContext — snapshot the response loop reads each turn
# ---------------------------------------------------------------------------

@dataclass
class PersonContext:
    """
    All graph data relevant to one child, gathered in a single O(neighbors) pass.

    person_attribute_edges — mood, attention, current_topic, trait, preference
    relationship_edges     — rapport, trust, disclosure_depth, interaction_count
                             (Robot→Person; both robots' data lives here)
    """
    person_attribute_edges: List[AnyEdge] = field(default_factory=list)
    relationship_edges: List[AnyEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Merge helper (shared by InMemoryGraphStore and PluggableGraphStore)
# ---------------------------------------------------------------------------

def _merge_edge(existing: AnyEdge, incoming: AnyEdge) -> AnyEdge:
    """
    Merge an incoming write onto a stored edge.

    Rules:
    - InteractionCountEdge: accumulate count, do not overwrite.
    - All other types: replace value/weight.
    - Provenance is replaced only when the incoming timestamp >= stored.
    - Returns existing unchanged if the incoming write is stale.
    """
    inc_ts = incoming.provenance.timestamp
    sto_ts = existing.provenance.timestamp
    # Normalise to UTC-aware for safe comparison
    if inc_ts.tzinfo is None:
        inc_ts = inc_ts.replace(tzinfo=timezone.utc)
    if sto_ts.tzinfo is None:
        sto_ts = sto_ts.replace(tzinfo=timezone.utc)

    if inc_ts < sto_ts:
        return existing  # stale write — discard

    if isinstance(existing, InteractionCountEdge) and isinstance(incoming, InteractionCountEdge):
        return existing.model_copy(
            update={
                "count": existing.count + incoming.count,
                "provenance": incoming.provenance,
            }
        )

    # Replace value/weight and provenance; preserve the stored edge's canonical id
    return incoming.model_copy(update={"id": existing.id})


# ---------------------------------------------------------------------------
# GraphStore — abstract interface
# ---------------------------------------------------------------------------

class GraphStore(abc.ABC):
    """
    Backend-agnostic interface for the dual-cluster graph.

    No SQL, no file paths, no cloud SDK identifiers appear in any method
    signature.  Inject a concrete backend via PluggableGraphStore.
    """

    @abc.abstractmethod
    def upsert_node(self, node: AnyNode) -> AnyNode:
        """Create or fully replace a node.  Returns the stored node."""
        ...

    @abc.abstractmethod
    def get_node(self, node_id: str) -> Optional[AnyNode]:
        """Return the node or None."""
        ...

    @abc.abstractmethod
    def upsert_edge(self, edge: AnyEdge) -> AnyEdge:
        """
        Create or update the edge identified by (source_id, target_id, edge_type).

        Merge rules are defined in _merge_edge:
        - InteractionCountEdge accumulates; all others replace.
        - Provenance replaced when incoming timestamp >= stored.
        """
        ...

    @abc.abstractmethod
    def get_edge(
        self, src_id: str, dst_id: str, edge_type: str
    ) -> Optional[AnyEdge]:
        """Return the edge matching (src_id, dst_id, edge_type), or None."""
        ...

    @abc.abstractmethod
    def query_neighbors(
        self, node_id: str, edge_type: Optional[str] = None
    ) -> List[Tuple[AnyEdge, AnyNode]]:
        """
        Return (edge, neighbor) pairs for all edges touching node_id.
        Filtered to edge_type when provided.  O(neighbors) — not O(graph).
        """
        ...

    @abc.abstractmethod
    def get_person_context(self, person_id: str) -> PersonContext:
        """
        Return all graph data for one child in a single O(neighbors) pass.
        This is the primary read the response loop calls each turn.
        """
        ...

    @abc.abstractmethod
    def apply_delta(
        self,
        nodes: Optional[List[AnyNode]] = None,
        edges: Optional[List[AnyEdge]] = None,
    ) -> None:
        """
        Batch-upsert a turn delta written by the learning loop.

        INVARIANT: implementations must only touch nodes/edges in the passed-in
        lists — never iterate the full graph.
        """
        ...


# ---------------------------------------------------------------------------
# InMemoryGraphStore
# ---------------------------------------------------------------------------

class InMemoryGraphStore(GraphStore):
    """
    Dict-backed, fully working implementation.  No I/O — for dev and tests.

    Two indexes keep all edge reads O(neighbors):
      _node_edge_index      node_id  -> {edge_ids touching that node}
      _endpoint_type_index  (src, dst, type) -> edge_id
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, AnyNode] = {}
        self._edges: Dict[str, AnyEdge] = {}
        self._node_edge_index: Dict[str, Set[str]] = defaultdict(set)
        self._endpoint_type_index: Dict[Tuple[str, str, str], str] = {}

    # -- nodes ----------------------------------------------------------------

    def upsert_node(self, node: AnyNode) -> AnyNode:
        self._nodes[node.id] = node
        return node

    def get_node(self, node_id: str) -> Optional[AnyNode]:
        return self._nodes.get(node_id)

    # -- edges ----------------------------------------------------------------

    def upsert_edge(self, edge: AnyEdge) -> AnyEdge:
        if edge.source_id not in self._nodes:
            raise ValueError(f"source_id '{edge.source_id}' not in store")
        if edge.target_id not in self._nodes:
            raise ValueError(f"target_id '{edge.target_id}' not in store")

        key = (edge.source_id, edge.target_id, edge.edge_type)
        existing_id = self._endpoint_type_index.get(key)

        if existing_id is None:
            self._edges[edge.id] = edge
            self._endpoint_type_index[key] = edge.id
            self._node_edge_index[edge.source_id].add(edge.id)
            self._node_edge_index[edge.target_id].add(edge.id)
            return edge

        merged = _merge_edge(self._edges[existing_id], edge)
        self._edges[existing_id] = merged
        return merged

    def get_edge(
        self, src_id: str, dst_id: str, edge_type: str
    ) -> Optional[AnyEdge]:
        edge_id = self._endpoint_type_index.get((src_id, dst_id, edge_type))
        return self._edges.get(edge_id) if edge_id else None

    def delete_edge(self, src_id: str, dst_id: str, edge_type: str) -> bool:
        """Remove one edge by (src, dst, type). Returns True if it existed."""
        key = (src_id, dst_id, edge_type)
        edge_id = self._endpoint_type_index.pop(key, None)
        if edge_id is None:
            return False
        edge = self._edges.pop(edge_id)
        self._node_edge_index[edge.source_id].discard(edge_id)
        self._node_edge_index[edge.target_id].discard(edge_id)
        return True

    def query_neighbors(
        self, node_id: str, edge_type: Optional[str] = None
    ) -> List[Tuple[AnyEdge, AnyNode]]:
        # O(neighbors) — iterates only edge_ids indexed for this node
        result = []
        for eid in self._node_edge_index.get(node_id, set()):
            edge = self._edges[eid]
            if edge_type is not None and edge.edge_type != edge_type:
                continue
            neighbor_id = (
                edge.target_id if edge.source_id == node_id else edge.source_id
            )
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is not None:
                result.append((edge, neighbor))
        return result

    def get_person_context(self, person_id: str) -> PersonContext:
        # O(neighbors of person_id) — uses node_edge_index, not a full graph scan
        attr: List[AnyEdge] = []
        rel: List[AnyEdge] = []
        for eid in self._node_edge_index.get(person_id, set()):
            edge = self._edges[eid]
            if edge.edge_type in _PERSON_ATTRIBUTE_TYPES:
                attr.append(edge)
            elif edge.edge_type in _RELATIONSHIP_TYPES:
                rel.append(edge)
        return PersonContext(person_attribute_edges=attr, relationship_edges=rel)

    def apply_delta(
        self,
        nodes: Optional[List[AnyNode]] = None,
        edges: Optional[List[AnyEdge]] = None,
    ) -> None:
        # INVARIANT: iterates only the passed-in lists — never walks self._nodes or self._edges
        for node in nodes or []:
            self.upsert_node(node)
        for edge in edges or []:
            self.upsert_edge(edge)

    # -- JSON persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the full graph to a JSON file using Pydantic model_dump."""
        import json
        data = {
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges.values()],
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        print(f"[KG] saved {len(self._nodes)} nodes, {len(self._edges)} edges → {path}")

    def load(self, path: str, *, quiet: bool = False) -> bool:
        """Load a JSON file written by save() (merges into current state)."""
        import json, os
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            data = json.load(fh)
        for nd in data.get("nodes", []):
            node = _node_adapter.validate_python(nd)
            self._nodes[node.id] = node
        for ed in data.get("edges", []):
            edge = _edge_adapter.validate_python(ed)
            self._edges[edge.id] = edge
            key = (edge.source_id, edge.target_id, edge.edge_type)
            self._endpoint_type_index[key] = edge.id
            self._node_edge_index[edge.source_id].add(edge.id)
            self._node_edge_index[edge.target_id].add(edge.id)
        if not quiet:
            print(f"[KG] loaded {len(self._nodes)} nodes, {len(self._edges)} edges ← {path}")
        return True

    def reload(self, path: str) -> bool:
        """REPLACE all in-memory state with the on-disk graph.

        Used to re-sync with external edits (e.g. deletions made in the viz)
        between turns. Validates the file fully BEFORE clearing, so a missing or
        mid-write file leaves the current graph untouched (returns False).
        """
        import json, os
        if not os.path.exists(path):
            return False
        try:
            with open(path) as fh:
                data = json.load(fh)
            nodes = [_node_adapter.validate_python(nd) for nd in data.get("nodes", [])]
            edges = [_edge_adapter.validate_python(ed) for ed in data.get("edges", [])]
        except Exception:
            return False  # keep current state on any read/parse/validate error

        self._nodes.clear()
        self._edges.clear()
        self._node_edge_index.clear()
        self._endpoint_type_index.clear()
        for node in nodes:
            self._nodes[node.id] = node
        for edge in edges:
            self._edges[edge.id] = edge
            key = (edge.source_id, edge.target_id, edge.edge_type)
            self._endpoint_type_index[key] = edge.id
            self._node_edge_index[edge.source_id].add(edge.id)
            self._node_edge_index[edge.target_id].add(edge.id)
        return True


# ---------------------------------------------------------------------------
# BackendAdapter — seam for concrete storage backends
# ---------------------------------------------------------------------------

class BackendAdapter(abc.ABC):
    """
    Interface for a concrete storage backend.

    All methods pass and return plain dicts (via schema's model_dump / validate)
    so implementations have no Pydantic dependency.  Merge logic lives in
    PluggableGraphStore._merge_before_write; adapters only do raw reads/writes.

    Implement this to drop in:
      - SQLite / SQLAlchemy (on-device)
      - An embedded graph DB (e.g. Kuzu, DuckDB)
      - A cloud store (e.g. Firestore, DynamoDB)
    """

    @abc.abstractmethod
    def upsert_node(self, node_dict: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: persist node_dict keyed by node_dict["id"]; return the stored dict
        raise NotImplementedError

    @abc.abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        # TODO: return the stored node dict or None
        raise NotImplementedError

    @abc.abstractmethod
    def upsert_edge(self, edge_dict: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: persist edge_dict keyed by edge_dict["id"]; return the stored dict.
        #       Merge rules are applied before this call — just write the merged dict.
        raise NotImplementedError

    @abc.abstractmethod
    def get_edge(
        self, src_id: str, dst_id: str, edge_type: str
    ) -> Optional[Dict[str, Any]]:
        # TODO: return the stored edge dict matching (src_id, dst_id, edge_type) or None.
        #       Requires an index on (source_id, target_id, edge_type).
        raise NotImplementedError

    @abc.abstractmethod
    def query_neighbors(
        self, node_id: str, edge_type: Optional[str]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        # TODO: return [(edge_dict, neighbor_node_dict), ...] for all edges touching node_id.
        #       Must be O(neighbors) — requires a node→edges index in the backend.
        raise NotImplementedError

    @abc.abstractmethod
    def get_person_context_raw(
        self, person_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # TODO: return (person_attribute_edge_dicts, relationship_edge_dicts).
        #       O(neighbors of person_id) — same index requirement as query_neighbors.
        raise NotImplementedError

    @abc.abstractmethod
    def apply_delta(
        self,
        node_dicts: List[Dict[str, Any]],
        edge_dicts: List[Dict[str, Any]],
    ) -> None:
        # TODO: batch-write the delta. Must not scan the full backing store.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PluggableGraphStore
# ---------------------------------------------------------------------------

class PluggableGraphStore(GraphStore):
    """
    Thin delegation layer: converts between Pydantic schema types and the
    BackendAdapter's plain-dict interface, and applies merge logic before writes.

    Swap backends by injecting a different BackendAdapter subclass — no
    call-site changes required.
    """

    def __init__(self, adapter: BackendAdapter) -> None:
        self._adapter = adapter

    def upsert_node(self, node: AnyNode) -> AnyNode:
        result = self._adapter.upsert_node(node.model_dump(mode="json"))
        return _node_adapter.validate_python(result)

    def get_node(self, node_id: str) -> Optional[AnyNode]:
        d = self._adapter.get_node(node_id)
        return _node_adapter.validate_python(d) if d is not None else None

    def upsert_edge(self, edge: AnyEdge) -> AnyEdge:
        existing_dict = self._adapter.get_edge(
            edge.source_id, edge.target_id, edge.edge_type
        )
        if existing_dict is not None:
            existing = _edge_adapter.validate_python(existing_dict)
            edge = _merge_edge(existing, edge)
        result = self._adapter.upsert_edge(edge.model_dump(mode="json"))
        return _edge_adapter.validate_python(result)

    def get_edge(
        self, src_id: str, dst_id: str, edge_type: str
    ) -> Optional[AnyEdge]:
        d = self._adapter.get_edge(src_id, dst_id, edge_type)
        return _edge_adapter.validate_python(d) if d is not None else None

    def query_neighbors(
        self, node_id: str, edge_type: Optional[str] = None
    ) -> List[Tuple[AnyEdge, AnyNode]]:
        pairs = self._adapter.query_neighbors(node_id, edge_type)
        return [
            (_edge_adapter.validate_python(e), _node_adapter.validate_python(n))
            for e, n in pairs
        ]

    def get_person_context(self, person_id: str) -> PersonContext:
        attr_dicts, rel_dicts = self._adapter.get_person_context_raw(person_id)
        return PersonContext(
            person_attribute_edges=[_edge_adapter.validate_python(d) for d in attr_dicts],
            relationship_edges=[_edge_adapter.validate_python(d) for d in rel_dicts],
        )

    def apply_delta(
        self,
        nodes: Optional[List[AnyNode]] = None,
        edges: Optional[List[AnyEdge]] = None,
    ) -> None:
        # INVARIANT: only the passed-in lists are written — no full-graph scan
        self._adapter.apply_delta(
            [n.model_dump(mode="json") for n in (nodes or [])],
            [e.model_dump(mode="json") for e in (edges or [])],
        )
