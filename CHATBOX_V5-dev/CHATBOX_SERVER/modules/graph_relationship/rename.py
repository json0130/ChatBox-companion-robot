"""
Rename a person's identity across the graph in place.

When a face is first seen it is enrolled under a provisional id (e.g. ``guest_3``);
once the person tells the robot their name we rewrite that provisional id to the
real one everywhere it is embedded:

    person node             guest_3                     -> jay
    interaction node        interaction:guest_3:chatbox -> interaction:jay:chatbox
    conversation node       conversation:guest_3:chatbox-> conversation:jay:chatbox
    every incident edge     source_id / target_id       -> remapped

Pure graph surgery — no LLM / embeddings. Deterministic-id nodes are recreated
under the new id and all their edges are re-pointed; type-specific edge fields
(weight / value / count / provenance) are preserved via ``model_copy``.
"""

from __future__ import annotations

import uuid

from .interactions import interaction_id
from .store import GraphStore
from .topics import conversation_id


def rename_person(
    store: GraphStore, old_id: str, new_id: str, robot_id: str, *,
    display_name: str | None = None,
) -> bool:
    """Rewrite ``old_id`` -> ``new_id`` for the person and every node/edge whose
    id embeds it (interaction, conversation). Returns True if anything changed.

    Missing nodes are skipped, so this is safe to call whether or not the person
    has an interaction/conversation yet.
    """
    if old_id == new_id:
        return False

    # Deterministic-id nodes whose ids embed the person id.
    id_map = {
        old_id:                              new_id,
        interaction_id(old_id, robot_id):    interaction_id(new_id, robot_id),
        conversation_id(old_id, robot_id):   conversation_id(new_id, robot_id),
    }

    # Gather every edge touching any of the old ids (dedup across shared edges).
    edge_ids: set[str] = set()
    for oid in id_map:
        edge_ids |= set(store._node_edge_index.get(oid, set()))  # noqa: SLF001

    new_edges = []
    for eid in edge_ids:
        edge = store._edges.get(eid)  # noqa: SLF001
        if edge is None:
            continue
        new_edges.append(edge.model_copy(update={
            "id":        str(uuid.uuid4()),
            "source_id": id_map.get(edge.source_id, edge.source_id),
            "target_id": id_map.get(edge.target_id, edge.target_id),
        }))

    # Recreate the renamed nodes (preserving all other fields).
    new_nodes = []
    changed = False
    for oid, nid in id_map.items():
        node = store.get_node(oid)
        if node is None:
            continue
        changed = True
        update = {"id": nid}
        if node.node_type == "person":
            update["display_name"] = display_name or new_id
        new_nodes.append(node.model_copy(update=update))

    if not changed:
        return False

    # Drop the old nodes (delete_node also removes their incident edges), then
    # re-add the renamed nodes + re-pointed edges.
    for oid in id_map:
        store.delete_node(oid)
    store.apply_delta(nodes=new_nodes, edges=new_edges)
    return True
