"""
World construction for a trial: a headless loop, a seeded person, a derived tier.

`TIER_RECIPE` and `seed_person` are MOVED here from tools/pad_prompt_grid.py so
the tier-derivation recipe has exactly one home; that tool now imports them back.

The tier is never forced as a string. Each person is given rapport / trust / turn
count and the tier is DERIVED through the real kg_bridge path, then asserted —
forcing the string would test a string rather than the system.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple

# rapport/trust/count chosen to land on each tier through _tier_from_scores:
#   score=(r+t)/2 >0.70 close | >0.45 known | count>5 known | count>0 visitor
TIER_RECIPE: Dict[str, Tuple[float, float, int]] = {
    "unknown": (0.00, 0.00, 0),
    "visitor": (0.10, 0.05, 1),
    "known":   (0.55, 0.50, 3),
    "close":   (0.80, 0.75, 9),
}


def seed_person(store, pid: str, robot: str, tier: str) -> str:
    """Create a person with an IDENTICAL memory seed, then dial the relationship
    to `tier` through the real derivation path. Returns the derived tier."""
    from modules.graph_relationship.interactions import (
        get_or_create_interaction, set_closeness,
    )
    from modules.graph_relationship.kg_bridge import derive_tier
    from modules.graph_relationship.schema import Embodiment, PersonNode, RobotNode
    from modules.graph_relationship.topics import add_person_topic

    if store.get_node(robot) is None:
        store.upsert_node(RobotNode(id=robot, name=robot,
                                    embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id=pid, display_name=pid))
    for label, cat, aff in (("guitar", "music", 0.85), ("space", "science", 0.70)):
        add_person_topic(store, pid, label, cat, affinity=aff, confidence=0.9,
                         source="padeval-seed")
    rapport, trust, count = TIER_RECIPE[tier]
    set_closeness(store, pid, robot, rapport=rapport, trust=trust, source="padeval")
    node = get_or_create_interaction(store, pid, robot)
    store.upsert_node(node.model_copy(update={"interaction_count": count}))
    return derive_tier(pid, robot, store)


def build_world(robot: str, tmp_root: str | None = None):
    """A headless WebcamKGLoop against throwaway state.

    `embed_fn=None` disables SessionRAG entirely, which removes an Ollama
    round-trip per turn AND makes the RAG block deterministically absent — a
    declared deviation from the deployed prompt (plan R11). Never call run();
    never call _apply_chat_result, which would write a turn and make every later
    trial for that person non-independent.
    """
    from modules.face_webcam.webcam_loop import WebcamKGLoop

    tmp = tmp_root or tempfile.mkdtemp(prefix="padeval_")
    return WebcamKGLoop(
        robot_id=robot, show_window=False, seed=True, llm_client=None,
        kg_path=os.path.join(tmp, f"kg_{robot}.json"),
        faces_path=os.path.join(tmp, "faces.npz"),
        sessions_db=os.path.join(tmp, f"s_{robot}.db"),
        embed_fn=None, pad_enabled=True, emotion_enabled=True,
    )


def strip_capabilities(store, robot_id: str) -> int:
    """Remove the robot's has_capability edges. Returns how many were dropped.

    The IDENTITY block renders these as "You can: tells stories, knows jazz,
    knows about space, ..." — a standing list of things the robot could raise,
    IDENTICAL in every experimental condition. Measured over 256 replies it
    dominates the behavioural directive: at the `unknown` tier, where the
    directive says "answer only what they ask, and do not introduce a topic of
    your own", the robot proposed a topic in ~72% of turns anyway.

    This is the A7 arm: quantify the floor by removing the competing invitation.
    """
    doomed = [e.id for e in list(store._edges.values())
              if e.edge_type == "has_capability" and e.source_id == robot_id]
    for eid in doomed:
        edge = store._edges.pop(eid, None)
        if edge is None:
            continue
        store._endpoint_type_index.pop(
            (edge.source_id, edge.target_id, edge.edge_type), None)
        store._node_edge_index[edge.source_id].discard(eid)
        store._node_edge_index[edge.target_id].discard(eid)
    return len(doomed)


def seed_person_no_memory(store, pid: str, robot: str, tier: str) -> str:
    """`seed_person` WITHOUT the two seeded interest topics.

    The A8 arm. `space` and `guitar` come from this seed, not from the
    capability list, and they are what the robot actually reaches for — so
    removing capabilities alone would not isolate the directive.
    """
    from modules.graph_relationship.interactions import (
        get_or_create_interaction, set_closeness,
    )
    from modules.graph_relationship.kg_bridge import derive_tier
    from modules.graph_relationship.schema import Embodiment, PersonNode, RobotNode

    if store.get_node(robot) is None:
        store.upsert_node(RobotNode(id=robot, name=robot,
                                    embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id=pid, display_name=pid))
    rapport, trust, count = TIER_RECIPE[tier]
    set_closeness(store, pid, robot, rapport=rapport, trust=trust, source="padeval")
    node = get_or_create_interaction(store, pid, robot)
    store.upsert_node(node.model_copy(update={"interaction_count": count}))
    return derive_tier(pid, robot, store)
