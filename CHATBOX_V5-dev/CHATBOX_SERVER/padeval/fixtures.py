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
