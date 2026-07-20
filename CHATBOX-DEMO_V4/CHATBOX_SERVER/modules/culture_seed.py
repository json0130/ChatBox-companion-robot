"""
App-layer seed for the culture DEMO layer (Command A).

Seeds ONE culture ("Korean") with demo topics and DUMMY priors, reusing the pure
graph_relationship helpers. Topics are created via `resolve_topic` so any topic
ALREADY in the graph (e.g. `hiking`) is REUSED, not duplicated, and its category
fills only if it was still `other` (existing first-non-other-wins rule).

Seeding is idempotent (deterministic ids) and does NOT assign any person to the
culture — that's a separate manual step (`assign_person_culture`).

NOTE: the prior values below are HAND-SET PLACEHOLDER DEMO DATA — a rough starting
guess for prompt/plumbing testing, NOT research claims about any group.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.graph_relationship.cultures import (
    culture_id, ensure_culture, assign_culture, set_culture_prior,
)
from modules.graph_relationship.store import GraphStore
from modules.graph_relationship.topics import resolve_topic

_CULTURE_LABEL = "Korean"

# (label, category, DUMMY prior in [0,1]) — demo placeholders, not research claims.
_KOREAN_DEMO: List[Tuple[str, str, float]] = [
    ("kimchi",        "food",     0.80),
    ("korean bbq",    "food",     0.70),
    ("kpop",          "music",    0.70),
    ("kdrama",        "media",    0.65),
    ("bibimbap",      "food",     0.65),
    ("hiking",        "activity", 0.60),
    ("son heung-min", "person",   0.55),
    ("noraebang",     "activity", 0.55),
    ("chuseok",       "activity", 0.50),
    ("esports",       "activity", 0.50),
    ("baseball",      "sport",    0.50),
    ("taekwondo",     "sport",    0.45),
]


def seed_korean_demo(store: GraphStore, *, source: str = "culture-seed") -> dict:
    """Seed the Korean culture + its demo topic priors. Idempotent.

    Returns {'culture': culture_id, 'topics': N, 'priors': N, 'reused': [labels]}.
    `reused` lists topics that already existed in the graph (not created here).
    """
    cnode = ensure_culture(store, _CULTURE_LABEL)
    reused: List[str] = []
    for label, category, prior in _KOREAN_DEMO:
        pre = store.get_node(f"topic:{_slug(label)}")
        if pre is not None:
            reused.append(label)
        topic = resolve_topic(store, label, category=category)
        set_culture_prior(store, cnode.id, topic.id, prior, source=source)
    return {
        "culture": cnode.id,
        "topics":  len(_KOREAN_DEMO),
        "priors":  len(_KOREAN_DEMO),
        "reused":  reused,
    }


def assign_person_culture(store: GraphStore, person_id: str, culture_label: str,
                          *, source: str = "culture-seed") -> str:
    """Manually link a person to a culture (creating the culture if needed).
    Returns the culture_id. The person node must already exist in the store."""
    cnode = ensure_culture(store, culture_label)
    assign_culture(store, person_id, cnode.id, source=source)
    return cnode.id


def _slug(label: str) -> str:
    from modules.graph_relationship.topics import normalize_label
    return normalize_label(label)
