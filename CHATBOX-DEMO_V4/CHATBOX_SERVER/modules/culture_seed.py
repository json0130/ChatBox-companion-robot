"""
App-layer seed for the culture DEMO layer (Command A).

Seeds ONE culture ("Korean") as the ROBOT's prior knowledge:
    robot --knows_culture--> Korean --culture_prior--> CultureTopic(...)

Culture topics are the robot's OWN nodes (`ck:korean:<slug>`), deliberately kept
SEPARATE from shared person-interest topics — so seeding the culture never touches
or couples any person. A person is tagged with the culture only via a separate,
manual `assign_person_culture` step.

Seeding is idempotent (deterministic ids).

NOTE: the prior values below are HAND-SET PLACEHOLDER DEMO DATA — a rough starting
guess for prompt/plumbing testing, NOT research claims about any group.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.graph_relationship.cultures import (
    ensure_culture, ensure_culture_topic, knows_culture, assign_culture,
    set_culture_prior,
)
from modules.graph_relationship.store import GraphStore

_CULTURE_LABEL = "Korean"
_DEFAULT_ROBOT = "chatbox"

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


def seed_korean_demo(store: GraphStore, *, robot_id: str = _DEFAULT_ROBOT,
                     source: str = "culture-seed") -> dict:
    """Seed the Korean culture as `robot_id`'s prior knowledge + its demo topic
    priors (robot-owned CultureTopic nodes). Idempotent. Does NOT touch any person
    or any shared person-interest topic.

    Returns {'culture', 'robot', 'topics': N, 'priors': N}.
    """
    cnode = ensure_culture(store, _CULTURE_LABEL)
    if store.get_node(robot_id) is not None:
        knows_culture(store, robot_id, cnode.id, source=source)
    for label, category, prior in _KOREAN_DEMO:
        ct = ensure_culture_topic(store, cnode.id, label, category=category)
        set_culture_prior(store, cnode.id, ct.id, prior, source=source)
    return {
        "culture": cnode.id,
        "robot":   robot_id if store.get_node(robot_id) is not None else None,
        "topics":  len(_KOREAN_DEMO),
        "priors":  len(_KOREAN_DEMO),
    }


def assign_person_culture(store: GraphStore, person_id: str, culture_label: str,
                          *, source: str = "culture-seed") -> str:
    """Manually tag a person with a culture (creating the culture if needed).
    Returns the culture_id. The person node must already exist in the store.
    Does NOT link the person to any culture topics."""
    cnode = ensure_culture(store, culture_label)
    assign_culture(store, person_id, cnode.id, source=source)
    return cnode.id
