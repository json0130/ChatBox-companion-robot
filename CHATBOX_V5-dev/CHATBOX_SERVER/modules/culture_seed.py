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

# HAND-WRITTEN DEMO SEED — the static "how to talk" manner hint for this culture, NOT
# a research claim. Same text for every interaction (no tier/affect/situation
# variation — that is Approach 2). Injected into HOW-TO-REPLY as soft, secondary
# guidance when a person is tagged to this culture.
_KOREAN_STYLE_HINT = (
    "Be polite, warm, and a little formal, especially early on. Addressing elders or "
    "new acquaintances respectfully is valued. Compliments are sometimes deflected out "
    "of modesty — offer them once and don't insist. Sharing food and small gestures of "
    "care read as friendly."
)

# (label, category, DUMMY prior in [0,1], [facts...]) — demo placeholders, not
# research claims. `facts` are short, shareable bits the robot can mention when it
# brings the topic up (kept general and light, never asserted about the person).
_KOREAN_DEMO: List[Tuple[str, str, float, List[str]]] = [
    ("kimchi",        "food",     0.60, [
        "Korea's iconic fermented vegetable dish, usually napa cabbage with chilli.",
        "There are hundreds of varieties and it's served with almost every meal."]),
    ("korean bbq",    "food",     0.50, [
        "Meat grilled right at your table — a social, shared way to eat.",
        "Often wrapped in lettuce with garlic and side dishes (banchan)."]),
    ("kpop",          "music",    0.50, [
        "Korean pop music known for polished songs and synchronised dance.",
        "Groups like BTS, BLACKPINK, TWICE and NewJeans have huge global fandoms."]),
    ("kdrama",        "media",    0.45, [
        "Korean TV dramas with big international followings via streaming.",
        "Genres span romance, thriller and historical (sageuk)."]),
    ("bibimbap",      "food",     0.45, [
        "A rice bowl topped with seasoned vegetables, egg and gochujang.",
        "You mix everything together before eating."]),
    ("hiking",        "activity", 0.40, [
        "Extremely popular in Korea — mountains are everywhere, even in Seoul.",
        "Weekend hiking clubs and well-marked trails are common."]),
    ("son heung-min", "person",   0.35, [
        "Korean footballer, a captain at Tottenham Hotspur and a national hero.",
        "One of the Premier League's standout forwards."]),
    ("noraebang",     "activity", 0.35, [
        "Korean karaoke — private singing rooms rented by the hour.",
        "A staple social outing with friends or after dinner."]),
    ("chuseok",       "activity", 0.30, [
        "Korea's harvest/thanksgiving holiday, a major family gathering.",
        "People share songpyeon (rice cakes) and honour ancestors."]),
    ("esports",       "activity", 0.30, [
        "Korea is the heart of competitive gaming — StarCraft and League of Legends.",
        "Pro leagues, star players and PC bangs (gaming cafés) are part of the culture."]),
    ("baseball",      "sport",    0.30, [
        "One of Korea's most popular sports, with the lively KBO league.",
        "Games are famous for organised cheering, chants and fan songs."]),
    ("taekwondo",     "sport",    0.25, [
        "A Korean martial art and the national sport, now an Olympic event.",
        "Known for fast, high kicks."]),
]


def seed_korean_demo(store: GraphStore, *, robot_id: str = _DEFAULT_ROBOT,
                     source: str = "culture-seed") -> dict:
    """Seed the Korean culture as `robot_id`'s prior knowledge + its demo topic
    priors (robot-owned CultureTopic nodes). Idempotent. Does NOT touch any person
    or any shared person-interest topic.

    Returns {'culture', 'robot', 'topics': N, 'priors': N}.
    """
    cnode = ensure_culture(store, _CULTURE_LABEL)
    # Set/overwrite the static manner hint idempotently (same text every seed).
    if cnode.style_hint != _KOREAN_STYLE_HINT:
        cnode = cnode.model_copy(update={"style_hint": _KOREAN_STYLE_HINT})
        store.upsert_node(cnode)
    if store.get_node(robot_id) is not None:
        knows_culture(store, robot_id, cnode.id, source=source)
    for label, category, prior, facts in _KOREAN_DEMO:
        ct = ensure_culture_topic(store, cnode.id, label, category=category,
                                  facts=facts)
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
