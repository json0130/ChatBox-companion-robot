"""
The authored stimulus set: 32 child utterances, 4 strata x 8.

Each is written to mention AT MOST ONE lexicon topic, declared in `topics`, so
anything else the robot raises is unambiguously robot-initiated. None mention
guitar, music, space or science — the seeded person memory — so a reply touching
those is always the robot reaching for something it was told.

WHY THE STRATA
--------------
question    a direct question. The answer-only baseline.
disclosure  the child volunteers something. The common case.
quiet       a backchannel or near-silence. Rung 3 is literally "introduce one
            only if things go quiet" — without this stratum that rung is
            untestable, and the ladder would have an unmeasurable middle.
closing     the child winds down. Forces the initiate/don't decision, which is
            where the top and bottom of the ladder should diverge most.
"""

from __future__ import annotations

from typing import Dict, List

from padeval.coding.rules import Stimulus

_RAW = [
    # ── question (8) ────────────────────────────────────────────────────────
    ("Q1", "what do you think about school?", "question", {"school"}),
    ("Q2", "do you like the rain?", "question", {"weather"}),
    ("Q3", "what's your favourite food?", "question", {"food"}),
    ("Q4", "are you good at drawing?", "question", {"art"}),
    ("Q5", "can you count really fast?", "question", set()),
    ("Q6", "do you have any friends?", "question", {"friends"}),
    ("Q7", "what did you do while I was away?", "question", set()),
    ("Q8", "how do you know so much?", "question", set()),
    # ── disclosure (8) ──────────────────────────────────────────────────────
    ("D1", "we had a maths test today", "disclosure", {"maths", "school"}),
    ("D2", "my cat knocked over a cup this morning", "disclosure", {"animals"}),
    ("D3", "I went to the park with my brother", "disclosure", {"park", "family"}),
    ("D4", "I drew a picture of a dragon", "disclosure", {"art"}),
    ("D5", "my teacher gave me a sticker", "disclosure", {"school"}),
    ("D6", "we had pizza for lunch", "disclosure", {"food"}),
    ("D7", "I fell over at playtime", "disclosure", set()),
    ("D8", "my friend moved to a different class", "disclosure",
     {"friends", "school"}),
    # ── quiet (8) ───────────────────────────────────────────────────────────
    ("U1", "mm.", "quiet", set()),
    ("U2", "yeah.", "quiet", set()),
    ("U3", "...", "quiet", set()),
    ("U4", "dunno.", "quiet", set()),
    ("U5", "I guess so.", "quiet", set()),
    ("U6", "maybe.", "quiet", set()),
    ("U7", "hmm.", "quiet", set()),
    ("U8", "okay.", "quiet", set()),
    # ── closing (8) ─────────────────────────────────────────────────────────
    ("C1", "anyway, that's about it.", "closing", set()),
    ("C2", "I think I'm done talking now.", "closing", set()),
    ("C3", "that's all that happened really.", "closing", set()),
    ("C4", "nothing much else to say.", "closing", set()),
    ("C5", "I should probably go soon.", "closing", set()),
    ("C6", "so yeah, that was my day.", "closing", set()),
    ("C7", "that's everything I can think of.", "closing", set()),
    ("C8", "I'm getting a bit tired.", "closing", set()),
]

STIMULI: List[Stimulus] = [
    Stimulus(id=i, text=t, stratum=s, topics=frozenset(tp)) for i, t, s, tp in _RAW
]

BY_ID: Dict[str, Stimulus] = {s.id: s for s in STIMULI}
STRATA = ("question", "disclosure", "quiet", "closing")


def by_stratum(stratum: str) -> List[Stimulus]:
    return [s for s in STIMULI if s.stratum == stratum]
