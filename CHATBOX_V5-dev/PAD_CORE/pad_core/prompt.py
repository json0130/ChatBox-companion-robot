"""
The one sentence the relationship tier contributes to the system prompt.

This module used to assemble a WHOLE system prompt, and the webcam loop used it
INSTEAD of its own — which meant enabling PAD silently dropped the identity
block, the KG memory, the RAG hits and the anti-hallucination rules. Prompt
assembly now lives in the loop (`_build_system_prompt`), which is the only place
that has all of those; this file keeps just the tier wording, which was good.

Deliberately no numbers. An earlier version injected
`rapport=0.62 trust=0.55 interactions=14`, which invites the model to narrate its
own metrics back at the child.
"""

_TIER_NOTES = {
    "close":   "You know this person well and feel very comfortable with them.",
    "known":   "You recognise this person from previous interactions.",
    "visitor": "You have met this person only briefly before.",
    "unknown": "You do not recognise this person yet.",
}

_FIRST_TIME = "You are meeting this person for the first time; be friendly and open."


def tier_note(tier: str, interaction_count: int = 0) -> str:
    """One sentence describing the relationship, for the prompt.

    `interaction_count` gates the first-time wording rather than the tier doing
    it: tier is derived from (rapport + trust) / 2 and a turn count, so a person
    with a full page of remembered conversations can still derive as 'visitor' or
    'unknown'. Keying "first time" off the tier would put that claim directly
    above five things the robot remembers about them.
    """
    if interaction_count <= 0:
        return _FIRST_TIME
    return _TIER_NOTES.get(tier, _TIER_NOTES["unknown"])


# ── Dominance -> a behavioural directive ────────────────────────────────────
# The three descriptor words tell the model what to BE. This tells it what to
# DO, which is what actually moves an LLM's output.
#
# Why not just the adjective: a trait label ("reserved") has to be interpreted
# before it can change anything, and models interpret trait labels loosely and
# inconsistently — measured on this project, the adjective alone barely moved
# CHATBOX's replies at all. Why not the raw number: a scalar is even weaker
# guidance, and numbers in a prompt invite the model to narrate them back at
# the child ("since we're close, ...").
#
# Dominance encodes social standing between the two parties, and standing shows
# up in conversation as WHO TAKES INITIATIVE — who proposes topics, who asserts
# versus asks, how hedged a suggestion is. So the directive is written in those
# terms. Each one is deliberately observable, which also makes it measurable:
# hedge count, question rate and imperative count all move with it.
#
# PROPOSAL. The bands and the wording are ours. Reads the FELT coordinate, not
# the shown one — `show` is expressive bandwidth for the BODY, and a robot with
# fewer servos is not thereby less articulate.
# Edge placement is not arbitrary: the two robots occupy DIFFERENT parts of the
# D axis (CHATBOX -1.00..-0.24, ELLEBOT +0.02..+0.82, because Extraversion swings
# the baseline by more than a unit), so a ladder that resolves only one of them
# leaves the other flat. These seven edges give each robot four distinct rungs
# across its own tier range, while staying ABSOLUTE — the same D always yields the
# same instruction, so "share the lead" means one thing on both bodies.
_DIRECTIVES = (
    (+0.62, "lead the exchange — say what you think plainly and suggest what to "
            "do next"),
    (+0.32, "take the initiative — make a concrete suggestion and give your own "
            "view"),
    (+0.12, "share the lead — answer plainly, and offer an idea when it helps"),
    (-0.44, "mostly follow their lead — answer directly, and suggest something "
            "only now and then"),
    (-0.74, "follow their lead, though you can offer one gentle suggestion if it "
            "fits"),
    (-0.92, "let them lead — ask about what they bring up rather than proposing "
            "topics, and keep any suggestion tentative"),
    (-9.00, "stay quiet and responsive — answer what they ask, ask a little back, "
            "and do not propose topics of your own"),
)


def manner_directive(dominance: float) -> str:
    """What the robot should DO at this Dominance, as one actionable clause."""
    for edge, text in _DIRECTIVES:
        if dominance >= edge:
            return text
    return _DIRECTIVES[-1][1]
