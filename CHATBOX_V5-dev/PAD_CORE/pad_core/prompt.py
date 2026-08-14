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
# Known ABOUT but barely talked TO. This is the normal state for a second robot:
# memory hangs off the PERSON and is shared, while rapport/trust/turn count hang
# off the (person, robot) pair — so a robot can hold a page of memories about
# someone it has personally exchanged three sentences with.
_KNOWN_OF = ("You know a lot about this person already, though the two of you "
             "have not talked much yet.")


def tier_note(tier: str, interaction_count: int = 0,
              has_memory: bool = False) -> str:
    """One sentence describing the relationship, for the prompt.

    The first-time claim is gated on BOTH the pair's turn count and whether the
    robot actually remembers anything. Gating on the tier alone printed "you are
    meeting this person for the first time" directly above eight remembered facts
    and five turns of prior conversation — the tier is per (person, robot) while
    memory is per person, so the two legitimately disagree and the wording has to
    say so rather than contradict the block underneath it.
    """
    if interaction_count <= 0:
        return _KNOWN_OF if has_memory else _FIRST_TIME
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
