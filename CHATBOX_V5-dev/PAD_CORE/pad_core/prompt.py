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
# Edge placement is not arbitrary: the two robots occupy DISJOINT parts of the D
# axis (CHATBOX -1.00..-0.24, ELLEBOT +0.02..+0.82 — a 0.26 gap, wider than the
# 0.20 a single tier step moves), because Extraversion swings the baseline by
# more than a unit. A ladder tuned to one leaves the other flat, so these seven
# edges give each robot four distinct rungs across its own range while staying
# ABSOLUTE: the same D always yields the same instruction. Exactly one band is
# shared — CHATBOX at 'close' (-0.243) and ELLEBOT at 'unknown' (+0.021) — which
# is the only pair where D is nearly matched and only the persona differs.
#
# Each rung is phrased as WHO INTRODUCES A TOPIC, and that is deliberate. An
# earlier version graded the wording instead ("let them lead" / "follow their
# lead" / "mostly follow their lead") and three consecutive rungs were the same
# instruction in different words — 0.60 of CHATBOX's 0.757 range, 79% of it,
# producing nothing anyone could observe. If a rater watching two clips cannot
# say which directive was in force, the claim that this is the measurable half
# of the model does not hold. "Did the robot introduce a topic the person had
# not mentioned?" is a binary that can be coded straight off a transcript.
_DIRECTIVES = (
    (+0.62, "open with something you already know about them"),
    (+0.32, "propose the next topic yourself"),
    (+0.12, "offer a topic if they do not bring one"),
    (-0.44, "either of you may open a topic — introduce one only if things go "
            "quiet"),
    (-0.74, "follow their topic, and you may add one follow-up question about it"),
    (-0.92, "ask about what they bring up, and do not introduce a topic of your "
            "own"),
    (-9.00, "answer only what they ask, and do not introduce a topic of your own"),
)


def manner_directive(dominance: float) -> str:
    """What the robot should DO at this Dominance, as one actionable clause."""
    for edge, text in _DIRECTIVES:
        if dominance >= edge:
            return text
    return _DIRECTIVES[-1][1]
