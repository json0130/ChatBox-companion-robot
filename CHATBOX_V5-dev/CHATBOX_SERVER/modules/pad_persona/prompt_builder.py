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
    "family":  "This person is like family to you; speak with warmth and ease.",
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
