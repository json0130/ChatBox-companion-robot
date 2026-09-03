"""
Rule-based DISCLOSURE detection — the deterministic trust signal.

WHY THIS EXISTS
---------------
`trust` and `rapport` are separate fields and separate axes of the same
InteractionNode, and the tier reads `(rapport + trust) / 2`
(kg_bridge.py:116). Since 2be86bb the live per-tick path moves rapport ONLY
(webcam_loop.py:1584-1585), which was the right fix: rapport is warmth, warmth
is readable from a face, and a frame cannot show what a child chose to tell you.

But that left trust with exactly one non-manual write path: an end-of-session
LLM asked for a free-form `"trust_delta": <number between -0.2 and 0.2>`
(extraction.py:39-50). So "trust is disclosure-gated" was true only in the sense
that a language model was asked to judge disclosure. Every measurement built on
it inherits that model's variance, its prompt, and its version — the same
problem that demoted E1.

This module replaces the judgement with a rule. Zero LLM calls, pure function of
the text, auditable in full.

THE DISTINCTION THE DETECTOR HAS TO MAKE, AND THE PRINCIPLE BEHIND IT
---------------------------------------------------------------------
Rapport already has a trigger: felt Pleasure above a threshold, read off the
face. For trust to be a genuinely different signal and not a second copy, its
trigger must fire on something the FACE CANNOT SHOW.

That gives the rule its shape, and it is not an arbitrary line:

    A first-person statement counts as disclosure when it carries information
    the camera does not already have.

So "I'm happy" is NOT disclosure — the emotion detector is looking straight at
that, and counting it would make trust a laundered copy of valence, which is the
exact failure 2be86bb fixed. But "I was scared when the lights went out" IS
disclosure: it is about a moment the camera never saw. The discriminator is a
DISPLACEMENT marker — past tense, a subordinator, a time shift — which moves the
statement off the visible present.

Non-facial states need no such marker. "I feel left out" is not on the list of
seven emotions the detector classifies and never will be; it is disclosed or it
is unknown.

THREE ROUTES, ORDERED BY DEPTH
------------------------------
    1  PREFERENCE     "I like jazz"                — a standing fact about them
    2  PERSONAL FACT  "my dad works nights"        — a fact about their life
    3  INTERNAL STATE "I felt left out at school"  — how they actually are

Depth orders the trust delta, because these are not equivalent acts. Naming a
preference is cheap; naming an internal state is the thing the design calls
closeness. A flat delta would say they are the same disclosure.

NEGATION IS DELIBERATELY *NOT* SCOPED OUT — THE OPPOSITE OF THE TOPIC CODER
---------------------------------------------------------------------------
rules.py discards a topic mentioned under negation, because "I don't know much
about space" does not raise space. Here the reverse holds: "I don't like maths"
and "I don't have any friends" are disclosures, and strong ones. The child is
still telling you about themselves. Reusing `_negated_spans` here would have
silently dropped the most revealing utterances in the set, so it is not reused,
and this paragraph exists so that reads as a decision rather than an oversight.

WHAT IT WILL MISS, STATED UP FRONT
----------------------------------
Sarcasm, third-party framing ("my teacher says I'm lazy"), and disclosure
carried entirely by implicature ("everyone went to the party") are not detected.
The errors run toward false NEGATIVES by construction — every route needs an
explicit first-person marker — which is the safe direction: a missed disclosure
slows trust accrual, whereas a false one would let trust drift up on ordinary
chat and re-collapse it onto rapport.

No nltk, no POS tagger, no model. Token matching over hand-written lists, so a
reviewer can check every rule by reading it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

DETECTOR_VERSION = "disclosure_rule_v1"

# ── First-person markers. Every route requires one. ─────────────────────────
FIRST_PERSON_SUBJ: FrozenSet[str] = frozenset({"i"})
FIRST_PERSON_POSS: FrozenSet[str] = frozenset({"my"})
# "we"/"our" admitted for the FACT route only. "we could talk about space" is a
# hedge, not a disclosure, and hedges take that form far more often than facts do.
FIRST_PERSON_PLURAL: FrozenSet[str] = frozenset({"we", "our"})

# ── Route 1: preference ─────────────────────────────────────────────────────
PREF_VERBS: FrozenSet[str] = frozenset({
    "like", "likes", "liked", "love", "loves", "loved", "hate", "hates",
    "hated", "enjoy", "enjoys", "enjoyed", "prefer", "prefers", "preferred",
    "adore", "dislike", "want", "wanted", "wish", "wished", "hope", "hoped",
    "dream", "dreamed", "miss", "missed", "fancy", "into",
})
PREF_PHRASES: Tuple[str, ...] = (
    "my favourite", "my favorite", "i'm into", "i am into", "im into",
    "i'd rather", "i would rather", "i can't stand", "i cant stand",
)

# ── Route 2: personal fact ──────────────────────────────────────────────────
# Nouns that make "my <noun>" a fact about the child's life rather than a turn
# of phrase. "my turn", "my go", "my point" are excluded by omission.
PERSONAL_NOUNS: FrozenSet[str] = frozenset({
    "mum", "mom", "mother", "dad", "father", "parents", "brother", "sister",
    "family", "grandma", "grandpa", "nan", "cousin", "aunt", "uncle",
    "dog", "cat", "pet", "rabbit", "hamster", "fish", "bird",
    "house", "home", "room", "bedroom", "garden", "street", "neighbour",
    "school", "teacher", "class", "classmate", "friend", "friends", "team",
    "birthday", "name", "age", "bike", "phone", "guitar", "toy", "toys",
    "holiday", "weekend", "job", "club", "coach", "doctor",
})
FACT_VERBS: FrozenSet[str] = frozenset({
    "have", "had", "got", "live", "lived", "went", "moved", "started",
    "broke", "made", "built", "joined", "won", "lost", "saw", "visited",
    "stayed", "learned", "learnt", "found", "bought", "met",
})

# ── Route 3: internal state ─────────────────────────────────────────────────
# States the emotion detector CANNOT read. These need no displacement marker:
# they are disclosed or they are unknown.
NONFACIAL_STATES: FrozenSet[str] = frozenset({
    "lonely", "alone", "worried", "worry", "anxious", "nervous", "stressed",
    "embarrassed", "ashamed", "guilty", "jealous", "proud", "homesick",
    "overwhelmed", "confused", "bored", "frustrated", "shy", "awkward",
    "left out", "fed up", "unsure", "insecure", "hopeless", "helpless",
    "exhausted", "tired", "restless", "uncomfortable", "unwanted", "useless",
    # Added after the authored case set caught them missing. All are ways a
    # child names how they were, and none is one of the seven classes the
    # emotion head predicts, so none of them lets valence in through the back
    # door. "i never feel included" and "i felt awful yesterday" both scored
    # zero without these.
    "awful", "terrible", "horrible", "rubbish", "miserable", "included",
    "welcome", "wanted", "safe", "supported", "ignored", "picked", "bullied",
})
# States the camera IS already reading — the seven-class emotion head plus its
# obvious synonyms. These count ONLY with a displacement marker (see below).
FACIAL_STATES: FrozenSet[str] = frozenset({
    "happy", "sad", "angry", "mad", "cross", "scared", "afraid", "frightened",
    "surprised", "shocked", "disgusted", "upset", "unhappy", "glad",
    "excited", "cheerful", "furious", "annoyed",
})
STATE_VERBS: FrozenSet[str] = frozenset({
    "cried", "cry", "crying", "panicked", "struggled", "struggling",
    "worry", "worried", "hurt", "ache", "miss", "missed",
})
STATE_COPULAS: FrozenSet[str] = frozenset({
    "am", "'m", "m", "was", "feel", "felt", "feeling", "get", "got", "getting",
    "been", "be", "is", "keep", "kept", "sometimes", "always", "really",
    "so", "very", "a", "bit", "kind", "of", "quite", "pretty", "still", "not",
    "never", "don't", "dont", "didn't", "didnt",
})

# A facial state counts only when one of these moves it off the visible present.
DISPLACEMENT_MARKERS: Tuple[str, ...] = (
    "when", "because", "after", "before", "since", "while", "until",
    "yesterday", "last night", "last week", "last time", "earlier",
    "this morning", "the other day", "at school", "at home", "on the way",
    "used to", "always get", "always feel", "sometimes",
)
DISPLACEMENT_TOKENS: FrozenSet[str] = frozenset({
    "was", "were", "felt", "got", "had", "did", "cried", "used",
})

# Clauses opening with one of these are the child ASKING, not telling.
# "I like jazz, do you?" survives because the split is clause-level: only the
# "do you" half is discarded.
INTERROGATIVE_OPENERS: FrozenSet[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which", "whose",
    "do", "does", "did", "is", "are", "was", "were", "have", "has", "had",
    "can", "could", "would", "will", "shall", "should", "am",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
})

_CLAUSE_SPLIT = re.compile(r"[.!?;,]+|\s+but\s+|\s+and\s+|\s+so\s+|\s+because\s+")
_WORD = re.compile(r"[a-z']+")

# Contracted first-person forms, expanded at token level. Found by the authored
# case set: "i'm worried about moving school" scored ZERO, because `_WORD` keeps
# "i'm" whole and "i'm" is not "i". Every route requires a first-person marker,
# so the contraction silently disabled all three of them — and children contract
# constantly, which would have made the detector look calm while missing most of
# what it exists to catch. Expanding beats loosening the marker set: "i've got a
# dog" then reaches the FACT route through a real "have".
_CONTRACTIONS: Dict[str, List[str]] = {
    "i'm": ["i", "am"], "im": ["i", "am"],
    "i've": ["i", "have"], "ive": ["i", "have"],
    "i'd": ["i", "would"], "i'll": ["i", "will"],
    "we're": ["we", "are"], "we've": ["we", "have"],
    "my": ["my"],
}

# How far after the first-person marker a predicate still counts as attached to
# it. 5 covers "I was really quite worried" without reaching the next clause,
# and clause splitting already bounds the search.
ATTACH_WINDOW = 5

# Trust accrued per disclosing TURN, by depth. Per turn, not per tick: the
# rapport path runs at pipeline rate (webcam_loop.py:1584) while this one can
# only fire when the child actually says something, which is already the
# order-of-magnitude separation 6a measured. Magnitudes are set so a session
# with a handful of real disclosures moves trust by ~0.1, keeping `close` a
# multi-session state rather than a single-conversation one.
DEPTH_TRUST_DELTA: Dict[int, float] = {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.06}

# Maximum trust a single session may add, whatever is said in it.
#
# This is not a new idea — it restores a property the LLM extractor had for free.
# That path ran ONCE per session and clamped its output to +/-0.2
# (extraction.py:46-47, `_clamp_delta`), so "close takes at least 3 sessions" was
# guaranteed by arithmetic: reaching score > 0.70 with rapport saturated needs
# trust > 0.4, and trust could not gain more than 0.2 in a sitting.
#
# Moving to a per-TURN rule silently removed that guarantee, because nothing
# bounded how many turns a session has. Measured before the cap was added: at 8
# turns per session `close` arrives in ONE session, and the 3-session floor 6a
# reported as "independent of session length or how warm the interaction is" was
# simply false. The cap is what makes that sentence true again, and it is set to
# the extractor's own number so the floor is unchanged rather than re-tuned.
SESSION_TRUST_CAP: float = 0.20


@dataclass(frozen=True)
class DisclosureResult:
    disclosed: bool
    depth: int                  # 0 none | 1 preference | 2 fact | 3 internal state
    trust_delta: float
    features: Dict = field(default_factory=dict)


def _clauses(text: str) -> List[str]:
    """Split into clauses. Conjunctions split too, so one clause per claim."""
    return [c.strip() for c in _CLAUSE_SPLIT.split(text.lower()) if c.strip()]


def _tokens(clause: str) -> List[str]:
    out: List[str] = []
    for tok in _WORD.findall(clause):
        out.extend(_CONTRACTIONS.get(tok, [tok]))
    return out


def _is_question_clause(tokens: Sequence[str]) -> bool:
    return bool(tokens) and tokens[0] in INTERROGATIVE_OPENERS


def _window(tokens: Sequence[str], start: int) -> List[str]:
    return list(tokens[start + 1:start + 1 + ATTACH_WINDOW])


def _has_displacement(clause: str, tokens: Sequence[str], full_text: str) -> bool:
    """Is this statement about something other than the visible present?

    Checked against the WHOLE utterance, not just the clause: "I was so sad.
    The lights went out." puts the marker in a different sentence, and clause
    splitting would otherwise throw away the thing that makes it a disclosure.
    """
    low = full_text.lower()
    if any(m in low for m in DISPLACEMENT_MARKERS):
        return True
    return any(t in DISPLACEMENT_TOKENS for t in tokens)


def detect(text: str) -> DisclosureResult:
    """Code one CHILD utterance for disclosure. Pure and deterministic.

    Returns the deepest route that fires — the routes are not exclusive, and a
    single sentence can be a preference and a fact at once.
    """
    text = (text or "").strip()
    if not text:
        return DisclosureResult(False, 0, 0.0, {"empty": True})

    low = text.lower()
    depth = 0
    hits: List[str] = []

    # Multi-word preference phrases are matched on raw text; clause splitting
    # would cut "i can't stand" at the apostrophe-free boundary in some inputs.
    for phrase in PREF_PHRASES:
        if phrase in low:
            depth = max(depth, 1)
            hits.append(f"pref_phrase:{phrase}")

    # Multi-word non-facial states ("left out", "fed up") likewise.
    for state in NONFACIAL_STATES:
        if " " in state and state in low:
            depth = max(depth, 3)
            hits.append(f"state_multiword:{state}")

    for clause in _clauses(text):
        tokens = _tokens(clause)
        if not tokens or _is_question_clause(tokens):
            continue

        for i, tok in enumerate(tokens):
            win = _window(tokens, i)

            if tok in FIRST_PERSON_SUBJ:
                # Route 3 — internal state.
                for w in win:
                    if w in NONFACIAL_STATES:
                        depth = max(depth, 3)
                        hits.append(f"state_nonfacial:{w}")
                    elif w in STATE_VERBS:
                        depth = max(depth, 3)
                        hits.append(f"state_verb:{w}")
                    elif w in FACIAL_STATES:
                        if _has_displacement(clause, tokens, text):
                            depth = max(depth, 3)
                            hits.append(f"state_facial_displaced:{w}")
                        else:
                            hits.append(f"state_facial_present_ignored:{w}")
                # Route 2 — personal fact.
                for w in win:
                    if w in FACT_VERBS:
                        depth = max(depth, 2)
                        hits.append(f"fact_verb:{w}")
                # Route 1 — preference.
                for w in win:
                    if w in PREF_VERBS:
                        depth = max(depth, 1)
                        hits.append(f"pref_verb:{w}")

            elif tok in FIRST_PERSON_POSS:
                for w in win:
                    if w in PERSONAL_NOUNS:
                        depth = max(depth, 2)
                        hits.append(f"fact_poss:{w}")

            elif tok in FIRST_PERSON_PLURAL:
                for w in win:
                    if w in FACT_VERBS:
                        depth = max(depth, 2)
                        hits.append(f"fact_plural:{w}")

    return DisclosureResult(
        disclosed=depth > 0,
        depth=depth,
        trust_delta=DEPTH_TRUST_DELTA[depth],
        features={"hits": hits, "detector": DETECTOR_VERSION},
    )


class SessionAccrual:
    """Per-session closeness accrual, with the cap the extractor used to provide.

    One instance per session — construct a new one at each session boundary,
    exactly as `KGBridge` takes a fresh `_session_start` per process
    (kg_bridge.py:257). Keeping the budget in an object rather than in a module
    global is what makes the cap a SESSION property instead of a process one; a
    global would silently share the budget across people.

    `rapport_cap` defaults to None — uncapped, i.e. the deployed behaviour. It is
    a parameter rather than a constant because rapport reaching its ceiling
    quickly is not obviously a defect the way uncapped trust was: it tops out at
    score 0.5, which cannot reach `close`. See the 8c report for the measurement
    that establishes that, and for why the `unknown -> known` rungs are
    nonetheless not a "slow" axis in any useful sense.
    """

    def __init__(self, trust_cap: float = SESSION_TRUST_CAP,
                 rapport_cap: Optional[float] = None):
        self.trust_cap = trust_cap
        self.rapport_cap = rapport_cap
        self.trust_accrued = 0.0
        self.rapport_accrued = 0.0

    def turn(self, child_text: str, felt_pleasure: float,
             ticks: int = 1, **kw) -> Tuple[float, float]:
        """(d_rapport, d_trust) for one turn, after this session's budget."""
        d_rapport, d_trust = turn_deltas(child_text, felt_pleasure, **kw)
        d_rapport *= ticks
        if self.trust_cap is not None:
            d_trust = max(0.0, min(d_trust, self.trust_cap - self.trust_accrued))
        if self.rapport_cap is not None:
            d_rapport = max(0.0, min(d_rapport,
                                     self.rapport_cap - self.rapport_accrued))
        self.trust_accrued += d_trust
        self.rapport_accrued += d_rapport
        return d_rapport, d_trust


def turn_deltas(child_text: str, felt_pleasure: float,
                rapport_gain: float = 0.025,
                pleasure_threshold: float = 0.05) -> Tuple[float, float]:
    """(d_rapport, d_trust) for one turn — the two signals, fired independently.

    Mirrors the deployed rapport rule exactly (webcam_loop.py:1583-1585:
    `if p > 0.05: d_rapport = 0.025 * p`) so a trace through this function is a
    trace through the live accrual, and adds the disclosure route beside it.

    Nothing couples the two arguments. A turn can move either, both, or neither,
    which is what makes the divergence in test_disclosure.py a demonstration
    rather than an assertion.
    """
    d_rapport = rapport_gain * felt_pleasure if felt_pleasure > pleasure_threshold else 0.0
    d_trust = detect(child_text).trust_delta
    return d_rapport, d_trust
