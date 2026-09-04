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

# ── Route 2b: identity, activity and opinion (added in 10b) ────────────────
#
# The gold set's largest error group was not a mis-scoped exclusion. It was
# ABSENT COVERAGE: "i am maori", "i am hj", "i do some research on space" and
# "i think countdowns is pretty cool" produced NO hits at all — no rule matched
# and none was suppressed. Three whole categories of camera-invisible
# self-report had no route:
#
#   identity/attribute   "i am maori"                      -> depth 2
#   activity/habit       "i do some research on space"     -> depth 2
#   opinion              "i think countdowns is cool"      -> depth 1
#
# Each needs a frame verb AND a content word after it, so "i think so", "i do"
# and "i am sure" stay silent. The predicate stoplist below is what keeps the
# identity route from firing on every filler that can follow a copula.
IDENTITY_COPULAS: FrozenSet[str] = frozenset({"am", "was", "is"})
ACTIVITY_VERBS: FrozenSet[str] = frozenset({
    "do", "did", "study", "studied", "work", "worked", "play", "played",
    "read", "write", "wrote", "watch", "watched", "build", "built", "run",
    "train", "trained", "practice", "practise", "collect", "draw", "drew",
    "fell", "ride", "rode", "swim", "dance", "sing", "cook", "bake",
})
OPINION_VERBS: FrozenSet[str] = frozenset({
    "think", "thought", "believe", "reckon", "guess", "suppose", "figure",
})
# Predicates that follow a copula but say nothing about the speaker. Without
# this, "i am sure" and "i am here" would read as identity claims.
EMPTY_PREDICATES: FrozenSet[str] = frozenset({
    # light verbs / progressive auxiliaries: "i am DOING good" is not an
    # identity claim about doing, and letting it read as one re-admitted the
    # camera-redundant affect statement 10c had just excluded.
    "doing", "having", "getting", "being", "feeling", "going", "trying",
    # temporal anchors belong to the displacement test, not to identity
    "today", "tonight", "yesterday", "tomorrow", "when", "while", "since",
    "now", "later", "earlier", "lately", "recently", "always", "sometimes",
    "never", "often", "again", "ever",
    # pronouns: "i am doing good you?" is a question bounced back, not an
    # identity claim about "you". A pronoun is never what someone IS.
    "i", "you", "we", "they", "he", "she", "me", "us", "them", "him", "her",
    "sure", "here", "there", "back", "done", "ready", "sorry", "right",
    "wrong", "fine", "ok", "okay", "alright", "all", "just", "still", "not",
    "no", "yes", "so", "very", "really", "too", "also", "gonna", "going",
    "about", "the", "a", "an", "it", "that", "this", "then", "now", "up",
    "on", "in", "at", "out", "off", "one", "some", "any", "my", "your",
})
# Function words that never count as the "content" a frame needs.
FUNCTION_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "with", "from",
    "some", "any", "my", "your", "his", "her", "its", "our", "their", "that",
    "this", "these", "those", "it", "so", "and", "but", "or", "is", "are",
    "was", "were", "be", "been", "am", "do", "does", "did", "not", "no",
    "very", "really", "quite", "pretty", "too", "also", "just", "like",
    "about", "up", "out", "off", "then", "now", "here", "there", "all",
    "more", "most", "much", "many", "lot", "lots", "bit", "kind", "sort",
})


def _has_content(window: Sequence[str]) -> bool:
    """Does anything in this window carry content, or is it all scaffolding?"""
    return any(w not in FUNCTION_WORDS and w not in EMPTY_PREDICATES and len(w) > 1
               for w in window)


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
    "included", "welcome", "wanted", "safe", "supported", "ignored",
    "picked", "bullied",
})
# NOTE 10c: "awful"/"terrible"/"horrible"/"rubbish"/"miserable" were here and
# have MOVED to FACIAL_STATES. They are general-valence terms, and the FER head
# outputs valence — so counting them as camera-invisible was inconsistent with
# treating "sad" as camera-visible. The line that survives scrutiny is
# general valence = the camera has it; a SPECIFIC state with no valence
# signature ("lonely", "embarrassed", "left out", "included") is what it cannot
# read. They still fire whenever a temporal anchor is present, which is how
# "i felt awful yesterday" is caught.
# States the camera IS already reading — the seven-class emotion head plus its
# obvious synonyms. These count ONLY with a displacement marker (see below).
FACIAL_STATES: FrozenSet[str] = frozenset({
    "happy", "sad", "angry", "mad", "cross", "scared", "afraid", "frightened",
    "surprised", "shocked", "disgusted", "upset", "unhappy", "glad",
    "excited", "cheerful", "furious", "annoyed",
    # general-valence terms (10c) — the FER head reads valence, so a bare
    # "i am doing good" is redundant with what the camera already reported.
    # With a temporal anchor they count, exactly like "sad".
    "good", "great", "well", "bad", "fine", "okay", "ok", "alright",
    "down", "low", "better", "worse", "awful", "terrible", "horrible",
    "rubbish", "miserable", "rough", "lovely", "amazing",
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
#
# 10c generalises this from a patch into the principle it always implied. The
# stated rule is "information the camera does not already have", and the camera
# sees the PRESENT INSTANT. So any affect statement anchored to another time
# carries information the camera cannot supply, whatever the face is doing:
# "just had a bad day", "I was scared when...", "I've been feeling...",
# "yesterday". These count regardless of face, and need no camera dependency to
# evaluate — they are a property of the text alone.
#
# Note this deliberately includes spans that CONTAIN the present, like "today"
# and "lately". A day is not an instant; the camera saw a few seconds of it. A
# child saying "I'm sad today" is reporting hours the robot did not observe, and
# treating that as camera-redundant was the reason the detector missed it.
DISPLACEMENT_MARKERS: Tuple[str, ...] = (
    "when", "because", "after", "before", "since", "while", "until",
    "yesterday", "last night", "last week", "last time", "last month",
    "earlier", "this morning", "the other day", "at school", "at home",
    "on the way", "used to", "always get", "always feel", "sometimes",
    # added in 10c — spans wider than the observed instant
    "today", "tonight", "this week", "this afternoon", "this year",
    "all day", "lately", "recently", "for a while", "the whole time",
    "just had", "have been", "has been", "been feeling", "been having",
)
DISPLACEMENT_TOKENS: FrozenSet[str] = frozenset({
    "was", "were", "felt", "got", "had", "did", "cried", "used",
    "been",          # present perfect spans time the camera did not see
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
    # Interrogative contractions. Added in 10a: the question guard tests
    # `tokens[0] in INTERROGATIVE_OPENERS`, which holds "what" but not "what's",
    # so "what's my name?" was read as a declarative and fired the possessive
    # FACT route. Five of the six false positives on the 71-item gold set were
    # this one line. Same class of defect as the "i'm" bug in 8a: the guard was
    # correct and the tokenizer never gave it the form it tests.
    "what's": ["what", "is"], "whats": ["what", "is"],
    "who's": ["who", "is"], "whos": ["who", "is"],
    "where's": ["where", "is"], "wheres": ["where", "is"],
    "when's": ["when", "is"], "whens": ["when", "is"],
    "why's": ["why", "is"], "how's": ["how", "is"],
    "hows": ["how", "is"], "there's": ["there", "is"],
    # 10d needs these: the abstain fragment test asks whether a finite verb is
    # present, and "you're funny" has one only once the contraction is opened.
    # Left contracted, it looked like a verbless fragment and abstained on a
    # sentence the detector should call confidently.
    "you're": ["you", "are"], "youre": ["you", "are"],
    "it's": ["it", "is"], "its": ["it", "is"],
    "that's": ["that", "is"], "thats": ["that", "is"],
    "they're": ["they", "are"], "he's": ["he", "is"], "she's": ["she", "is"],
    "we'll": ["we", "will"], "let's": ["let", "us"],
}

# Discourse markers that can sit in front of a question word. The guard tests
# position 0, so "so what's my name?" and "yo, what's my name?" slipped past it
# with "so"/"yo" occupying the slot. Stripped before the test rather than added
# to INTERROGATIVE_OPENERS, because they are not interrogative themselves and
# putting them there would suppress genuine declaratives like "so i live with
# my grandma".
DISCOURSE_PREFIXES: FrozenSet[str] = frozenset({
    "so", "yo", "hey", "hi", "hello", "well", "ok", "okay", "um", "uh", "erm",
    "like", "actually", "anyway", "right", "oh", "ah", "yeah", "yes", "no",
    "please", "just", "and", "but", "then",
})

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


# ── 10d: abstain ────────────────────────────────────────────────────────────
#
# A two-way coder must call every utterance, including the ones it has no basis
# for calling. On real traffic that is not a rare edge: the gold pool contains
# ASR garble ("Non, il s'agit, il s'agit...") and bare fragments continuing an
# earlier turn ("like spaceship assembly") where the disclosing subject is in a
# sentence the detector never sees.
#
# Forcing those to `not_disclosed` does not make the detector cautious. It makes
# it wrong in a direction that HIDES ITSELF: the misses land in the false-
# negative column, kappa absorbs them, and nothing distinguishes "I looked and
# there is nothing here" from "I cannot tell". Abstain separates the two, and is
# reported as coverage so a high score on an easy subset cannot pass for a high
# score.
#
# The confidence signal is what a RULE-BASED coder can actually justify —
# structural facts about what fired and what was parseable. No invented
# probability: these rules do not carry one and pretending otherwise would be
# the same overclaim as an LLM judge's self-reported certainty.
#
# ABSTAIN when, and only when:
#
#   A1  No first-person marker anywhere, no finite verb, but content words are
#       present. A fragment like "like spaceship assembly" may well continue a
#       disclosure from the previous turn; the detector has no subject to
#       attribute it to and cannot rule it either way.
#
#   A2  The utterance is not parseable as English at the token level — most of
#       its alphabetic tokens are unknown to every list the detector holds AND
#       it is long enough that this cannot be chance. This is the ASR-garble
#       case.
#
#   A3  Routes conflict: an affect term was seen and SUPPRESSED as
#       camera-redundant, while some other route fired at a lower depth. The
#       verdict then hinges on the camera comparison the detector cannot make
#       (see 10c's deferred valence-contradiction rule).
#
# Everything else is a decision. "mm", "okay" and "what's my name?" are
# confident negatives, not abstentions — the detector looked and there is
# genuinely nothing there.
ABSTAIN_MIN_TOKENS = 4           # below this, unknown words are not evidence
ABSTAIN_UNKNOWN_FRACTION = 0.75  # A2 threshold

FINITE_VERB_HINTS: FrozenSet[str] = frozenset({
    "is", "are", "was", "were", "am", "be", "been", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall",
    "may", "might", "must", "went", "got", "said", "think", "know",
    "want", "feel", "felt", "see", "saw", "make", "made", "take", "took",
    "come", "came", "give", "gave", "tell", "told", "ask", "asked", "play",
    "played", "live", "lived", "work", "worked", "study", "read", "watch",
})
# NOTE: "like" is deliberately NOT in the set above. It is a discourse marker at
# least as often as a verb in this corpus ("like spaceship assembly"), and
# counting it as a finite verb defeated the fragment test on exactly the items
# that test exists for. Utterances where "like" IS the verb ("i like jazz")
# carry a first-person marker, which exempts them from the fragment test anyway.


@dataclass(frozen=True)
class DisclosureResult:
    disclosed: bool
    depth: int                  # 0 none | 1 preference | 2 fact | 3 internal state
    trust_delta: float
    features: Dict = field(default_factory=dict)
    abstained: bool = False

    @property
    def outcome(self) -> str:
        """Three-way. Never collapse abstain into not_disclosed."""
        if self.abstained:
            return "abstain"
        return "disclosed" if self.disclosed else "not_disclosed"


def _clauses(text: str) -> List[str]:
    """Split into clauses. Conjunctions split too, so one clause per claim."""
    return [c.strip() for c in _CLAUSE_SPLIT.split(text.lower()) if c.strip()]


def _tokens(clause: str) -> List[str]:
    out: List[str] = []
    for tok in _WORD.findall(clause):
        out.extend(_CONTRACTIONS.get(tok, [tok]))
    return out


def _is_question_clause(tokens: Sequence[str]) -> bool:
    """Does this clause ASK rather than tell?

    A question requests information; it cannot supply it. "what's my name?" is
    the child asking the robot to recall, not the child disclosing.

    Leading discourse markers are skipped before the test — see
    DISCOURSE_PREFIXES for why they are not simply added to the opener set.
    """
    i = 0
    while i < len(tokens) and tokens[i] in DISCOURSE_PREFIXES:
        i += 1
    return i < len(tokens) and tokens[i] in INTERROGATIVE_OPENERS


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


def _known_vocab() -> FrozenSet[str]:
    """Every token any rule can act on, plus the scaffolding words."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        v = set()
        for group in (PREF_VERBS, PERSONAL_NOUNS, FACT_VERBS, NONFACIAL_STATES,
                      FACIAL_STATES, STATE_VERBS, STATE_COPULAS,
                      INTERROGATIVE_OPENERS, DISCOURSE_PREFIXES, FUNCTION_WORDS,
                      EMPTY_PREDICATES, IDENTITY_COPULAS, ACTIVITY_VERBS,
                      OPINION_VERBS, FIRST_PERSON_SUBJ, FIRST_PERSON_POSS,
                      FIRST_PERSON_PLURAL, FINITE_VERB_HINTS):
            v |= {w for w in group if " " not in w}
        _VOCAB_CACHE = frozenset(v)
    return _VOCAB_CACHE


_VOCAB_CACHE: Optional[FrozenSet[str]] = None


def _should_abstain(text: str, hits: Sequence[str], depth: int):
    """A1/A2/A3 from the block above. Returns (abstain, reason)."""
    tokens = [t for c in _clauses(text) for t in _tokens(c)]
    if not tokens:
        return False, ""

    has_first_person = any(t in FIRST_PERSON_SUBJ or t in FIRST_PERSON_POSS
                           or t in FIRST_PERSON_PLURAL for t in tokens)
    has_verb = any(t in FINITE_VERB_HINTS for t in tokens)
    vocab = _known_vocab()
    # "Content" for the fragment test means SPECIFIC content — the kind that
    # could be a topic continued from a previous turn. Evaluatives and
    # scaffolding do not qualify, or "nice one" abstains alongside "like
    # spaceship assembly", and it should not: there is genuinely nothing
    # personal in "nice one" and the detector can say so.
    content = [t for t in tokens
               if t not in FUNCTION_WORDS and t not in DISCOURSE_PREFIXES
               and t not in EMPTY_PREDICATES and t not in FACIAL_STATES
               and len(t) > 2]

    # A3 was specified and then DROPPED. It would have abstained whenever an
    # affect term was suppressed as camera-redundant, on the grounds that the
    # verdict really turns on a camera comparison the detector cannot make.
    # Two reasons it is not here:
    #
    #   * It contradicts the published Group 3 definition (10e), which settles
    #     present-tense affect as NOT disclosure — a decision, not a doubt.
    #     Abstaining on it would quietly relitigate a rule already published.
    #   * Where another route fires, the question is moot: "i am doing good but
    #     my dad works nights" discloses either way, and abstaining threw away a
    #     confident verdict to honour an ambiguity that changed nothing.
    #
    # The case is not lost: `state_facial_present_ignored` is already logged on
    # every such utterance, so the deferred valence-contradiction work can find
    # them exactly.

    # A1 — fragment with content but no subject and no verb.
    # Two content words, not one: a single specific noun is as likely to be a
    # one-word answer ("space") as a continued disclosure, and answers are
    # confident negatives.
    if not has_first_person and not has_verb and len(content) >= 2:
        return True, (f"fragment: {len(content)} content words, no subject, "
                      f"no finite verb")

    # A2 — token-level unparseable, only where there is enough text to judge.
    if len(tokens) >= ABSTAIN_MIN_TOKENS and depth == 0:
        unknown = [t for t in tokens if t not in vocab]
        if len(unknown) / len(tokens) >= ABSTAIN_UNKNOWN_FRACTION:
            return True, (f"unparseable: {len(unknown)}/{len(tokens)} tokens "
                          f"unknown to every rule list")
    return False, ""


def detect(text: str) -> DisclosureResult:
    """Code one CHILD utterance for disclosure. Pure and deterministic.

    Returns the deepest route that fires — the routes are not exclusive, and a
    single sentence can be a preference and a fact at once.
    """
    text = (text or "").strip()
    if not text:
        return DisclosureResult(False, 0, 0.0, {"empty": True})

    depth = 0
    hits: List[str] = []

    for clause in _clauses(text):
        tokens = _tokens(clause)
        if not tokens or _is_question_clause(tokens):
            continue

        # Multi-word phrases are matched PER CLAUSE, not on the raw text.
        #
        # 10a: they were matched on the whole utterance before the clause loop
        # began, which meant they bypassed the interrogative guard entirely.
        # "Hello chat box, what is my favorite song?" split correctly, and the
        # question clause was correctly flagged — and then `my favorite` matched
        # the raw string anyway and fired. The guard was never wrong; one code
        # path simply ran before it and never consulted it.
        for phrase in PREF_PHRASES:
            if phrase in clause:
                depth = max(depth, 1)
                hits.append(f"pref_phrase:{phrase}")
        for state in NONFACIAL_STATES:
            if " " in state and state in clause:
                depth = max(depth, 3)
                hits.append(f"state_multiword:{state}")

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
                # Route 2b — identity / activity / opinion (10b).
                for j, w in enumerate(win):
                    rest = win[j + 1:]
                    if w in IDENTITY_COPULAS:
                        head = [x for x in rest
                                if x not in FUNCTION_WORDS
                                and x not in EMPTY_PREDICATES]
                        # An affect predicate is handled by route 3, which knows
                        # about displacement; identity must not pre-empt it.
                        head = [x for x in head
                                if x not in FACIAL_STATES
                                and x not in NONFACIAL_STATES]
                        if head:
                            depth = max(depth, 2)
                            hits.append(f"identity:{head[0]}")
                    elif w in ACTIVITY_VERBS and _has_content(rest):
                        depth = max(depth, 2)
                        hits.append(f"activity:{w}")
                    elif w in OPINION_VERBS and _has_content(rest):
                        depth = max(depth, 1)
                        hits.append(f"opinion:{w}")

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

    abstain, reason = _should_abstain(text, hits, depth)
    if abstain:
        # Abstain means trust does not move. Same false-negative-safe direction
        # the detector has had since 8a, but now visible instead of silent.
        return DisclosureResult(
            disclosed=False, depth=0, trust_delta=0.0, abstained=True,
            features={"hits": hits, "detector": DETECTOR_VERSION,
                      "abstain_reason": reason},
        )
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
    # An abstain carries trust_delta = 0.0 by construction, so this line needs no
    # special case — but the distinction is preserved in the result object, and
    # `outcome` is what callers should log if they want an abstain RATE.
    d_trust = detect(child_text).trust_delta
    return d_rapport, d_trust
