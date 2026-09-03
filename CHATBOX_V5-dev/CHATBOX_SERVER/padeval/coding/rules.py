"""
The rule coder — PRIMARY outcome measure for E1.

Deterministic, auditable, and fixed before the data is generated. It is primary
precisely because it cannot be tuned after seeing a result: an LLM judge can be
re-prompted until the numbers look better, and nobody can prove it wasn't.

TWO OUTCOMES PER REPLY
----------------------
`initiated` (bool) is the pre-registered binary: did the robot bring up a topic
the person had not mentioned? It is the observable the directive ladder was
rewritten to produce (prompt.py:76-83).

`initiative_level` (0-5) exists because the binary SATURATES. Rungs 4/5/6 all
predict "no new topic" and rungs 0/1/2 all predict "new topic", so a binary can
resolve at most three of seven commanded levels and would understate the
controller. The ordinal maps monotonically onto the ladder and is what Kendall's
tau is computed on.

  0  answers only — no question, no novel content
  1  asks back about THEIR topic
  2  follow-up question plus expansion, still their topic
  3  hedged / conditional offer of a new topic
  4  asserts a new topic outright
  5  opens with something it REMEMBERS about this person

Level 5 is exactly rung 0's wording ("open with something you already know about
them"), which only ELLEBOT at `close` can command — a targeted check that the
top of the ladder does what it says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

import nltk

from padeval.coding.lexicon import (
    CLAUSE_BREAKS, HEDGE_MARKERS, MEMORY_RECALL_MARKERS, NEGATION_CUES,
    FRAME_SCOPE_TOKENS, NEGATION_SCOPE_TOKENS, NON_TOPIC_NOUNS,
    PERSON_MEMORY_TOPICS, TOPIC_FRAMES, VERB_LIKE, multiword_forms,
    normalise, topic_surface_index,
)

_SURFACE = topic_surface_index()
_MULTIWORD = multiword_forms()
_NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_SENT_SPLIT = re.compile(r"[.!?]+")

CODER_VERSION = "rule_v1"


@dataclass(frozen=True)
class Stimulus:
    """One authored utterance and the topics it mentions."""
    id: str
    text: str
    stratum: str                       # question | disclosure | quiet | closing
    topics: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class CodeResult:
    initiated: bool
    initiative_level: int
    features: Dict = field(default_factory=dict)


def _negated_spans(tokens: Sequence[str]) -> Set[int]:
    """Indices falling inside the scope of a negation cue.

    A mention inside "I don't know much about space" must not count as raising
    space. Scope runs NEGATION_SCOPE_TOKENS forward, or until a clause break —
    whichever comes first, so "not today, but tell me about space" is not
    swallowed.
    """
    out: Set[int] = set()
    for i, tok in enumerate(tokens):
        t = tok.lower()
        if t in NEGATION_CUES or t.endswith("n't"):
            for j in range(i + 1, min(i + 1 + NEGATION_SCOPE_TOKENS, len(tokens))):
                if tokens[j].lower() in CLAUSE_BREAKS:
                    break
                out.add(j)
    return out


def mentioned_topics(text: str) -> Set[str]:
    """Topic ids mentioned in `text`, ignoring negated mentions."""
    low = text.lower()
    found: Set[str] = set()
    for form, topic in _MULTIWORD.items():
        if form in low:
            found.add(topic)
    tokens = nltk.word_tokenize(text)
    negated = _negated_spans(tokens)
    for i, tok in enumerate(tokens):
        if i in negated:
            continue
        topic = _SURFACE.get(normalise(tok))
        if topic:
            found.add(topic)
    return found


def _framed_spans(text: str, tokens: Sequence[str]) -> Set[int]:
    """Token indices inside the scope of a topic-introducing frame.

    Frames are matched on the lower-cased raw text, then mapped back to token
    positions by walking the tokens and tracking a character offset — the
    tokenizer drops and reshapes punctuation, so index arithmetic on the raw
    string alone would drift.
    """
    low = text.lower()
    hits = [low.find(f) for f in TOPIC_FRAMES]
    starts = sorted(h for h in hits if h >= 0)
    for f in TOPIC_FRAMES:                       # every occurrence, not just first
        pos = low.find(f)
        while pos >= 0:
            starts.append(pos + len(f))
            pos = low.find(f, pos + 1)
    if not starts:
        return set()

    # character offset of each token
    offsets, cur = [], 0
    for tok in tokens:
        idx = low.find(tok.lower(), cur)
        if idx < 0:
            idx = cur
        offsets.append(idx)
        cur = idx + len(tok)

    out: Set[int] = set()
    for start in starts:
        began = None
        for i, off in enumerate(offsets):
            if off >= start:
                began = i
                break
        if began is None:
            continue
        for j in range(began, min(began + FRAME_SCOPE_TOKENS, len(tokens))):
            out.add(j)
    return out


def _content_nouns(text: str, framed_only: bool = False) -> Set[str]:
    """Normalised nouns that could be a topic, before any subtraction.

    `framed_only` restricts to nouns inside a topic-introducing frame, which is
    what makes the open detector mean "introduced a subject" rather than "used a
    word we had not seen".
    """
    tokens = nltk.word_tokenize(text)
    negated = _negated_spans(tokens)
    framed = _framed_spans(text, tokens) if framed_only else None
    out: Set[str] = set()
    # Tag the LOWER-CASED tokens. Sentence-initial capitals otherwise get tagged
    # NNP — "Was it hard?" tagged `Was` as a proper noun, which coded a pure
    # ask-back as introducing a topic and would have inflated the initiation rate
    # on exactly the rungs that are supposed to suppress it. Lower-casing costs
    # nothing here: a genuine proper-noun topic still tags NN ("tim" -> NN), and
    # the detector accepts NN as readily as NNP.
    for i, (tok, tag) in enumerate(nltk.pos_tag([t.lower() for t in tokens])):
        if i in negated or tag not in _NOUN_TAGS:
            continue
        if framed is not None and i not in framed:
            continue
        n = normalise(tok)
        if (n and n not in NON_TOPIC_NOUNS and n not in VERB_LIKE
                and n.isalpha() and len(n) > 2):
            out.add(n)
    return out


def _is_question(text: str) -> bool:
    return "?" in text


def _n_questions(text: str) -> int:
    return text.count("?")


def _n_sentences(text: str) -> int:
    return len([s for s in _SENT_SPLIT.split(text) if s.strip()])


def _first_sentence(text: str) -> str:
    parts = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts[0] if parts else text


def _has_any(text: str, markers: Sequence[str]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


def rule_code(reply: str,
              stimulus: Stimulus,
              person_memory: FrozenSet[str] = PERSON_MEMORY_TOPICS,
              person_name: Optional[str] = None) -> CodeResult:
    """Code one reply. Pure, deterministic, no model calls."""
    reply = (reply or "").strip()
    if not reply:
        return CodeResult(False, 0, {"empty": True})

    reply_topics = mentioned_topics(reply)
    novel_topics = reply_topics - set(stimulus.topics)

    stimulus_nouns = _content_nouns(stimulus.text)
    known_words = set(_SURFACE)                     # lexicon words are handled above
    novel_nouns = _content_nouns(reply, framed_only=True) - stimulus_nouns - known_words
    if person_name:
        novel_nouns.discard(normalise(person_name))

    # If the reply is still ON the person's topic, a specific noun inside it is
    # ELABORATION, not a new topic: "Did the maths test cover fractions?" was
    # coding as initiating because `fractions` is not in the lexicon. Follow-up
    # questions are exactly what the suppressing rungs (4-6) are supposed to
    # produce, so a false positive there would destroy the effect being measured.
    #
    # The cost is a known false NEGATIVE: a reply that stays on their topic AND
    # bolts on an off-lexicon new one ("Maths is fun! Let's talk dinosaurs")
    # reads as not-initiated. Accepted because the stimuli are authored to raise
    # at most one lexicon topic and the robot's plausible new topics are its
    # capability list, which IS in the lexicon — so the off-lexicon case is rare
    # by construction. Novel LEXICON topics are never suppressed this way.
    if reply_topics & set(stimulus.topics):
        novel_nouns = set()

    novel = bool(novel_topics or novel_nouns)
    hedged = _has_any(reply, HEDGE_MARKERS)
    recalls = _has_any(reply, MEMORY_RECALL_MARKERS)
    recall_first = _has_any(_first_sentence(reply), MEMORY_RECALL_MARKERS)
    n_q = _n_questions(reply)
    n_s = _n_sentences(reply)

    # Ordered most-specific first: level 5 is a special case of 4, and 3 of 4.
    if novel and recall_first and (novel_topics & person_memory):
        level = 5
    elif novel and hedged:
        level = 3
    elif novel:
        level = 4
    elif n_q >= 1 and n_s >= 2:
        level = 2
    elif n_q >= 1:
        level = 1
    else:
        level = 0

    return CodeResult(
        initiated=novel,
        initiative_level=level,
        features={
            "reply_topics": sorted(reply_topics),
            "novel_topics": sorted(novel_topics),
            "novel_content_nouns": sorted(novel_nouns),
            "stimulus_topics": sorted(stimulus.topics),
            "is_question": bool(n_q),
            "n_questions": n_q,
            "n_sentences": n_s,
            "hedged_offer": hedged,
            "memory_recall_marker": recalls,
            "memory_recall_first_sentence": recall_first,
            "n_words": len(re.findall(r"[A-Za-z']+", reply)),
        },
    )
