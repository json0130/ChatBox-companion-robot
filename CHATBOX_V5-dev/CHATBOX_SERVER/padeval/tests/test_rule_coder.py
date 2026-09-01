"""
Phase 4 gate (a): the rule coder on hand-written cases with known labels.

48 cases, written to be HARD rather than flattering. The easy direction — a
reply that plainly names a new topic — is not where an automatic coder fails.
It fails on:

  * a follow-up question about THEIR topic (must not count as initiating)
  * a topic the stimulus already raised (not novel)
  * a negated mention ("I don't know much about space")
  * a hedged offer vs an outright assertion (level 3 vs 4)
  * a remembered fact vs a capability pitch (level 5 vs 4)
  * backchannels and empty replies

Every case here is authored, so this is a unit test, not a study. The study-grade
check is kappa against a human-coded gold set (test in gold.py's harness), which
is gated separately and must clear 0.70 BEFORE any GPU hours are spent.

No LLM, no network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.coding.lexicon import (                       # noqa: E402
    TOPICS, normalise, topic_surface_index,
)
from padeval.coding.rules import Stimulus, rule_code       # noqa: E402

MATHS = Stimulus("S1", "we had a maths test today", "disclosure",
                 frozenset({"maths", "school"}))
QUIET = Stimulus("S2", "mm.", "quiet", frozenset())
SPACE = Stimulus("S3", "i read about the planets today", "disclosure",
                 frozenset({"space"}))
CLOSE = Stimulus("S4", "anyway, that's about it.", "closing", frozenset())

# (stimulus, reply, expected_initiated, expected_level)
CASES = [
    # ── level 0: answers only ───────────────────────────────────────────────
    (MATHS, "That sounds tough.", False, 0),
    (MATHS, "Maths tests can be hard.", False, 0),
    (QUIET, "Okay.", False, 0),
    (CLOSE, "Alright then.", False, 0),
    (MATHS, "I hope it went well.", False, 0),
    (SPACE, "Planets are fascinating.", False, 0),
    # ── level 1: asks back about their topic ────────────────────────────────
    (MATHS, "How did it go?", False, 1),
    (MATHS, "Was it hard?", False, 1),
    (SPACE, "Which planet did you read about?", False, 1),
    (QUIET, "What's on your mind?", False, 1),
    (MATHS, "Want to tell me more?", False, 1),
    (CLOSE, "Are you sure?", False, 1),
    # ── level 2: follow-up plus expansion, still their topic ────────────────
    (MATHS, "That sounds tough. How did it go?", False, 2),
    (MATHS, "Tell me more about the test. What was the hardest question?", False, 2),
    (SPACE, "Planets are amazing. Which one was your favourite?", False, 2),
    (MATHS, "Oh no. Did you finish it in time?", False, 2),
    # ── level 3: hedged offer of a new topic ────────────────────────────────
    (MATHS, "Maths can be tricky! Want to hear a joke instead?", True, 3),
    (MATHS, "We could talk about space if you like.", True, 3),
    (QUIET, "How about a story?", True, 3),
    (CLOSE, "Shall we play a word game?", True, 3),
    (MATHS, "Maybe we could listen to some jazz after?", True, 3),
    (QUIET, "Do you want to hear about rockets?", True, 3),
    # ── level 4: asserts a new topic ────────────────────────────────────────
    (MATHS, "Let's talk about space exploration!", True, 4),
    (QUIET, "I've been thinking about elephants all day.", True, 4),
    (CLOSE, "Here's a joke: why did the astronaut bring a map?", True, 4),
    (MATHS, "Tennis is a great way to unwind after a test.", True, 4),
    (SPACE, "I love telling stories about brave explorers.", True, 4),
    (QUIET, "Jazz music always cheers me up.", True, 4),
    # ── level 5: opens with a remembered fact ───────────────────────────────
    (QUIET, "You told me you love guitar — shall we talk about that?", True, 5),
    (MATHS, "I remember you're into space. Want to hear about Mars?", True, 5),
    (CLOSE, "You mentioned your guitar last time. How's the practice going?", True, 5),
    (QUIET, "You said you like music. Shall we talk songs?", True, 5),
    # ── hard negatives: the stimulus already raised it ──────────────────────
    # a tag question still asks back — the coder was right, the
    # original expectation here was wrong.
    (SPACE, "Space is wonderful, isn't it?", False, 1),
    (SPACE, "Which planets did you read about?", False, 1),
    (MATHS, "Maths is my favourite subject too.", False, 0),
    (MATHS, "Did the maths test cover fractions?", False, 1),
    # ── hard negatives: negated mentions ────────────────────────────────────
    (MATHS, "I don't know much about space, but how was the test?", False, 1),
    (MATHS, "I can't tell jokes right now.", False, 0),
    (QUIET, "No stories today then.", False, 0),
    (MATHS, "I never talk about tennis during homework.", False, 0),
    # ── hard negatives: backchannels and degenerate input ───────────────────
    (QUIET, "Mm.", False, 0),
    (QUIET, "", False, 0),
    (QUIET, "   ", False, 0),
    (CLOSE, "Okay!", False, 0),
    (QUIET, "Yeah?", False, 1),
    # ── hedge vs assert on the SAME topic — the level 3/4 boundary ──────────
    (QUIET, "We could talk about animals.", True, 3),
    (QUIET, "Animals are my favourite thing.", True, 4),
    # ── recall vs pitch on the SAME topic — the level 4/5 boundary ──────────
    (QUIET, "Guitars are wonderful instruments.", True, 4),
]


def test_all_48_cases():
    assert len(CASES) == 48, f"expected 48 authored cases, have {len(CASES)}"
    wrong = []
    for stim, reply, want_init, want_level in CASES:
        got = rule_code(reply, stim)
        if got.initiated != want_init or got.initiative_level != want_level:
            wrong.append({
                "reply": reply, "stimulus": stim.text,
                "want": (want_init, want_level),
                "got": (got.initiated, got.initiative_level),
                "novel_topics": got.features.get("novel_topics"),
                "novel_nouns": got.features.get("novel_content_nouns"),
            })
    for w in wrong:
        print(f"  MISS {w['want']} -> {w['got']}  {w['reply'][:56]!r}  "
              f"novel={w['novel_topics']}{w['novel_nouns']}")
    acc = (len(CASES) - len(wrong)) / len(CASES)
    print(f"1. rule coder on 48 authored cases: {acc:.1%} "
          f"({len(wrong)} miss(es)) ✓" if not wrong else
          f"1. rule coder on 48 authored cases: {acc:.1%} — {len(wrong)} MISS")
    assert not wrong, f"{len(wrong)}/48 cases mis-coded"


def test_binary_and_ordinal_are_consistent():
    """level >= 3 must mean initiated, and level <= 2 must mean not."""
    for stim, reply, _wi, _wl in CASES:
        c = rule_code(reply, stim)
        assert c.initiated == (c.initiative_level >= 3), (reply, c)
    print("2. initiated <=> level >= 3, on every case ✓")


def test_lexicon_has_no_surface_collisions():
    """Two topics sharing a normalised surface form would make attribution
    ambiguous and silently mis-assign novelty."""
    seen = {}
    collisions = []
    for topic, forms in TOPICS.items():
        for form in forms:
            if " " in form:
                continue
            key = normalise(form)
            if key in seen and seen[key] != topic:
                collisions.append((key, seen[key], topic))
            seen[key] = topic
    assert not collisions, f"surface-form collisions: {collisions}"
    print(f"3. no surface-form collisions across {len(TOPICS)} topics "
          f"({len(topic_surface_index())} forms) ✓")


def test_negation_scope_stops_at_a_clause_break():
    """Scope must not swallow the whole sentence, or a legitimate proposal after
    a contrast would be discarded."""
    s = Stimulus("X", "hello", "question", frozenset())
    swallowed = rule_code("I don't like tennis.", s)
    assert not swallowed.initiated, swallowed.features
    survives = rule_code("I don't mind, but let's talk about tennis.", s)
    assert survives.initiated, survives.features
    print("4. negation scope stops at a clause break ✓")


def test_coder_is_deterministic():
    s = Stimulus("X", "we had a maths test today", "disclosure",
                 frozenset({"maths"}))
    r = "Want to hear about space instead?"
    first = rule_code(r, s)
    for _ in range(20):
        again = rule_code(r, s)
        assert (again.initiated, again.initiative_level) == \
               (first.initiated, first.initiative_level)
    print("5. rule coder is deterministic over 20 repeats ✓")


if __name__ == "__main__":
    test_all_48_cases()
    test_binary_and_ordinal_are_consistent()
    test_lexicon_has_no_surface_collisions()
    test_negation_scope_stops_at_a_clause_break()
    test_coder_is_deterministic()
    print("\nrule coder passes the authored-case gate.")
