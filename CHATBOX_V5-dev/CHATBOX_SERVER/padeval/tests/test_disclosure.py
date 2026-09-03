"""
Phase 8a gate. Two separable claims, and the second is the one that matters.

1. The disclosure detector separates DISCLOSURE from WARMTH. Tested against
   authored cases laid out as a 2x2 — disclosure-without-warmth,
   warmth-without-disclosure, both, neither — because a detector that fires on
   warmth would make trust a second copy of rapport, which is the exact defect
   2be86bb fixed on the write side and this module is meant to fix on the
   trigger side.

2. Rapport and trust ACTUALLY DIVERGE in a scripted session. Until this test
   existed, "trust and rapport can differ" was a property of the type signature
   — two floats, two parameters — and nothing more. A signature permits
   divergence; it does not exhibit it.

REGISTERED PREDICTIONS — written before the detector was run on this set
-----------------------------------------------------------------------
P1  Disclosure with flat affect raises trust and leaves rapport within 0.01 of
    its starting value.
P2  Positive affect without disclosure is the exact mirror: rapport rises,
    trust is unchanged to the float.
P3  Over the 2x2 case set the detector is correct on the binary in >= 90% of
    cases, and its errors are false NEGATIVES rather than false positives.
P4  In the scripted session rapport and trust end at DIFFERENT values, and the
    turn at which they first differ is the scripted disclosure turn, not the
    scripted smile turn.

No LLM anywhere in this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.coding.agreement import cohens_kappa, kappa_ci   # noqa: E402
from padeval.coding.disclosure import (            # noqa: E402
    DEPTH_TRUST_DELTA, SESSION_TRUST_CAP, SessionAccrual, detect, turn_deltas,
)
from padeval.stimuli import STIMULI                # noqa: E402

# ── The 2x2. (text, expect_disclosed, note) ─────────────────────────────────
# Authored to the definition in the module docstring: disclosure is information
# the CAMERA does not already have.

DISCLOSURE_NO_WARMTH = [
    ("i felt really left out at school today", True, "internal state, non-facial"),
    ("i don't have any friends in my class", True, "negated fact — still disclosure"),
    ("my dad works nights so i don't see him much", True, "family fact"),
    ("i get nervous before tests", True, "non-facial state, habitual"),
    ("i was scared when the lights went out", True, "facial state + displacement"),
    ("i don't like maths at all", True, "negated preference"),
    ("i live with my grandma", True, "fact verb + personal noun"),
    ("i hate it when people laugh at me", True, "preference, negative valence"),
    ("my brother broke my guitar", True, "possessive fact"),
    ("i cried after the match", True, "state verb + displacement"),
    ("i'm worried about moving school", True, "non-facial state"),
    ("we moved house last year", True, "plural fact verb"),
    ("i feel like nobody listens to me", True, "state via feel"),
    ("i got in trouble yesterday", True, "fact verb + displacement"),
]

WARMTH_NO_DISCLOSURE = [
    ("you're really funny", False, "warmth directed outward"),
    ("haha that's brilliant", False, "affect, no first person"),
    ("that's such a good one", False, "evaluation of the robot"),
    ("yeah!", False, "minimal positive"),
    ("i'm happy", False, "facial state, no displacement — camera has this"),
    ("i'm so excited", False, "facial state, present tense"),
    ("that was a great joke", False, "past tense but about the robot"),
    ("you always make me laugh", False, "second-person subject"),
    ("cool", False, "bare token"),
    ("nice one", False, "bare evaluation"),
]

BOTH = [
    ("i love your jokes, my dad tells them too", True, "preference + family fact"),
    ("that's so funny, i like when you do voices", True, "warmth + preference"),
    ("haha! i felt awful yesterday but this helped", True, "warmth + displaced state"),
    ("you're the best, my birthday is on friday", True, "warmth + personal fact"),
]

NEITHER = [
    ("what's the weather like", False, "question"),
    ("do you know any jokes", False, "question"),
    ("mm", False, "backchannel"),
    ("okay", False, "backchannel"),
    ("tell me a story", False, "imperative, no self-reference"),
    ("what do you like", False, "question, second person"),
    ("is it raining", False, "question"),
    ("that one", False, "fragment"),
    ("i don't know", False, "no content disclosed"),
    ("hmm let me think", False, "filler"),
]

ALL_CASES = (
    [(t, e, n, "disclosure_no_warmth") for t, e, n in DISCLOSURE_NO_WARMTH]
    + [(t, e, n, "warmth_no_disclosure") for t, e, n in WARMTH_NO_DISCLOSURE]
    + [(t, e, n, "both") for t, e, n in BOTH]
    + [(t, e, n, "neither") for t, e, n in NEITHER]
)


def test_case_set_accuracy_and_error_direction():
    """P3: >= 90% correct, and errors skew to false negatives."""
    wrong, fp, fn = [], [], []
    for text, expect, note, quadrant in ALL_CASES:
        got = detect(text).disclosed
        if got != expect:
            wrong.append((text, expect, got, note, quadrant))
            (fp if got else fn).append(text)
    acc = 1.0 - len(wrong) / len(ALL_CASES)
    print(f"1. detector accuracy {acc:.1%} over {len(ALL_CASES)} authored cases "
          f"({len(fp)} false positive, {len(fn)} false negative)")
    for text, expect, got, note, quadrant in wrong:
        print(f"      [{quadrant}] {text!r}\n        expected {expect}, got {got}  ({note})")
    assert acc >= 0.90, f"accuracy {acc:.1%} below the registered 90% bar"
    assert len(fp) <= len(fn), (
        f"errors skew to FALSE POSITIVES ({len(fp)} fp vs {len(fn)} fn) — a false "
        f"positive lets trust drift up on ordinary chat and re-collapses it onto "
        f"rapport, which is the failure this detector exists to prevent"
    )


def test_present_tense_facial_affect_is_not_disclosure():
    """The load-bearing rule. If this fails, trust launders valence."""
    for text in ("i'm happy", "i am so excited", "i'm sad"):
        r = detect(text)
        assert not r.disclosed, f"{text!r} counted as disclosure (depth {r.depth})"
    # ...but the same state, displaced off the visible present, does count.
    for text in ("i was happy when we went camping", "i felt sad after school"):
        assert detect(text).disclosed, f"{text!r} should be disclosure"
    print("2. present-tense facial affect ignored; the same state displaced counts ✓")


def test_depth_orders_the_trust_delta():
    assert detect("i like jazz").depth == 1
    assert detect("my dad is a nurse").depth == 2
    assert detect("i feel lonely").depth == 3
    deltas = [DEPTH_TRUST_DELTA[d] for d in (0, 1, 2, 3)]
    assert deltas == sorted(deltas) and len(set(deltas)) == 4
    print(f"3. depth 1/2/3 -> trust delta {deltas[1]}/{deltas[2]}/{deltas[3]}, "
          f"strictly increasing ✓")


def test_negation_does_not_suppress_disclosure():
    """Opposite of the topic coder's rule, on purpose (see module docstring)."""
    for text in ("i don't like maths", "i don't have any friends",
                 "i never feel included"):
        assert detect(text).disclosed, f"negated disclosure lost: {text!r}"
    print("4. negated statements still count as disclosure ✓")


def test_questions_do_not_disclose_but_trailing_statements_still_do():
    assert not detect("do you like jazz").disclosed
    assert not detect("what do you want to talk about").disclosed
    assert detect("i like jazz, do you?").disclosed, (
        "clause-level splitting should keep the 'i like jazz' half"
    )
    print("5. questions ignored; a statement beside a question survives ✓")


# ── P1/P2/P4: the divergence demonstration ─────────────────────────────────

def _run_session(script, r0=0.0, t0=0.0):
    """Walk a scripted session through the real delta function.

    `script` is a list of (child_text, felt_pleasure). Uses turn_deltas, which
    reproduces the deployed rapport rule verbatim, so this is a trace of the
    accrual mechanism rather than a re-implementation of it.
    """
    r, t, rows = r0, t0, []
    for i, (text, p) in enumerate(script):
        dr, dt = turn_deltas(text, p)
        r, t = min(1.0, r + dr), min(1.0, t + dt)
        rows.append({"turn": i, "text": text, "pleasure": p,
                     "d_rapport": dr, "d_trust": dt,
                     "rapport": r, "trust": t})
    return rows


def test_P1_disclosure_with_flat_affect_moves_trust_only():
    script = [("i felt left out at school", 0.0),
              ("my dad works nights", 0.0),
              ("i get nervous before tests", 0.0)]
    rows = _run_session(script)
    end = rows[-1]
    print(f"6. P1 flat affect + disclosure -> rapport {end['rapport']:.4f}, "
          f"trust {end['trust']:.4f}")
    assert end["trust"] > 0.0, "trust did not move on three disclosures"
    assert abs(end["rapport"] - 0.0) < 0.01, "rapport moved without warmth"


def test_P2_affect_without_disclosure_moves_rapport_only():
    script = [("you're really funny", 0.8),
              ("haha that's brilliant", 0.9),
              ("nice one", 0.7)]
    rows = _run_session(script)
    end = rows[-1]
    print(f"7. P2 warmth, no disclosure -> rapport {end['rapport']:.4f}, "
          f"trust {end['trust']:.4f}")
    assert end["rapport"] > 0.0, "rapport did not move on three warm turns"
    assert end["trust"] == 0.0, "trust moved with nothing disclosed"


def test_P4_the_two_diverge_at_the_scripted_disclosure_turn():
    """The demonstration. Warmth and disclosure are scheduled on DIFFERENT turns,
    so any divergence is attributable to a specific turn rather than to drift."""
    script = [
        ("mm", 0.0),                              # 0 neither
        ("haha you're funny", 0.9),               # 1 warmth only
        ("that's a good one", 0.8),               # 2 warmth only
        ("i felt left out at school today", 0.0), # 3 disclosure only  <- divergence
        ("okay", 0.0),                            # 4 neither
        ("my dad works nights", 0.0),             # 5 disclosure only
    ]
    rows = _run_session(script)
    print("8. P4 scripted session:")
    print("      turn  d_rapport  d_trust   rapport   trust   text")
    for r in rows:
        print(f"      {r['turn']:>4}  {r['d_rapport']:>9.4f}  {r['d_trust']:>7.4f}  "
              f"{r['rapport']:>7.4f}  {r['trust']:>6.4f}   {r['text']!r}")

    end = rows[-1]
    assert end["rapport"] != end["trust"], (
        "rapport and trust ended EQUAL — the divergence claim is not demonstrated"
    )

    first_trust_turn = next(r["turn"] for r in rows if r["d_trust"] > 0)
    assert first_trust_turn == 3, (
        f"trust first moved at turn {first_trust_turn}, not the scripted "
        f"disclosure turn 3"
    )
    # Warm turns must have left trust flat, or the divergence is not clean.
    assert all(r["d_trust"] == 0.0 for r in rows if r["turn"] in (1, 2))
    assert all(r["d_rapport"] == 0.0 for r in rows if r["turn"] in (3, 5))
    print(f"      -> diverge at turn {first_trust_turn}; end rapport "
          f"{end['rapport']:.4f} != trust {end['trust']:.4f} ✓")


def test_tier_score_is_no_longer_just_rapport():
    """Why the divergence matters: the tier reads (rapport+trust)/2, so if the
    two were identical the average would be rapport and trust decorative."""
    rows = _run_session([("haha you're funny", 0.9),
                         ("i felt left out at school today", 0.0)])
    r, t = rows[-1]["rapport"], rows[-1]["trust"]
    score = (r + t) / 2
    assert score != r, "tier score collapsed back onto rapport"
    print(f"9. tier score {score:.4f} differs from rapport {r:.4f} ✓")


if __name__ == "__main__":
    test_case_set_accuracy_and_error_direction()
    test_present_tense_facial_affect_is_not_disclosure()
    test_depth_orders_the_trust_delta()
    test_negation_does_not_suppress_disclosure()
    test_questions_do_not_disclose_but_trailing_statements_still_do()
    test_P1_disclosure_with_flat_affect_moves_trust_only()
    test_P2_affect_without_disclosure_moves_rapport_only()
    test_P4_the_two_diverge_at_the_scripted_disclosure_turn()
    test_tier_score_is_no_longer_just_rapport()
    print("\ndisclosure is a rule, and trust is demonstrably not rapport.")


# ── held-out validation, with an interval ──────────────────────────────────

def test_held_out_agreement_on_the_E1_stimuli():
    """The only unbiased number in this file, reported WITH its interval.

    The 32 E1 stimuli were authored months earlier for a different experiment and
    stratified independently; the detector never saw them during development.
    `stratum == "disclosure"` is the gold label.

    The point estimate alone would read as far more precise than n=32 with 8
    positives supports, which is why the CI is asserted rather than just printed.
    The two misses are past-tense action verbs absent from FACT_VERBS ("I drew…",
    "I fell…") and are deliberately LEFT UNFIXED: patching against a held-out set
    turns it into a second training set and destroys the only honest number here.
    """
    gold = [s.stratum == "disclosure" for s in STIMULI]
    pred = [detect(s.text).disclosed for s in STIMULI]
    k = cohens_kappa(gold, pred)
    point, lo, hi = kappa_ci(gold, pred, n_boot=10000, seed=0)
    n_pos = sum(gold)
    print(f"10. held out (n={len(STIMULI)}, {n_pos} positive): kappa {k:.3f}, "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    assert k >= 0.70, f"held-out kappa {k:.3f} below the 0.70 bar"
    assert lo > 0.0, "CI includes zero — agreement is not distinguishable from chance"


def test_session_cap_restores_the_floor_the_extractor_used_to_provide():
    """The cap is a restoration, not a new tuning knob.

    The LLM extractor ran once per session and clamped to +/-0.2
    (extraction.py:46-47), which is what made "close takes >= 3 sessions" true
    by arithmetic. Moving to a per-turn rule removed that bound silently, because
    nothing limits how many turns a session has.
    """
    a = SessionAccrual()
    total = sum(a.turn("i felt left out at school", 0.5, ticks=20)[1]
                for _ in range(50))
    assert abs(total - SESSION_TRUST_CAP) < 1e-9, (
        f"50 deep disclosures accrued {total:.3f} trust, cap is "
        f"{SESSION_TRUST_CAP}")
    # A fresh session gets a fresh budget, or the cap would bound the whole run.
    b = SessionAccrual()
    assert b.turn("i felt left out at school", 0.5, ticks=20)[1] > 0
    # Rapport is uncapped by default — the deployed behaviour, preserved.
    c = SessionAccrual()
    rap = sum(c.turn("you're funny", 0.5, ticks=20)[0] for _ in range(50))
    assert rap > 1.0, "rapport was capped; that is a separate design decision"
    print(f"11. trust caps at {total:.2f}/session and resets per session; "
          f"rapport left uncapped ({rap:.1f} over 50 turns) ✓")


# ── 9.0: the gold-set item file must leak nothing ──────────────────────────

def test_gold_items_are_blinded_and_unsampled():
    """The item file is what a human will read. It must carry the utterance and
    an id, and nothing that could anchor them on the thing being validated.

    Also asserts the pool was taken WHOLE. A stratified draw would have to
    stratify on the detector's own output — the only signal available — and
    validating a detector against a sample it helped choose is the circularity a
    gold set exists to break.
    """
    import json
    import os
    import sqlite3

    path = "runs/eval/disclosure_gold_items.jsonl"
    if not os.path.exists(path):
        pytest.skip("gold items not built yet (--build)")
    items = [json.loads(l) for l in open(path)]

    allowed = {"gold_id", "child_said"}
    for it in items:
        assert set(it) == allowed, f"item leaks fields: {set(it) - allowed}"

    # Unsampled: every unique non-empty child utterance in the db is present.
    con = sqlite3.connect("sessions.db")
    rows = con.execute("SELECT child FROM turns WHERE child IS NOT NULL "
                       "AND TRIM(child) <> ''").fetchall()
    pool = {t.strip().lower() for (t,) in rows}
    present = {it["child_said"].strip().lower() for it in items}
    assert present == pool, (
        f"item set is not the whole pool: {len(pool - present)} missing, "
        f"{len(present - pool)} extra — a selection step crept in")
    print(f"12. gold items: {len(items)} blinded to {sorted(allowed)}, "
          f"covering the entire {len(pool)}-utterance pool with no sampling ✓")
