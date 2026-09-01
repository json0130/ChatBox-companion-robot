"""
Phase 3 gate. Two separable claims.

1. The change to `LLMClient.respond` is BEHAVIOUR-PRESERVING. Adding
   `temperature`/`seed` must not alter a single existing call site. Asserted by
   capturing the request kwargs with a stub client — no Ollama needed — so this
   half runs in CI.

2. Seeded generation actually reproduces, and the pairing holds. Needs a live
   Ollama; skipped cleanly without one.

The determinism half is a MEASUREMENT, not an assumption: `replication_rate`
reports the fraction that reproduced, which is the number the paper's methods
section should quote. A prefix-cache effect means the first call after a prompt
change can differ, so `warm_up` runs first and the residual rate is reported.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.face_webcam.webcam_loop import LLMClient        # noqa: E402
from padeval.llm import (                                    # noqa: E402
    SeededLLM, pair_key, replication_rate, sample_diversity, seed_for,
    trial_key,
)


# ── behaviour preservation (no Ollama) ──────────────────────────────────────

class _CaptureClient:
    """Stands in for the OpenAI client and records the request kwargs."""

    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = "[HAPPY] hi there"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _client_with_capture():
    c = LLMClient(model="test-model")
    cap = _CaptureClient()
    c._client = cap
    c.available = True
    return c, cap


def test_defaults_are_untouched_for_spoken_replies():
    """The live path must produce exactly the request it always produced."""
    c, cap = _client_with_capture()
    c.respond("sys", "hello")
    kw = cap.calls[-1]
    assert kw["temperature"] == 0.7
    assert kw["stop"] == ["<|im_start|>", "<|im_end|>",
                          "\nuser", "\nUser", "\nassistant"]
    assert "seed" not in kw, "seed must be absent unless explicitly requested"
    assert "response_format" not in kw
    print("1. spoken-reply defaults unchanged; no seed key emitted ✓")


def test_defaults_are_untouched_for_json_mode():
    c, cap = _client_with_capture()
    c.respond("sys", "hello", json_mode=True, max_tokens=900)
    kw = cap.calls[-1]
    assert kw["temperature"] == 0.0
    assert kw["response_format"] == {"type": "json_object"}
    assert "stop" not in kw
    assert "seed" not in kw
    assert kw["max_tokens"] == 900
    print("2. json_mode defaults unchanged; no seed key emitted ✓")


def test_overrides_are_applied_after_the_defaults():
    """An explicit value must win over the built-in one, in both modes."""
    c, cap = _client_with_capture()
    c.respond("sys", "hello", temperature=0.0, seed=1234)
    kw = cap.calls[-1]
    assert kw["temperature"] == 0.0 and kw["seed"] == 1234
    c.respond("sys", "hello", json_mode=True, temperature=0.9, seed=7)
    kw = cap.calls[-1]
    assert kw["temperature"] == 0.9 and kw["seed"] == 7
    print("3. explicit temperature/seed override the defaults in both modes ✓")


def test_history_is_still_expanded_the_same_way():
    c, cap = _client_with_capture()
    c.respond("sys", "now", history=[("u1", "a1"), ("u2", "a2")])
    roles = [m["role"] for m in cap.calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    print("4. history expansion unchanged ✓")


# ── pairing (no Ollama) ─────────────────────────────────────────────────────

_SPEC = {"robot": "CHATBOX", "tier": "known", "emotion": "happy",
         "stimulus_id": "Q07", "arm": "A1_full", "replicate": 0,
         "assignment": "identity"}


def test_paired_assignments_share_a_seed():
    """The point of the pairing: same seed across the manipulation."""
    a = dict(_SPEC, assignment="identity")
    b = dict(_SPEC, assignment="perm_emotion_D")
    assert seed_for(a) == seed_for(b), "paired trials must share a seed"
    assert pair_key(a) == pair_key(b)
    assert trial_key(a) != trial_key(b), "but must remain distinguishable rows"
    print("5. identity and perm_emotion_D share a seed, keep distinct trial keys ✓")


def test_seed_varies_with_every_design_field():
    """A seed that ignored a factor would silently reuse a sample across cells."""
    base = seed_for(_SPEC)
    for field, alt in (("robot", "ELLEBOT"), ("tier", "close"),
                       ("emotion", "sad"), ("stimulus_id", "Q08"),
                       ("arm", "A0_pad_off"), ("replicate", 1)):
        assert seed_for(dict(_SPEC, **{field: alt})) != base, field
    print("6. seed responds to every pair field ✓")


def test_seed_is_order_independent_and_in_range():
    assert seed_for(_SPEC) == seed_for(dict(reversed(list(_SPEC.items()))))
    assert 0 <= seed_for(_SPEC) < 2 ** 32
    print("7. seed is order-independent and fits 32 bits ✓")


# ── live determinism (needs Ollama) ─────────────────────────────────────────

def _ollama_ready():
    c = LLMClient()
    return c.connect() and c.available


_LIVE = pytest.mark.skipif(not _ollama_ready(), reason="Ollama not reachable")


@_LIVE
def test_same_seed_reproduces_byte_for_byte():
    """The hard requirement. Reproducibility is what the seed is FOR."""
    c = LLMClient()
    c.connect()
    llm = SeededLLM(c)
    sysp = ("You are a playful companion robot talking to a child. Reply in one "
            "or two short sentences. Begin with an action tag in [BRACKETS].")
    for usr in ("mm.", "what should we do today?"):
        llm.warm_up(sysp, usr)
        a = llm.generate(sysp, usr, seed=42)
        b = llm.generate(sysp, usr, seed=42)
        assert a.ok and b.ok, (a.error, b.error)
        assert a.text == b.text, f"seed 42 not reproducible for {usr!r}"
    print("8. same seed reproduces byte-for-byte ✓")


@_LIVE
def test_reply_diversity_is_measured_not_assumed():
    """Diversity across seeds is a property of the PROMPT, not the sampler, so it
    is measured rather than asserted.

    An earlier version of this test asserted that two arbitrary seeds must give
    different replies. That failed — and it was the test that was wrong, not the
    system. Measured here: a tag-constrained prompt answering "mm." gave 2
    distinct replies across 5 seeds, while the same prompt answering an open
    question gave 5.

    This matters for the design, not just for the test. Replicates on a
    low-entropy cell buy almost no information, so a power calculation assuming
    every cell contributes equally overstates the effective n. The stimulus set
    must be screened with this and the distribution reported.
    """
    c = LLMClient()
    c.connect()
    llm = SeededLLM(c)
    sysp = ("You are a playful companion robot talking to a child. Reply in one "
            "or two short sentences. Begin with an action tag in [BRACKETS].")
    seeds = [41, 42, 43, 44, 45]
    results = {}
    for usr in ("mm.", "what should we do today?"):
        llm.warm_up(sysp, usr)
        results[usr] = sample_diversity(
            lambda s, u=usr: llm.generate(sysp, u, seed=s), seeds)
    for usr, d in results.items():
        print(f"   {usr!r:32s} {d['n_distinct']}/{d['n_seeds']} distinct, "
              f"modal share {d['modal_share']:.0%}")
    # The sampler must be capable of variation SOMEWHERE, or the seed is inert.
    assert max(d["n_distinct"] for d in results.values()) > 1, (
        "no prompt produced more than one distinct reply — seed may be inert"
    )
    print("9. reply diversity measured per prompt; sampler is not inert ✓")


@_LIVE
def test_replication_rate_is_measured_not_assumed():
    """Report the byte-level rate; do NOT demand perfection.

    An earlier version of this gate asserted 100% byte-identical. That is not
    achievable on this backend: interleaving distinct prompts gives ~80%
    replication at temperature 0.7 and the SAME ~80% at temperature 0.0, so the
    residue is backend batching / cache state rather than the sampler. Asserting
    1.0 would either fail forever or force a false claim into the methods
    section.

    The floor here is a smoke test for the seed being wired up at all — a broken
    seed would sit near chance, not near 80%. The reproducibility claim the paper
    makes is the outcome-level one (see outcome_replication), which is measured
    in Phase 4 once a coder exists.
    """
    c = LLMClient()
    c.connect()
    llm = SeededLLM(c)
    sysp = ("You are a companion robot talking to a child. Reply in one or two "
            "short sentences.")
    stimuli = ["mm.", "we had a maths test today", "what do you think about school?",
               "anyway, that's about it.", "yeah."]
    specs = [dict(_SPEC, stimulus_id=f"S{i}", replicate=0)
             for i in range(len(stimuli))]
    for s in stimuli:
        llm.warm_up(sysp, s)

    def gen(spec):
        return llm.generate(sysp, stimuli[int(spec["stimulus_id"][1:])],
                            seed=seed_for(spec))

    rate, mismatches = replication_rate(gen, specs)
    print(f"10. byte-level replication {rate:.0%} over {len(specs)} trials "
          f"({len(mismatches)} paraphrase mismatch(es)) — reported, not asserted ✓")
    for m in mismatches:
        print(f"      {m['first'][:56]!r}\n   vs {m['second'][:56]!r}")
    assert rate >= 0.5, (
        f"replication {rate:.0%} is near chance — the seed is probably not "
        f"reaching the backend at all: {mismatches}"
    )


if __name__ == "__main__":
    test_defaults_are_untouched_for_spoken_replies()
    test_defaults_are_untouched_for_json_mode()
    test_overrides_are_applied_after_the_defaults()
    test_history_is_still_expanded_the_same_way()
    test_paired_assignments_share_a_seed()
    test_seed_varies_with_every_design_field()
    test_seed_is_order_independent_and_in_range()
    if _ollama_ready():
        test_same_seed_reproduces_byte_for_byte()
        test_reply_diversity_is_measured_not_assumed()
        test_replication_rate_is_measured_not_assumed()
    else:
        print("   (skipped live determinism — Ollama not reachable)")
    print("\nthe LLM path is seeded, paired, and behaviour-preserving.")
