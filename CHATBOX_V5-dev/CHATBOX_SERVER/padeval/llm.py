"""
Seeded generation, and the pairing that makes E1 a within-subject comparison.

TWO KEYS PER TRIAL, AND THE DIFFERENCE IS THE WHOLE POINT
---------------------------------------------------------
`trial_key` identifies a row uniquely. It includes the axis assignment, so a
resumed run never regenerates a completed trial and never confuses `identity`
with `perm_emotion_D`.

`pair_key` deliberately EXCLUDES the assignment. The sampling seed derives from
`pair_key`, so the `identity` trial and the `perm_emotion_D` trial for the same
(robot, tier, emotion, stimulus, arm, replicate) draw the same seed and differ
only by the manipulation under study.

That turns E1 from two runs you have to argue are comparable into a paired
comparison where the pairing is structural. Run sequentially instead and every
difference is confounded with whatever drifted between the batches — model
version, machine load, prefix-cache state. Cheap to arrange now; not recoverable
afterwards, because the seeds would already be wrong.

BIT-REPRODUCIBILITY IS NOT AVAILABLE ON THIS BACKEND. MEASURED, NOT ASSUMED.
----------------------------------------------------------------------------
Ollama honours `seed` in the narrow sense: the same seed repeated back-to-back on
the same prompt reproduces byte for byte, and different seeds sample different
replies. Both verified.

It does NOT survive interleaving. Running ten different prompts and repeating
each gave **80% byte-level replication at temperature 0.7 AND the identical 80%
at temperature 0.0** — greedy decoding does not rescue it, so the residue is
backend batching / KV-cache state, not the sampler. Warming every distinct
prompt first does not close it either.

So the harness cannot promise a bit-identical rerun, and claiming otherwise in a
methods section would be false. Two consequences, both handled here:

  * `replication_rate()` reports the byte-level figure as a number to publish.
  * The figure that actually matters is OUTCOME-level: whether the coder assigns
    the same label to both generations. Every observed mismatch is a
    near-paraphrase — "learn new things" vs "learn lots of new things" — which no
    topic-initiation coder would separate. `outcome_replication()` measures that
    once a coder exists (Phase 4), and it is the reproducibility claim the paper
    should make.

The pairing above still earns its keep regardless: sharing a seed across the
compared assignments removes the sampler as a source of between-arm difference
even when it cannot remove it entirely.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# The design fields that identify a comparison PAIR. `assignment` and `arm` are
# handled explicitly by the caller: `arm` is part of the pair for an
# assignment-comparison, and part of the key for an arm-comparison.
PAIR_FIELDS = ("robot", "tier", "emotion", "stimulus_id", "arm", "replicate")


def pair_key(spec: Dict) -> str:
    """Stable identity of a comparison pair — everything but the manipulation."""
    parts = [f"{f}={spec[f]!r}" for f in PAIR_FIELDS]
    return "|".join(parts)


def trial_key(spec: Dict, manipulation: str = "assignment") -> str:
    """Unique identity of one row: the pair plus the manipulation it varies."""
    return f"{pair_key(spec)}|{manipulation}={spec[manipulation]!r}"


def seed_for(spec: Dict) -> int:
    """A reproducible 32-bit seed derived from the PAIR, not the trial.

    Independent per pair, immune to execution order (so shuffling the trial list
    for drift control cannot change any output), and identical across the
    assignments being compared — which is what makes the comparison paired.
    """
    digest = hashlib.sha256(pair_key(spec).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class Generation:
    text: str
    latency_ms: int
    seed: int
    ok: bool          # False for the "[LLM error: ...]" / not-connected sentinels
    error: str = ""


# LLMClient never raises — it returns these as ordinary strings, so a harness
# that only catches exceptions would silently record them as replies.
_SENTINELS = ("[LLM error:", "[LLM not connected")


class SeededLLM:
    """Thin wrapper over the live LLMClient. Owns no prompt logic.

    Deliberately wraps rather than subclasses: the stop-string list and the
    _clean_reply call stay in one place, so what the harness measures is the
    deployed path and not a copy of it that has drifted.
    """

    def __init__(self, client, temperature: Optional[float] = None,
                 max_tokens: int = 200):
        # 200, not the live default of 140: at 140 a reply proposing a topic can
        # be truncated mid-proposal, and if truncation rate differs by arm it
        # biases the outcome (plan R4). finish_reason is recorded per row.
        self._client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._warmed: set = set()

    def warm_up(self, system_prompt: str, user_msg: str) -> None:
        """Absorb the cold-prefix call for this exact prompt, once.

        Keyed on system AND user text. Keying on the system prompt alone was a
        bug: the backend's prefix cache is over the whole token sequence, so
        every distinct user message is its own cold start. With the earlier
        keying, warming a batch of stimuli warmed only the first and the rest
        went in cold — which showed up directly as a fall in replication rate.
        """
        prefix = hashlib.sha256(
            (system_prompt + "\x00" + user_msg).encode()).hexdigest()[:16]
        if prefix in self._warmed:
            return
        self._client.respond(system_prompt, user_msg,
                             max_tokens=self.max_tokens,
                             temperature=self.temperature, seed=0)
        self._warmed.add(prefix)

    def generate(self, system_prompt: str, user_msg: str, seed: int) -> Generation:
        t0 = time.time()
        text = self._client.respond(system_prompt, user_msg,
                                    max_tokens=self.max_tokens,
                                    temperature=self.temperature, seed=seed)
        ms = int((time.time() - t0) * 1000)
        bad = any(text.startswith(s) for s in _SENTINELS)
        return Generation(text=text, latency_ms=ms, seed=seed,
                          ok=not bad, error=text if bad else "")


def replication_rate(gen: Callable[[Dict], Generation],
                     specs: Sequence[Dict]) -> Tuple[float, List[Dict]]:
    """Generate every spec twice and report the byte-identical fraction.

    This is the Phase 3 gate AND a methods-section number. Determinism is a
    property of a particular model, backend and machine, not a promise, so it is
    measured on the actual run rather than assumed.
    """
    mismatches: List[Dict] = []
    for spec in specs:
        a = gen(spec)
        b = gen(spec)
        if a.text != b.text:
            mismatches.append({"pair_key": pair_key(spec),
                               "first": a.text, "second": b.text})
    rate = 1.0 - len(mismatches) / max(1, len(specs))
    return rate, mismatches


def sample_diversity(gen: Callable[[int], Generation],
                     seeds: Sequence[int]) -> Dict:
    """How many DISTINCT replies a cell produces across seeds.

    Not a diagnostic — a design input. Reply diversity is a property of the
    PROMPT, not of the sampler: measured on this machine, a tag-constrained
    system prompt answering "mm." yielded 2 distinct replies across 5 seeds,
    while the same prompt answering an open question yielded 5.

    Replicates on a low-entropy cell therefore buy almost no information, and a
    power calculation that assumes otherwise overstates the effective n. Screen
    the stimulus set with this before the main run and report the distribution,
    rather than assuming every cell contributes equally.
    """
    texts = [gen(s).text for s in seeds]
    distinct = len(set(texts))
    return {"n_seeds": len(seeds),
            "n_distinct": distinct,
            "diversity": distinct / max(1, len(seeds)),
            "modal_share": max(texts.count(t) for t in set(texts)) / max(1, len(texts)),
            "texts": texts}


def outcome_replication(gen: Callable[[Dict], Generation],
                        code: Callable[[str], object],
                        specs: Sequence[Dict]) -> Tuple[float, List[Dict]]:
    """Replication measured on the CODED OUTCOME rather than the raw text.

    The honest reproducibility claim for this backend. Byte-level replication is
    ~80% and cannot be improved by seed or temperature, but the mismatches are
    paraphrases that carry the same label, so the experiment can still be
    reproducible at the level of its dependent variable.

    `code` takes a reply and returns whatever the analysis treats as the outcome
    (the initiation bool, or the 0-5 ordinal). Anything comparable with == works.
    """
    disagreements: List[Dict] = []
    for spec in specs:
        a, b = gen(spec), gen(spec)
        ca, cb = code(a.text), code(b.text)
        if ca != cb:
            disagreements.append({"pair_key": pair_key(spec),
                                  "first": a.text, "second": b.text,
                                  "code_first": ca, "code_second": cb})
    rate = 1.0 - len(disagreements) / max(1, len(specs))
    return rate, disagreements
