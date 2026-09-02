"""
Prompt VARIANTS for Phase 5. Additive: nothing here changes how a directive is
worded, how PAD maps to words, or anything in pad_core.

The deployed builder stays exactly as E1 measured it. E1's bounded null is only
reproducible if the thing it was measured against is untouched, so every variant
is a wrapper that takes the deployed prompt as input and returns a new string.

REGISTERED PREDICTIONS — written before any generation was spent.
-----------------------------------------------------------------
D1 (directive position + duplication):
    Appending the directive as the FINAL line before generation, in addition to
    its existing mid-prompt position, raises the magnitude of the rung-ordering
    effect by at least |0.10| in tau relative to the deployed baseline.

    Baseline for comparison: the L_full ladder arm (persona and tier PINNED,
    Dominance forced per rung, memory present) — rung 3 = 66%, rung 6 = 75%.
    That arm is the correct comparator because it is the only published one
    where the directive is the SOLE manipulation; the tier-level tables confound
    rung with tier and persona.

    Falsification: if |tau_D1| - |tau_baseline| < 0.10, D1 is a second null and
    is reported as such rather than dropped.

D2 (model capacity):
    A larger, more instruction-tuned model shows a non-zero, correctly-signed
    rung effect (tau < 0: suppressing rungs producing LESS initiation) at the
    same screen size where qwen2.5:7b does not.
"""

from __future__ import annotations

from typing import Optional

# The two rungs the D1/D2 screen uses. Rung 3 and rung 6 are the extremes of the
# range CHATBOX can actually reach, which is why the deployed baseline exists for
# exactly this pair.
SCREEN_RUNGS = (3, 6)


def deployed(prompt: str, directive: str) -> str:
    """The prompt exactly as E1 measured it. The control arm."""
    return prompt


def directive_last(prompt: str, directive: str) -> str:
    """D1: repeat the directive as the final line before generation.

    Not a replacement — the directive keeps its existing mid-prompt position, so
    this tests POSITION and REDUNDANCY, not rewording. A rewording variant would
    confound the two and could not be compared to the E1 baseline.
    """
    return (f"{prompt}\n\n"
            f"━━━ MOST IMPORTANT ━━━\n"
            f"Before you reply, re-read this: {directive}.\n"
            f"This instruction overrides anything above it about what you know "
            f"or what you could talk about.")


def directive_last_neutral(prompt: str, directive: str) -> str:
    """D1b: same POSITION and DUPLICATION as `directive_last`, neutral framing.

    `directive_last` added "This instruction overrides anything above it about
    what you know or what you could talk about." That sentence discourages
    topic-talk by itself, independently of which rung's directive it introduces —
    a confound I introduced, not a property of position or duplication.

    This variant repeats the directive in the same final slot with no override
    clause and no mention of topics, so the difference between the two isolates
    my framing from the manipulation under test.
    """
    return (f"{prompt}\n\n"
            f"━━━ REMINDER ━━━\n"
            f"Your instruction for this turn: {directive}.")


VARIANTS = {
    "deployed": deployed,
    "directive_last": directive_last,
    "directive_last_neutral": directive_last_neutral,
}
