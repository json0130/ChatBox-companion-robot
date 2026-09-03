"""
Closed-loop directive verification: check the reply against the rung's rule
before accepting it, instead of trusting one open-loop instruction.

WHY THIS RATHER THAN A BIGGER MODEL
-----------------------------------
Phase 5a found the directive effect is model-capacity-dependent: undetectable at
7B, present with the predicted sign at 14B. But even at 14B the suppressing rung
still initiated in 32% of replies — roughly one in three violating an explicit
instruction. A larger model shifts a TENDENCY; it does not give a GUARANTEE.
This wrapper is model-independent, so it composes with the capacity finding
instead of competing with it.

REGISTERED PREDICTION — written before any generation was spent.
----------------------------------------------------------------
    The closed-loop condition reduces the topic-initiation rate at the two
    suppressing rungs by at least 20 PERCENTAGE POINTS relative to the matched
    open-loop baseline (same stimuli, same seeds).

    Falsification: if the reduction is < 20pp, that is a null ON THE FIX and is
    reported as such. The threshold is not renegotiated after seeing the number.

WHAT COUNTS AS A VIOLATION
--------------------------
Checked in both directions, using the rung's own wording:

    rungs 5, 6   SUPPRESS  — violation if the reply DID initiate
    rungs 0, 1   REQUIRE   — violation if the reply did NOT initiate
    rungs 2, 4             — weaker wording ("offer a topic if they do not
                             bring one" / "you may add one follow-up"), so no
                             hard rule is asserted; left unchecked rather than
                             invented.
    rung 3       CONDITIONAL — "introduce one only if things go quiet" is a rule
                             ONLY on the `quiet` stratum, where it requires
                             initiation. Elsewhere it permits either, so it is
                             unchecked.

Inventing a rule where the directive does not state one would make the violation
rate a measure of my interpretation rather than of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from padeval.coding.rules import Stimulus, rule_code

SUPPRESS_RUNGS = (5, 6)
REQUIRE_RUNGS = (0, 1)
CONDITIONAL_RUNG = 3

MAX_RETRIES = 1          # hard cap: an unbounded loop is unmeasurable

# Fallback text per rung family, used only when the correction pass ALSO
# violates. Deliberately bland: the point is to be safe and obviously templated,
# so a fallback is visible in a transcript rather than passing as a real reply.
# NOTE: a fallback must SATISFY the rule it stands in for, or it is not a
# fallback — it is a violation with a friendlier face. The first version used
# "What would you like to talk about?" for `require`, which asks a question but
# introduces nothing, so it failed the very rule it existed to guarantee: 48 of
# 1,536 emitted replies (3.12%) violated, ALL of them this template. Caught by
# re-coding the final emitted text rather than trusting the path label.
FALLBACKS = {
    "suppress": "Okay.",
    "require": "Shall we talk about space?",
}


def _assert_fallbacks_satisfy_their_rules() -> None:
    """Import-time guard so this class of bug cannot recur silently."""
    from padeval.coding.rules import Stimulus
    probe = Stimulus("_fb", "mm.", "quiet", frozenset())
    for rule, text in FALLBACKS.items():
        initiated = rule_code(text, probe).initiated
        ok = (not initiated) if rule == "suppress" else initiated
        if not ok:
            raise AssertionError(
                f"fallback for {rule!r} does not satisfy its own rule: {text!r}")


@dataclass
class VerifiedResult:
    text: str
    path: str                     # "clean" | "corrected" | "fallback" | "unchecked"
    violated_initially: bool
    violated_after: bool
    attempts: int
    initial_text: str = ""
    rule: str = ""
    log: List[str] = field(default_factory=list)


_assert_fallbacks_satisfy_their_rules()


def rule_for(rung: int, stimulus: Stimulus) -> Optional[str]:
    """"suppress" | "require" | None (no rule asserted at this rung)."""
    if rung in SUPPRESS_RUNGS:
        return "suppress"
    if rung in REQUIRE_RUNGS:
        return "require"
    if rung == CONDITIONAL_RUNG and stimulus.stratum == "quiet":
        return "require"
    return None


def _violates(reply: str, stimulus: Stimulus, rule: str) -> bool:
    initiated = rule_code(reply, stimulus).initiated
    return initiated if rule == "suppress" else not initiated


def _correction_note(rule: str, directive: str) -> str:
    """The retry instruction. Names the BEHAVIOUR, never the underlying state.

    No PAD coordinate, no rapport or trust value, no tier name — the deployed
    system already refuses to put numbers in front of the model (they invite it
    to narrate its own metrics back at a child), and a diagnostic wrapper must
    not create the leak the product avoids.
    """
    if rule == "suppress":
        return ("Your previous reply introduced a topic the person had not "
                f"mentioned. Your instruction for this turn is: {directive}. "
                "Reply again, without bringing up anything new.")
    return ("Your previous reply did not offer anything to talk about. Your "
            f"instruction for this turn is: {directive}. Reply again, and this "
            "time bring up a topic.")


def verify(generate: Callable[[str], str],
           system_prompt: str,
           user_msg: str,
           stimulus: Stimulus,
           rung: int,
           directive: str) -> VerifiedResult:
    """Generate, check, correct once, else fall back.

    `generate` takes a system prompt and returns a reply, so the caller keeps
    control of seeding — the correction pass must be reproducible too.
    """
    rule = rule_for(rung, stimulus)
    first = generate(system_prompt)

    if rule is None:
        return VerifiedResult(first, "unchecked", False, False, 1,
                              initial_text=first, rule="none")

    if not _violates(first, stimulus, rule):
        return VerifiedResult(first, "clean", False, False, 1,
                              initial_text=first, rule=rule)

    note = _correction_note(rule, directive)
    retry_prompt = f"{system_prompt}\n\n━━━ CORRECTION ━━━\n{note}"
    second = generate(retry_prompt)

    if not _violates(second, stimulus, rule):
        return VerifiedResult(second, "corrected", True, False, 2,
                              initial_text=first, rule=rule,
                              log=["violation on attempt 1, fixed on retry"])

    return VerifiedResult(FALLBACKS[rule], "fallback", True, True, 2,
                          initial_text=first, rule=rule,
                          log=["violation on attempt 1 AND on retry; "
                               "template substituted"])
