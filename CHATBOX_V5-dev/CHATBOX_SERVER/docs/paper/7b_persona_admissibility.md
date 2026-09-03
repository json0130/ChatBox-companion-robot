# 7b — Persona admissibility, generalized to the trait population

**What this generalizes.** 6c established `|D_baseline| ≤ 1 − m` exactly, and
reported it about one persona: CHATBOX exceeds the deployed bound by 0.043. That
is a bug-shaped observation. This makes it a statement about the design space —
which personas are safe to assign, and why.

## Two structural facts that simplify the geometry

**Neuroticism has zero weight on Dominance.** `affect.WEIGHTS["D"]` is
`{O: 0.25, C: 0.17, E: 0.60, A: −0.32}` (`affect.py:41-43`) — no `N` term. So
admissibility over `[-1,1]⁵` is *exactly* the question over `[-1,1]⁴` in
(O, C, E, A), and **a persona's Neuroticism cannot affect whether it clamps.**
Verified against the live `to_pad`, and pinned by a test that fails if a future
weights change introduces an N term.

**D is linear in the traits**, so the admissible set `{t : |w·t| ≤ 1−m}` is a
slab between two parallel hyperplanes through the trait hypercube. Admissibility
is a single scalar question, not a shape needing numerical exploration.

## Admissible fraction of the design space

Uniform over `[-1,1]⁴`, computed two independent ways: Monte Carlo (400k draws,
binomial CI) and near-exact numerical convolution of the four box densities. They
agree at every point.

| m | bound (1−m) | admissible fraction |
|---|---|---|
| 0.10 | 0.90 | 97.6% |
| 0.20 | 0.80 | 94.7% |
| 0.30 | 0.70 | 89.8% |
| **0.40 (deployed)** | **0.60** | **82.7%** |
| 0.50 | 0.50 | 73.2% |
| 0.60 | 0.40 | 61.3% |

**At the deployed magnitude, 17.3% of assignable personas would clamp.**

*On the distribution:* uniform is used because this is a **design-space coverage**
question ("what fraction of assignable personas are safe?"), not a claim about how
human traits are distributed. Real Big Five scores are roughly normal and
concentrated near the mean, which would make this fraction look considerably
better — uniform is the conservative, neutral choice for a design rule rather
than the flattering one.

## The intuition about which traits drive violation is half wrong

The natural expectation — E has the largest positive weight (0.60) and A the
largest opposing one (−0.32), so **high-E/low-A** is the risky combination —
describes only the *upper* violation. The constraint is `|D| ≤ 1−m`, **two-sided**,
so the mirror profile violates the lower bound just as often:

| | fraction | mean O | mean C | mean E | mean A |
|---|---|---|---|---|---|
| over-assertive (D > +0.60) | 8.6% | +0.41 | +0.28 | **+0.74** | **−0.51** |
| over-deferential (D < −0.60) | 8.6% | −0.40 | −0.27 | **−0.74** | **+0.52** |
| admissible | 82.7% | ~0.00 | ~0.00 | ~0.00 | ~0.00 |

Exactly symmetric, as a two-sided bound on a zero-centred linear form must be.

**And the deployed system's clamping persona is an instance of the mirror, not
the intuited case.** CHATBOX has E = −0.6 (low) and A = +0.6 (high), giving
D = −0.643: it clamps for being **too deferential**, not too assertive. An
analysis that had only checked the intuited direction would have reported the
intuition confirmed and missed the case actually present in the shipped system.

## The design rule, in usable form

Since E is the dominant lever, the question a designer actually asks is: *given
this persona's Agreeableness, how much Extraversion can I assign?*

Safe E range at the deployed m = 0.40 (with O = C = 0):

| A | safe E range | width |
|---|---|---|
| −1.0 | [−1.00, +0.47] | 1.47 |
| −0.3 | [−1.00, +0.84] | 1.84 |
| 0.0 | [−1.00, +1.00] | 2.00 (unconstrained) |
| +0.6 | [−0.68, +1.00] | 1.68 |
| +1.0 | [−0.47, +1.00] | 1.47 |

At A = +0.6 — the value **both** deployed personas share — safe E is
`[−0.68, +1.00]`. CHATBOX sits at **E = −0.60**, just 0.08 inside the boundary,
which is why a persona that was not designed with this constraint in mind landed
so close to it. Tightening the offset magnitude shrinks the window further: at
m = 0.60 and A = +0.6, safe E is only `[−0.35, +0.99]`.

**Stated as a rule:** for a target offset magnitude `m`, a persona is admissible
iff `|0.25·O + 0.17·C + 0.60·E − 0.32·A| ≤ 1 − m`. Neuroticism is unconstrained.
Personas near the *extremes of the Extraversion axis in either direction* — with
Agreeableness reinforcing rather than opposing — are the ones that lose the
bottom or top rung of the relationship ladder to saturation.

## Verified — `padeval/tests/test_persona_admissibility.py`, 7/7 (new), 64 padeval tests total

Pins the N-independence against live `to_pad`, Monte-Carlo/exact agreement,
monotonicity in m, consistency with 6c's per-persona numbers (CHATBOX's 0.043
margin reproduced exactly), the two-sided violation structure, the worst corner,
and the design rule against direct evaluation.

## For the paper

> The admissibility constraint `|D_baseline| ≤ 1 − m` was generalized from the
> deployed personas to the trait design space. Because Dominance is a linear
> function of (O, C, E, A) with no Neuroticism term, admissibility is a slab
> through the four-dimensional trait cube and is independent of N. Evaluating the
> admissible fraction by Monte Carlo and by exact convolution (agreeing at every
> point) gives 82.7% of the uniform trait cube at the deployed offset magnitude
> m = 0.40, falling to 61.3% at m = 0.60. Violation is two-sided and symmetric:
> 8.6% of personas exceed the upper bound (high Extraversion, low Agreeableness)
> and 8.6% the lower (the mirror profile), the latter being the case realized by
> the deployed CHATBOX persona, which saturates for being excessively deferential
> rather than excessively assertive. The resulting design rule — for a target
> offset magnitude, the admissible region of trait space — makes explicit a
> constraint linking persona specification to relationship-ladder design that was
> previously implicit.
