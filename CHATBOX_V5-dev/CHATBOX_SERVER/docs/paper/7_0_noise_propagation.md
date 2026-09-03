# 7.0 — Finishing 6d: measured noise propagated, cross-checked, reconciled with E3

**Registered before running:** under `identity`, propagated std on D is exactly
0 (no noise path exists). Under the permutations, propagated std on D is
strictly positive, comparable to or exceeding the smallest gap between adjacent
directive rungs (0.20 units, actually 0.18 once measured — see below), which
would make frequent switching an expected, not surprising, consequence of
E3's simulation.

## The noise, and its caveat

σ_v = 0.279, σ_a = 0.292 — pooled within-class dispersion of the deployed
regression head over the 320-face corpus (`docs/PROGRESS.md:192-193`,
`padeval/traces.py:measure_model_dispersion`). Stated every time these numbers
are used, not just once: this mixes identity, pose and lighting variation with
sensor noise, since the corpus is not a repeated video of one held expression.
It **bounds** the operating point rather than naming it exactly.

## Two independent routes, and they agree exactly

**Closed form**: the pipeline from raw camera (v,a) to felt PAD is affine up to
the clamp, so variance propagates through the E2 axis Jacobian's valence/arousal
columns by the ordinary sum-of-squares rule, then divides by √5 for the deployed
5-sample `AffectStream` window.

**Monte Carlo**: draw 20,000 independent noisy (v,a) pairs, run each through the
*real* `AffectStream` + `compose_offsets` — not a linearisation — and measure the
empirical std.

| assignment | axis | closed-form std | Monte Carlo std | agree |
|---|---|---|---|---|
| identity | D | 0.00000 | 0.00000 | yes |
| perm_emotion_D | D | 0.07486 | 0.07544 | yes (<0.01 apart) |
| perm_arousal_D | D | 0.07835 | 0.07809 | yes (<0.01 apart) |

Agreement to <0.01 on every axis, both permutations. This cross-validates the
implementation before trusting either route for the harder question below —
disagreement here would have meant one of the two was simply wrong.

## The registered prediction, as literally stated, is NOT met — reported as such

Smallest gap between adjacent rungs: **0.18** (`prompt._DIRECTIVES`, the
`0.32→0.12` step). Propagated D std at the deployed (5-sample-smoothed) level:
**0.0749–0.0784**, well below 0.18. The naive form of the registered prediction
fails, and this is pinned by a test
(`test_registered_prediction_as_literally_stated_is_not_met`) precisely so it
cannot quietly become a pass by editing the prediction after the fact.

## The reconciliation with E3 — not a discrepancy, a different question

E3 measured real switching (18–328/min depending on assignment and σ) via
direct time-domain simulation. The propagated std computed here is a
**marginal** statistic — how far D typically sits from its mean at any instant.
Switching **count** is a **level-crossing-rate** question, which depends on the
smoothing filter's autocorrelation timescale, not on the marginal variance
alone: a process can have a small marginal spread and still cross a nearby
boundary often if it moves slowly and lingers near the boundary (clustered
crossings) rather than a memoryless process making one clean jump per unit
variance. A comparison of "propagated std vs. rung gap" is at best a rough
first-order heuristic for whether switching is plausible; it is not the
rigorous test, which is E3's direct simulation.

**Checked directly, not asserted:** if E3's switching were driven mainly by its
schedule's scripted emotion transitions (real, large swings — e.g. happy→sad is
a 1.5-unit valence swing, dwarfing any noise term), a small marginal std would
be no puzzle at all — the scripted transitions would explain everything and
noise would be a footnote. That is **not** what is happening:

| assignment | switches (σ=0.20) | fraction near a scripted transition | fraction genuinely mid-block |
|---|---|---|---|
| `perm_emotion_D` | 26 | 31% | **69%** |
| `perm_arousal_D` | 39 | 26% | **74%** |
| `collapse_D` | 46 | 17% | **83%** |

The large majority of switches occur **away from any scripted transition** —
genuine steady-state noise-driven crossing, not an artefact of the block
schedule. This is what a marginal-std comparison cannot see, and it is why the
naive heuristic under-predicted the effect: the mechanism generating E3's
switching is real and is exactly what the routing argument says it should be
(camera noise reaching D under a permutation), it is just not well summarised by
a single "std vs. gap" number.

## What survives unconditionally

**`identity`'s zero is exact and unaffected by any of the above.** It does not
rest on a std-vs-gap comparison at all — D receives literally no contribution
from valence or arousal under the correct routing, confirmed by both the closed
form and 20,000 Monte Carlo draws. The permutations' nonzero, noise-driven,
mostly-mid-block switching is the positive counterpart: not merely "big scripted
swings get misrouted" but "ordinary camera jitter, continuously, gets misrouted
into rung changes a bystander would read as the robot changing its mind for no
reason."

## Verified — `padeval/tests/test_noise_propagation.py`, 5/5 (new), 49 padeval tests total

Pins identity's exact zero (both methods), the closed-form/Monte-Carlo
agreement, nonzero permutation noise, the registered prediction's literal
failure (so it can't silently be redefined), and the mid-block-majority
boundary-clustering check that resolves the apparent tension with E3.

## For the paper

> Camera noise at its measured magnitude (σ_v=0.279, σ_a=0.292, pooled
> within-class dispersion over 320 faces) was propagated through the affect
> pipeline by two independent methods — a closed-form variance propagation
> through the deployed smoothing and fusion stages, and a 20,000-draw Monte
> Carlo simulation of the same stages — which agreed to within 0.01 on every
> axis. Under the correct (identity) routing, Dominance carries exactly zero
> noise-driven variance, confirmed by both methods; under the permutations, it
> carries a non-trivial marginal std (~0.075–0.078) that, while smaller than a
> single rung gap in isolation, drives real, frequent rung-switching because
> switch frequency is a level-crossing-rate property of the smoothing filter's
> autocorrelation rather than of the marginal variance alone — confirmed by
> showing the large majority (69–83%) of simulated switches occur away from any
> scripted emotion transition, i.e. from steady-state noise, not from the
> synthetic trace's schedule.
