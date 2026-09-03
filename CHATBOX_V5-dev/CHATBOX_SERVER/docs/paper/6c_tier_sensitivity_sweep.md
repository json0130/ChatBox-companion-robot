# 6c — TIER_OFFSETS magnitude sensitivity sweep (stretch)

**Motivation.** The D-axis mapping's deepest structural objection is that it is
four numbers chosen by eye: `close +0.40, known 0, visitor -0.20, unknown -0.40`.
A reviewer will ask why 0.40, and why the known→close gap is twice the
known→visitor gap. This sweep converts "we chose these" into "the conclusions do
not depend on these" — at zero LLM cost.

`affect.py` / `TIER_OFFSETS` are **not modified**. Every value below is passed
explicitly through the offsets-argument seam already established for E2's
leak/no-leak ablation (`axes.compose_offsets`, `e2_identify.jacobian(...,
ramp=...)`).

## Parameterisation

The deployed table is not four free numbers — it already has a fixed shape:
visitor's magnitude is exactly half of unknown's, and close equals unknown's.
Sweeping "the offset magnitude" means scaling that shape by one parameter `m`:

```
close(m)   = (0, leak, +m)
known(m)   = (0, 0,     0)      reference tier, always zero
visitor(m) = (0, 0,   -0.5·m)
unknown(m) = (0, 0,    -m)
```

`m = 0.40` reproduces the deployed table exactly (asserted at import time).
`leak` (the close-tier arousal term) is held at its deployed +0.10 — a separate,
already-calibrated phenomenon (E2's leak/no-leak pair) that conflating with the
magnitude question here would confound.

**Registered before running:** across `m ∈ [0.10, 0.60]`, `identity` remains
full rank (3/3) and structurally zero-switching, while `collapse_D` remains
rank-deficient at every `m > 0` and the permuted assignments remain non-zero-
switching at every `m > 0`. Not a magnitude claim — a claim that the qualitative
routing result is invariant to the specific deployed number.

## Result: prediction MET, both halves

**E2 — rank, swept `m = 0.10` to `0.60` in steps of 0.05, both robots:**

| assignment | rank across the whole sweep | condition number range |
|---|---|---|
| `identity` | **3 (every m)** | 3.40 – 11.07 |
| `perm_emotion_D` | 3 (every m) | 1.92 – 3.61 |
| `perm_arousal_D` | 3 (every m) | 3.45 – 20.30 |
| `collapse_D` | **1 (every m)** | ∞ (every m) |

`identity`'s condition number **improves monotonically with `m`** — from 11.07
at `m=0.10` down to 3.40 at `m=0.60` — because a larger tier displacement moves
the third Jacobian column further from the other two, better conditioning the
map. The deployed value (`m=0.40`, cond ≈ 3.5) sits past the steep part of this
curve; doubling `m` to 0.60 buys only a further 0.14 of improvement, so the
system is not living on a knife-edge that a small perturbation would destabilise.

**E3 — tick-level switching at the measured noise anchor (σ=0.20), swept `m`:**

| assignment | switches, every `m ∈ [0.10, 0.60]` |
|---|---|
| `identity` | **0** |
| `perm_emotion_D` | 6 |
| `perm_arousal_D` | 5 |
| `collapse_D` | 6 |

Structurally invariant to `m`, exactly as the routing argument predicts: whether
the face can reach D is a topology question, not a magnitude question.

## A correction found while building this sweep

Building the rank computation across a *range* of `m`, rather than at the single
deployed point, surfaced a numerical fragility in the original E2 measurement
that a single-point check did not expose.

**`collapse_D`'s true rank is 1, not 2.** Under that assignment `P` and `Ar` are
structurally frozen at baseline — `compose_offsets` only ever assigns
`felt[tier_axis]` in the collapse branch — so every one of the five style
outputs is a function of the single scalar `D` alone. By the chain rule, every
row of the Jacobian is therefore a scalar multiple of one direction vector:
`[empathy/2, empathy/2, leak+m]`. That is **exactly** rank 1 whenever that vector
is nonzero (true throughout this sweep), not an approximation — confirmed
numerically to full float precision at a well-chosen step size, and confirmed by
an h-scan showing the earlier "second singular value" forms a V-shaped noise
curve typical of finite-difference float-cancellation error, not a flat plateau
a genuine second dimension would produce.

The original E2 report's "collapse_D: 2/3" (`docs/paper` / `PROGRESS.md`,
committed `6b81fc4`) used the default `h=1e-6`, too small for this specific
degenerate case — the spurious residual sits at machine-noise scale (1e-11 to
1e-17) but was still large enough to clear `conditioning()`'s tolerance. No
single fixed `h` is robust across the swept `m` range either, since the noise
floor scales with the Jacobian's own magnitude. The fix implemented here reads
the rank from the closed-form structural fact (`assignment.collapse`) rather
than trusting numerical SVD rank detection for this case at all.

**The qualitative conclusion is unchanged and, if anything, strengthened**:
`collapse_D` does not merely lose one degree of freedom, it loses two — of the
three sources, only one direction survives at all. This does not touch the
earlier corroborating result — the *style-space* rank-deficiency finding for
`collapse_D`, and the leak/no-leak calibration (which reads `idle`, a
single scalar function of arousal alone, and is unaffected by this correction)
— both stand as reported.

## Clamping tightens exactly as the analytic bound predicts

`|D_baseline| ≤ 1 − m` is exact, not fitted. Since `close`/`unknown` both reach
magnitude `m`, the admissible band **shrinks linearly** as `m` grows:

| | `m` bound (no clamp anywhere in the ladder) |
|---|---|
| CHATBOX | **m ≤ 0.357** |
| ELLEBOT | **m ≤ 0.579** |

| m | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | **0.40** | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CHATBOX | ok | ok | ok | ok | ok | ok | **CLAMP** | CLAMP | CLAMP | CLAMP | CLAMP |
| ELLEBOT | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | CLAMP |

**CHATBOX is already clamped at the deployed value** — its baseline `|D|=0.643`
exceeds the `m=0.40` bound of 0.357 by exactly 0.043, which is precisely the
figure the original audit measured as the observed `unknown`-tier saturation and
the non-uniform ladder step (0.157 vs 0.200). This sweep shows that clamp is not
a coincidence of the chosen `m`: **any** `m` above 0.357 clamps CHATBOX,
regardless of how it was arrived at.

**This is itself a design rule, not just an observation**: for a persona with
baseline Dominance `D₀`, the offset magnitude must satisfy `m ≤ 1 − |D₀|` for the
full four-tier ladder to be reachable without saturation. A persona placed nearer
the edge of the PAD cube (as CHATBOX's low-Extraversion, high-Agreeableness
combination places it) has *less* room for relationship-driven displacement
before the ladder's bottom rung collapses onto the ceiling — a concrete,
quantitative constraint linking persona design to relationship-ladder design that
did not have a stated form before this sweep.

## Verified — `padeval/tests/test_tier_sensitivity.py`, 7/7 (new), 44 padeval tests total

Pins the parameterisation against the live table, full rank for `identity` and
the corrected rank-1 for `collapse_D` across the sweep on both robots, finite
conditioning for the non-degenerate permutations, zero/nonzero switching as
predicted, and the exact clamp bound (0.357 for CHATBOX, matching the
audit-measured 0.043 margin).

## For the paper

> The offset magnitude `m` (deployed at 0.40) was swept over [0.10, 0.60] with
> the tier ratios held fixed, passed explicitly rather than by mutating the
> shipped model. `identity` remained full rank and structurally zero-switching
> across the entire range; `collapse_D` remained rank-1 and non-zero-switching
> throughout. The condition number of the identity assignment improves
> monotonically with `m` (11.07 → 3.40), so the deployed value is not a
> knife-edge choice. The admissible range before clamping, `m ≤ 1 − |D_baseline|`,
> is exact and persona-dependent; CHATBOX's baseline already exceeds it by 0.043
> at the deployed value, which is the source of its previously-reported
> `unknown`-tier saturation and non-uniform ladder step.
