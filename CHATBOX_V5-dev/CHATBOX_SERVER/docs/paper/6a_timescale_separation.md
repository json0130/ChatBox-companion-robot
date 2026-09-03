# 6a — Timescale separation and cross-talk

**Registered before computing:** τ_D/τ_P > 10³. **Result: NOT MET** — the measured
separation is ~10², two orders of magnitude, not three. Reported as-is rather than
reframed; the paper's wording needs revising, not the threshold.

## τ_P and τ_Ar — the fast axes

Measured as the end-to-end step response of the deployed path, not the bare
fusion equation. All three live stages included: the 0.7/0.3 mood blend
(`kg_bridge.py:270`), the 5-sample `AffectStream` mean (`stream.py:28`, called at
`adapter.py:93`), and the empathy fusion (`affect.py:82`).

Integer tick-counting quantises at τ≈1 and is too coarse, so τ is taken from the
**effective pole** — log-linear fit to the geometric error decay, skipping the
first 6 ticks while the 5-sample window fills.

| configuration | pole a | τ (ticks) |
|---|---|---|
| fusion only | — | 0 (memoryless) |
| + 5-sample AffectStream mean | — | 0 (FIR, settles in-window) |
| + 0.7/0.3 mood blend only | 0.1800 | 0.583 |
| **FULL live path** | **0.6103** | **2.025** |
| bare-fusion analytic, −1/ln(1−e) | — | 1.091 |

**τ_P = τ_Ar = 2.025 controller ticks** (1 tick = 1 s, `_DEFAULT_TICK`,
`webcam_loop.py:99`). The smoothing stages lengthen τ by **1.86×** over the bare
fusion figure, so the honest number is 2.025, not 1.091.

Two structural points worth stating. **The fast axes have no dynamics of their
own** — `affect.feel` is memoryless; every bit of their memory comes from the two
smoothing stages. And the mood-blend-only pole of **0.1800** reproduces the
`0.3·e` closed form derived in the earlier audit pass, which cross-validates both.

## τ_D — the slow axis

Post-Fix-3 accrual: the live tick moves **rapport only** (+0.025·P), trust moves
only at end-of-session extraction, capped at ±0.2. Warm profile (happy face, felt
P = +0.586):

| step | reached | sessions |
|---|---|---|
| unknown → visitor | session 1 | 1 |
| visitor → known | session 1 | 0 (same session) |
| known → close | session 3 | 2 |
| **unknown → close** | **session 3** | **3** |

**τ_D = 3 sessions** to traverse, 2 sessions for the slowest single step.

The ≥3-session floor is set by the ±0.2 extraction cap and is **independent of
session length or how warm the interaction is**: rapport saturates from live
ticks, so score > 0.70 then requires trust > 0.4, and trust can only move +0.2 per
session. That makes it a property of the controller, not of the deployment.

## The ratio — and why it needs a stated conversion

τ_P is in **ticks**; τ_D is in **sessions**. Neither converts without a
session-duration assumption, which is an *operational* parameter, not a property
of the controller. So the ratio is reported as a function of it:

| session length | τ_D (ticks) | ratio | orders |
|---|---|---|---|
| 0.5 min | 90 | 44 | 10^1.65 |
| 1 min | 180 | 89 | 10^1.95 |
| **2 min** | **360** | **178** | **10^2.25** |
| 5 min | 900 | 444 | 10^2.65 |
| 10 min | 1800 | 889 | 10^2.95 |
| 30 min | 5400 | 2667 | 10^3.43 |

In turns, at the empirical median of 2 turns/session: 30 (10 s/turn), 89
(30 s/turn), 178 (60 s/turn).

**The prediction of 10³ is only reached at session lengths of ~30 minutes**, which
is not a plausible child-interaction episode. Across every realistic operating
point the separation is **10²**.

### Caveat on the empirical conversion

`sessions.db` holds **31 sessions, 97 turns, 4 persons**; median **2** turns per
session, IQR **[1, 4]**, with 11 of 31 sessions being single-turn.

```sql
SELECT session_id, COUNT(*) AS turns FROM turns GROUP BY session_id
```

This is development and test traffic, not deployment data. It is quoted for
transparency and **is not adequate to fix the conversion**, which is why the ratio
is reported across a range rather than as one number.

## Cross-talk matrix

|d(PAD axis)/d(input channel)|, identity assignment, dimensionless gain. Computed
by reusing `e2_identify.leakage_matrix` — not recomputed. Identical for both
robots.

**As deployed** (with the `close`-tier +0.10 arousal offset):

| | valence | arousal | tier |
|---|---|---|---|
| **P** | 0.6000 | 0 | 0 |
| **Ar** | 0 | 0.6000 | **0.1000** |
| **D** | 0 | 0 | 0.4000 |

**Counterfactual** (that one offset zeroed): **exactly diagonal**, off-diagonal
mass 0.0000.

This is the numerical statement of "each signal owns the axis it can carry". The
architecture is diagonal by construction; the single 0.1000 entry is the one
documented, deliberate exception, and it is confined to one tier of four.

**On the trait channel:** it is *not* a column. Traits set the baseline point via
`affect.to_pad`; they are not a per-turn input, so their contribution is a one-off
offset rather than a derivative. Rendering them as a gain column would imply a
dynamic the system does not have.

## What this changes for the paper

The separation claim survives but must be **restated as two orders of magnitude,
not three**, and the paper should give the ratio with its session-length
assumption attached rather than as a bare number. The stronger and more defensible
form of the claim is the structural one: τ_D's floor is set by a cap in the update
rule, so the separation cannot be tuned away by changing session length — only its
*magnitude* depends on deployment.
