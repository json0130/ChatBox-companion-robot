# 8b — The whole controller, traced end to end

Phases 6 and 7 each established one property in isolation. None of them showed the
components running together. This does: four scripted sessions, both robots, every
established quantity logged per turn, **no camera, no LLM, no network**, entirely
reproducible from a seed.

The face signal is synthetic, drawn with the per-frame dispersion measured on the real
corpus in 7.0 (σ_V = 0.279, σ_A = 0.292, `noise_propagation.py:71-72`). The utterances are
authored. The coordinate, the effectors, the tier thresholds and both closeness deltas are
the deployed code paths, imported rather than reimplemented.

## Four predictions, registered before the run

| | prediction | result |
|---|---|---|
| P1 | D moves in discrete steps at session boundaries only, never within a session | **FALSIFIED** |
| P2 | mood visibly decays between sessions; no session-old value reaches the next first turn at full weight | **held** |
| P3 | rapport and trust diverge at least once, exactly at the scripted disclosure | **held** |
| P4 | CHATBOX's D clamps at the tier 7b predicts; ELLEBOT's does not | **held** |

## The trace — CHATBOX

Disclosure is scheduled **away** from the warmest turns on purpose. Had the two co-occurred,
a divergence between rapport and trust could not be attributed to either.

| s | t | emotion | tier | D | clamp | mood in | used | P | rapport | trust | disc | ampl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | neutral | `unknown` | −1.000 | **Y** | — | — | +0.106 | 0.0532 | 0.0000 | · | 0.300 |
| 0 | 1 | happy | `visitor` | −0.843 | · | +0.106 | +0.106 | +0.447 | 0.2768 | 0.0000 | · | 0.355 |
| 0 | 2 | happy | `visitor` | −0.843 | · | +0.447 | +0.447 | +0.499 | 0.5264 | 0.0000 | · | 0.355 |
| 1 | 0 | happy | `visitor` | −0.843 | · | +0.499 | **gated** | +0.591 | 0.8218 | 0.0000 | · | 0.355 |
| 1 | 1 | neutral | `visitor` | −0.843 | · | +0.591 | +0.591 | +0.187 | 0.9153 | 0.0000 | · | 0.355 |
| 2 | 0 | neutral | `known` | −0.643 | · | +0.187 | **gated** | +0.143 | 0.9869 | 0.0600 | **d3** | 0.425 |
| 2 | 1 | sad | `known` | −0.643 | · | +0.143 | +0.143 | −0.156 | 0.9869 | 0.1000 | **d2** | 0.425 |
| 2 | 2 | neutral | `known` | −0.643 | · | −0.156 | −0.156 | +0.077 | 1.0000 | 0.1600 | **d3** | 0.425 |
| 2 | 3 | happy | `known` | −0.643 | · | +0.077 | +0.077 | +0.386 | 1.0000 | 0.1600 | · | 0.425 |
| 3 | 0 | happy | `known` | −0.643 | · | +0.386 | **gated** | +0.444 | 1.0000 | 0.1600 | · | 0.425 |
| 3 | 1 | neutral | `known` | −0.643 | · | +0.444 | +0.444 | +0.090 | 1.0000 | 0.2000 | **d2** | 0.425 |

ELLEBOT runs the same script with no clamping at any rung, and reaches `known` one session
earlier (its higher baseline P accrues rapport faster).

## P1 — falsified, and the reason is structural

**D steps within a session, on both robots.** `pre_turn` re-derives the tier *every turn*
(`kg_bridge.py:296`), and **neither kind of threshold is session-scoped**:

- **count-gated** — `count > 0 → visitor` (`kg_bridge.py:112-131`) counts *turns across all
  sessions*, so it fires on turn 2 of the very first session. This is CHATBOX's step.
- **score-gated** — `score > 0.45 → known`. Rapport accrues once per *tick* (~20 per turn),
  so a warm run crosses the threshold mid-session. This is ELLEBOT's step at s1t1, where
  rapport had reached 0.96 with trust still at 0.

Reporting CHATBOX alone would have been comfortable and wrong: it shows only the count-gated
case, which reads like a tidy first-contact exception. ELLEBOT shows the score-gated rung
doing it too. **Nothing in the tier derivation makes D a per-session quantity.**

This does not overturn the architecture's claim — D still moves ~100× slower than P, which
is what 6a measured and what the separation argument rests on. But the paper must say *"D
changes on the order of sessions"* and not *"D changes only at session boundaries"*. The
second is a stronger sentence and it is false.

## P2 — held, and shown as a difference rather than asserted

All **3/3** session-opening turns had their stale mood gated out, none mid-session. The
counterfactual matters here: running the same seed with the gate disabled (the pre-`f5bfac9`
behaviour) shifts the opening valence of each later session by **+0.0925, −0.0348, +0.0531**.
Without computing both branches, "mood is gated" would be a property of a flag rather than a
visible consequence.

Note that the gate is **session-scoped, not time-decayed** (`kg_bridge.py:242-256`). A
five-minute gap and a five-day gap are the same event to it. That is why this trace does not
invent an inter-session gap distribution: there is no time constant here to defend.

## P3 — held

On both robots trust first moves at global turn 5 — **s2t0, the scripted disclosure turn**.
Every turn before it moved rapport and left trust exactly 0. Final: **rapport 1.0000, trust
0.2000**, tier score 0.6000, which is neither of them.

## P4 — held, at the tier named in advance

7b's rule: admissible iff `|D_baseline| ≤ 1 − m`. CHATBOX's D_baseline = −0.643 against the
`unknown` offset of −0.40 gives −1.043, so `unknown` saturates and the other three rungs do
not. ELLEBOT's +0.421 leaves headroom everywhere.

Observed: CHATBOX clamped at exactly `{unknown}` and nowhere else, **losing 0.043 of
commanded D**; ELLEBOT clamped at no rung. The predicted set was computed from 7b's bound
before the trace was read, not recovered from it.

The visible cost is in the effectors: amplitude is 0.300 at `unknown` where the true command
wanted lower, against 0.355/0.425 at the rungs above.

## τ_D recomputed under the 8a mechanism — and a regression 6a's method would have missed

8a replaced end-of-session LLM trust with per-turn rule-based trust, so 6a's τ_D cannot be
carried over. Recomputed by 6a's own method (sessions to traverse to `close`, warmest
profile, deepest disclosure every turn):

| turns/session | sessions to `close` |
|---|---|
| 1 | 7 |
| **2 (6a median)** | **4** |
| 4 (6a IQR upper) | **2** |

**At the median, τ_D = 4 sessions, up from 6a's 3 — the separation is slightly stronger.**
Re-deriving 6a's ratio table with τ_D = 4:

| session length | τ_D (ticks) | ratio to τ_P = 2.025 | orders |
|---|---|---|---|
| 1 min | 240 | 119 | 10^2.07 |
| **2 min** | **480** | **237** | **10^2.37** |
| 5 min | 1200 | 593 | 10^2.77 |

**The two-order-of-magnitude separation holds, and improves.**

**But a property was lost, and it is not visible at the median.** 6a's 3-session figure was a
*hard floor*: trust moved once per session and the extractor capped it at ±0.2, so the floor
held "independent of session length or how warm the interaction is" — a property of the
controller rather than of the deployment. 8a accrues trust **per turn with no per-session
cap**, so that independence is gone: at 4 turns/session, `close` arrives in **2** sessions,
below 6a's floor.

Reporting only the median (4 sessions, an improvement) would have concealed this. The fix is
one line — cap cumulative trust gain per session at the extractor's old ±0.2 — and it would
restore the floor while keeping the determinism 8a bought. **This is flagged, not applied:**
it changes the accrual mechanism a third time, and the test
(`test_tau_D_recomputed_under_the_8a_mechanism`) pins the current behaviour so the change
cannot land silently.

## Verified — `padeval/tests/test_system_trace.py`, 9/9 (new), 82 padeval tests total

Includes a seed-reproducibility and seed-sensitivity check, the gated/ungated counterfactual,
and an effector check that amplitude is constant within a tier and differs across tiers — the
trace must reach the servos or it is not end to end.
