# 9a — Complete reachability of the relationship ladder

8b established one path precisely: maximal warmth, zero disclosure, one session — `close`
provably unreachable. Correct, but one path through a larger space. This characterises the
whole space with the discipline 7a applied to the axis assignments: **a closed form for every
cell, and an independent simulation confirming every closed form.**

The three rules everything below derives from, quoted rather than paraphrased:

- `_tier_from_scores` (`kg_bridge.py:112-131`): `score = (rapport+trust)/2`; `score > 0.70 →
  close`; `score > 0.45 → known`; `count > 5 → known`; `count > 0 → visitor`. **All four
  comparisons are strict**, which is worth exactly one session at the `close` boundary.
- Rapport: `0.025 · P` per tick when `P > 0.05` (`webcam_loop.py:1583-1585`), 1 tick = 1 s
  (`webcam_loop.py:99`), hard-clamped at 1.0 (`webcam_loop.py:742`).
- Trust: per turn by disclosure depth, capped at 0.20/session (`SESSION_TRUST_CAP`, 8c).

With `P = 0.4·P_base + 0.6·v` (`affect.feel`, EMPATHY = 0.6), so `ρ = 0.025·P_max` per second.

## The table

| transition | channel path | minimum | unit | gated on |
|---|---|---|---|---|
| `unknown -> visitor` | count > 0 | **1** | turns | nothing — any affect, any content |
| `visitor -> known` | count > 5 | **6** | turns | nothing — neutral, disengaged turns qualify |
| `visitor -> known` | score > 0.45 (rapport only) (CHATBOX) | **51** | seconds | continuous v=+1 face; no disclosure needed |
| `visitor -> known` | score > 0.45 (rapport only) (ELLEBOT) | **47** | seconds | continuous v=+1 face; no disclosure needed |
| `known -> close` | score > 0.70 (joint) (CHATBOX) | **3** | sessions | 4 depth-3 disclosures per session, plus ~57s of warmth once |
| `known -> close` | score > 0.70 (joint) (ELLEBOT) | **3** | sessions | 4 depth-3 disclosures per session, plus ~52s of warmth once |

- **CHATBOX**: P_max = 0.7064, rho = 0.017660/s, two-path crossover at **8.49 s/turn**
- **ELLEBOT**: P_max = 0.7700, rho = 0.019250/s, two-path crossover at **7.79 s/turn**

| robot | to `known` at v=+1 | at v=0 (neutral face) | rapport stops below v |
|---|---|---|---|
| CHATBOX | 51s | 339s | -0.0940 |
| ELLEBOT | 47s | 212s | -0.2000 |
Every cell is derived closed-form **and** confirmed by stepping the deployed rules forward
and asking the live `_tier_from_scores` what tier resulted
(`padeval/tests/test_tier_reachability.py`, 8/8).

## Why the cross-check was not decoration

The `close` closed form initially returned **2 sessions** against the simulation's **3**. The
simulation was right. `2*0.70 - 1.0` evaluates to `0.3999999999999999`, not `0.40`, so the
exact-multiple branch in `_strict_ticks` never fired and the derivation lost an off-by-one on
the single most-quoted number in the phase.

The underlying arithmetic is worth stating because it is a genuine boundary case, not just a
float artefact: two capped sessions give trust **exactly 0.40**, hence score **exactly
0.70** — which does not satisfy `> 0.70`. The third session is what crosses. A closed form
alone, or a simulation alone, would each have been believed.

## `close` is a corner solution, not a tradeoff

The boundary is `rapport + trust > 1.40`, which invites looking for the cheapest point on
that line — rapport pushed higher to demand less trust, or the reverse. **There is no
interior optimum.** Rapport is hard-clamped at 1.0, so the feasible set is
`rapport ∈ [0, 1]`, and since trust is the expensive channel (0.20 per session against
rapport's ~0.9 per minute) the cheapest feasible point is always the corner `rapport = 1.0`.
The constraint therefore collapses to an exact statement:

> **`close` ⟺ `trust > 0.40`**

This is strictly stronger than 8b's "trust must be nonzero" — that was necessary, but it
understated the bar by the entire trust budget. Verified at both sides of the boundary:
`trust = 0.40` gives `known`, `trust = 0.41` gives `close`.

## Two findings to see clearly, both design properties rather than defects

### 1. `known` has two fast paths, and one requires no engagement whatsoever

`known` is reachable either by **6 turns of anything at all** — confirmed with `v = −1`, where
felt pleasure is below the accrual gate so rapport contributes exactly zero — or by
**~50 seconds of sustained warmth**. Six hostile, disengaged turns unlock the same tier as a
minute of genuine warmth.

Which is faster depends on how fast the conversation moves, and the crossover is exact:

| robot | crossover |
|---|---|
| CHATBOX | **8.49 s/turn** |
| ELLEBOT | **7.79 s/turn** |

Below it the volume path wins; above it the warmth path does. Stated plainly: **half the
four-rung ladder can be climbed by turn-taking alone.** This is reported, not fixed — whether
`count > 5` should gate a *relationship* tier is a design question, and it is the user's.

### 2. A neutral face is not affectively neutral to this system

Felt pleasure is `0.4·P_base + 0.6·v` and the accrual gate is `P > 0.05`. Both robots carry a
positive baseline P, so **a blank face still builds rapport**:

| robot | felt P at v = 0 | rapport rate | `known` on a neutral face | rapport stops below |
|---|---|---|---|---|
| CHATBOX | 0.1064 | 0.00266/s | 338 s | v = −0.094 |
| ELLEBOT | 0.1700 | 0.00425/s | 212 s | v = −0.200 |

The no-rapport regime requires a mildly **negative** face, not an absent one. This was found
by a test that assumed the opposite and failed — "disengaged" and "neutral" are not the same
state here, because the robot's own temperament supplies most of the felt value and the face
only moves it 60% of the way.

## 9c — The disjointness lemma

> **Lemma.** Let two accrual channels `a` and `b` drive a single scalar `s = (a+b)/2`. If
> `a` and `b` are driven by observationally-equivalent signals — that is, if the information
> determining `b` is a function of the information determining `a` — then `b` carries no
> information about the world that `a` does not, and `s` is an affine function of `a` alone.
> The second channel is then not a second channel but a disguised copy of the first, and the
> pair cannot be independently identified from `s`. **Disjoint information content is
> therefore necessary for `a` and `b` to compose rather than collapse.**
>
> This is the same necessary condition 7a proved for the three PAD axes, applied one level
> down: there, sources routed onto a shared axis produced a rank-deficient Jacobian and were
> provably unrecoverable by any observer; here, rapport and trust driven by the same signal
> reduce `score` to `rapport` and make trust decorative. Defining disclosure as *information
> the camera does not already have* (8a) is precisely the disjointness condition that
> discharges the lemma's hypothesis, which is why the pre-`2be86bb` system — writing the same
> delta to both — was `collapse_D` at a smaller scale.
