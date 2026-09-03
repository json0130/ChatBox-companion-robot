# 7a — The general identifiability theorem

**What this replaces.** E2 previously read as *"we checked four assignments;
three were fine and one was not."* That is a spot-check. This states and
verifies the general fact of which those four are instances, turning an
empirical observation into a characterization.

## The two statements

Let an assignment route each of three input sources (valence, arousal, tier)
onto one of three PAD axes (P, Ar, D). Let `T = {valence_axis, arousal_axis,
tier_axis}` as a **set**, and `j = 3 − |T|` the number of axes never written.

> **Theorem 1 (freezing bounds rank).** If `j` axes are frozen at baseline —
> receiving zero contribution from every source — then `rank(J) ≤ 3 − j` for the
> axis-space Jacobian `J = ∂(P,Ar,D)/∂(valence,arousal,tier)`, regardless of how
> the remaining axes are driven.
>
> *Proof.* A frozen axis is a constant function of the inputs, so its Jacobian
> row is identically zero by the chain rule — exactly, not approximately, and
> independent of any step size. `J` then has at most `3 − j` nonzero rows. ∎

> **Theorem 2 (permutations preserve rank).** If the assignment is a bijection —
> every source to a distinct axis, no axis frozen — then `rank(J) = 3`.
>
> *Proof.* Each axis is written by exactly one source, with gain `empathy` for
> the face sources and the routed offset magnitude for tier. `J` is a
> permutation matrix scaled by a nonzero diagonal, invertible by construction
> whenever `empathy ≠ 0` and the offset magnitude `≠ 0`. ∎

Theorem 1's proof is the generalization of the argument that corrected
`collapse_D`'s rank in 6c — the same chain-rule fact, stated once for all cases
rather than rediscovered per case. It is also precisely why the finite-difference
rank detection failed there and the structural argument was authoritative.

## The complete table: 30 cases, all predicted

Every structurally distinct assignment — 27 non-collapse (3 sources × 3 axes) and
3 collapse variants — with rank computed **two independent ways**: from a
closed-form Jacobian derived directly from `axes.compose_offsets`, and from
finite differences via the existing `e2_identify.jacobian`.

| class | cases | predicted rank | measured rank | agree |
|---|---|---|---|---|
| bijection | 6 | 3 | 3 | 6/6 |
| one-axis-frozen | 18 | 2 | 2 | 18/18 |
| two-axes-frozen | 3 | 1 | 1 | 3/3 |
| collapse | 3 | 1 | 1 | 3/3 |
| **total** | **30** | | | **30/30** |

The analytic and numerical Jacobians agree to **3.3 × 10⁻¹⁴** (machine
precision) across all 30 — confirming the 6c failure mode does not recur once
the step size is chosen deliberately, and that the closed form is correct.

**24 of the 30 possible assignments are rank-deficient**, i.e. unidentifiable.
Only the 6 bijections are not.

### The four originally-tested assignments, as corollaries

| assignment | collapse | \|T\| | predicted | measured | class |
|---|---|---|---|---|---|
| `identity` | no | 3 | 3 | 3 | bijection |
| `perm_emotion_D` | no | 3 | 3 | 3 | bijection |
| `perm_arousal_D` | no | 3 | 3 | 3 | bijection |
| `collapse_D` | yes | 1 | **1** | **1** | collapse |

Every previously-measured value is now *predicted* rather than observed —
including the corrected rank-1 for `collapse_D` from 6c, which is no longer a
special case but an instance of Theorem 1 with `j = 2`.

## What this does NOT establish

Identifiability is **necessary but not sufficient** for correctness. All six
bijections are equally full-rank and therefore equally identifiable — **nothing
in this section singles out `identity` as the correct routing.** A test
(`test_identifiability_does_not_single_out_identity`) pins this so the claim
cannot quietly drift.

The overall argument is two-part and must be reported as such:

- **`collapse` and every degenerate routing are ruled out by identifiability
  alone** (this section, Theorem 1) — 24 of 30 assignments fall here.
- **Among the 6 identifiable bijections, `identity` is singled out by dynamics**
  — E3's setpoint stability and 7.0's noise propagation, where a per-frame source
  driving a per-session effector produces continuous rung-switching that the
  correct routing structurally cannot.

## A semantic subtlety, found by reading the code

`axes.compose_offsets` assigns `felt[valence_axis]` then `felt[arousal_axis]`
**sequentially**. When both route to the same axis, the second **overwrites** the
first — so valence is *discarded entirely*, not mixed with arousal. Those cases
mean "one source is silently dropped", not "two sources share an axis". The rank
arithmetic is unaffected (a dropped source gives a zero *column*, a frozen axis a
zero *row*; both reduce rank), but the interpretation differs, and **only the
explicit `collapse=True` branch actually averages sources.** Recorded because a
reader would otherwise reasonably assume duplicate routing implies mixing.

## Verified — `padeval/tests/test_identifiability.py`, 8/8 (new), 57 padeval tests total

Pins complete enumeration, both theorems, all 30 predictions against two
independent rank computations, machine-precision Jacobian agreement, the four
original assignments as corollaries, the anti-overclaim guard, and the
overwrite semantics.

## For the paper

> The identifiability of a source→axis assignment is fully determined by its
> structure. If `j` of the three PAD axes receive no contribution from any source,
> the axis-space Jacobian has at most `3 − j` nonzero rows and rank at most
> `3 − j` (chain rule, exact); if the assignment is a bijection, the Jacobian is
> an invertible scaled permutation and has full rank. Enumerating all 30
> structurally distinct assignments and computing rank both in closed form and by
> finite differences confirms the prediction in every case (30/30, Jacobians
> agreeing to 3×10⁻¹⁴): 24 assignments are rank-deficient and hence
> unidentifiable, and the 6 bijections are not. The four assignments evaluated in
> §E2 are corollaries of this result rather than independent measurements. Note
> that identifiability is necessary but not sufficient: all six bijections are
> equally full-rank, and the selection of the deployed routing among them rests
> on the timescale-separation and setpoint-stability results of §E3.
