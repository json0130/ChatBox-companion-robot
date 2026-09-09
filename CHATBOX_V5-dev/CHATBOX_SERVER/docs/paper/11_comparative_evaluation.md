# 11 — Comparative evaluation: an external baseline and three ablations

Every prior comparison (E3, 7a) permuted the deployed system's OWN axis assignment. That is
an ablation, not a baseline, and it invites the obvious reviewer question: measured against
what published method?

## The baseline, and what it actually shares with us

WASABI (Becker-Asano & Wachsmuth, AAMAS 2010) routes at the axis level exactly as we do,
**verified from the primary source**: two valences map into PAD continuously from embodied
emotion dynamics, and Dominance is "derived from the situational context in the cognition of
the architecture" — a separate, third channel, same as ours. The axis assignment is not a
point of difference; it is corroboration. An appraisal-theoretic architecture from 2008–2010
converged on the same routing our own identifiability criterion (7a) selects independently.

**The difference is timescale, not assignment.** In WASABI's Skip-Bo scenario, their agent MAX
"feels dominant whenever it is his turn and non-dominant, i.e. submissive, otherwise" —
turn-taking state with no memory across episodes. Ours accumulates across sessions with a
provable floor (9a: τ_D ≥ 3 sessions, gated at trust > 0.40).

This phase implements **WASABI's D-sourcing rule** as one comparison arm inside our own
pipeline, holding the axis assignment fixed. **It does not reimplement WASABI's architecture**,
and no claim below should be read as if it does.

## The five arms

All five keep the identical axis routing: face → P, face → Ar, "the relationship, however
sourced" → D. Only the D-generating mechanism varies.

| Arm | D source | Isolates |
|---|---|---|
| **A1 — Full (deployed)** | tier(accumulated rapport + trust) | the proposed method |
| **A2 — No trust channel** | tier(accumulated rapport only) | is disclosure load-bearing? |
| **A3 — No accumulation** | tier(this turn's warmth only, no memory) | is accumulation load-bearing, or just the axis? |
| **A4 — WASABI-style** | ±0.40 by turn ownership, no memory | the published external comparison |
| **A5 — No relationship** | persona baseline, constant | the floor: what does routing D at all buy? |

A2 never calls the disclosure detector — the trust channel does not exist for A2, so Phase 10's
known defects in that detector cannot leak into this arm's result even in principle. A4's ±M
deliberately excludes the deployed ladder's +0.10 arousal leak: that leak is our own documented
calibration target (E2), not part of WASABI's rule, and including it would attribute our
violation to their design.

## The five-arm, five-metric table

| | Metric 1: rank | Metric 2: frame-noise switches/min | Metric 3: std(D), axis-level | Metric 4: τ_D | Metric 5: ownership switches/min |
|---|---|---|---|---|---|
| **A1 full** | 3/3 | 0.00 | 0.0000 | **4 sessions** (τ_D/τ_P ~ 10²–10³ in ticks, 9a) | 0.00 |
| **A2 no trust** | 3/3 | 0.00 | 0.0000 | `known`: 2 sessions. **`close`: unreachable** | 0.00 |
| **A3 no accumulation** | 3/3 | **116.25 (CHATBOX) / 41.25 (ELLEBOT)** | 0.0000 | **0 ticks** settling (ratio to τ_P = 0.0) | 0.00 |
| **A4 WASABI-style** | 3/3 | 0.00 | 0.0000 | 10-tick ownership period (ratio to τ_P = 4.94) | **5.85** |
| **A5 no relationship** | **2/3** | 0.00 | 0.0000 | undefined — D never changes | 0.00 |

Both robots agree on every column except the magnitude of metric 2 for A3 (CHATBOX's lower,
more central baseline P crosses the instantaneous band edges more often under the same noise).

## What metrics 1 and 3 cannot show, and why that is stated rather than hidden

Both operate on the axis-level Jacobian (`e2_identify.jacobian`), which treats "the
relationship" as an external scalar ramping toward a target offset. That scalar's *temporal
source* — accumulated across sessions (A1), accumulated without trust (A2), or recomputed
fresh every turn (A3) — is invisible to a Jacobian evaluated at one static point. **A1, A2 and
A3 are therefore bit-identical on metrics 1 and 3, by construction, not by coincidence** — the
new ramp-taking helper used for A4/A5 was regression-tested to reproduce the existing,
already-published `closed_form_std` bit for bit at the point they overlap. This is exactly why
the brief specified metric 5 rather than metric 2 as the arm-separating measurement: metrics 1
and 3 test the axis assignment, which every arm shares.

## The registered predictions, verdicted against the results above

| # | Prediction | Verdict |
|---|---|---|
| 1 | All five arms rank 3 | **NOT MET, as literally stated.** A5 is rank 2/3 — nothing reaches D under it, by design. The comparability check is *working*: it disqualifies exactly the arm built to have no relationship channel, for a different reason than `collapse_D`'s axis collision (7a). |
| 2 | Structural zero switching under frame noise, all five | **MET, with the stated exception.** A1/A2/A4/A5 are zero. **A3 is not** (62–116/min) — the brief's own prediction names this as expected, and it is confirmed rather than assumed: A3 is the one arm that routes the face to D directly, so frame noise reaches it by the same path felt Pleasure does. |
| 3 | τ_D/τ_P: A1 at 10²–10³, A3 and A4 at 10⁰ | **MET.** A3's ratio is exactly 0.0 — *at least* as reactive as the deployed P axis, not merely the same order, because A3 excludes the mood-blend lag the deployed P path carries. A4's ratio is 4.94 — same order as P, categorically not session-scale. |
| 4 | Metric 5 separates A4 from the rest | **MET.** A4 switches on 39/39 possible ownership flips; every other arm holds at exactly zero under a script that holds the relationship fixed on purpose. This does **not** contradict Phase 9's finding that A1 can move mid-session from rapport accrual — that is a different scenario (the relationship itself changing) from this one (isolating ownership sensitivity with the relationship held fixed). |
| 5 | A2 vs A1 distinguishable only if the trust channel matters | **Distinguishable on exactly one metric.** Metrics 1–3 are identical, exactly as predicted. Metric 4 differs: A1 reaches `close`; A2 **provably never does** (9a: score caps at 0.50 against a 0.70 threshold). Trust is load-bearing, and the claim is attributed to the one metric that shows it rather than left as an unlocated "A2 is worse." |

**The gate the brief singles out as the headline case if it triggers:** does A3 match A1 on
every metric? **It does not.** A3 diverges from A1 on metrics 2 and 4 — the two metrics that
involve time, which an axis-only ablation could never have distinguished. **Accumulation is
load-bearing; the contribution is not merely the axis choice WASABI already shares.**

## 11e — Metric 6: the scenario where WASABI's rule should win

Mehrabian's Dominance conflates two distinct things: **social standing** (who this person is to
me, accumulated over the relationship) and **situational control** (who holds rhetorical
authority right now). This phase's D routes only the first; WASABI's routes only the second.
Neither is complete, and metrics 1–5 were all built to characterise *our* axis, so none of them
could have surfaced this.

**Scenario:** the robot is mid-explanation, correcting a factual error the child stated. It
legitimately holds the floor for one turn, independent of how close the relationship is.
`robot_turn=True` stands for holding that authority for the one turn the scenario describes
(not for every turn indiscriminately, which is what made A4 chatter in metric 5).

| | A1/A2/A3/A5 gain | A4 gain |
|---|---|---|
| At every tier, both robots | **0** (bit-identical D and directive text) | **+3** (constant across tier) |

Quoted, tier=`known`:

> **CHATBOX A1** — ordinary: *"follow their topic, and you may add one follow-up question about
> it."* Correcting: **the identical sentence.**
> **CHATBOX A4** — ordinary: *"answer only what they ask, and do not introduce a topic of your
> own."* Correcting: *"either of you may open a topic — introduce one only if things go quiet."*

**This is the flip side of metric 5, not a contradiction of it.** Metric 5 showed A4 chattering
on ownership changes that carry no relational meaning; this shows the same mechanism correctly
serving a moment that does. Reported together: each rule captures a different sense of
Dominance, with evidence for both.

## Metric 5 as the paper's figure candidate

No plotting library is added to this project (established convention: CSV/LaTeX only). The
data for a bar chart — switches/min by arm, both robots — is emitted at
`runs/eval/11_metric5_ownership.csv`:

```
robot,arm,switches_per_min
CHATBOX,A1_full,0.00        CHATBOX,A2_no_trust,0.00        CHATBOX,A3_no_accumulation,0.00
CHATBOX,A4_wasabi,5.85      CHATBOX,A5_no_relationship,0.00
ELLEBOT,A1_full,0.00        ELLEBOT,A2_no_trust,0.00        ELLEBOT,A3_no_accumulation,0.00
ELLEBOT,A4_wasabi,5.85      ELLEBOT,A5_no_relationship,0.00
```

A single bar at ~5.85/min against four bars flat at zero is the whole finding in one glance,
and it is the metric that most directly maps onto the contribution: everything else in this
table involves some structural blindness or a unit mismatch that needs a sentence to interpret;
this one does not.

## What this phase does not claim

- **Not "we beat WASABI."** WASABI's turn-taking dominance rule is well matched to its own
  scenario — a two-player card game where holding the floor to act *is* the dominance signal.
  The finding is a statement about **fit to a companion-robot dialogue setting**, where floor-
  holding alternates every few seconds for reasons that have nothing to do with the
  relationship, not a statement about the quality of WASABI's design.
- **Not that A4 is WASABI.** A4 is WASABI's D-sourcing rule, run inside our pipeline with
  everything else held fixed. It is described that way everywhere in this report and the
  accompanying code.
- **Not that the structural-zero result (metric 2, for A1/A2/A4/A5) distinguishes our method
  from WASABI's.** It cannot — both share the axis assignment, and metric 2 tests the axis
  assignment. The separating measurement is metric 5, and the report says so rather than
  implying the zero itself is the win.

## Verified

`padeval/tests/test_dsourcing.py` (8), `test_dsourcing_metrics.py` (13),
`test_dsourcing_predictions.py` (6) — 27 new tests, 125/125 padeval tests total.
`affect.py`, `WEIGHTS`, `TIER_OFFSETS` and the existing E2/E3 code paths are unmodified; every
arm is passed to the existing machinery as an explicit `ramp` argument, the same seam E2's
leak/no-leak pair and 7a's enumeration already use. Zero LLM calls throughout.
