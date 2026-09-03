# Progress Log

Running record of what was **tried**, what **worked**, and what **didn't work / was fixed** — one entry
per commit, newest first. Companion to `RND_KG_Companion_System.md` (the design report). Kept so the
research write-up can reference which approaches were attempted and why.

---

## exp(padeval): 7b — persona admissibility as a design rule  *(branch `feature/pad-affect-core`)*

Generalizes 6c's `|D_baseline| <= 1-m` from one persona to the trait design space.

**Two structural simplifications.** Neuroticism has **zero weight** on D (`affect.py:41-43`), so admissibility
over the 5-D trait cube is exactly the 4-D question in (O,C,E,A) and a persona's N cannot affect whether it
clamps — pinned against live `to_pad`, with a test that fails if a future weights change adds an N term. And D
is linear, so the admissible set is a slab, not a shape needing exploration.

**Admissible fraction**, by Monte Carlo (400k, with CI) and near-exact convolution, agreeing at every point:
97.6% at m=0.10, **82.7% at the deployed m=0.40**, 61.3% at m=0.60. So 17.3% of assignable personas would
clamp as deployed. Uniform is used deliberately — this is a design-space coverage question, not a claim about
human trait distributions, and a concentrated distribution would flatter the design.

**The stated intuition is half wrong, and the missing half is the one that shipped.** "High E, low A is the
risky combination" describes only the UPPER violation. The bound is two-sided, so the mirror (low E, high A)
violates the lower bound equally often — 8.6% each, exactly symmetric. **CHATBOX is the mirror case**: E=-0.6,
A=+0.6, D=-0.643, clamping for being too *deferential*, not too assertive. Checking only the intuited direction
would have reported the intuition confirmed and missed the case actually present in the shipped system.

**The rule, usable form:** at m=0.40 and A=+0.6 (the value both deployed personas share), safe E is
[-0.68, +1.00]. CHATBOX sits at E=-0.60 — 0.08 inside the boundary, which is why a persona designed without
this constraint landed so near it.

Verified — `padeval/tests/test_persona_admissibility.py` 7/7 (new), 64 padeval tests total.

Report for the draft: `docs/paper/7b_persona_admissibility.md`.

---

## exp(padeval): 7a — identifiability characterized, not spot-checked  *(branch `feature/pad-affect-core`)*

Turns E2 from "we checked four assignments" into a proof with those four as corollaries.

**Two theorems.** (1) If `j` of the 3 PAD axes are frozen at baseline, the axis-space Jacobian has at most
`3-j` nonzero rows, so `rank <= 3-j` — by the chain rule, exactly, independent of step size. This is the
generalization of the argument that corrected `collapse_D`'s rank in 6c, and precisely why finite-difference
rank detection failed there while the structural argument was authoritative. (2) A bijective routing gives an
invertible scaled permutation matrix, hence full rank.

**All 30 structurally distinct assignments enumerated** (27 non-collapse + 3 collapse) with rank computed two
independent ways — a closed-form Jacobian derived from `compose_offsets`, and finite differences via the
existing `e2_identify.jacobian`. **30/30 agree**, Jacobians matching to 3.3e-14. 24 of 30 assignments are
rank-deficient; only the 6 bijections are not. The four originally-tested assignments are now *predicted*,
including the corrected rank-1 `collapse_D` (an instance of Theorem 1 with j=2, no longer a special case).

**Explicitly does NOT overclaim.** All 6 bijections are equally full-rank, so identifiability alone does not
single out `identity` — a dedicated test pins this so the claim cannot drift. The argument is two-part:
degenerate routings are ruled out by identifiability (24/30); among the identifiable bijections, `identity` is
singled out by *dynamics* (E3 setpoint stability, 7.0 noise propagation).

**A semantic subtlety found by reading the code:** `compose_offsets` assigns valence then arousal
*sequentially*, so when both route to one axis the second **overwrites** the first — valence is discarded, not
mixed. Those cases mean "a source is silently dropped", not "two sources share an axis". Rank arithmetic is
unaffected; the interpretation is not. Only `collapse=True` actually averages sources.

Verified — `padeval/tests/test_identifiability.py` 8/8 (new), 57 padeval tests total.

Report for the draft: `docs/paper/7a_identifiability_theorem.md`.

---

## exp(padeval): 7.0 — measured noise propagated, and an apparent E2/E3 tension resolved  *(branch `feature/pad-affect-core`)*

Finishes 6d (previously timeboxed as stretch) with an unhurried pass. **Registered prediction NOT met as
literally stated** — reported honestly rather than redefined after the fact — but the resolution is a real,
useful finding in its own right.

**Two independent routes agree exactly.** Closed-form variance propagation (through the E2 axis Jacobian's
valence/arousal columns, scaled for the deployed 5-sample `AffectStream` window) and a 20,000-draw Monte Carlo
simulation of the actual pipeline agree to <0.01 on every axis, for both permutations. Identity's D-std is
exactly 0 by both methods — no noise path to D exists under correct routing, full stop.

**The registered threshold failed at the deployed smoothing level**: propagated D std under the permutations
is 0.075-0.078, below the smallest directive-rung gap (0.18). Pinned by a test specifically so this can't
quietly become a pass later.

**Chased down rather than left as a discrepancy with E3's switching result** (18-328/min, previously measured).
The resolution: propagated std is a *marginal* statistic (how far D typically sits from its mean); switch
*count* is a level-crossing-RATE question, which depends on the smoothing filter's autocorrelation timescale,
not on marginal variance alone. Checked directly: if E3's switching were mostly driven by its schedule's
scripted emotion transitions (real swings up to 1.5 units), a small marginal std would need no explanation. It
is not — 69-83% of simulated switches occur **away from any scripted transition**, confirming genuine
steady-state noise-driven crossing is the real mechanism, which a std-vs-gap comparison simply cannot see.

Verified — `padeval/tests/test_noise_propagation.py` 5/5 (new), 49 padeval tests total.

Report for the draft: `docs/paper/7_0_noise_propagation.md`.

---

## exp(padeval): 6c — tier-offset magnitude sweep, and a rank correction  *(branch `feature/pad-affect-core`)*

STRETCH item, timeboxed. **Registered prediction met**: across offset magnitude `m in [0.10,0.60]` (deployed
`m=0.40`, tier ratios held fixed, passed explicitly — `TIER_OFFSETS`/`affect.py` untouched), `identity` stays
full rank (3/3) and structurally zero-switching at every `m`; `collapse_D` stays rank-deficient and switching
at every `m`; the permuted assignments stay switching at every `m`. `identity`'s condition number improves
monotonically with `m` (11.07 -> 3.40), so the deployed value is not a knife-edge.

**A correction surfaced by sweeping a range rather than one point.** `collapse_D`'s true rank is **1, not the
2 the earlier E2 report gave** (`6b81fc4`). Under that assignment P and Ar are structurally frozen at baseline,
so every style output is a function of the scalar D alone, and every Jacobian row is provably a scalar multiple
of one direction vector — exactly rank 1, not approximately. The earlier "2" was a finite-difference artifact:
`h=1e-6` left a spurious second singular value at machine-noise scale that still cleared the rank tolerance: an
h-scan showed the classic V-shaped noise curve of float-cancellation error, not the flat plateau a real second
dimension would produce, and no single fixed `h` was robust across the swept `m` range either, since the noise
floor scales with the Jacobian's own magnitude. Fixed by reading the rank from the closed-form structural fact
(`assignment.collapse`) rather than trusting numerical SVD rank detection for this case at all. The qualitative
conclusion is unchanged and strengthened: collapse loses two degrees of freedom, not one.

**Clamping tightens exactly on the analytic bound**, not merely correlates with it: `m <= 1-|D_baseline|` gives
CHATBOX `m<=0.357`, already exceeded by the deployed `m=0.40` by precisely the 0.043 the original audit
measured as the `unknown`-tier saturation. Stated as a design rule: a persona nearer the edge of the PAD cube
has less room for relationship-driven displacement before the ladder's bottom rung saturates.

Verified — `padeval/tests/test_tier_sensitivity.py` 7/7 (new), 44 padeval tests total.

Report for the draft: `docs/paper/6c_tier_sensitivity_sweep.md`.

---

## exp(padeval): 6b result — human gold set, kappa 0.861, bias-free  *(branch `feature/pad-affect-core`)*

The number `docs/paper/6b_gold_set_protocol.md` was built to produce. Coded by a human, blind, on all 30
items (1 skip, n=29 scored):

| | value |
|---|---|
| binary agreement | 93.1% |
| **Cohen's kappa** | **0.861**  95% CI **[0.649, 1.000]** |
| McNemar b/c | 1/1, p = 1.0000 |

**Gate passed.** The CI lower bound (0.649) sits just under the 0.70 registered bar — reported honestly rather
than rounded away, since at n=29 that is the number a reviewer checks first.

**The Phase 4b frame-restriction fix holds against a real human**, not only against the earlier AI blind pass:
McNemar is symmetric (1/1), against the 6/0 directional over-fire (p=0.031) measured before that fix.

Both disagreements were read individually rather than aggregated away. One (`"what's on your mind right
now?"`) is a boundary in the *operational definition* — no topic is named, so the coder's "not initiated" is
defensible under the letter of the question being asked, even though the human read the redirect itself as a
form of initiative. The other (`"chatting about space **again**"`) is a precise, nameable lexicon gap: `"again"`
signals a callback that nothing in `MEMORY_RECALL_MARKERS` catches. Not patched here — two items is not a
pattern, they cancel in direction, and tuning a coder to its own 30-item gold set this close to submission
would be circular. The fix is identified for after submission.

This closes 6b. Both blocking subphases (6a, 6b) are now complete and reported.

---

## feat(client-link): the unmodified v4 robot client speaks to the V5 pipeline  *(branch `feature/pad-affect-core`)*

**Goal (user):** the Jetson robot client (`CHATBOX-DEMO_V4/CHATBOX_CLIENT/client.py` on `main`) had no server
left to talk to — V5's work all happened behind the webui. Give it one, without editing the client: only
`server_url` in its `client_config.json` should have to change.

**New `modules/webui/robot_link.py`.** A Socket.IO server run alongside the webui's HTTP server, answering on
exactly the events that client already emits — `client_init` / `chat_message` / `speech` in, and
`client_init_response` / `chat_response` / `speech_response` back. Mic audio arrives as base64 16 kHz mono WAV
and is transcribed **server-side** with faster-whisper, the same placement v4 used in
`Modules/speech_processor.py`.

The decision that mattered: a spoken turn is pushed into **the same queue the web page's text box feeds**,
not a parallel one. So voice and typed turns take an identical path through face-reco → PAD → KG → prompt
builder. A second chat path would have drifted out of step with the first within a commit or two — that is
how the tag-synonym rot below happened.

`enqueue` grew an `on_reply` callback (the web page passes `None`; it just reads state). The reply goes back
in main's wire format — the tag at the **head** of the response text, `[TAG] words`, which is the only place
the v4 client looks: its TTS strips that span before speaking, its Arduino output reads it for the gesture.
The servo STYLE line does **not** go out; this server drives the servos over its own ESP32 socket.

Both optional deps degrade instead of crashing: no `flask-socketio` disables the link, no `faster-whisper`
disables speech and the link still serves typed chat (`--no-stt` forces it). Whisper loads on a background
thread so the webui keeps serving frames through a multi-hundred-MB model load, and reports not-ready rather
than queueing, so a client that speaks too early gets an answer instead of a hang.

**Didn't work / fixed on the way** — four bugs, all found by the two read-only audit passes in
`AUDIT_REPORT.md` / `AUDIT_REPORT_2.md`, and all of them about *who the robot thinks it is talking to*.
They matter more once turns arrive from off-machine, because identity is decided by the **server's** camera,
not by whoever sent the audio.

1. **A stranger inherited the last person's identity.** The 60 s `PID_GRACE` was meant to cover a dropout —
   nobody in frame. It was also firing when a face *was* in frame and simply did not match, which is not a
   dropout: `FaceIdentifier._confirm_identity` has already spent its own miss-grace on the hard-pose case and
   concluded this is someone else. Holding the old id through that handed a stranger the previous person's
   tier, memories and name — literally "Your name is Jay". The grace now covers only the no-face case.

2. **`dets[0]` was not the person being spoken to.** The identifier votes on the largest box; the overlay read
   index 0, which is largest-first only during the hold phase. In a sample window `dets[0]` can be a
   bystander, so the two disagreed about who "the person" was. Both now take the largest box.

3. **An unidentified speaker's history was written and never read back.** Turns were filed under
   `"__unknown__"` but read with `pid or ""` — two different buckets. Every access now goes through one
   `_history_key()`, so the writer and the readers cannot disagree again.

4. **Dead expression tags were being sent to the ESP32, and the robot wore a departed person's mood.**
   `_TAG_SYNONYMS` still mapped `SMILING`→`HAPPY`, `CURIOUS`→`NOD` and friends after those expressions were
   retired from `_TAG_TO_ESP32`; the dead name sailed through the lookup and out to the robot. One
   `_resolve_tag()` now gates both callers on the real map, so a synonym can never outlive its target. The
   families with no honest survivor were **dropped rather than remapped** — forcing `JOY` onto `IDLE` would
   make an enthusiastic line stand perfectly still, which is worse than the second LLM call `score_tag` spends
   choosing from the real list.

   Separately, with nobody identified the PAD pipeline could not run (no person → no interaction record → no
   tier), so `style`/`wire` stayed at their `None` defaults and the servos held whatever mood the last person
   put them in — and `_last_pad_result` kept **their** tier's manner line, which `_build_system_prompt` then
   stamped onto a stranger's prompt (only the WHO block is gated on `pid`). `neutral_pad()` now publishes the
   robot's own resting temperament at a neutral face. It is built on a **throwaway** adapter: the live one
   carries a smoothing stream, and feeding it invented neutral frames every second while the room is empty
   would drag the next real person's first reading toward zero. It depends only on (robot, tier), so it is
   computed once and cached.

**Verified:** 66 + 15 tests still green, nothing regressed. The link itself is not unit-tested — it is I/O
against a client on another machine; it was exercised by hand.

---

## feat(padeval): evaluation harness core + E3 chattering, the first zero-LLM result  *(branch `feature/pad-affect-core`)*

**Goal (user):** build the evaluation harness for the ICRA submission. Plan in
`~/.claude/plans/then-can-we-plan-glowing-beaver.md`; it front-loads the results that need **no LLM calls**,
because those cannot fail for infrastructure reasons and are likely the strongest figures. This commit is
Phase 0 (the permutation core) and Phase 1 (the chattering metric).

**New package `padeval/`**, deliberately outside `modules/`. Dependency direction is one-way — `padeval`
imports `modules.*`, nothing under `modules/` or `PAD_CORE/` imports `padeval`, so the runtime never depends
on the harness. Core modules import **numpy only**; scipy/sklearn/pandas stay behind `padeval.analysis` and a
separate `requirements-eval.txt`.

**`padeval/axes.py` — the permutation object.** `AxisAssignment` parameterises which source writes which PAD
axis, and `compose()` routes baseline + face + relationship accordingly. Four assignments: `identity` (the
deployed system), `perm_emotion_D`, `perm_arousal_D`, `collapse_D`.

`affect.py` is **not modified**. `compose` calls `affect.to_pad` and routes the deltas itself, and the gate is
that under `identity` it reproduces `affect.feel_with_relationship` **bit for bit** — asserted over all 56
reachable cells plus 1000 random points across four empathy values. That mirrors the guarantee
`affect.pipeline` already makes about `tier="known"` (affect.py:365-369) and is what makes this a strict
superset rather than a second implementation of the model. Without it, every E3 number would be a comparison
between the real system and a slightly different one.

One routing decision worth recording: `TIER_OFFSETS` is authored in AXIS space, not source space. The
principal displacement (index 2) follows `tier_axis`; the secondary nudges follow whichever axes valence and
arousal own. So the `close` tier's +0.10 arousal leak stays attached to *arousal* under permutation, which is
its semantics. Under `identity` the routing is the identity map, asserted.

**`padeval/analysis/e3_permute.py` — the chattering metric.** The behavioural directive is a per-SESSION
control. Under the deployed assignment the face cannot write Dominance, so the commanded rung is invariant to
the face; under `perm_emotion_D` a 30 Hz signal drives a per-session effector. Measured by driving a V/A frame
trace through the **real** `AffectStream` (giving the permuted assignment the benefit of the deployed smoother,
so the comparison is fair rather than rhetorical) and counting rung switches.

**Result — and it is categorical, not marginal.** Two rates are reported, because conflating them would
overstate it. The frame rate describes the *signal*; the tick rate is what the controller actually samples
(`_DEFAULT_TICK` = 1.0 s, and the directive only reaches the LLM once per conversational turn, slower still).

| sigma | identity (frame / tick) | perm_emotion_D ELLEBOT (frame / tick) | collapse_D CHATBOX (frame) |
|---|---|---|---|
| 0.00 | **0.0 / 0.0** | 18.8 / — | 11.2 |
| 0.10 | **0.0 / 0.0** | 101.2 / 15.0 | 71.2 |
| 0.20 | **0.0 / 0.0** | 240.0 / 35.6 | 86.2 |
| 0.28 | **0.0 / 0.0** | 328.1 / 41.2 | 93.8 |

`identity` is **exactly zero at every noise level and at both rates**, by construction rather than by tuning —
the face has no path to D at all — while the permutations degrade monotonically with noise.

The frame-level median dwell falls from the full 32 s trace to 0.10 s, and that is the number most open to
being overstated: it is a property of the signal, not a claim that the robot visibly changes behaviour ten
times a second. The honest framing is the tick row. When dwell falls below the sampling interval, the
directive that reaches the model is decided by whichever instant happened to be sampled — it does not chatter
visibly, **it becomes arbitrary**. At sigma = 0.20 the sampled directive still changes ~36x/min, and more than
one rung is reachable within a single fixed relationship tier.

**The noise range is empirically anchored, not chosen.** `traces.measure_model_dispersion` runs the deployed
regression head over the real corpus: pooled within-class dispersion over 320 faces is **sd_v = 0.279,
sd_a = 0.292**. That is an upper bound (it also mixes identity, pose and lighting), so it bounds the sweep
rather than naming an operating point — and sigma = 0.20 sits inside the plausible range.

**Stated before a reviewer says it:** this does **not** show the prompt-to-behaviour map breaks under
permutation. That map still works; it is merely commanded from the wrong source. E1 measures the map, this
measures the signal driving it. Expect E1 fidelity to largely survive permutation — permutation breaks the
*system-level* properties (setpoint stability, and identifiability in E2), which is the more interesting
result anyway.

**Verified — `padeval/tests/` 15/15 (new):** bit-identity on the grid and on 1000 random points; tier routing
is the identity map under `identity`; permutation actually moves the signal (a permutation that changed
nothing would make E3 vacuous); `collapse_D` leaves unwritten axes at baseline; clamp flags fire at
CHATBOX/`unknown` (D wants −1.043) and nowhere at `known`; `rung_of` agrees with `prompt.manner_directive`
across [−1,+1]; zero switches at every sigma under `identity`; monotone worsening under permutation. A
tripwire test also pins the `close`-tier +0.10 arousal leak — kept deliberately so E2 can demonstrate its
leakage metric resolves it, so if anyone zeroes it the E2 sensitivity result must be regenerated rather than
silently becoming a null. All 75 pre-existing tests unaffected.

**Artifacts:** `runs/eval/e3_chatter.{md,jsonl}`, regenerable from one command.

---

## exp(padeval): powered ladder — a BOUNDED null, and the measured ICC  *(branch `feature/padeval-phase3`)*

The run that settles the open half. Persona and tier pinned, memory and capability blocks removed, all seven
rungs, **32 stimuli x 8 replicates = 256 per rung, 1,792 generations, 724 s.** Every reply generated (1792/1792
ok).

| rung | directive | init% | mean ordinal |
|---|---|---|---|
| 0 | open with something you already know | 9.0% | 1.12 |
| 1 | propose the next topic yourself | 9.4% | 1.55 |
| 2 | offer a topic if they do not bring one | 5.5% | 1.16 |
| 3 | either of you may open a topic | 6.2% | 1.12 |
| 4 | follow their topic, one follow-up | 4.3% | 1.77 |
| 5 | ask about what they bring up, do not introduce | 3.9% | 1.22 |
| 6 | answer only what they ask | 7.8% | 1.02 |

**The 9% -> 3% trend from the n=32 probe did not replicate.** At 256 per rung it is 9.0% -> 7.8%, and the
series is not monotone. That earlier hint was noise, which is exactly what the underpowered warning said it
might be.

### Cluster-robust inference — bootstrap over the 32 stimuli, not the 1,792 replies

| statistic | point | 95% CI |
|---|---|---|
| tau(rung, initiated) | −0.0393 | [−0.0884, +0.0068] |
| tau(rung, ordinal) | −0.0137 | [−0.0615, +0.0324] |
| rate difference, rung 0 − rung 6 | +0.0117 | [−0.0391, +0.0586] |

This is a **bounded null, not merely a non-significant one**. The interval on the rung-0 vs rung-6 difference
excludes any effect larger than about **6 percentage points in either direction**. The claim is no longer
"we failed to detect an effect" but "any effect is smaller than 6 points" — a far stronger statement, and the
one worth writing down.

(The naive Fisher p of 0.75 is reported alongside only to show it agrees; it ignores clustering and should not
be the quoted figure.)

### The ICC, measured rather than assumed

| outcome | ICC | design effect (m=8) | effective n | per rung |
|---|---|---|---|---|
| initiated | **0.128** | 1.90 | 944 of 1792 (53%) | 135 of 256 |
| ordinal level | **0.193** | 2.35 | 763 of 1792 (43%) | 109 of 256 |

Lower than the 0.3 the plan assumed, so replicates are worth more than feared — but still nearly half the
nominal sample is lost to clustering. **Any future budget must be quoted in effective n.** At ICC 0.13 with 8
replicates, 256 replies per rung buys 135, so a design needing 245 effective per rung would need ~465 raw, or
more stimuli instead — and adding stimuli is the better buy, since it raises the cluster count rather than the
cluster size.

This closes the ICC item that was outstanding before any generation budget was committed.

### Where the E1 claim now rests

**Settled, both directions:**
1. *Deployed configuration:* the directive does not control topic initiation, and the point estimate runs
   backwards. Adequately powered.
2. *Memory and capabilities removed:* no effect either, and now bounded at <6 points. The previous "open
   question with a costed answer" is closed, and the answer is no.

**The headline is therefore a clean, well-powered negative result:** a behavioural instruction resident in an
LLM system prompt did not measurably control the behaviour it names, across the full seven-rung range of the
instruction, with and without competing prompt content, at 1,792 + 448 + 512 + 256 generations.

That is a genuine contribution about prompt-based behavioural control, and it is more useful stated plainly
than dressed up. It does **not** touch the architecture's other two claims: E3's setpoint stability and E2's
rank/leakage results are structural properties of the equations and stand independently of what any LLM does
with a prompt.

---

## exp(padeval): the full-rung ladder — no directive effect, and the honest power bound  *(branch `feature/padeval-phase3`)*

The decisive controllability probe. Persona and tier **pinned** (CHATBOX, `known`), Dominance forced to the
midpoint of each of the seven directive bands, descriptor words deliberately left **unchanged** so the only
thing varying is the directive clause. Every midpoint asserted against `prompt.manner_directive` before use.
This removes the persona/rung confound (plan R1) and isolates the directive channel. 448 generations, 363 s.

| rung | D | directive | init% (deployed) | init% (distractions removed) |
|---|---|---|---|---|
| 0 | +0.81 | open with something you already know | 66% | 9% |
| 1 | +0.47 | propose the next topic yourself | 62% | 16% |
| 2 | +0.22 | offer a topic if they do not bring one | 56% | 6% |
| 3 | −0.16 | either of you may open a topic | 66% | 9% |
| 4 | −0.59 | follow their topic, one follow-up | 66% | 9% |
| 5 | −0.83 | ask about what they bring up, do not introduce | 75% | 3% |
| 6 | −0.96 | answer only what they ask | 75% | 3% |

**Deployed prompt: no effect, and the sign is backwards.** tau = +0.083 (p = 0.16) on the binary; rung 6
("answer only what they ask") initiates *more often* than rung 0 ("open with something you already know") —
75% vs 66%, Fisher p = 0.59. At a ~70% base rate with n = 32/rung the minimum detectable difference is ~26
points, so this is not a precision problem: an effect large enough to matter would have shown.

**Distractions removed: the right direction, but genuinely underpowered.** tau = −0.093 (p = 0.11), rung 0 at
9% falling to 3% at rung 6. That is the ordering the ladder predicts, and rungs 5/6 are the lowest — but with
a 3-16% base rate, n = 32 gives 1-5 events per cell and a minimum detectable difference of ~23 points. **This
arm cannot resolve the effect it is hinting at, and reporting it as a null would be wrong.**

Power for the rung-0 vs rung-6 contrast at alpha .05, power .80:

| arm | observed | n needed per rung | total generations |
|---|---|---|---|
| deployed | 66% vs 75% | 402 | ~2,800 (~70 min) |
| distractions removed | 9% vs 3% | **245** | **~1,715 (~30 min)** |

**Where this leaves the claim.** Two separate statements, and they must not be merged:

1. *In the deployed configuration the behavioural directive does not control topic initiation.* Adequately
   powered for any effect worth claiming, and the point estimate runs the wrong way. This is a real finding.
2. *With the memory and capability blocks removed, a small effect in the predicted direction may exist and
   this experiment cannot resolve it.* Underpowered by roughly 8x. Not a null — an open question with a
   costed answer.

The honest headline is (1): **a prompt-resident behavioural instruction is overridden by the same prompt's
memory block**, which is a result about prompt-based control rather than about PAD. The architecture's other
two claims are untouched by it — E3's setpoint stability and E2's rank result are structural and stand.

**Next, if pursued:** the ~30-minute powered run on the stripped arm settles (2) properly. Nothing else should
be written about controllability until it does.

---

## fix(padeval): cry-wolf coder fixed; the distractor is MEMORY, not capabilities; no tier gradient  *(branch `feature/padeval-phase3`)*

### Coder fix — the directional bias is gone

The open-noun detector now fires only on nouns inside a **topic-introducing frame** ("talk about X", "did you
know X", "how about X"). Introducing a topic is a framing act; a bare noun inside an answer is not an
introduction however novel it is. That is a fix to the concept, not a patch on the symptom — the previous
version had nothing to constrain it on the 20 of 32 stimuli that declare no topics, so `laugh`, `kitty`, `guy`
and `alright` all read as new subjects.

Against the same 60 blinded items (coded by `claude_ai_v1` — an AI adjudicator, **not** a human gold set):

| | before | after frames | + lexicon gap closed |
|---|---|---|---|
| binary agreement | 90% | 93% | **97%** |
| kappa (binary) | 0.74 [0.52, 0.91] | 0.85 [0.68, 0.96] | **0.92 [0.79, 1.00]** |
| kappa_w (ordinal) | 0.65 | 0.82 | **0.84 [0.62, 0.96]** |
| McNemar b/c, p | **6/0, p=0.031** | 1/3, p=0.625 | **1/1, p=1.000** |

The bias is symmetric now. The lexicon gap was `constellation`/`spaceship`/`universe` and similar missing from
the `space` topic — plainly space words, added on their face rather than tuned to a result. 48/48 authored
cases still pass.

### Finding — the competing block is the seeded MEMORY, not the capability list

512 new generations across two suppression arms, same grid as the baseline:

| arm | mean initiation |
|---|---|
| `A1_full` (as deployed) | **70%** |
| `A7_no_caps` (capability list removed) | **66%** |
| `A9_no_caps_no_mem` (also no seeded interests) | **10%** |

Removing the "You can: tells stories, knows jazz, knows about space..." list **barely moves anything** — 70%
to 66%. Removing the seeded person memory as well collapses initiation to 10%. So the earlier diagnosis was
wrong in its target: plan risk R2 named the capability list, but `space` and `guitar` reach the robot through
the MEMORY block, and that is what overrides the directive.

### Finding — no tier gradient survives in ANY arm

Kendall's tau between tier rank and behaviour, per robot:

| arm | robot | binary tau | ordinal tau | p (ordinal) |
|---|---|---|---|---|
| A1_full | chatbox | +0.040 | −0.002 | 0.975 |
| A1_full | ellebot | +0.044 | +0.057 | 0.451 |
| A7_no_caps | chatbox | +0.013 | +0.003 | 0.968 |
| A7_no_caps | ellebot | +0.014 | +0.054 | 0.474 |
| A9_no_caps_no_mem | chatbox | −0.016 | +0.141 | 0.065 |
| A9_no_caps_no_mem | ellebot | +0.070 | +0.076 | 0.320 |

Nothing reaches significance, and nothing is monotone. Clearing the floor from 70% to 10% did **not** reveal a
hidden gradient — so the null is not simply a ceiling artefact.

**The honest statement, scoped:** *at the four reachable tiers*, the behavioural directive does not measurably
control topic initiation, on either the binary or the ordinal, with or without competing prompt blocks.

**What is NOT yet tested, and is the decisive remaining probe:** the reachable tiers cover only 4 of the 7
rungs per robot, and rung is confounded with persona (plan R1 — only rung 3 is shared). The `A1_full_ladder`
arm, which pins one persona and forces Dominance across all seven rungs, decouples the two and gives the
prompt-to-behaviour map its strongest test. ~224 generations, about four minutes. That should run before any
claim about controllability is written down in either direction.

---

## feat(padeval): Phase 4b — gold pool, agreement maths, and two blocking findings  *(branch `feature/padeval-phase3`)*

**Built:** `padeval/stimuli.py` (32 authored utterances, 4 strata x 8, each raising at most one lexicon topic
and none touching the seeded memory), `padeval/fixtures.py` (`TIER_RECIPE` + `seed_person` moved out of
`tools/pad_prompt_grid.py`, plus `build_world`), `padeval/coding/agreement.py` (Cohen's kappa with bootstrap
CI, quadratic-weighted kappa for the ordinal, exact McNemar, per-stratum report — numpy only, checked against
a worked example).

**Pool generated:** 256 replies = 2 robots x 4 tiers x 32 stimuli, real prompts through the deployed
`_build_system_prompt`, seeded, 252 s. `runs/eval/gold_pool.jsonl`.

### A note on who coded what

The blind sample was coded by **Claude, labelled `claude_ai_v1`** — an LLM, not a human. Reported as an AI
adjudicator and nothing else. It cannot stand in for `kappa(rule, human)`: an LLM's errors correlate with the
generator's in exactly the way a gold set exists to rule out, and a methods section claiming human coding when
an LLM did it would be false about provenance. **The human gold set is still required before any campaign.**
The AI pass is useful for what it did here — it caught a coder bug early, for the cost of 60 items.

### Finding 1 (blocking) — the rule coder over-fires, and directionally

60 blinded items, stratified: binary agreement 90%, `kappa = 0.74 [0.52, 0.91]`, ordinal
`kappa_w = 0.65 [0.38, 0.85]`. Kappa clears the 0.70 bar — but **McNemar is 6/0, p = 0.031**. Every single
disagreement runs the same way: the rule coder says "initiated" where the AI coder says it did not. A
systematic bias in a rate is worse than symmetric noise, because it inflates the outcome everywhere at once.

Root cause found: **20 of the 32 stimuli declare no topics**, and the elaboration guard added in Phase 4a only
suppresses the open-noun detector when the reply mentions a *stimulus* topic. On a stimulus with no declared
topics nothing suppresses it, so any noun in the reply counts as a new topic — `laugh`, `guy`, `kitty`,
`alright`, `course`. Four of the six over-fires are exactly this case.

The fix is not more stoplist entries; that is whack-a-mole. The open detector should fire only on nouns inside
a **topic-introducing frame** ("talk about X", "tell you about X", "did you know X"), which is what
introducing a topic actually looks like. Deferred rather than rushed, because it changes the primary outcome
measure and wants its own authored cases.

### Finding 2 (blocking, and bigger) — the directive is being swamped

Initiation rate by tier, rule coder, over the whole 256-reply pool:

| robot | unknown | visitor | known | close |
|---|---|---|---|---|
| chatbox | 72% | 69% | 75% | 69% |
| ellebot | 75% | 75% | 84% | 81% |

**No ordering by tier, and the floor is ~70%.** At `unknown` the directive reads *"answer only what they ask,
and do not introduce a topic of your own"* — and the robot proposes space anyway, in roughly three replies out
of four. Inspection confirms these are genuine initiations, not coder artefacts: *"mm."* -> *"Did you know
space is filled with billions of stars?"*

This is plan risk R2 confirmed and quantified. The IDENTITY block's capability list and the seeded person
memory are a standing invitation, identical in every condition, and they dominate the directive. Two
consequences: the `A7_no_capabilities` arm is **essential, not optional**, and a memory-suppressed arm is
probably needed too, since `space` and `guitar` come from the seed rather than the capability list.

The stimulus stratum also dwarfs the tier — disclosure 38% vs quiet 98% vs closing 95% — so the stratum has to
enter the model as a factor, not be averaged over.

Also observed: **ordinal levels 1 and 5 never occur** in 256 replies. Level 5 is rung 0's own wording ("open
with something you already know about them"), commandable only by ELLEBOT at `close`, and it never appears —
the top of the ladder does not produce its distinctive behaviour.

### What this means for the campaign

Neither finding is a reason to abandon E1, but both must be resolved before spending GPU hours: a directional
coder bias would inflate every cell, and a 70% floor with no tier ordering would produce a null that says more
about the prompt's other blocks than about the directive. The floor is itself a publishable result — it is
just not the result E1 was designed to measure.

---

## feat(padeval): Phase 4a — the rule coder, 100% on 48 authored cases  *(branch `feature/padeval-phase3`)*

**Goal (user):** build E1's primary outcome coder. Rule-based and PRIMARY on purpose: it is deterministic and
fixed before the data exists, so it cannot be tuned after seeing a result — an LLM judge can be re-prompted
until the numbers improve and nobody can prove it wasn't.

**Two corrections to the plan's assumptions, found by checking rather than trusting.**

*The "free labelled data" is 36 replies, not 250.* The plan claimed ~8,700 qwen replies were already on disk;
that counted JSONL **lines** (each carrying a ~2,000-char prompt), not replies. Actual inventory:
`A_adj.jsonl` 18 + `B_dir.jsonl` 18 = **36 distinct replies**, and the other two run files were `--no-llm` with
empty replies. So the 250-reply gold set cannot come from existing artifacts; it needs a fresh unlabelled pool,
which is cheap (~250 seeded generations) and does not require the full campaign.

*NLTK's tagger was missing.* `averaged_perceptron_tagger_eng` (the name newer NLTK requires) was absent, so
`pos_tag` raised. Downloaded, and `nltk==3.9.4` is now pinned in a new `requirements-eval.txt` — a tagger
revision would silently change coded outcomes, which is not something to leave floating.

**Design — why a rule coder is tractable here.** The experiment controls the topic universe on all three
sides: person memory is seeded identically in every cell (guitar/music, space/science), robot capabilities are
authored in the spec YAMLs, and stimulus topics are authored by us. So the coder matches against sets we wrote
down rather than guessing at open-domain semantics. Two detectors OR'd: a closed-lexicon match (20 topics, 121
surface forms) and an open-noun detector for the residual case. Morphology is a hand-written irregular map plus
an explicit suffix rule, **not a stemmer** — Porter folds `space` and `spacing`, and "we used the Porter
stemmer" is not something a reviewer can audit.

**Binary AND ordinal.** The binary saturates: rungs 4/5/6 all predict "no new topic" and 0/1/2 all predict
"new topic", so it resolves at most three of seven commanded levels and would understate the controller. The
0-5 ordinal (answers-only / asks-back / follow-up+expansion / hedged offer / asserts / opens-with-a-remembered-
fact) maps monotonically onto the ladder and is what Kendall's tau gets computed on. Level 5 is exactly rung
0's wording, reachable only by ELLEBOT at `close`.

**Four real bugs found by the authored cases, which is what they are for.**

1. *`Was` tagged NNP.* Sentence-initial capitals fooled the tagger, so `"How did it go? Was it hard?"` coded as
   introducing a topic — a false positive on precisely the ask-back behaviour the suppressing rungs produce,
   which would have destroyed the effect. Fixed by tagging lower-cased tokens; a genuine proper-noun topic
   still tags NN (`tim` -> NN), so nothing is lost.
2. *Verbs mis-tagged as nouns.* `want`, `let`, `cover` and friends fired the open detector. Added a `VERB_LIKE`
   guard, kept separate from `NON_TOPIC_NOUNS` so the two reasons for exclusion stay legible.
3. *`"you like"` collided with the hedge `"if you like"`,* so *"We could talk about space if you like"* coded
   as a remembered fact (level 5) instead of a hedged offer (level 3). Removed `you like`/`you love` from the
   recall markers — a recall marker has to be unambiguous about who said it.
4. *The open detector fired on elaboration.* `"Did the maths test cover fractions?"` counted as initiating
   because `fractions` is not in the lexicon. Now suppressed when the reply still mentions a stimulus topic:
   a specific noun inside their topic is elaboration, not initiation. **The cost is a documented false
   negative** — a reply that stays on their topic and bolts on an off-lexicon new one reads as not-initiated —
   accepted because the stimuli raise at most one lexicon topic and the robot's plausible new topics are its
   capability list, which IS in the lexicon. Novel lexicon topics are never suppressed this way.

One "miss" was **not** a bug: `"Space is wonderful, isn't it?"` was coded level 1 against an expected 0. A tag
question does ask back, so the coder was right and the authored expectation was wrong. Corrected in the test
rather than worked around in the coder.

**Verified — `padeval/tests/test_rule_coder.py`, 48/48 (100%), 37 padeval tests total.** The cases are written
to be hard rather than flattering: follow-up questions about their topic, topics the stimulus already raised,
negated mentions, the hedge/assert boundary, the recall/pitch boundary, backchannels and empty input. Also
pinned: binary and ordinal stay consistent (`initiated` iff `level >= 3`), the lexicon has no surface-form
collisions, negation scope stops at a clause break rather than swallowing the sentence, and the coder is
deterministic over repeats.

**Still open before the gold set:** kappa against human-coded replies must clear 0.70 before any GPU hours,
and the pool has to be generated first (see the 36-vs-250 correction above).

---

## feat(padeval): Phase 3 — seeded, paired generation, and bit-reproducibility measured rather than assumed  *(branch `feature/padeval-phase3`)*

**Worktree.** Built in `../chatbox-phase3` on its own branch. A second session had committed to
`webcam_loop.py` (`dc87d93`) mid-flight, and Phase 3 is the first phase whose measured artifact depends on
that exact file, so the campaign runs isolated and merges deliberately.

`dc87d93` was diffed before proceeding: five hunks, all outside the harness's contract surface —
`_build_system_prompt`, `LLMClient`, `respond`, `manner_directive`, the pad-dict reads, `_adapter` and
`gesture_style` had **0 changed lines** each. The one overlap is `_parse_llm_response` routing through the new
`_resolve_tag` with pruned synonyms: the recorded `tag` changes, but `_TAG_ANY.sub` strips every bracketed
span regardless, so the reply text E1 codes is byte-identical. Consequence to note: the `tag` column in
pre-existing `runs/*.jsonl` predates that change and is not comparable across it.

**`LLMClient.respond` gained optional `temperature` / `seed`,** applied after the built-in defaults so they
win only when passed, and omitted from the request entirely when `None`. Wrapped rather than subclassed in
`padeval/llm.py`: a subclass would have restated the stop-string list and `_clean_reply` and drifted, which
would mean measuring something other than the deployed path. Behaviour preservation is asserted with a stub
client capturing request kwargs — no Ollama needed, so that half runs in CI.

**Pairing (requested).** `pair_key` deliberately excludes the assignment and the seed derives from it, so the
`identity` and `perm_emotion_D` trials for the same (robot, tier, emotion, stimulus, arm, replicate) draw the
**same seed** and differ only by the manipulation. E1 is therefore a paired comparison by construction rather
than two runs argued to be comparable afterwards. Run sequentially instead and every difference is confounded
with whatever drifted between batches; not recoverable later, because the seeds would already be wrong.

**Finding 1 — bit-reproducibility is NOT available on this backend, and the approved gate is withdrawn.**

| | byte-level replication, 10 interleaved prompts |
|---|---|
| temperature 0.7, fixed seeds | **80%** |
| temperature 0.0, fixed seeds | **80%** |

Same seed repeated back-to-back on one prompt reproduces perfectly (10/10 identical). It is **interleaving**
distinct prompts that breaks it, and greedy decoding does not rescue it — identical 80% — so the residue is
backend batching / KV-cache state, not the sampler. Warming every distinct prompt first does not close it
either. A `warm_up` bug was found and fixed on the way (it keyed on the system prompt alone, but the prefix
cache is over the whole token sequence, so warming a batch warmed only its first member).

The Phase 3 gate of "100% byte-identical" would either fail forever or force a false claim into a methods
section, so it is replaced by: report the byte-level rate, and assert only a floor (>= 0.5) as a smoke test
that the seed reaches the backend at all — a broken seed would sit near chance, not near 80%.

**The reproducibility claim the paper should make is OUTCOME-level.** Every observed mismatch is a
near-paraphrase — *"learn new things"* vs *"learn lots of new things"* — which no topic-initiation coder would
separate. `outcome_replication()` measures agreement on the coded label rather than the raw text and is the
figure to publish; it runs in Phase 4 once a coder exists.

**Finding 2 — reply diversity is a property of the PROMPT, and it is not uniform.**

| prompt | distinct replies across 5 seeds | modal share |
|---|---|---|
| tag-constrained + `"mm."` | 2/5 | 80% |
| tag-constrained + open question | 5/5 | 20% |

A low-entropy stimulus collapses toward one continuation, so replicates on that cell buy almost no
information. The approved power calculation assumed 32 stimuli x 3 replicates with a uniform ICC; that
overstates the effective n for low-entropy cells. `sample_diversity()` exists to screen the stimulus set
before the main run so the distribution can be reported rather than assumed. **This needs a design decision
before Phase 6** — see the open question below.

**Verified — `padeval/tests/test_llm_determinism.py` 10 checks (7 offline, 3 live), 32 padeval tests total,
43 pre-existing green.** Offline half pins that spoken-reply and json_mode requests are byte-identical to
before, that overrides win when passed, that history expansion is unchanged, that paired assignments share a
seed while keeping distinct trial keys, and that the seed responds to every design field and is
order-independent.

**Open for the next session:** with replicates buying little on low-entropy cells and bit-reproducibility
capped at ~80% regardless, the stimulus/replicate split needs revisiting — most likely more stimuli and fewer
replicates, since generalisation rests on `n_stimuli` anyway.

---

## feat(padeval): E2 analytic Jacobian — a calibrated leakage metric, and collapse proved degenerate  *(branch `feature/pad-affect-core`)*

**Goal (user):** Phase 2 of the harness, run as a PAIR — once against the deployed system with the
`close`-tier +0.10 arousal term present, once with that single term zeroed. A lone leakage number would have
to be taken on faith; a pair shows the metric responds to a known manipulation.

**Why this is algebra and not a classifier.** The style vector is a closed-form function of the felt
coordinate, so a classifier on noiseless style data scores 100% and demonstrates nothing (R10 in the plan).
The real question is whether the forward map preserves the distinction between sources, which is a question
about a Jacobian and is answerable exactly:

    J = d[amplitude, tempo, posture, droop, idle] / d[valence, arousal, tier]      (5x3)

The relationship is a discrete factor, so a derivative w.r.t. it is undefined as stated. It is made continuous
by ramping the tier offset from the `known` reference along s in [0,1] — the path the system actually
traverses as a relationship deepens. `axes.compose_offsets` takes the offset triple explicitly, so the
ablation is a caller's argument rather than a monkey-patch of the shipped table.

**Result 1 — `collapse_D` is provably unidentifiable.**

| assignment | rank | cond | sigma_min |
|---|---|---|---|
| identity | 3/3 | 3.5 | 2.44e-01 |
| perm_emotion_D | 3/3 | 2.3 | 2.57e-01 |
| perm_arousal_D | 3/3 | 5.1 | 1.70e-01 |
| **collapse_D** | **2/3** | **inf** | **7.75e-18** |

Rank-deficient means two sources move the effectors along the same direction, so no observer — however
clever, however noise-free — can attribute a change to one of them. Unidentifiability *proved*, with no
dataset, no noise model and no train/test split.

**Stated plainly because it would be easy to overclaim:** the condition number does **not** favour the
deployed assignment. `perm_emotion_D` is better conditioned (2.3) than `identity` (3.5). Conditioning
separates the degenerate assignment from the non-degenerate ones; it does not rank the non-degenerate ones
against each other. What distinguishes `identity` from the permutations is the timescale argument (E3) and
the noise-threshold sweep in E2's empirical half — not this table. The report says so in the artifact itself.

**Result 2 — the leakage metric is calibrated.** `idle = 0.45 + 0.4·Ar` is a pure function of arousal
(affect.py:355), asserted against the real `gesture_style` rather than trusted from a comment, so any
tier→idle sensitivity *is* arousal leakage with no attribution ambiguity:

| | d(Ar)/d(tier), leak present | leak zeroed | recovered |
|---|---|---|---|
| identity | **+0.1000** | **+0.0000** | **0.1000** |

Exactly 0.10 in, exactly 0.10 out, to 1e-6. And with the leak removed the source→axis map is **exactly
diagonal** — the disjoint-axis claim as an equation rather than a sentence — while with it present there is
**exactly one** off-diagonal term (Ar ← tier, 0.10). One named, located violation, not a diffuse smear.

Two rows read `n/a` and that is correct, not a failure: under `perm_arousal_D` the tier's *principal*
displacement is routed to arousal, so d(Ar)/d(tier) = 0.40 by design and the 0.10 secondary term is not
separable from it; under `collapse_D` nothing writes arousal at all.

**Verified — `padeval/tests/test_e2_jacobian.py` 6/6 (new), 22 padeval tests total.** Pins both structural
facts the attribution rests on (droop pure pleasure, idle pure arousal) against the real `gesture_style`;
rank deficiency for collapse and full rank for the rest, on both robots; recovery of the injected 0.10;
diagonality with the leak removed; exactly one off-diagonal term with it present; and Jacobian stability
across step sizes 1e-5..1e-7, since central differences on a clamped piecewise function can be fragile.

**Also in this commit — two E3 closeouts.**

*`collapse_D` at sigma = 0.28 was never missing.* The sweep ran all 4 assignments x 2 robots x 2 tiers x 5
sigmas = 80 rows; the em-dash was a transcription gap in a hand-written summary table in the previous
PROGRESS entry, now corrected. The value is 93.8/min (CHATBOX/known) rising to 172.5 (ELLEBOT/close).

*Frame rate vs tick rate.* The frame-level "0.10 s median dwell" describes the SIGNAL and could be misread as
the robot visibly twitching ten times a second. `directive_volatility` now also decimates to the controller's
actual sampling rate (`tick_hz`, default 1.0 Hz = `_DEFAULT_TICK`; the directive only reaches the LLM once per
conversational turn, slower still) and reports both:

| sigma | identity frame/tick | perm_emotion_D ELLEBOT frame/tick |
|---|---|---|
| 0.10 | 0.0 / 0.0 | 101.2 / 15.0 |
| 0.20 | 0.0 / 0.0 | 240.0 / 35.6 |
| 0.28 | 0.0 / 0.0 | 328.1 / 41.2 |

The honest framing is the tick column, and it survives: identity is zero at **both** rates while the
permutation still moves ~36x/min at 1 Hz with more than one rung reachable inside a single fixed tier. When
dwell falls below the sampling interval the directive is decided by whichever instant was sampled — it does
not chatter visibly, **it becomes arbitrary**. A new test pins the claim at the decimated rate so it cannot
survive only as a frame-rate artefact.

---

## fix(closeness): rapport and trust are two axes, not one written twice  *(branch `feature/pad-affect-core`)*

**Goal (user):** the tier reads `score = (rapport + trust) / 2`, but both live write paths called
`_update_rapport_trust(delta=…)` with a **single** delta applied to both fields. In live operation the two were
therefore numerically identical, the average collapsed to `score = rapport`, and trust was a decorative second
copy. Only the once-per-session extractor ever separated them.

**The intended distinction was already written down** — in the extractor's own brief (`extraction.py`):
*"rapport_delta rises with warmth and positive affect; trust_delta rises with the child sharing personal
things."* The per-tick path violated it: its delta is `0.025 × felt P`, i.e. warmth read off a face, and it was
being credited to trust as well. No single frame can show what a child chose to disclose.

**Fix.** `_update_rapport_trust` now takes independent `d_rapport` / `d_trust`, so every call site must state
which axis it moves:

| path | before | after |
|---|---|---|
| live tick (1 Hz, gated on felt P > 0.05) | `+0.025·P` to **both** | `+0.025·P` to **rapport only** |
| end-of-session extraction | independent ±0.2 | unchanged |
| operator B key | `+0.15` to both | unchanged — an explicit override, not an inference |

**Consequence, and it is the point.** `close` needs `score > 0.70`. With rapport alone that is unreachable: an
unbroken hour of smiling saturates rapport at 1.0 and still scores 0.50 → `known`. Trust must come from the
extractor at ±0.2 per session, so **three sessions is the floor**, and only at maximum judged disclosure each
time. Measured with the real code:

| | before (Δ→both) | after (Δ→rapport) |
|---|---|---|
| happy face → close | **48 s** | not within a session; 3 sessions minimum |
| neutral face → close | 264 s | not within a session |
| → visitor / → known | 1 s / 6 s | unchanged (both ride `interaction_count`) |

This is what the design always claimed. `RELATIONSHIP_TO_DOMINANCE.md` §6 carried the caveat that "a
relationship the design describes as *slow-moving* is in practice the fastest-changing signal in the system";
it is now the slowest. Demos needing a specific rung on demand already have `tier_override`.

**Considered and rejected:** (a) retuning the `close` threshold so it stays reachable in one sitting — buys a
demo convenience at the cost of another by-eye constant to defend, and re-breaks the slow-signal claim;
(b) collapsing the two fields into a single `closeness` scalar — honest, and behaviourally identical today, but
it discards self-disclosure as a separate dimension, which for a child-companion system is arguably the more
interesting of the two.

**Verified — `test_closeness.py` 6/6 (new):** this rule previously had **no coverage at all**. Tests pin
independent movement, the [0,1] clamp, an hour of smiling stopping at `known` with trust still 0.0, `close`
arriving on the third +0.2 trust increment, five B presses still reaching `close` for testing, and a direct
regression guard asserting `score` has not collapsed back onto rapport. Full sweep green: graph suite 51,
`test_pad_prompt` 10, `test_emotion_va` 8, `test_affinity` 5, `test_first_impression` 6, `pad_core` and
`servo_style` unchanged.

**Docs corrected:** `RELATIONSHIP_TO_DOMINANCE.md` §3 (the "to both" row and the lockstep consequence), §6 (the
time-to-tier table, whose 48 s / 264 s figures are now wrong), and §8 (findings — two entries move to *fixed*,
and the `show = 0.30` entry is marked *stale* since `show` no longer feeds the descriptor words). The
`webcam_loop` header's tier-progression block was also wrong independently of this change: it advertised
"~34 s" where the code gave 48 s.

---

## fix(kg): a mood is FAST, so only the session that wrote it may blend it  *(branch `feature/pad-affect-core`)*

**Goal (user):** close the cross-session mood leak found by the audit. `pre_turn` blends the graph's stored
`MoodEdge` into camera valence at weight 0.3. That edge is declared FAST — `schema.Timescale.FAST` says mood
and attention "decay within a session", and `MoodEdge` itself says it "decays between sessions" — but **no
decay was ever implemented**, and the graph is persisted whole with no timescale filter on either `save`
(`store.py:313-314`) or `load` (`store.py:330-336`, which writes `self._edges` directly and would bypass a
filter in `upsert_edge` anyway).

**What that actually did,** measured on the repo's own `kg_state.json`: it holds a `MoodEdge` for `jay` of
−0.0545 written 2026-08-19. Loading that graph and calling `pre_turn("jay","chatbox","happy")` on a fresh
`KGBridge` returned valence **0.5436** instead of the camera's **0.80** — a 12-day-old mood steering turn 1 of
a new process, with nothing able to decay it. Beyond being wrong, it silently contaminates the first turns of
any experiment: every A/B run would have inherited state from whenever the system last ran.

**Fix — session-scope the read, not the write.** `KGBridge.__init__` records `_session_start`; `pre_turn`
ignores mood edges written before it. Deliberately a session boundary and not a decay curve:

- It is the semantics the schema already documents, so it makes the code match the comment rather than
  inventing new behaviour.
- It introduces **no tuned time constant**. The model already carries more by-eye constants than it can
  defend, and a half-life here would be one more with no evidence behind it.
- Within a session the question does not arise: the loop rewrites mood every tick (1 Hz), so a live mood is
  never stale.

**Why not stop persisting FAST edges instead:** the per-tick save exists so the live viz server can poll
`kg_state.json` within ~1 s (`webcam_loop.py:2306-2313`). Dropping mood at save would fix the leak and break
the visualiser. Gating the read fixes the leak and leaves the viz untouched.

**Deliberately unchanged:** SLOW state still persists in full. After the fix the same `jay` graph still derives
`tier == "close"` from stored rapport/trust — long-term relationship memory is exactly as it was, which is the
point of the FAST/SLOW split. `AttentionEdge` is also FAST but is written and never read back into the
pipeline, so it needs no gate; noted in the code rather than pre-emptively handled.

**Verified — `tests_kg_bridge.py` 19/19 (one new):** `test_A_stale_mood_from_a_previous_session_is_ignored`
pins both halves — a mood timestamped 2020 is ignored on turn 1 (raw camera value passes through), and a mood
written by the current session still blends to the documented 0.24, so the gate scopes the feature rather than
disabling it. It also asserts `tier` is untouched. Naive timestamps are read as UTC rather than raising, since
crashing a turn over a mood value is the worse failure. Full graph suite 51 passed; `test_pad_prompt.py`,
`test_emotion_va.py` and `pad_core` all green.

**Closes** `AUDIT_REPORT_2.md` OPEN #4 and #5 (outcome was `PERSISTS-UNDECAYED`).

---

## feat(emotion): the camera regresses V/A instead of manufacturing it from a lookup  *(branch `feature/pad-affect-core`)*

**Goal (user):** the emotion axis of the PAD model was supposed to be a dimensional signal. An audit of the
live path found it was not: the loaded model was `enet_b0_8_best_vgaf`, an 8-class classifier, and valence /
arousal were reconstructed downstream as a softmax-weighted average over seven hardcoded points in
`emotion_detector._VA_TABLE`. A categorical model wearing a dimensional coat.

**Why it mattered, not just tidiness.** Every blended output is a convex combination of those seven points, so
it can only land inside their hull. Consequence, measured: four of the ten P/Ar descriptor band words in
`affect.BANDS` were **unreachable at any input** — `cold` and `languid` for both robots, `placid` for ELLEBOT.
`CONCEPT.md` §4 already *claimed* the camera used `enet_b0_8_va_mtl` and that "no lookup table is needed"; that
was false when written. The code has now caught up to the documentation rather than the reverse.

**The blocker, and how it was resolved without guessing.** The library returns a bare 10-vector and documents
nothing about the last two elements — not in `hsemotion_onnx`, not in the maintainer's PyTorch source, not in
either README. The ONNX graph is one fused `Gemm` emitting a single tensor named `output`, no per-head names,
no metadata. Ordering was settled from the author's own training notebook for this exact checkpoint
(`face-emotion-recognition`, `training_and_examples/affectnet/train_emotions-pytorch.ipynb`):

```python
loss_valence = self.loss_valence(preds[:, num_classes],   target[1])   # index 8
loss_arousal = self.loss_arousal(preds[:, num_classes+1], target[2])   # index 9
```

with `num_classes = 8`. Targets are raw AffectNet annotations on `[-1, +1]`, unnormalised, fitted with a
Concordance-Correlation-Coefficient loss — so the head is already in PAD's units and needs no rescaling. An
independent check agreed: correlating each output against class probabilities, anger (the one emotion whose
valence and arousal have opposite signs) came out −0.51 on `[-2]` and +0.55 on `[-1]`.

**Tried and rejected:** a live webcam probe. `/dev/video0` is a wide room-view camera, not a user-facing
webcam, so a frontal cascade finds nothing; the probe blocked in its positioning loop. Also unnecessary once
the training source settled the ordering. Synthetic faces were built and discarded — anger separated correctly
but Fear and Surprise contradicted, and both outputs stayed in a narrow positive band, because the model was
never trained on schematic drawings.

**Measured over 480 real faces** (60/class, `Affectnet-HQ + RAF-DB`, folder names as truth):

| | categorical acc | valence range | arousal range | 20×20 grid cells |
|---|---|---|---|---|
| `best_vgaf` + lookup | 67.5% | [−0.698, +0.800] | [−0.390, +0.799] | 93 / 400 |
| `va_mtl` regressed | **68.8%** | [−1.063, +0.985] | [−0.649, +1.484] | **204 / 400** |

The blend's observed range lands within a rounding error of the table's hull, confirming the analysis on real
data. Coverage of the V/A plane rises 2.2×; distinct PAD descriptor triplets over that face distribution roughly
double (CHATBOX 33 → 63 across all tiers, ELLEBOT 35 → 50). Classification does **not** regress, so the
regression head is not bought at the price of the label the KG and prompt paths still use.

**Caveats, stated rather than buried:** the dataset carries categorical labels only — no V/A ground truth — so
these are *capacity* measurements, not accuracy ones; no claim is made that the regression is more *correct*,
only that it spans a space the lookup structurally cannot. Its shipped `labels.csv` also disagrees with the
folder names on 37% of rows, so 67.5 / 68.8 are a relative comparison, not absolute benchmarks.

**Fixed along the way:**
- A latent `KeyError`. `_infer` looped `for i, score in enumerate(scores)` and indexed `self._labels[i]`; on a
  10-element MTL vector that raises at index 8. Now slices `scores[:n]`, which also stops a V/A value being
  read as class confidence.
- V/A is clamped to `[-1, +1]` at the detector boundary. The CCC loss does not bound the head — real faces
  produced arousal `+1.484` and valence `−1.063` — and everything downstream (`affect.feel`, then the published
  Eq. 9 / Eq. A1 mapping) assumes a coordinate in that interval.
- A comment claiming `_VA_TABLE` "mirrors `kg_bridge._EMOTION_VA`" was false: they disagree on six of seven
  shared labels, and so does `affect.CATEGORY_VA`. Left un-retuned deliberately (the blended path's published
  numbers were taken with these values) but the divergence is now recorded and pinned by a test.

**Verified — `test_emotion_va.py` 7/7 (new):** the V/A path had **no automated coverage at all**. A comment in
`kg_bridge` claimed `test_pad_affect` guarded the tables; that file was deleted in `324f569` and the claim went
stale without anything failing. New tests pin the default backend, the clamp, both paths staying inside the PAD
domain, confidence excluding the V/A tail, the documented head indices, and the three-table divergence.
`test_pad_core.py` 7/7 and `test_affect.py` unchanged and green.

**Kept for comparison:** the old path is selectable as `--emotion hsemotion-lookup`, so the A/B above stays
reproducible — only the model differs between it and the default.

---

## feat: first-impression integration — auto-enrol strangers into the culture pipeline  *(branch `feature/cultural-awareness`)*

**Goal (user):** port the `first-impression` branch's "meet a stranger" pipeline into the culture branch, but
as a HYBRID: an unknown face is auto-enrolled as a new person, and then the FULL culture/interest pipeline
(auto-attach, extraction, viz highlight) runs on them — not the first-impression branch's stripped
fast-node-only mode.

**Why a manual port, not a merge:** a `git merge first-impression` is unworkable — ~130 rename/delete
conflicts from committed runtime data (rag_indexes/faiss, sessions.db, faces.npz) + a real `webcam_loop.py`
conflict (both branches rewrote it). Culture already had the infra it needed (`identify_all`, `enroll`,
4-value VA emotion), so only the auto-enrol + name-learning was ported. The `first-impression` branch is left
intact for further research.

**Ported (new, no conflicts):**
- `graph_relationship/rename.py::rename_person` — pure graph surgery: re-key `guest_N` → real name across the
  person node + `interaction:`/`conversation:` ids, re-pointing every incident edge (rapport/trust/mood/
  culture preserved). `SessionStore.rename_person` (SQLite rows). `FaceIdentifier.rename` (move / weighted-merge
  the enrolled embedding).
- `webcam_loop`: module-level `_extract_name` (regex self-intro, stoplist rejects feelings/filler + culture
  words) + `_slug_name`; methods `_next_guest_id`, `_auto_enroll` (enrol unknown face as `guest_N` + save
  faces.npz), `_learn_name` (re-key across face DB + graph + transcripts + in-memory maps, under the store
  lock). Wired into the chat Enter handler: unknown face in view → auto-enrol as `guest_N` (this turn is
  attributed to them); if they introduce themselves → re-key to the real name — all BEFORE the reply is
  dispatched, so the prompt + KG writes use the right id.
- A `guest_N` is a normal `PersonNode`, so culture auto-attach, interest extraction, cross-namespace bridges,
  and the face-driven viz highlight all work on them unchanged.

**Verified — `test_first_impression.py` 6/6:** name extraction; graph rename_person (ids re-keyed, culture +
relationship edges follow, no dangling guest id, counts stable); session rename; face rename (move + merge);
`_learn_name` re-keys all four stores; a guest gets the full culture/interest prompt. Full culture regression
green (multi-culture, style-hint, soft-evidence, cross-namespace, affinity, preference, culture-seed/extract,
+ graph_relationship pytest 47).

**Deferred (inherited from first-impression):** if topics are extracted (via `X`) BEFORE the guest gives their
name, `interest:guest_N:*` node ids aren't re-keyed (rename_person re-keys person/interaction/conversation
only). The normal flow (name learned during chat, extraction at session end) attaches to the real name, so
this only bites an unusual ordering. Fix would extend rename_person to also re-key interest ids.

---

## feat: multi-culture selector — Māori added + active-culture (auto-attach) + viz toggle  *(branch `feature/cultural-awareness`)*

**Goal (research/testing):** let the robot hold MULTIPLE cultures (Korean + Māori) and adapt to whoever it's
talking to, with a viz toggle to A/B "robot has cultural knowledge" vs "generic" — to see if cultural
grounding improves UX.

**What was built:**
- **Māori seed** (`culture_seed`): refactored the Korean seeder into a generic `_seed_culture`; added
  `_MAORI_DEMO` (12 topics: haka, marae, waiata, te reo, hangi, rugby, pepeha…) + `_MAORI_STYLE_HINT`
  (manaakitanga/whanaungatanga, respect for kaumātua, humility). `seed_maori_demo` + `seed_all_cultures`;
  `--mode seed-culture-demo` now seeds BOTH. Demo data, not research claims. The person-driven culture path
  was already generic, so a Māori-tagged person "just works" once seeded.
- **Active-culture resolver** (`webcam_loop._active_culture_id`): the culture that shapes the prompt =
  **override → the current person's `belongs_to_culture` → generic**. So the active lens auto-attaches to /
  detaches from whoever is recognised, and the persistent per-person tag never flips. `rank_suggestions`
  gained an optional `culture_id`; `_culture_block` + the style-hint injection now use the ACTIVE culture.
  Framing: recall-as-fact only when the active culture is what the person SELF-DECLARED, else it reads as the
  robot's "knowledge lens" (a starting point, never a fact about them).
- **Mid-session auto-attach**: on self-declaration ("I'm Korean"), the culture attaches IMMEDIATELY (async
  detection off the main thread, guarded by a cheap origin-cue filter + "only while untagged"), so it shapes
  the rest of that session. Self-declaration/manual ONLY — never inferred from face/name/appearance.
- **Override backend (A/B, dormant)**: `GET/POST /culture` on the viz server writes a `culture_override.json`
  sidecar (kept OUT of kg_state.json); the loop reads it each turn. `auto` (person-driven) · `generic`
  (culture off, the A/B control) · a culture name (force Korean/Māori). The UI selector button was REMOVED —
  culture is now fully automatic (driven by the recognised person) — but the endpoint/file remain so A/B is
  still possible via curl/file without cluttering the page.
- **Viz active-highlight (display-only attach/detach, face-driven)**: the loop writes the current
  (face-recognised) person + resolved active culture to `active_state.json`; the viz server folds it into
  `/graph.json` as `active`. The page lights the robot + that person's OWN subgraph (BFS through their
  interests → topics → bridges) + the ACTIVE culture cluster, and dims everything else (`.inactive` CSS). So
  switching who's in front detaches the previous person's bridges; a SECOND person of the same culture keeps
  the culture nodes lit and only dims the previous person's own bridges (shared topics stay lit). Purely
  visual — `knows_culture` and graph data are never mutated (seeded prior knowledge stays true).

**Verified — `test_multi_culture.py` 5/5:** both cultures seeded + idempotent; person-driven lens+manner per
person with no cross-leak; resolver auto→person / forced / generic-off + viz labels; mid-session attach
(cue+untagged→attach, tagged→skip, no-cue→skip); viz round-trip flips the loop's prompt. Existing suites
green: test_style_hint, test_soft_evidence, test_cross_namespace, test_affinity, test_preference_model,
test_culture_seed (headers updated to the knowledge-lens wording), test_culture_extraction, +
graph_relationship pytest (47). `graph_relationship/` purity unchanged.

---

## feat: thin per-culture static style hint — Approach 1, STEP 4 (final)  *(branch `feature/cultural-awareness`)*

**What:** the "how to talk" half of cultural adaptation — a single short, STATIC manner paragraph per culture,
injected into HOW-TO-REPLY when a person is tagged to that culture. Deliberately dumb: hand-written seed
data, identical for every interaction; NO tier/affect/situation/dynamics (that is Approach 2's policy vector,
kept a clean separable step-up).

**Change:**
- **schema** (`CultureNode`): one optional field `style_hint: str = ""` — default "" so old graphs load
  unchanged and inject nothing.
- **seed** (`culture_seed`): Korean CultureNode gets a hand-written `_KOREAN_STYLE_HINT` (polite/warm/slightly
  formal; deflected compliments; food as friendliness). Set/overwritten idempotently on seed; counts & hint
  unchanged on re-seed.
- **prompt** (`_build_system_prompt`): if the person is tagged to a culture with a non-empty hint, append
  `"• Cultural manner (soft guidance, secondary to answering their actual question): <hint>"` to HOW-TO-REPLY
  — SEPARATE from the content topic-offer block (which stays in the WHO/CULTURAL BACKGROUND section). Pure read
  `cultures.person_culture_style_hint`. Untagged / empty hint → injects nothing (no leakage, no empty header).
- **viz**: CultureNode carries `styleHint` for its tooltip (optional).

**Static by construction:** the injected text depends ONLY on which culture the person is tagged with — same
string for a "visitor" vs a "close" person (verified).

**Verified — `test_style_hint.py` 6/6:** backward-compat (pre-feature culture → `style_hint=""`, round-trip
byte-identical); seed sets it, re-seed idempotent (counts + hint unchanged); injected when tagged &
manner/offer blocks stay separate; not injected when untagged; empty hint injects nothing; static across
tiers. Existing suites green: test_soft_evidence, test_cross_namespace, test_affinity, test_preference_model,
test_culture_seed, test_culture_extraction, + graph_relationship pytest (47). `graph_relationship/` purity
unchanged.

**This completes the four Approach-1 upgrades** (continuous affinity + hedging · cross-namespace bridges ·
confidence-weighted clamp · static style hint). Approach 2 (communication-policy vector / tier-driven,
dynamic manner) is the separable next milestone.

---

## feat: confidence-weighted BN clamp — Approach 1, STEP 3  *(branch `feature/cultural-awareness`)*

**Problem:** an observed topic clamped to its `affinity` alone, so a rock-solid "I love jazz" (aff 1.0,
conf 0.95) and a hedged "jazz is okay I guess" (aff 1.0, conf 0.62) clamped to the SAME value and
propagated equally hard across the bridges — the BN conflated *how much they like it* with *how sure we
are they said it*. The 0.6 extraction gate is binary (0.61 and 0.99 look alike once past it).

**Change (calibration, not a new model — `modules/preference_model.py` only):** the observed-topic clamp is
now `clamp_from(affinity, confidence) = 0.5 + (affinity - 0.5) * confidence`. An uncertain observation is
pulled toward neutral (0.5) so it moves the posterior less; confidence 1.0 → clamp == affinity (identical to
Step 2). Signed & symmetric (a confident dislike still pins low; an unsure one sits nearer 0.5). This is the
ONLY change — rounds (2), damping (0.8), floor, bridge traversal, and the candidate set are untouched, and
observed topics stay excluded from suggestions. Now the BN's number agrees with the prompt's Step-1 hedge
word (both treat a shaky observation as shaky).

**Verified — `test_soft_evidence.py` 6/6:** (1) full trust reproduces the EXACT Step-2 lift (clamp 1.0 →
kpop 0.4960); (2) uncertainty shrinks the lift (conf 0.6 → 0.3968 < 0.4960); (3) confident dislike still
drags a neighbour down; (4) a weak low-confidence dislike (clamp 0.47) leaves a strong culture prior (0.60)
above the 0.35 floor while a confident one (clamp 0.40) moves it more; (5) neutral (0.5) is a fixed point at
any confidence; (6) read-only/deterministic/graceful. Existing suites green: test_cross_namespace,
test_affinity, test_preference_model (confidence=1.0 → unchanged), test_culture_seed, test_culture_extraction,
+ graph_relationship pytest (47). `graph_relationship/` purity unchanged.

**Live-KG copy demo:** same 0.9 "like" on `football_player` bridged to `ck:korean:baseball` (prior 0.30) →
posterior 0.459 (conf 1.00), 0.448 (0.95), 0.357 (0.50), 0.300 (0.20) — lower confidence lifts it less; a
very-unsure signal barely moves it off the prior.

**Deferred (Step 4, untouched):** style hint.

---

## perf: thread the chat pipeline + async topic label + debounced saves  *(branch `feature/cultural-awareness`)*

**Problem:** the webcam loop felt unresponsive — the OpenCV window froze during every reply. Investigation:
vision (face/emotion/demographics) already runs in a daemon thread on CPU (forced off the RTX-5060/sm_120
GPU), but the **whole chat turn ran synchronously on the display thread** — 3 serialized Ollama round-trips
(RAG embed + qwen reply + a 2nd qwen call for the live-topic label) plus a full-graph `store.save()` — so the
display couldn't read frames until the LLM finished. Disabling emotion didn't help (it's off the chat path).

**Fixes (one at a time, agreed order):**
- **Option B — async live-topic label.** The 2nd LLM call (`_detect_topic`, a metadata label, NOT the reply)
  moved to a background thread. The turn is recorded immediately; the label is backfilled to the session DB
  (`SessionStore.set_turn_topics`, thread-safe) and handed to the main thread via a queue to update the
  conversation node. Zero reply-quality loss (still LLM-generated), just off the critical path.
- **#2 — threaded chat pipeline.** RAG + prompt build + reply now run on a dedicated `chat-worker` daemon
  thread; the display shows `…` and keeps rendering. The `InMemoryGraphStore` isn't thread-safe, so a single
  `RLock` serialises the worker's *fast* prompt-build read against every main-thread store mutation/save
  (per-tick, chat-apply, topic-drain, hotkeys, shutdown). **All store writes stay on the main thread**; the
  lock is never held across the slow LLM/RAG calls. Verified: 40 concurrent chats + a thread hammering
  store writes/saves → 0 errors, no corruption.
- **#3 — debounced persistence.** The frequent per-tick/chat mutations now just mark the graph dirty; the
  loop flushes `kg_state.json` at most once/second (or forced on shutdown), instead of rewriting the whole
  file every tick (which produced the "8 saves per turn" churn).

**Also (data tuning):** lowered the Korean demo culture priors ~0.20 (`culture_seed._KOREAN_DEMO`) into a
0.25–0.60 band, so a person's liked interest bridging to a culture topic (Step 2) can actually lift it above
the 0.35 suggestion floor instead of being capped by an already-high prior.

**Still TODO:** #4 — GPU for vision (needs a torch build with sm_120/Blackwell support; deferred because
vision is already threaded and would contend with Ollama on the GPU). Thread model now: main/display +
detection-worker + chat-worker + short-lived topic-detect threads.

---

## feat: cross-namespace culture↔person bridges — Approach 1, STEP 2  *(branch `feature/cultural-awareness`)*

**Problem:** culture priors live on `ck:<culture>:<slug>` CultureTopic nodes; person interests live on
`topic:<slug>` nodes. These were disconnected islands — only a label-join, no edge — so a person's OBSERVED
interest could never propagate to a culturally-adjacent culture topic and the BN degraded to a base-rate
lookup. Step 2 adds `related_topic` BRIDGE edges across the namespaces so the existing 2-round noisy-OR
carries person evidence into culture topics (observed `topic:jazz` lifts `ck:korean:kpop` above its prior).

**What was built (bridges only — propagation math untouched):**
- **topics.py** `link_related_cross()` — same `related_topic` edge/storage as `link_related_topic`, but each
  endpoint may be a `topic` OR a `culture_topic` node. A relatedness LINK only: never merges, never changes
  identity — a `ck:` node and a same-slug `topic:` node stay two DISTINCT nodes, now traversable.
- **kg_extraction.py** `link_cross_namespace_bridges()` — two idempotent, LLM-free passes: (1) exact-slug
  bridges (weight 1.0) for a person `topic:<slug>` whose slug equals a `ck:<culture>:<slug>`; (2) embedding
  bridges in the same `[0.60, 0.86)` related band, `same_category_only`, exact-slug pairs excluded. Reuses the
  existing `_embed`/`_pairs`/`_same_category` machinery pointed at the cross-namespace pair set.
- **preference_model.py** — the ONE traversal change: when gathering a person topic's `related_topic`
  neighbours, accept `culture_topic` neighbours too (they resolve into the same slug space). No change to
  rounds, damping (0.8), floor, or the signed clamp — only which neighbours the existing walk may visit.
- **Wiring:** the bridge pass runs wherever related-linking already runs — `--mode consolidate` and the
  auto-consolidate path.
- **Viz:** cross-namespace bridges render as `related_topic` with a dashed stroke + "culture bridge" tooltip.

**Finding (reported honestly):** the BN joins by SLUG, so an EXACT-slug bridge (`topic:hiking ~
ck:korean:hiking`, same slug) is a self-loop in slug space and does not by itself move a distinct node — it
is created/idempotent/merge-safe and makes the join an explicit traversable edge, but the measurable
cross-namespace LIFT comes from DISTINCT-slug (embedding) bridges. Carrying an exact-slug bridge two hops
(person→ck:hiking→ck:kpop) would need a hop-count change, which this step deliberately scoped out.

**Verified — `test_cross_namespace.py` (synthetic store, fake embed) 6/6:** (1) embedding bridge lifts kpop
0.30→0.496, delete→0.30 fallback; (2) exact-slug bridge created/idempotent/merge-safe/traversable; (3) merge
invariant — consolidation never merges ck↔topic, per-namespace counts unchanged, only +1 related edge; (4)
dislike crosses — disliked jazz pulls kpop 0.30→0.151 (Step-1 signed clamp composes with Step-2 bridge); (5)
read-only + deterministic + graceful degradation; (6) embedding-band linker creates the bridge idempotently.
Existing suites still green: test_affinity, test_preference_model, test_culture_seed, test_culture_extraction,
and the graph_relationship pytest suite (47). `graph_relationship/` purity unchanged.

**Deferred (Steps 3–4, untouched):** confidence-weighting the BN clamp; style hint.

---

## fix: self-declared culture is a recallable fact, not just a tentative hint  *(branch `feature/cultural-awareness`)*

**User point:** jay said *"i am from korea"* in a session, yet next time the robot answered *"I'm not sure if
I can remember that."* Investigation: the graph DID hold `jay --belongs_to_culture--> Korean` (provenance
`self-declared:<sid>`, confidence 1.0), and the prompt DID include it — but `_culture_block` worded EVERY
culture tag as *"a starting guess about their background, not a fact about them as a person"* + *"Never
assert…"*, so the LLM correctly refused to state it. The anti-stereotyping framing (right for an INFERRED
culture) was being applied to a background the child had EXPLICITLY stated.

**Fix (Approach A — graph-driven, no new data/mechanism):** branch the framing on the belongs_to_culture
edge's existing PROVENANCE.
- New pure reads in `graph_relationship/cultures.py`: `person_culture_source()` (the edge's `source`) and
  `person_culture_self_declared()` (True iff `source` starts `self-declared`).
- `_culture_block` now emits: self-declared → *"Background (they told you themselves): Korean… you CAN recall
  it as a fact if they ask — e.g. 'you mentioned you're Korean'. Don't assume what they like from it — ask."*;
  manual/seed-assigned → unchanged tentative hint. Either way it still never ASSUMES preferences from the
  background. Provenance was already stored (self-declaration writes `self-declared:<sid>`, manual writes
  `culture-seed`) — A just reads it.

**Verified:** `test_culture_seed.py` new case 5b — self-declared person gets recall-as-fact wording (no
"starting guess"); manually-assigned person keeps the tentative hint (no recall permission); neither assumes
preferences. Existing cases 1–5 + purity still green (cultures.py still imports only schema/store/topics).
Live: jay's rebuilt block now reads *"Background (they told you themselves): Korean…"*.

**Deferred (Approach B, later):** a general biographical-facts memory (origin/age/family/pet…) separate from
topics/culture, so first-person facts beyond culture also persist and are recallable.

---

## feat: continuous affinity (0–10) + confidence hedging — Approach 1, STEP 1  *(branch `feature/cultural-awareness`)*

**Goal:** replace the flat "observed = positive" assumption with a SIGNED, graded signal per topic, feeding
two consumers side by side: the preference BN (affinity → signed clamp) and the person-memory prompt
(affinity → like/neutral/dislike word, confidence → "clearly/probably/possibly" hedge). Human-facing scale
is 0–10; stored internally as `affinity ∈ [0,1]` (0 dislike / 0.5 neutral / 1 like) so it drops into the BN
clamp with no conversion. Confidence is model-agnostic and (this step) affects prompt wording ONLY — it does
NOT weight the BN clamp yet (deferred).

**What was built:**
- **Schema (pure, `graph_relationship/schema.py`):** `AboutEdge` gains `affinity: float = 0.5` and
  `confidence: float = 1.0` (both `[0,1]`). Backward-compat: old `kg_state.json` about edges load as
  neutral/fully-trusted. Robot capability→topic about edges just carry the unused defaults.
- **Boundary helpers (pure, `graph_relationship/scales.py`, new):** exactly `aff01_from_10` / `aff10_from_01`
  — the ONLY place the 0–10 ↔ [0,1] conversion happens.
- **topics.py:** `add_person_topic` / `reinforce_person_topic` / `add_person_interest` now accept + write
  `affinity`/`confidence` on the about edge (re-mention = simple overwrite; EWMA blending noted as future
  work). New pure read `person_topic_affinity(store, pid) → [(TopicNode, affinity, confidence)]` — the
  shared "observed evidence" reader for BOTH consumers.
- **Extraction (`modules/kg_extraction.py`):** prompt asks for a per-topic `sentiment` 0–10 (5 = neutral/
  unstated) alongside the existing `confidence`; mapped via `aff01_from_10`. NEW guard: missing/out-of-range
  sentiment → 0.5 (item KEPT, not dropped); all existing drops (bad JSON/category/hallucination/conf<0.6)
  unchanged.
- **BN clamp (`modules/preference_model.py`):** the flat `_OBSERVED_P = 0.90` clamp is gone — an observed
  topic now clamps at its stored affinity. Propagation is signed: a LIKED neighbour raises others (noisy-OR,
  as before); a DISLIKED one (below neutral) pulls related neighbours DOWN toward its low clamp, scaled by
  `0.8·weight`. Observed nodes stay fixed and remain excluded from suggestions. Still read-only, 2-round, no
  BN library, no confidence-weighting.
- **Prompt (`modules/affinity_phrasing.py`, new + `_person_memory`):** learned topics render as signed,
  hedged lines — "They clearly like jazz." / "They probably dislike baseball — avoid raising it." / "They may
  be neutral on pasta." The old flat `Interests:` summary is replaced by a `How they feel about topics:`
  block; specific notes still lead.
- **Viz (minimal):** person interest→topic about edges expose `affinity` (shown 0–10) + `confidence` in the
  edge tooltip.

**Verified — `test_affinity.py` (headless, fake LLM), 6/6:** (1) real `kg_state.json` copy loads with every
about edge neutral/fully-trusted, save→load→save byte-identical; (2) sentiment mapping love-jazz→1.0,
can't-stand-baseball→0.1, unstated-pasta & oob→0.5 kept, idempotent; (3) helpers round-trip exact at
0/5/10; (4) disliked topic drags a `related_topic` neighbour BELOW prior, liked one lifts ABOVE, both
observed excluded, read-only + deterministic; (5) verb×hedge wording correct; (6) `graph_relationship/`
purity (no LLM/PAD/app/embedding imports). Existing `test_preference_model.py` updated to pass `affinity=0.9`
for its observed-positive topic (preserves the 0.468 propagation number) — still green; culture-seed/
culture-extraction and the `graph_relationship` pytest suite still green.

**Didn't do (out of scope for STEP 1, per plan):** cross-namespace bridges, confidence-weighting of the BN
clamp, style hint / communication-policy vector, EWMA blending of re-mentions, any PAD change.

**Live-smoke follow-up (extraction robustness):** first real run surfaced two issues, both fixed:
1. *Intermittent "LLM JSON parse failed"* — the extraction call reused the spoken-reply settings (temp 0.7 +
   ChatML `stop` strings + greedy first-`{`/last-`}` slicing), so qwen2.5:7b occasionally wrapped the JSON in
   prose or truncated it, silently dropping the whole turn's topics (and thus all affinity). Added a
   `json_mode` path to `LLMClient.respond`: `response_format={"type":"json_object"}`, temperature 0, no stop
   strings, and skips `_clean_reply`; the extraction wrapper (`_json_llm`) now uses it. Live: 10/10 parses OK
   (was intermittent).
2. *Timid sentiment* — qwen rated "used to like the spicy beef soup" as 5 (neutral). Added full-range anchors
   to the sentiment rule in `build_system_prompt` ("hate it"→0-1 … "love it"→9-10). Live: the same utterance
   now scores 7 → affinity 0.70 → "like"; "don't really like sushi" → 3 → 0.30 → "dislike".

---

## fix: tag a person's culture ONLY on explicit self-declaration  *(branch `feature/cultural-awareness`)*

**User point:** jay shouldn't be connected to Korean "at first place unless jay said he is korean" — the demo
had assigned jay via the manual `--assign-culture` flag with no basis, which is exactly the stereotyping the
layer is meant to avoid.
**Two fixes:**
1. Removed jay's unfounded `belongs_to_culture` edge from the live KG. ChatBox still `knows_culture` Korean
   (its own prior knowledge) and the culture + priors remain — only the *person* tag was wrong.
2. `belongs_to_culture` is now driven by SELF-DECLARATION, not an operator hunch:
   - new `modules/culture_extraction.py` — `detect_self_declared_culture(turns, llm_fn)`: reads ONLY the
     person's own words and returns a culture label ONLY on an explicit self-statement ("I'm Korean", "my
     parents are from Korea"). Returns None for liking kimchi/K-dramas, visiting Korea, speaking Korean, or
     any ambiguity. Conservative guards (NONE/empty/sentence/tag/non-demonym → None); robot replies never fed.
   - wired into `_extract_session` step (c): after topic extraction, if the person self-identifies and isn't
     already tagged, `assign_person_culture`. (This is NOT the prohibited auto-detection from face/name/
     language/appearance — it's the person explicitly telling us; the earlier "manual only" guard was really
     anti-stereotyping, which self-declaration honours.)
**Verified:** real LLM 7/7 — "I'm Korean"/"my parents are from Korea" → tagged; "i love kimchi and
k-dramas" / "I visited Korea" / "recommend korean bbq?" / "I'm learning Korean" → NOT tagged. Headless
`test_culture_extraction.py` (fake LLM): parse/guards, robot-reply excluded, end-to-end (kimchi-lover not
tagged, explicit declaration tags once, idempotent). culture-seed + preference-model tests still green.
**Note:** the manual `--assign-culture` flag stays as an admin override. The seed still only sets ChatBox's
`knows_culture` + priors; no person is tagged by seeding.

---

## feat: Bayesian preference overlay — recommend topics on read (Command B)  *(branch `feature/cultural-awareness`)*

**Goal:** rank topics the robot could tentatively bring up, blending the person's CULTURE priors (base
rates) with OBSERVED interests (strong evidence) and soft propagation over `related_topic` links — a simple
noisy-OR, no external BN library.
**Module (`modules/preference_model.py`, app layer):** `rank_suggestions(store, person_id, k=3,
floor=0.35) -> [(node_id, posterior)]`. Compiled ON READ, mirrors the `_tier_from_edges` pattern:
READ-ONLY (zero writes/creates), no live embeddings (consumes only STORED `related_topic` weights), no LLM,
no PAD. `graph_relationship/` is untouched (imports its pure reads only).
**Algorithm:** (1) candidate set = culture-prior topics ∪ the person's own topics ∪ their one-hop
`related_topic` neighbours; (2) init unobserved = culture prior (default 0.30); (3) clamp observed
(interest/about) = 0.90; (4) 2 rounds noisy-OR `p[b]=max(p[b], p[a]·0.8·w)`, observed stay clamped;
(5) top-k UNOBSERVED with posterior ≥ floor, tie-break posterior→prior→lexicographic id. Dislike/negative
evidence explicitly OUT OF SCOPE (noted in a comment as future work).
**Namespace join (post-redesign):** culture priors live on `ck:` CultureTopic nodes, DISTINCT from a
person's `topic:` nodes — so the model works in a unified SLUG space (a culture "kimchi" prior and a person
"kimchi" topic are the same concept, joined by normalized label). This is the label-join flagged when the
culture layer became ChatBox-owned.
**Prompt (`_culture_block`):** the raw-prior top-4 list is replaced by `rank_suggestions(k=4)` output;
observed exclusion now comes for free (step 5). All other wording unchanged.
**Pre-step:** `--mode consolidate` still creates `related_topic` links among PERSON topics (culture topics
are a different node_type, so consolidation skips them — verified it runs cleanly and leaves the 12 culture
topics untouched). The overlay degrades gracefully to prior-only when no related links exist.
**Verified (`test_preference_model.py`, 6 synthetic checks):** prior-only == culture priors desc;
propagation raises a linked topic (kpop 0.30→0.468 via kdrama·0.8·0.65) while an unlinked one is unchanged;
observed topic never suggested; READ-ONLY (node/edge counts + save bytes identical across 3 calls);
deterministic; graceful degradation when all `related_topic` edges are deleted. `test_culture_seed.py` +
`tests_schema.py` still green. Real-LLM pipeline (qwen2.5:7b): fresh person tagged Korean → "i've been
watching a lot of kdramas" → extraction adds the `kdramas` interest → "what should we talk about?" → robot
offers Korean dramas AS A QUESTION, no false assertion. (Minor: extraction produced plural "kdramas" vs the
culture "kdrama" slug — a labelling variance, harmless.)
**Deferred (future):** negative/polarity evidence; confidence-weighted soft evidence; cross-namespace
culture↔person `related_topic` links (would let a jazz-lover get kpop *propagated*, not just base-rate).

---

## refactor: culture as ChatBox's prior knowledge — separate nodes, HJ decoupled  *(branch `feature/cultural-awareness`)*

**User feedback on the first culture cut:** the viz showed HJ *near* Korean even though HJ never discussed
anything Korean. **Root cause:** the culture reused the SHARED `topic:hiking` node (Command A's "reuse, don't
duplicate" rule), and HJ already liked hiking — so both pointed at the one node and the force layout pulled
them together (a 2-hop path, not a real HJ→Korean edge). **User's reframing (approved):** a culture is the
ROBOT's *prior knowledge*; ChatBox should own it, and a person only links to a topic by actually talking
about it over time. So: (1) anchor the culture under ChatBox, (2) give the culture its OWN topic nodes so no
unrelated person is ever coupled in.
**Redesign:**
- schema: new `CultureTopicNode` (id `ck:<culture>:<slug>`, own `node_type` so topic consolidation / interest
  machinery never touch or merge it into a person topic) + `KnowsCultureEdge` (robot→culture, SLOW).
  `CulturePriorEdge` now targets a CultureTopic, not a shared Topic.
- `cultures.py`: `culture_topic_id`, `ensure_culture_topic`, `knows_culture`, `culture_knowers`;
  `culture_priors` now returns `(ck_id, label, prior)`. Still schema/store/topics-only (pure).
- `culture_seed.py`: seeds `chatbox --knows_culture--> Korean --culture_prior--> ck:korean:*` — NO
  `resolve_topic`, so shared person topics are never created or reused. Robot-aware (`--robot`).
- prompt (`_culture_block`): offers now exclude the person's own topics by NORMALIZED LABEL (culture topics
  are separate nodes, so compare slugs, not ids). Wording unchanged.
- viz: `CultureTopic` → 6-point "sparkle" shape + legend; `knows_culture` → SLOW.
**Model now:** `chatbox --knows_culture--> Korean`; `Korean --culture_prior--> ck:korean:kimchi …`;
`jay --belongs_to_culture--> Korean` (tag only). `topic:hiking` (HJ's interest) and `ck:korean:hiking`
(ChatBox's knowledge) are DISTINCT nodes → HJ has zero culture edges.
**Verified (`test_culture_seed.py`, rewritten):** empty seed = chatbox-owns-Korean + 12 culture topics/priors
+ 0 person topics, round-trips identically; idempotent; on a copy of the real KG person topics are UNCHANGED
(+12 culture topics, no reuse); jay→Korean prompt block correct (≤4 offers, kpop excluded by slug, memory
leads, ChatBox owns culture); **HJ decoupled** (shares "hiking" label, zero culture edges, ck≠topic node);
purity. `tests_schema.py` still 5/5. Real-LLM smoke (qwen2.5:7b) unchanged in behaviour (bibimbap/kimchi
politely; offered jazz as a question, no false assertion). Live `kg_state.json` re-seeded to the new design.
**Worktree:** done in an isolated git worktree (`.claude/worktrees/cultural-awareness`) so another agent can
use `first-impression` in the main checkout concurrently.
**Deferred (gated):** Command B — read-only Bayesian preference overlay. Note for B: culture priors are now
`ck:` nodes distinct from person `topic:` nodes, so B must label-JOIN a culture prior to a person's observed
topic by slug (not node id).

---

## feat: Korean culture layer — dummy priors + prompt injection (Command A)  *(branch `feature/cultural-awareness`)*

**Goal:** a MANUAL cultural-background layer that gives the robot a weak, respectful hint about a person's
background + a few topics it could tentatively offer — plumbing/prompt only, NO Bayesian inference yet
(that's the gated Command B, held for approval). No auto-detection from face/name/appearance — assignment
is manual (CLI) only.
**Schema (pure):** `CultureNode` (id `culture:<slug>`), `BelongsToCultureEdge` (person→culture, SLOW,
idempotent), `CulturePriorEdge` (culture→topic, `prior∈[0,1]`, SLOW, upsert). Added to the node/edge unions
exactly like `ConversationNode`/`RelatedTopicEdge`; old `kg_state.json` loads unchanged (40 nodes/44 edges).
**Pure helpers (`graph_relationship/cultures.py`):** `ensure_culture`, `assign_culture`, `set_culture_prior`
(clamped), `culture_priors` (sorted desc), `person_culture`. Imports ONLY schema/store/topics.
**Seed (app `modules/culture_seed.py`):** seeds ONE culture `Korean` with 12 demo topics + DUMMY priors
(hand-set placeholders, marked as demo — not research claims) via `resolve_topic`, so topics already in the
graph (e.g. `hiking`) are REUSED not duplicated. Idempotent. Does NOT assign anyone. CLI:
`--mode seed-culture-demo` + a standalone `--assign-culture <person> <culture>`.
**Prompt (`_person_memory._culture_block`):** if the person has a `BelongsToCultureEdge`, append a
"CULTURAL BACKGROUND" block — a hint line ("starting guess … not a fact about them"), up to 4 highest-prior
topics EXCLUDING ones they already have an interest in, and tentative-offer phrasing (offer ONE if the
convo lulls, drop if uninterested, never assert what they like). Appended LAST so specific memories still
lead (applies the earlier mood-weighting lesson — weak hint, content first).
**Viz:** `Culture` node → heptagon + legend row; `belongs_to_culture`/`culture_prior` → SLOW colour; edge
thickness reflects `prior`.
**Verified (`test_culture_seed.py`, headless, fake/real LLM):** empty seed = 1 culture/12 topics/12 priors,
save→load→save byte-identical; idempotent; on a COPY of the real KG only `hiking` is reused (topics +11 not
+12, overlap computed from the file); jay→korean prompt has the block with ≤4 offers that exclude jay's
kpop interest and keep interests above the culture block; purity (pure `graph_relationship/` has no
LLM/PAD/app imports). Existing `tests_schema.py` still 5/5. Real-LLM smoke (qwen2.5:7b): "korean food?" →
bibimbap/kimchi politely; "what should we talk about?" → offered a topic as a question, no false assertion.
**Note:** the LIVE `kg_state.json` was never mutated — all tests used temp copies; seed it explicitly to see
the culture nodes in the viz.
**Deferred (gated):** Command B — the read-only Bayesian preference overlay (`preference_model.py`,
noisy-OR propagation over `related_topic`). Held until this is approved.

---

## feat: cultural-awareness webcam view — FairFace region/age (--culture, display-only)  *(branch `feature/cultural-awareness`)*

**Goal:** estimate a person's regional heritage (East Asian, European, …) + age from the webcam and show it
on the live view. Display-only this pass — deliberately NOT fed to the KG or the LLM prompt.
**Model (`modules/face_webcam/demographics.py`):** pluggable `DemographicsDetector` mirroring the
emotion/face backends; `FairFaceDetector` runs the FairFace ResNet-34 ONNX (7 race + 2 gender + 9 age in one
pass) via onnxruntime (~15 ms/face CPU, no torch dep). 7 races → friendly regions (East Asian, Southeast
Asian, European, African, South Asian, Middle Eastern, Latino/Hispanic); age band → coarse stage. Weights
(~85 MB) cached to `~/.fairface/`, auto-downloaded on first use. A confidence-weighted `_StableVote`
stabilises + "locks" the estimate (demographics don't change frame-to-frame); standalone CLI + `--image`.
**Webcam wiring (`webcam_loop.py`):** `--culture` flag; the detection worker runs the model per face
(throttled every 3rd cycle, last estimate cached between), draws `region conf* / age (stage)` under each
face box in cyan. Off by default; existing runs unchanged.
**Verified:** module smoke test (loads, infers, vote locks + re-locks on sustained flip); headless worker
→ overlay path attaches region/age and `draw_overlay` renders it; `--culture` parses.
**Note:** region is an uncertain appearance estimate (flickers ~1 s then locks); framed as non-identifying.
**Didn't / deferred:** no KG persistence, no prompt use, no auto-culture-assignment from it (kept manual).

---

## refactor: prune dead code + share consolidation core (compact/modular)  *(branch `KG-knowledge-extraction`)*

**Goal:** review the branch and remove unnecessary/redundant code. Net −109 lines, no behaviour change.
**Removed (dead):**
- `_mood_phrase`, `_MOOD_WEIGHT`, and the `mood`/`emotion` params of `_build_system_prompt` (+ the
  `cur_mood`/`cur_emotion` locals) — dead since mood was dropped from the prompt.
- `CurrentTopicEdge` (schema class + union member + store classification + viz timescale map + comments) —
  superseded by `ConversationNode`; 0 such edges in any persisted graph, so removal is backward-compatible.
- `SessionStore.person_turns` (never called); an unused `_dump_kg` import.
**Refactored (modular):** the three consolidation functions (`consolidate_topics`, `consolidate_interests`,
`link_related_topics`) shared ~120 lines of near-identical embed + pairwise-cosine + union-find loops. Pulled
out `_embed`, `_pairs`, `_merge_by_similarity`, `_same_category` helpers; the public functions are now thin
wrappers (kg_extraction.py −143 lines).
**Verified:** all modules compile; real `kg_state.json` loads (40 nodes / 44 edges, incl. related_topic);
merge/related behaviour identical on a controlled synthetic case (near-dups merge, related link, unrelated
skip); prompt has no mood line but keeps common-ground + related interests; real-LLM reply still recalls
memory ("R&B and jazz … SZA and Kendrick Lamar").

---

## feat: use topic↔topic relations in retrieval + common ground (2c points 1&2)  *(branch `KG-knowledge-extraction`)*

**Goal:** actually *use* the `related_topic` edges (they were structure-only). Wired points 1 (retrieval /
recall) and 2 (common ground); left 3–5 (recommendations / transitions / cold-start) for later.
**Implemented (pure helpers in topics.py):**
- `topic_related(topic_id)` — one-hop related topics.
- `related_common_ground(person, robot)` → `{direct, bridges}`: a bridge = a person topic `related_topic`-
  linked to a robot capability topic (indirect common ground, e.g. their *multiplication* ~ your *math
  problems*).
- `person_related_pairs(person)` — related pairs among the person's own topics (rap ~ hiphop).
**Prompt (`_person_memory`):**
- Point 2: "Common ground" now adds bridges — "You can also connect via related topics: their multiplication
  ~ your math problems".
- Point 1: note-gathering expands one hop across `related_topic`, so related memories surface; plus a
  "Related interests: hiphop ~ rap" line so the robot can bridge/recall across them.
**Verified (real LLM):** "do we have anything in common?" → "We both enjoy jazz and math problems…" (the
math-problems bridge now surfaces); "i love rap, can we talk about it?" → engages naturally. Pure modules
stay LLM-free.
**Deferred:** 2c points 3–5 (recommendations, smoother transitions, cold-start generalization); rapport/trust.

---

## feat: topic↔topic relations (Feature-2c) — link related-but-distinct topics  *(branch `KG-knowledge-extraction`)*

**User observation:** "rap" and "hiphop" didn't merge. **Finding:** cos(rap,hiphop)=0.678 — below the 0.86
merge floor, and there's NO safe merge threshold (tennis/basketball=0.654, dogs/cats=0.639 sit right below
rap/hiphop). **User's call:** don't merge them — **link** them instead (they're distinct but related).
**Implemented Feature-2c:**
- schema: `RelatedTopicEdge` (topic↔topic, SLOW, `weight`=similarity; conceptually undirected, stored with
  sorted endpoints).
- pure `topics.link_related_topic(a, b, weight)` — idempotent, only between topic nodes.
- app `kg_extraction.link_related_topics(store, embed_fn, related_floor=0.60, merge_floor=0.86,
  same_category_only=True)` — links same-category pairs whose cosine is in the "related" band
  [0.60, 0.86) (related but not near-duplicate). Embedding-only (no LLM), non-destructive.
- runs in `_auto_consolidate` (every extraction) and `--mode consolidate`; viz maps `related_topic` → SLOW.
- **merge stays for ≥0.86 only** (near-identical labels); the earlier gray-zone LLM-merge idea was dropped
  in favour of links.
**Verified + applied on real data:** links `hiphop ~ rap (0.678)` and `multiplication ~ math problems
(0.642)` — 2 clean links, no noise; new edge round-trips save/load.
**Note:** related links are non-destructive (both nodes kept), so a looser band is safe; lower
`related_floor` if you want more relations (e.g. jazz~r&b at 0.55).

---

## fix: consolidate every extraction + stop qwen Chinese/ChatML leak  *(branch `KG-knowledge-extraction`)*

**Two requests from a live run:**
1. Run consolidation **every extraction session** (was every 3 conversations).
2. A reply came out in **Chinese** and leaked ChatML tokens + a fake user turn
   (`…你在想什么新专辑？[CURIOUS]\n<|im_start|><|im_start|>user\nI don't really like it…`).
**Fixes:**
- `_maybe_auto_consolidate` (every-3 gate) → `_auto_consolidate`: runs topic + interest consolidation after
  **every** `_extract_session`. Dropped `_CONSOLIDATE_EVERY`.
- `LLMClient.respond`: added `stop=["<|im_start|>","<|im_end|>","\nuser",…]`, temperature 0.8→0.7, and a
  `_clean_reply()` that truncates at any leaked ChatML / next-turn marker. Prompt: "Reply in ENGLISH … output
  ONLY your single reply — never write the user's next turn."
**Verified:** `_clean_reply` cuts the exact leaked fake-turn; real-LLM replies now English + no leak, and
still use memory ("SZA's 'Open Arms'", previous-album preference).
**Note:** the qwen mid-sentence language switch is a model quirk — the stop tokens + English instruction +
lower temp make it far less likely, and the trailing fake-turn leak is always trimmed.

---

## feat: consolidation also merges near-duplicate INTERESTS  *(branch `KG-knowledge-extraction`)*

**User report:** two nodes "sports" and "sport" never merged despite auto-consolidate running every 3
conversations.
**Root cause:** they're **Interest** nodes, not Topics — `interest:jay:sports` (old LLM-chosen label) vs
`interest:jay:sport` (new: topics wire under an interest named after their category, and the enum value is
"sport"). Consolidation only merged *topics*, so the interest-level dup was never touched.
**Fix:** extend consolidation to interests.
- Pure `topics.merge_interests(canonical, duplicate)`: redirect has_interest (person→) and about (→topic)
  edges onto the canonical, delete the duplicate.
- App `kg_extraction.consolidate_interests(store, embed_fn, floor, dry_run)`: per person, embed interest
  labels, merge cosine ≥ floor groups (canonical = highest degree → shortest → lexicographic).
- webcam runs BOTH topic + interest consolidation everywhere (auto every 3 convos, `C` preview,
  `--mode consolidate`).
**Verified:** "sports"↔"sport" cosine **0.957 ≥ 0.86**; applied on real data → one "sport" interest holding
both tennis + football_player; unit test of merge_interests redirects edges and deletes the dup.
**Known limitation (deferred):** interests whose *labels* differ but mean the same (old "math" interest vs
the new "science" interest that now holds math topics) won't merge by label similarity — that's the old
LLM-label vs category-name scheme mismatch, a separate normalization task.

---

## fix: thread-safe session DB + drop mood from the prompt entirely  *(branch `KG-knowledge-extraction`)*

**Two issues from a live run:**
1. **Viz `/history` crashed** — `ThreadingHTTPServer` touched the one `sqlite3` connection from multiple
   threads ("SQLite objects created in a thread can only be used in that same thread").
2. **Replies were unnatural** — even down-weighted, the mood made the robot say two contradictory things in
   one reply (e.g. "Messi is amazing! [CONCERNED] It's okay to feel sad sometimes."). User asked to stop
   injecting mood/emotion into the prompt for now.
**Fixes:**
- `SessionStore`: open with `check_same_thread=False` + guard every DB op with a `threading.RLock` — verified
  6 threads × concurrent reads, no errors.
- Prompt: **remove the mood line and the per-turn emotion tag** from the LLM prompt entirely. Mood/emotion is
  still written to the graph/conversation node (viz unaffected). HOW-TO-REPLY simplified to "reply to what
  they said; don't comment on feelings or offer support unless they raise it." `_mood_phrase`/`_MOOD_WEIGHT`
  kept (unused) for easy re-enable later.
**Verified (real LLM):** "fav football player" → "Lionel Messi"; "what do you think about him?" → a natural
football answer — no emotional-support bolt-ons, no contradictions.
**Deferred:** improving the emotion model / re-introducing mood with better weighting; 2c; rapport/trust.

---

## fix: down-weight mood/emotion by a quarter (content over emotional support)  *(branch `KG-knowledge-extraction`)*

**Problem (user report):** even after the last fix the robot still led with emotional support and deflected
questions ("who is my fav sports player" → "I noticed you're feeling down…"), because the emotion detector
kept reading the user as sad and the prompt over-weighted it.
**Fix (reduce mood/emotion weight ~25%, per request):**
- `_MOOD_WEIGHT = 0.75`: the mood valence used in the prompt is damped ×0.75, so mild negatives fall under
  the ±0.15 threshold and read as "neutral".
- Mood line reframed from a directive ("Right now they seem low 🙁") to a weak, explicitly-unreliable
  background hint; the per-turn emotion tag likewise softened to "(weak camera hint: …)".
- HOW TO REPLY: "CONTENT FIRST — reply to what they said; mood is a faint hint, usually IGNORE it; don't
  open with or redirect to feelings, and don't offer emotional support unless they raise their feelings."
**Verified (real LLM, negative mood + Sadness):** "fav colour?" → honest "I don't remember" (on-topic);
"you should answer my questions" → "Of course, what's on your mind?"; "do you like jazz?" → jazz answer;
"fav sports player" → answers "Lionel Messi" (brief mood aside remains — expected at a quarter reduction,
no longer a deflection).
**Didn't / deferred:** fixing the upstream emotion detector reading neutral faces as sad; 2c; rapport/trust.

---

## fix: memory actually gets used in replies (retrieval + prompt tuning)  *(branch `KG-knowledge-extraction`)*

**Problem (user report):** the robot didn't use past info — asked "who's my favourite tennis player?" it
said "I'm fuzzy"; and it deflected every message into "you seem sad, let's listen to music". Data was all
present (69 embedded turns; notes with "Rafael Nadal", "SZA 'Open Arms'").
**Root causes found (by rebuilding the real prompt):** (a) the mood rule *"if they seem low, be gentle and
reassuring"* + a stuck-negative mood made the robot **console instead of answer**; (b) the notes cap (3, one
per topic, recency-sorted) **hid the specific facts** behind generic notes; (c) RAG on a meta-question
("do you remember X") retrieved other **questions**, and the block showed the robot's past replies (which
included "I don't have the name") — reinforcing forgetting.
**Fixes:**
- Prompt HOW-TO-REPLY: answer the actual question directly from memory and **state the name**; if it's NOT
  in memory, say so — **never invent a name**; only *note* mood, don't dwell/redirect.
- `_person_memory`: surface **specific** notes first (proper nouns / quoted titles score higher), then
  recency; caps raised to 8 notes / 2-per-topic. So "Rafael Nadal" and "SZA 'Open Arms'" lead.
- RAG block shows only **the person's own words** (not the robot's past replies), top_k 3→5.
**Verified with the REAL LLM + data:** "favourite tennis player?" → *Rafael Nadal*; "favourite r&b artist?"
→ *SZA*; "favourite colour?" (unknown) → *"I don't remember"* (no hallucination, no deflection).

---

## feat: RAG over transcripts + topic-click history (Phase 2)  *(branch `KG-knowledge-extraction`)*

**Tried:** use the SQLite transcripts for (a) RAG retrieval into the live prompt and (b) a viz "click a topic
→ see the conversation history", per the approved plan.
**Worked:**
- `modules/session_rag.py` (`SessionRAG`): embeds each turn once (cached in the store's `embedding` column),
  searches with FAISS `IndexFlatIP` (numpy fallback), blends similarity with **recency** and returns hits in
  **timeline order**. Lazy `reindex()`; embedding failures skipped/retried. embed_fn injected — no Ollama
  import inside the module beyond numpy/faiss.
- SessionStore gained `turns_needing_embedding / set_embedding / embedded_turns`.
- webcam: builds `SessionRAG` when embeddings are on, `reindex()`es at startup, and injects the top-3
  relevant past turns for the current message into a new prompt block "Relevant things they've said before"
  (timeline-dated).
- viz server: `HistoryProvider` + `/history?topic=&person=` endpoint (RAG when an embed model is reachable,
  else keyword `turns_for_topic`); `--sessions-db` / `--embed-model` args. Frontend: clicking a Topic node
  fetches `/history` and renders the conversation turns (child/robot bubbles + timeline).
**Verified (headless, fake embeddings):** RAG search returns the right turns; prompt gains the RAG block;
keyword history works; FAISS present (1.13.2); HTML well-formed; graph_relationship pure modules import no
session/LLM code.
**Note:** the 62 migrated turns have no embeddings/topic-tags yet, so topic-click history needs one RAG run
with Ollama up (webcam startup reindex, or the viz server with `--embed-model`) before it populates.
**Didn't / deferred:** 2c topic↔topic relations; rapport/trust; removing the now-unused "Session" legend row.

---

## feat: externalize session transcripts to SQLite (Phase 1)  *(branch `KG-knowledge-extraction`)*

**Tried:** move conversation transcripts OUT of the knowledge graph into a dedicated SQLite store so the KG
focuses on relationships/topics/interests and the viz is no longer cluttered with per-session nodes
(user approved: SQLite backend; sessions removed from graph; topic-click history via RAG comes in Phase 2).
**Worked:**
- New app-layer `modules/session_store.py` (pure stdlib sqlite3, no graph/LLM/PAD imports): one row per turn
  (session_id, person, robot, turn_idx, ts, emotion, child, reply, topics, embedding-reserved, extracted);
  `append_turn / unextracted_turns / mark_extracted / person_turn_count / session_count / turns_for_topic`.
- Pure `interactions.set_interaction_count()` so the Interaction node's count comes from the transcript DB
  instead of graph SessionNodes.
- webcam rewired: chat turns write to SQLite (not the graph); no more SessionNode/`start_session`/graph
  `append_turn`; `_extract_session` reads un-extracted turns from SQLite and `mark_extracted`s them; auto-
  consolidate cadence counts `session_store.session_count()`. `_ensure_interaction` + a per-run uuid session id
  replace the old `_ensure_session`.
- `--mode migrate-sessions`: moved the real graph's **17 sessions / 62 turns** into `sessions.db` and removed
  all SessionNodes; graph node types now: person/robot/interaction/topic/interest/conversation/persona/role/
  capability. Migrated turns are marked extracted; interaction_count preserved.
**Verified (headless, fake LLM):** store ops; extraction reads SQLite + adds typed topic + Δrapport; zero
session nodes created in the graph; re-extract idempotent; tier unaffected; migration moves turns + strips
nodes + preserves counts.
**Didn't / deferred (Phase 2):** FAISS RAG retrieval into the prompt; topic-node click → conversation history
in the viz; removing the now-unused "Session" legend row. Rapport/trust still parked.

---

## fix(kg): category enum coercion + viz spread-out force defaults  *(branch `KG-knowledge-extraction`)*

**Tried:** (1) retype the 15 pre-existing `other` topics; (2) make the graph self-spread so no manual
dragging is needed.
**Worked:**
- **Bug found + fixed:** `resolve_topic`/`merge_topics` upgraded category via `model_copy(update=...)`, which
  in Pydantic v2 does NOT re-validate — so the category was left as a plain `str` in memory (only fixed itself
  after a save/load). Now coerced to `TopicCategory(...)` explicitly. Verified in-memory type is the enum.
- Retyped all 15 existing topics (data op on `kg_state.json`, backup `.pre-retype.bak`): science
  (math/space/mars), music (jazz/r&b/hiphop/favorite songs), sport (tennis), food (baking/pasta),
  activity (hiking/camping), place (landscapes), animals (dogs). None left `other`.
- Viz force defaults tuned to spread out: charge −320→−700 (distanceMax 600), link length 90→130, link
  force 0.4→0.35, collide 28→34, centre-gravity 0→0.04. Sliders + FORCE_DEFAULTS updated to match.
**Didn't / deferred:** LLM-based retyping (used a deterministic map for the known set); 2c; rapport/trust.

---

## feat: auto-consolidate every 3 conversations + Feature-2d category colours  *(branch `KG-knowledge-extraction`)*

**Tried:** (1) auto-run topic consolidation every 3 conversations at end-of-session; (2) Feature-2d — colour
topic nodes in the viz by their category.
**Worked:**
- `_maybe_auto_consolidate()` runs inside `_extract_session` after extraction: counts total SessionNodes in
  the graph (persists across runs) and, when `count % 3 == 0`, applies `consolidate_topics` (merges). Verified
  it fires only at 3, 6, … and is a no-op without embeddings. **Design change (user-approved):** consolidation
  is no longer strictly manual — it auto-applies every 3rd conversation; the standalone `--mode consolidate`
  and `C` preview still exist.
- 2d: viz server now emits `category` on topic nodes; `index.html` tints each Topic diamond by a 10-colour
  category palette (`CATEGORY_COLOR`) and adds a "Topic category → fill" legend. Live-updates when a topic's
  category changes. HTML well-formed; transform emits category (existing topics show `other` until re-typed).
**Note:** old topics created before Feature-1 are all `category=other` (grey) until a new extraction types
them — expected.
**Didn't / deferred:** 2c topic↔topic relations; rapport/trust (still parked).

---

## feat(kg): Feature-2 semantic topic consolidation (2a + 2b)  *(branch `KG-knowledge-extraction`)*

**Tried:** merge near-duplicate topics that exact-label reuse can't catch ("hiphop"/"hip hop",
"football"/"soccer"). **2a (pure, graph_relationship):** `merge_topics(canonical, duplicate)` — redirect all
incident edges onto the canonical, union notes (+ `merged_from` marker), upgrade category only if canonical
was `other`, delete the duplicate; plus `topic_degree()` and a new pure `store.delete_node()`.
**2b (app layer, kg_extraction):** `consolidate_topics(store, embed_fn, floor=0.86, same_category_only=True,
dry_run=False)` — embed each label, pair by cosine ≥ floor, union-find groups, canonical = highest degree
(tie → shortest, then lexicographic), call the pure merge. Triggers: standalone `--mode consolidate`
(+`--dry-run`, `--merge-floor`) and an in-window `C` hotkey (dry-run preview only). **Approved scope: 2a+2b
only** — topic↔topic relations (2c) and category viz grouping (2d) deferred.
**Worked (verified, fake embed_fn):** dry-run proposes merges and writes nothing; apply merges the two
near-dup pairs, keeps distinct "jazz", 5→3 topics; canonical picks the shorter label; notes unioned with
`merged_from`; **idempotent** re-run (no further merges); **cross-category never merges** even at high
similarity; save/load round-trips; `graph_relationship/` stays free of LLM/PAD/app imports.
**Decisions:** hard-merge (redirect + delete) not alias; consolidation is **manual/reviewable**, never
auto-run during live extraction; merge floor 0.86 (stricter than the 0.62 capability floor); `C` is
preview-only (apply via `--mode consolidate`).
**Didn't / deferred:** topic↔topic relations, clustering, category-based viz grouping (revisit after this),
and any change to rapport/trust (still deferred).

---

## feat(kg): fine-grained topic typing + graph-aware extraction  *(branch `KG-knowledge-extraction`)*

**Tried:** two improvements to LLM knowledge extraction. **Step 1** — `TopicNode` gains a `category` from a
CLOSED taxonomy (`TopicCategory`: music/science/animals/food/activity/place/person/media/sport/other).
**Step 2** — condition the extraction prompt on the person's *existing* topics so the LLM reuses established
nodes; output splits into `existing_topics_discussed` vs `new_topics`. Kept decoupling: all LLM/prompt/guard
logic in the new APP module `modules/kg_extraction.py`; `graph_relationship/` gained only pure helpers.
**Worked (all verification points):**
- category enum defined once; **TopicNode id stays label-only** (category is an attribute, not identity —
  two extractions disagreeing on category resolve to the SAME node).
- backward-compat: old `kg_state.json` untyped topics load and default to `other` (real file: 14 topics).
- graph-aware reuse: with "jazz" known, a transcript saying "jazz music" lands in
  `existing_topics_discussed` and creates **no** second node (before==after counts).
- new topic ("dinosaurs") → one typed `TopicNode(animals)` wired via the Interest layer (category→interest).
- guards write **nothing** on: malformed JSON (whole extraction discarded), invalid category (dropped),
  hallucinated "existing" not in the provided list (dropped), confidence < 0.6 (dropped).
- idempotent: re-running identical extraction gives identical node/edge counts.
- category round-trips through save→load→save; `graph_relationship/` has **zero** LLM/PAD imports.
**Decisions / deviations (flagged):**
- Invalid category → **drop** the item (not coerce), so "nothing written" holds for bad output.
- Closeness (rapport/trust) kept working by reusing the existing pure `extract()` for **deltas only** +
  the untouched `adjust_closeness` (its interest logic is not used). Closeness logic itself untouched.
- New/existing topics wire under an Interest named after the **category** (`person→Interest(category)→Topic`).
- `resolve_topic(category)` only fills a category when the node is still `other` (first non-other wins;
  TopicNode has no provenance field, so a conflict is not persisted — kept, not merged).
- Capability↔topic auto-linking (old embedding matcher path) is **not** run in the new topic extraction —
  embeddings/merge are explicitly out of scope for this step.
**Didn't / deferred:** embeddings, fuzzy/semantic merge, topic↔topic relations, clustering (Feature-2).

---

## docs: R&D system report + progress log

**Tried:** wrote a detailed R&D report (`RND_KG_Companion_System.md`) covering face-reco, emotion, the
FAST/SLOW/RELATIONSHIP graph, extraction, prompt structure, and pipeline; started this progress log.
**Worked:** report captures the current baseline (PAD disabled, emotion→mood only) accurately.
**Didn't / open:** no evaluation numbers yet; references + abstract still to add.
**Next:** improve the knowledge-extraction method (see report §7).

---

## `a31f30a` feat(viz): colour edges per person; chatbox edges blue

**Tried:** colour each person's edges with a distinct hue, shaded by timescale (FAST lighter / SLOW
darker), and force all robot (chatbox) edges to a single blue. Edge ownership inferred from the source
node id (`person` / `interest:` / `conversation:` / `interaction:` / `*:capability` / robot).
**Worked:** verified on the live graph — jay's 31 edges → jay hue, HJ's 10 → HJ hue, chatbox's 8 → blue,
0 unowned. Legend lists each person's colour dynamically.
**Didn't / watch:** blue is reserved for the robot and excluded from the person palette; if many people
are added the 8-colour palette will wrap (acceptable for now).

---

## `d093e00` feat(viz): Obsidian-style force sliders

**Tried:** top-right panel with live sliders — Repel (charge), Link length (distance), Link force
(strength), Center gravity (forceX/Y) — plus reset.
**Worked:** sliders drive the d3 sim live; centre-gravity re-centres on window resize; HTML well-formed.
**Didn't:** —

---

## `8eedb5f` feat: live conversation-status node + emotion/mood-aware prompt

**Tried:** (a) live "current topic" tracking; (b) emotion → FAST mood; (c) restructured, retrieval-augmented
prompt with affect.
**Worked:**
- Dedicated `ConversationNode` (rolling last-3 topic keywords + mood + emotion, linked to person **and**
  robot) — updates in place, verified via unit check.
- Prompt rebuilt into 3 blocks (IDENTITY / HOW TO REPLY / WHO YOU'RE TALKING TO); **dual affect signal**
  (mood valence in context + emotion label tagged on the current user turn); memory capped (top-4
  interests, ≤3 topics each, 3 recent notes one-per-topic).
- Raised embedding matcher floor 0.50 → 0.62.
**Didn't work → fixed (found during live testing):**
- *Save spam* returned once emotion was on — the raw detector valence jitters and kept tripping the
  0.04 dirty-gate. Fixed by widening the gate (save only on emotion-label change or ≥0.15 valence shift).
- *Spurious capability link* `tennis ↔ "good at math"`: the first design reused shared `TopicNode`s for
  the current topic, so the extraction embedding matcher attached bogus `about` edges. Root-caused and
  replaced with the dedicated `ConversationNode` (structurally cannot receive capability edges).
- *Accumulated artifacts*: one-time cleanup script purged old `current_topic` edges, orphaned
  current-topic topics, and non-keyword capability→topic links from `kg_state.json` (backup kept).
**Note:** `CurrentTopicEdge`/`set_current_topic` are now dead (kept in schema, unused).

---

## `aaa402d` feat: integrate webcam face-reco with the KG conversation pipeline

**Tried:** wire `webcam_loop` into the `graph_relationship` KG — recognize → retrieve into the prompt →
record turns → end-of-session extraction → update graph. PAD + emotion disabled by default; embedding
matcher default on.
**Worked:** full loop verified with a headless fake-LLM smoke test (seed → session → turns → extract →
graph update); seeds robot/human subgraphs from `specs/` on startup; retrieves interests / shared topics /
notes into the system prompt; records real turns onto `SessionNode.turns`.
**Didn't work → fixed:**
- *Crash on quit* `'Event' object is not callable`: `_DetectionWorker` stored its stop flag as
  `self._stop`, shadowing `threading.Thread._stop` (called by `join()`); surfaced once emotion was
  disabled and the worker finished fast. Renamed to `_stop_evt`.
- *Save spam* (identical snapshots each tick): added dirty-gating so KG-only ticks don't rewrite
  unchanged graphs.

## Unified OpenCV face detection (shared by face-reco + emotion)

**Goal:** one face-detection pass feeds both identity and emotion, using OpenCV.

- **Tried:** added a `detector` mode to `FaceIdentifier` (`face_id.py`) — `"opencv"`
  (new default) vs `"mtcnn"`. In opencv mode a single Haar `detectMultiScale` locates
  every face; each box is cropped (with a ~20% margin to mimic MTCNN's) and embedded by
  the *same* InceptionResnetV1 (facenet fixed-image-standardization) for identity, and the
  same box is handed to the emotion detector. MTCNN is not loaded in opencv mode.
- **Worked:** `identify_all` returns the shared Haar boxes → the detection worker already
  passes each box to emotion, so face-reco + emotion now share ONE detection. `--detector
  {opencv,mtcnn}` on the loop + enroll mode. 12/12 face/culture tests pass; embeddings stay
  L2-normalised (self-sim=1).
- **Didn't / caveat:** Haar boxes are unaligned vs MTCNN's landmark alignment, so recognition
  is a bit looser — **re-enroll people after switching modes** (a face enrolled under MTCNN
  matches poorly against an opencv-cropped probe). `--detector mtcnn` restores the old path.

## Sampled face identity (sample-and-hold) — stop 24/7 re-identification flicker

**Goal:** face recognition flickered (jay ↔ unknown frame-to-frame) and ran every
frame. Don't check identity 24/7 — check periodically and hold a confirmed result.

- **Tried:** duty cycle in `_DetectionWorker`. SAMPLE window (`--id-sample`, 1 s):
  identify every frame and vote on the primary (largest) face. Confirm the identity
  only if it was recognised in >= `--id-confirm` (0.6) of the sampled frames. Then
  HOLD (`--id-interval`, 3 s): detect boxes only (new `FaceIdentifier.detect_boxes`,
  no embedding) for emotion/display, and reuse the confirmed identity — never
  re-identify. Displayed label is always the held identity → zero flicker.
- **Worked:** helpers unit-tested (primary pick, relabel, boxes→raw, vote ratio
  confirm/reject); 12/12 face+culture tests still pass; opencv & mtcnn both get a
  box-only path. Big CPU saving too (embedding runs ~1 s in 4 instead of every frame).
- **Caveat / trade-off:** during a HOLD window the largest box is labelled with the
  held identity, so if a *different* person appears mid-hold they're mislabelled for
  up to `--id-interval` s until the next sample corrects it. Tune interval down if
  that matters. Multi-face identity is primary-focused (non-primary faces show as
  unknown during hold).

## Angle-robust, self-improving face recognition (multi-view gallery + adaptive capture)

**Problem:** recognition only worked at one head angle and flickered — the label dropped whenever
the person turned or looked away. Four separate causes, not one.

- **Tried / found (diagnosis first):**
  1. *One averaged prototype per person.* `faces.npz` held `jay = (1,512), counts=[50]` — 50 frames
     collapsed into ONE centroid. Averaging across poses matches no pose well.
  2. *Detection, not recognition.* The default `opencv` detector was the FRONTAL-only Haar cascade,
     so a turned head produced **no box at all**. `haarcascade_profileface.xml` measured **2.6 ms**
     (vs 12.7 ms frontal) — essentially free.
  3. *A real bug in the sample-and-hold code from the previous commit*: `_id_samples += 1` counted
     frames with NO face, inflating the vote denominator so looking away pushed the ratio under
     `id_confirm_ratio` and dropped a good identity.
  4. *Binary threshold* (0.75) with no hysteresis — an off-angle dip flipped the label instantly.

- **Worked:**
  - **Phase 0:** vote only counts frames containing a face; HOLD-phase boxes report `sim=-1` and
    render as `held` (they were fabricating `1.00`, which would have poisoned any calibration);
    `--id-debug` trace.
  - **Phase 1:** frontal + profile + mirrored-profile cascades, deduped by IoU **and**
    centre-containment (offset frontal/profile boxes on one head sit at IoU 0.25-0.35 and slip past
    pure IoU → phantom "unknown" face, which would also silently disable adaptive capture);
    source-aware crop margins; `_embed_opencv` now detects at the same 0.5 scale as `identify_all`.
  - **Phase 2:** `_protos`/`_meta` multi-view gallery (K=12 enrolled + 6 learned), merge-vs-insert on
    `tau_dup=0.92`, farthest-point eviction, `retain`/`acquire` (top-2) scoring, concat-not-average
    rename, npz **schema 2** with automatic schema-1 migration (jay survived losslessly).
  - **Phase 3:** acquire 0.75 / retain 0.62 / switch-margin 0.08 hysteresis in `_match`, applied to
    the largest face only; worker miss-grace of 2 windows, but instant drop on a rival or empty frame.
  - **Phase 4:** adaptive capture — learns a new view only from a confirmed, single-face, unanimous,
    quality-checked window, rate-limited 20 s/person and 20/session, in `[adapt_floor, tau_dup)` with
    `adapt_floor` defaulting to `threshold` (conservative). Worker queues, main thread applies →
    gallery keeps a single writer. `--reset-adaptive` undoes it; learned views can never displace
    enrolled ones.
  - **Phase 5:** guided five-pose enrollment (CLI + in-window E-key), which REJECTS frames that
    repeat a stored view — otherwise ignoring the prompts silently re-records the front.
  - `--faces-info` prints each person's views + closest-pair similarity ("only ONE view" diagnosis).
  - Tests: new `test_face_multiview.py` (16 cases incl. the centroid-vs-multi-view regression and 7
    poisoning cases the adaptive gate must refuse); `test_face_rename` updated for concat semantics.
    **68/68 repo tests green.** Integration-smoked the worker end-to-end with a stubbed detector:
    confirm → hold through an off-angle turn → learn the new view.

- **Didn't / watch:** thresholds are priors, NOT calibrated on real faces — run `--id-debug` through
  0/±30/±60/±90/chin-up/chin-down and tune `--retain-threshold` (10th pct of ±45 sims) and
  `--proto-dup` (40th pct of consecutive still-frame sims). With only ONE person enrolled nothing
  competes, so recall gains also raise false accepts — enrolling a decoy identity would turn the
  absolute threshold into a relative one. jay still has a single enrolled view until re-enrolled
  with the guided poses. Emotion now receives profile crops it was not validated on (emotion is off
  by default).

## 8.0 — reconciliation of the mood-decay and trust/disclosure claims (read-only)

**Tried.** Re-verified two claims from the original audits against the current HEAD of
every branch that has the code, rather than trusting the earlier summaries.

**Worked.** *Fix 2 (FAST-tier mood decay): CONFIRMED.* `_session_start` (kg_bridge.py:257),
`_written_since` (kg_bridge.py:194-205) and the `pre_turn` gate (kg_bridge.py:309-314) are
all present and identical on `feature/pad-affect-core` and `feature/padeval-phase3`.
`store.save`/`load` (store.py:309-339) remain unfiltered — by design, since the fix is a
read-side session gate, not a write-side filter.

*Fix 3 (rapport/trust decoupling): PARTIALLY DONE.* The decoupling is real — independent
`d_rapport`/`d_trust` (webcam_loop.py:711-747), live tick moves rapport only
(webcam_loop.py:1584-1585). But `disclosure_depth` is **not a real field**: it appears only
in two comments (schema.py:10, store.py:64), there is no `DisclosureEdge` among schema.py's
16 edge classes, and `_RELATIONSHIP_TYPES` is `{rapport, trust, interaction_count}`
(store.py:49-51). The only non-manual write path to trust was the end-of-session LLM
(extraction.py:39-50), so "disclosure-gated" meant "LLM-judged".

**Didn't.** No merge gap: hardware runs `feature/pad-affect-core`, the same branch that
carries both fixes. `main`/`origin/main` lack them, but also lack the whole
`modules/graph_relationship/` tree, so they were never a deployment candidate.

## 8a — a rule-based disclosure detector (`padeval/coding/disclosure.py`)

**Tried.** Replace the LLM trust judgement with a deterministic detector, following the
Phase 4a rule-coder discipline: authored cases first, zero model calls, every list
hand-written.

**Worked.** All four registered predictions held. The organising principle — *disclosure is
a first-person statement carrying information the camera does not already have* — does real
work: it is why "I'm happy" is excluded (the emotion head already has it; counting it would
launder valence into trust and undo `2be86bb`) while "I was scared when the lights went out"
is included. Held out against the 32 E1 stimuli, never seen during development: κ=0.739,
precision 0.857, recall 0.750, flagging 6/8 disclosure and 0/8 question, 0/8 quiet.
The divergence test now *exhibits* rapport and trust separating at the scripted disclosure
turn, where before the claim rested on the type signature alone. 73/73 padeval tests pass.

**Didn't.** The 100% on the 38 authored cases is spec-conformance, not evidence, and is
labelled that way — the set was patched against. The two held-out misses (past-tense action
verbs: "I drew…", "I fell…") were deliberately **left unfixed**; patching `FACT_VERBS`
against the held-out set would turn it into a second training set and destroy the only
unbiased number in the report. Not yet wired into the live loop — the detector is built to
be wireable (`turn_deltas` reproduces the deployed rapport rule verbatim) but hardware runs
this branch, so the live edit is held as a separate decision.

## 8b — end-to-end system trace, 4 sessions, both robots, zero LLM

**Tried.** Run the deployed mechanism forward over a scripted multi-session schedule and log
every established quantity per turn, to check whether the properties proved component-wise in
6/7 survive composition. Four predictions registered before running.

**Worked.** P2, P3, P4 all held. The mood gate fired on 3/3 session-opening turns and none
mid-session, shown as a *difference* against a gate-disabled run of the same seed (+0.0925,
−0.0348, +0.0531 valence) rather than asserted from a flag. Rapport and trust diverged at
exactly the scripted disclosure turn on both robots, ending 1.0000 vs 0.2000. CHATBOX clamped
at exactly `{unknown}` — the tier computed from 7b's bound *before* reading the trace — losing
0.043 of commanded D, while ELLEBOT clamped nowhere. 82/82 padeval tests pass.

**Didn't.** **P1 was falsified**: D steps *within* a session on both robots. `pre_turn`
re-derives the tier every turn and neither threshold kind is session-scoped — CHATBOX shows
the count-gated case (`count > 0 → visitor` counts turns across all sessions), ELLEBOT the
score-gated one (rapport hit 0.96 mid-session and crossed `score > 0.45`). Reporting CHATBOX
alone would have looked like a tidy first-contact exception; ELLEBOT is what shows the finding
is general. The paper must say "D changes on the order of sessions", not "only at session
boundaries".

Also found by recomputing 6a's τ_D under the 8a mechanism: at the median session length τ_D
rose 3 → 4 sessions and the 10² separation *improved*, but **6a's hard floor is gone**. That
floor came from the extractor's ±0.2-per-session cap, which 8a removed; at the IQR upper bound
(4 turns/session) `close` now arrives in 2 sessions. The median alone would have hidden this,
so the test asserts the regression explicitly rather than sampling a favourable parameter. Fix
is a one-line per-session trust cap — flagged, not applied, since it would change the accrual
mechanism a third time.

## 8c — chasing P1's falsification to the bottom, and restoring the floor

**Tried.** P1 failing raised a sharper question than "soften a sentence": if D moves
mid-session, is the old per-tick rapport mechanism still uncapped, and could one long cheerful
session sprint the ladder the way the original 48-second bug did? Measured directly rather
than reasoned about — 500 turns, one unbroken session, maximally warm face, zero disclosure.

**Worked.** Two-part answer. (a) It **cannot** reach `close`: rapport saturates at 1.0 with
trust at 0, so score tops out at 0.50 against a 0.70 threshold — the old bug is fixed by
*arithmetic*, trust being a required second term a face cannot supply, not by a tuned rate.
(b) It **does** sprint the lower rungs: `unknown → visitor → known` in 60–80 s of smiling,
half the ladder in under two minutes. So the honest claim is "D is slow where trust gates it
and fast where rapport alone does", which also explains P1's mid-session steps exactly.

Separately, swept sessions-to-close across session length instead of sampling the median, and
found the uncapped 8a rule reaches `close` in **one** session at ≥8 turns — worse than the
2-session figure reported earlier, which came from testing only the IQR upper bound. Added
`SESSION_TRUST_CAP = 0.20` via a per-session `SessionAccrual`; the floor is now 3 and **flat**
in session length (verified to 128 turns/session). The cap restores the LLM extractor's own
±0.2 clamp rather than introducing a new tuning knob, so the floor returns to 6a's value.
Held-out κ now reported with its interval: 0.739, 95% CI [0.389, 1.000] — wide, and quoted as
wide, since n=32 with 8 positives cannot support a point estimate. 85/85 padeval tests pass.

**Didn't.** Rapport is deliberately left **uncapped** — it tops out at score 0.50 and cannot
reach `close`, so capping it is a separate design decision, not a defect fix. Still not wired
into the live loop, per the standing decision to keep hardware off an unverified accrual
change; both open items (the cap and the rapport question) are now resolved and reverified, so
wiring is unblocked whenever wanted.
