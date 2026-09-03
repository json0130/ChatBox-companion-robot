# 9 — Pre-deploy regression check, and one open decision

## Pre-deploy: does the style wire output change?

Before updating the server on the CHATBOX rig. Same check performed once for `dc87d93`,
repeated here rather than assuming "the signal is pretty much the same".

**Probe.** A fixed set of 40 PAD inputs — both robots × all four tiers × five
representative face readings (neutral, warm-high-arousal, the `sad` corner, the `happy`
corner, and the hostile-high-arousal corner from `affect.CATEGORY_VA`) — pushed through
`affect.feel_with_relationship` → `affect.gesture_style`, plus both baselines. Values
captured as `repr()` of the float, so the comparison is **bit-exact, not rounded**.

**Method.** Each candidate commit checked out into a detached worktree (the shared directory
is never touched) and the same probe script run against it.

| commit | what it is | style output sha256 (first 16) |
|---|---|---|
| `2be86bb` | the rapport/trust decoupling fix | `4e3813100cfca45f` |
| `dc87d93` | client-link, the previously-checked point | `4e3813100cfca45f` |
| `0eb529f` | last commit touching runtime code at all | `4e3813100cfca45f` |
| `a6203d3` | last push before Phase 8 | `4e3813100cfca45f` |
| `de41f75` | **HEAD, the candidate for deploy** | `4e3813100cfca45f` |

**Result: byte-identical across every candidate.** `diff` between the earliest and HEAD is
empty.

Corroborated independently of the probe: `git diff 0eb529f..HEAD` over `PAD_CORE/`,
`modules/` and `tools/` is **empty**. Everything on this branch since `0eb529f` lands in
`padeval/` (24 files), `docs/` (9) and `requirements-eval.txt` (1) — none of which the robot
imports. The runtime the rig would receive is the runtime it is already running.

**Deploy verdict: no behavioural change to the style path. Safe on this evidence.** The one
caveat worth stating: this probe covers the PAD → style wire output, which is what was asked
for. It does not exercise the LLM prompt path, the KG persistence path, or the client
transport — those were unchanged by inspection (`git diff` empty over `modules/`) but not
independently probed.

## 9b — Open decision: should rapport also be session-capped?

**Not implemented.** This is the fourth time this quantity's accrual has been in question, and
the decision is the user's, as it was the previous three times.

### The tradeoff, in 9a's numbers

| | rapport **uncapped** (today) | rapport capped at 0.20/session |
|---|---|---|
| `unknown → visitor` | 1 turn | 1 turn (count-gated, unaffected) |
| `visitor → known` (warmth path) | **~50 s** | ~3 sessions |
| `visitor → known` (volume path) | 6 turns | 6 turns (count-gated, unaffected) |
| `known → close` | 3 sessions | 3 sessions (trust-bound, unchanged) |
| the claim you can make | "D is slow where trust gates it, fast where rapport alone does" | "D moves on the order of sessions" |

**For capping.** It restores a single, uniform claim. The current two-part statement is
honest but awkward to defend in a paper: a reviewer reading "the relationship axis is the
slow signal" and then finding half the ladder climbable in under a minute will not be
reassured by the distinction, even though the distinction is real. Capping also makes the
mechanism symmetric — both closeness channels bounded the same way, by the same constant,
for the same stated reason.

**Against capping.** Rapport is *provably* incapable of reaching `close` on its own (9a: it
tops out at score 0.50 against a 0.70 threshold, at any session length and any pacing). So
capping it fixes no defect — it changes a design property that is currently working as
specified. It would also make the robot feel unresponsive in exactly the situation the
per-tick accrual was written for: a child who is visibly enjoying the interaction *right now*
would see no relational movement for several sessions. And `visitor → known` would then be
reachable **only** by the volume path in practice, which would make `count > 5` the sole
live route to half the ladder — arguably worse than the current position.

**What would not change either way:** the `close` bound, since it is trust-limited and 9a
shows the optimum is the corner `rapport = 1.0` regardless of how fast rapport gets there.

### If told to proceed

One commit, all three together, since they cannot be correct separately:
`SessionAccrual(rapport_cap=0.20)` (the mechanism already exists and is tested), 9a's table
regenerated under the new rule, and 8b's trace regenerated. Until then rapport is unchanged.

## 9.0 — Gold set, ready for a human

Built and waiting. **Not coded by me** — the point of a gold set is a human.

```
cd CHATBOX_V5-dev/CHATBOX_SERVER
python3 -m padeval.coding.disclosure_gold_cli          # code the 71 items
python3 -m padeval.coding.disclosure_gold_cli --score  # kappa vs the detector
```

Resumable — quit with `q` and rerun to continue. ~5–8 minutes at a normal pace.

**Provenance, which is the part that matters.** The items are all 71 unique non-empty child
utterances in `sessions.db` — real development and test traffic, typed and spoken at the
running system, ASR garble included. None was used to build, tune or validate the detector.

**Every one is included; there is no sampling step.** That is stronger than sampling
carefully: a stratified draw would have to stratify on the detector's own output, since that
is the only signal available, and validating a detector against a sample it helped select is
the exact circularity a gold set exists to break. Taking the whole pool removes the question.
The base rate will be whatever the traffic contains, which is also correct — a deployment's
base rate is not the evaluator's to choose.

Blinding is asserted by test: the item file carries `gold_id` and `child_said` and nothing
else, and the test also re-derives the pool from the database to confirm no selection step
crept in.

Target: tighten 8a's κ = 0.739, 95% CI **[0.389, 1.000]** (n = 32, 8 positives). The interval
is wide because the sample is small; more items is the only fix.
