# 10 — Disclosure detector: what the gold set found, and what it cost to fix

## The three numbers, together

Reporting one of these without the others would be misleading, so none appears alone.

| | κ | 95% CI | n | coverage | McNemar b/c |
|---|---|---|---|---|---|
| **Pre-fix — the honest headline** | **0.128** | **[−0.099, 0.363]** | 64 | 100% | 17/6, p = 0.035 |
| Post-fix — **in-sample, not independently validated** | 0.734 | [0.532, 0.894] | 59 | **92.2%** | 6/1, p = 0.125 |
| *Sensitivity: Group 3 items removed* | *0.914* | *[0.774, 1.000]* | *54* | *84.4%* | *1/1, p = 1.0* |

**The pre-fix number is the measurement.** κ = 0.128 with a CI spanning zero, on 64 items of
real traffic the detector had never seen. It failed the 0.70 gate and that is the finding.
8a's κ = 0.739 came from 32 authored stimuli; the first contact with real utterances cost
0.61 of kappa, which is what a genuinely held-out sample is *for*.

**The post-fix number is not a validation and is not presented as one.** The same 71
utterances that exposed the bugs were used to confirm the fixes. `sessions.db` holds no more
real traffic, so no independent re-test is available and none is claimed. It says the known
defects are gone; it cannot say what remains unknown.

**Coverage is reported with κ because κ on a subset means nothing without it.** A detector
abstaining on 80% and scoring 0.9 on the remainder is not better than one scoring 0.5 at full
coverage. 92.2% is the fraction actually called.

**The McNemar check matters as much as κ.** Pre-fix skew was 17/6 toward false negatives — the
safe direction for a trust signal, since a miss only slows accrual while a false positive lets
trust drift up on ordinary chat and re-collapse onto rapport (9c's lemma). Post-fix is **6/1,
still false-negative skewed**. The fixes did **not** flip it toward the unsafe direction, which
was the specific risk of adding three new firing routes in 10b.

## Why 10a and 10b are spec corrections, not curve-fitting

A reviewer should assume a coder touched after seeing its validation set was tuned to it. The
argument that this was not:

**10a — the guard already existed and already worked.** The spec ("a question requests
information; it cannot supply it") predates the gold set, and 8a measured **0/8 on
question-form stimuli**. The guard was demonstrably correct on every form it had been given.
The gold set supplied three forms it had never seen: contracted interrogatives (`what's`),
leading discourse markers (`so what's...`), and one code path — multi-word phrase matching on
the raw utterance — that ran *before* the clause loop and structurally could not consult the
guard at all. Fixing an evasion of an existing rule is not fitting a new one.

**10b — the brief's hypothesis was wrong, and checking it is what found the real cause.** The
brief expected an exclusion built for "I'm happy" to have grown to match the whole
`I am [predicate]` frame. Checked before changing anything: `"i am maori"`, `"i am hj"`,
`"i do some research on space"` all produced **`hits=[]`** — nothing matched *and nothing was
suppressed*. There was no over-broad exclusion. Three categories of camera-invisible
self-report — identity, activity, opinion — had **no route at all**. The written spec covers
them and always did; the rules were incomplete against it.

The distinction is testable rather than rhetorical: the 38 authored cases from 8a still pass
at 100% with zero false positives, and the added routes stay silent on `"i think so"`,
`"i am sure"`, `"i am here"` and `"i don't know"`.

## Group 3, as published before recoding

The definition, settled in writing first and pinned by
`test_the_group_3_definition_as_published`:

> **Present-tense affect is camera-redundant and does not disclose. The same affect anchored
> to another time does, because the camera saw an instant of it at most.**

| item | published verdict | human said | why |
|---|---|---|---|
| `"i am doing good"` / `"I am doing well..."` / `"I am doing great"` / `"i am doing good you?"` | not disclosed | DISCLOSED | present-tense general valence — the FER head already reports it |
| `"actually i am not doing well..."` | not disclosed | DISCLOSED | the valence-contradiction case, **deferred** (below) |
| `"i am really sad today"` | **disclosed** | DISCLOSED | `today` spans hours the camera did not see |
| `"I'm feeling sad today. Just had a bad day."` | **disclosed** | DISCLOSED | temporal anchor |
| `"i felt awful yesterday"` | **disclosed** | DISCLOSED | temporal anchor |

**Five of the seven residual disagreements are these items** — a definition disagreement, not
a detector error. Removing them lifts κ from 0.734 to **0.914 [0.774, 1.000]** with McNemar
1/1, which locates the remaining gap precisely: outside Group 3 only **two** items disagree in
59 (`DG021 "i guess space?"`, a false positive; `DG024`, a fragment).

That the human consistently labelled `"I am doing well"` as disclosure is worth recording
rather than dismissing. It is a real signal that the camera-redundancy principle, however
well it serves the disjointness argument, does not match untrained intuition about what
"telling someone about yourself" means. The principle is kept because 9c's lemma requires it —
a trust channel that fires on what the camera already reports is rapport wearing a second name —
but the divergence is a limitation to state in the paper, not a coding error to correct.

## Deferred: the valence-contradiction rule

**Not implemented, and the reason is not schedule pressure.** Checked whether FER valence is
available where disclosure is evaluated: it is not. `turn_deltas` receives `felt_pleasure` —
the robot's *blended* coordinate, 40% its own temperament via `EMPATHY = 0.6` plus graph mood.
Raw camera valence exists at `webcam_loop.py:1564` and is never passed down. Comparing a
child's stated valence against the robot's own feelings would be the wrong comparison, so this
needs new plumbing rather than a new rule.

**The residual argument, recorded for that work:** trust would read the *disagreement* between
stated and observed valence — precisely what rapport structurally cannot see, since rapport is
a function of observed valence alone. Disjointness (9c) therefore still holds, so the extension
is principled rather than convenient. `state_facial_present_ignored` is logged on every
affected utterance so the candidates are already identified.

## Abstain (10d)

Three-way outcome; abstain accrues no trust. Fired on **5 of 64** items:

| item | text | reason |
|---|---|---|
| DG009, DG012 | `"then what aobut me?"` | subjectless fragment |
| DG023 | `"like spaceship assembly"` | subjectless fragment — human called it DISCLOSED |
| DG053 | `"goodnight chatbox"` | subjectless fragment |
| DG063 | `"about it and then look what happened"` | subjectless fragment |

Only one abstention (DG023) was a disclosure the human saw — a fragment continuing a topic
from the previous turn, which the detector has no subject to attach to. Four of five were
items the human also called not-disclosed, so abstention is not being used to dodge the hard
positives.

`A3` was specified and **dropped**: it would have abstained whenever an affect term was
suppressed as camera-redundant, but that relitigates a rule Group 3 settles as a *decision*,
and where another route fires the ambiguity changes nothing.

## Scope of the 71-item set

Not curated. The pool was taken whole in 9.0 precisely so it could not be selected against
detector output, and **no exclusion rule was applied after seeing which items failed**. The 7
excluded items are the human's own `[s]` skips, recorded during blind coding before any of
these fixes existed.
