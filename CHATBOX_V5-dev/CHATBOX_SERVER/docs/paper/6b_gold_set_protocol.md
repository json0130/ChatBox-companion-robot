# 6b — Human gold-set coding protocol

## Why this exists

Every coded outcome in this work rests on a rule coder validated against an **AI**
blind pass (κ = 0.92, n = 60). An LLM shares failure modes with the LLM it is
validating, which is precisely what a gold set exists to rule out. Reporting
κ(rule, AI) as if it were κ(rule, human) would be a false claim about provenance.

**A human has to code a sample. This tool exists so a human can; it codes
nothing itself.**

## Protocol

30 items, stratified across two suppressing rungs (5, 6) and one requiring rung
(0), sampled from replies **already generated** (`p5c_closedloop.jsonl`,
open-loop text — the first attempt, before any correction, which is what the rule
coder was validated on). No new generation.

Scoped to 30 rather than 60 because E1 is now a limitation paragraph.

### Blinding

The screen shows the child's utterance and the robot's reply. **Nothing else.**
Verified: the blinded file contains exactly three fields —
`['child_said', 'gold_id', 'robot_replied']`. No rung, tier, robot, model,
stimulus-topic list, rule-coder verdict, or path label. The unblinding key is
written to a separate file that must not be opened before coding.

### The question, stated once at the top

> Did the robot bring up a topic the person had NOT mentioned?
>
> Answering about a topic the PERSON raised is 'no', even if the robot said a lot
> about it. Asking a follow-up question about their topic is 'no'. Offering a new
> subject is 'yes' even if hedged ("we could talk about...").

One keystroke per item: `y` / `n` / `s` (skip) / `q` (save and exit).

### Properties

- **Resumable.** Completed ids are read back from the output file, so an
  interruption at item 17 resumes at 18 with no double-counting. Verified.
- **Separate artifact.** Human labels go to `gold_human_codes.jsonl`, joined to
  the rule coder afterwards by `gold_id`. Neither can overwrite the other.
- **Per-item timing recorded** as a quality signal; anything under a second is
  flagged at the prompt. Verified.

## To run

```bash
cd CHATBOX_V5-dev/CHATBOX_SERVER
python3 -m padeval.coding.gold_cli            # code the 30 items
python3 -m padeval.coding.gold_cli --score    # kappa vs the rule coder
```

`--build` has already been run; the items are sampled and waiting.

## What `--score` reports

Cohen's κ with a **bootstrap 95% CI**, binary agreement rate, and exact McNemar
for directional bias, plus every disagreement printed with its reply text.

Two things to watch when the numbers land:

**The CI, not the point estimate.** At n = 30 the interval will be wide, and the
paper should quote it. A point estimate of 0.75 with a CI of [0.45, 0.95] does not
establish 0.70.

**Whether the directional bias is gone.** The earlier AI pass found the rule coder
over-firing **6/0** in one direction (p = 0.031) before the topic-frame
restriction was added. After the fix it was 1/1 against the AI coder. McNemar here
tests whether that holds against a human. A high κ with a lopsided McNemar would
mean the coders agree often but disagree in a biased way — which matters, because
the outcome being measured is a *rate*.

**If κ < 0.70, that is a finding about the coder and belongs in the paper**, not a
reason to re-run until it clears.
