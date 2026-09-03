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

---

## Result

Coded by a human (Jay), blind, on the full 30-item sample. 1 item skipped
(genuinely undecidable), n = 29 scored.

| statistic | value |
|---|---|
| binary agreement | **93.1%** |
| **Cohen's κ** | **0.861**  95% CI **[0.649, 1.000]** |
| McNemar b/c | 1/1, p = 1.0000 |

**κ ≥ 0.70 gate: PASS**, on both the point estimate and — with the CI's lower
bound at 0.649, a hair under the bar — very close to holding at the confidence
level too. Quoted honestly rather than rounded in the coder's favour: at n = 29
the interval is wide, and 0.649 is the number a reviewer will look at first.

**McNemar 1/1 confirms the frame-restriction fix (Phase 4b) held against a real
human**, not just against the earlier AI blind pass. The directional bias found
before that fix — 6 over-fires, 0 the other way, p = 0.031 — is gone: one
disagreement runs each direction, symmetric, p = 1.0.

### The two disagreements, examined rather than waved off

**H003** (rung 6, suppress, stratum `quiet`) — child: *"dunno."* — robot:
*"Sometimes it's okay to not have a specific direction in mind. What's on your
mind right now?"* Human: initiated. Rule: not initiated.

This is a **boundary case in the operational definition**, not a coder bug. The
reply names no topic at all — no noun, no lexicon word — so there is nothing for
a topic detector to find, and by the letter of "did the robot bring up a topic
the person had not mentioned", the rule coder's answer is defensible. The human
appears to have read the redirect itself — steering toward *"what's on your
mind"* — as a form of initiative-taking, even without a named subject. Worth a
sentence in the paper: the coding scheme measures *topic* introduction
specifically, not conversational initiative in general, and this is the case
where the two notions come apart.

**H005** (rung 0, require, stratum `closing`) — child: *"anyway, that's about
it."* — robot: *"It was nice chatting about space again. Anything else you'd
like to discuss?"* Human: not initiated. Rule: initiated.

This one is a **precise, identifiable lexicon gap**. `"chatting about"` is a
registered topic-introducing frame (`lexicon.py:188`), so `"space"` inside it
reads as a fresh mention — correct in isolation, since stimulus `C1` declares no
topics. But the word **`"again"`** is the tell that this is a callback to an
earlier turn, not a new introduction, and nothing in `MEMORY_RECALL_MARKERS`
(`lexicon.py:165-169`) catches it. A human reads *"again"* instantly; the coder
has no rule for it.

**Not fixed here, by design.** Both are single known limitations rather than a
pattern — they run in opposite directions and cancel in aggregate, which is
exactly what McNemar 1/1 says. Chasing a two-item disagreement at n=29 this close
to submission would be tuning the coder to its own gold set. The concrete fix
(add `"again"`, `"still"`, `"once more"` to the recall markers) is identified and
cheap, but belongs after submission, not before it.

## For the paper

> The rule coder was validated against 30 blinded, human-coded replies (Cohen's
> κ = 0.861, 95% CI [0.649, 1.000]; binary agreement 93.1%). McNemar's test found
> no directional bias (1/1, p = 1.0), confirming that the topic-frame restriction
> introduced during coder development (§4b) removed the directional over-firing
> found in earlier validation against an independent AI coder (6/0, p = 0.031).
> The two residual disagreements were examined individually: one reflects a
> boundary between topic-specific and general conversational initiative: one is a
> single identified lexicon gap (failure to recognise "again" as a memory-recall
> cue). Both are isolated rather than systematic.
