# 8a — Trust as a rule, not a judgement

## The gap this closes

`trust` and `rapport` are separate fields on the pair's `InteractionNode`, and the
relationship tier reads their mean (`kg_bridge.py:116`). Commit `2be86bb` stopped the
live per-tick path writing the same delta to both — it now moves rapport only
(`webcam_loop.py:1584-1585`), on the correct grounds that rapport is warmth, warmth is
readable from a face, and a single frame cannot show what a child chose to tell you.

That fix was necessary and incomplete. It left `trust` with exactly one non-manual
write path: an end-of-session LLM asked for a free-form
`"trust_delta": <number between -0.2 and 0.2>` (`extraction.py:39-50`). "Trust is
disclosure-gated" was therefore true only in the sense that *a language model was asked
to judge disclosure*. Any measurement resting on it inherits that model's variance, its
prompt, and its version — the same dependency that demoted E1 to a limitation.

`padeval/coding/disclosure.py` replaces the judgement with a rule. No model call, pure
function of the text, auditable by reading it.

## The principle that gives the rule its shape

Rapport already has a trigger: felt Pleasure above threshold, read off the face. For
trust to be a *different* signal rather than a second copy, its trigger must fire on
something the camera cannot see. Hence:

> A first-person statement counts as disclosure when it carries information the camera
> does not already have.

This is why **"I'm happy" is not disclosure** — the emotion head is looking directly at
that, and counting it would make trust a laundered copy of valence, reintroducing the
exact collapse `2be86bb` fixed. But **"I was scared when the lights went out" is**: it
concerns a moment the camera never saw. The discriminator is a *displacement* marker
(past tense, a subordinator, a time shift) that moves the statement off the visible
present. Non-facial states — "left out", "lonely", "embarrassed" — need no such marker,
because they are not among the seven classes the detector predicts and never will be.

Three routes, ordered by depth, because these are not equivalent acts:

| depth | route | example | trust delta |
|---|---|---|---|
| 1 | preference | "I like jazz" | 0.02 |
| 2 | personal fact | "my dad works nights" | 0.04 |
| 3 | internal state | "I felt left out at school" | 0.06 |

Negation is deliberately **not** scoped out, the opposite of the topic coder's rule
(`rules.py:71-87`). "I don't like maths" and "I don't have any friends" are disclosures,
and strong ones. Reusing the topic coder's negation handling would have silently dropped
the most revealing utterances in the set.

## Registered predictions

Stated in `padeval/tests/test_disclosure.py` before the detector was run on the case set.

| | prediction | result |
|---|---|---|
| P1 | disclosure with flat affect raises trust, leaves rapport within 0.01 of start | **held** — rapport 0.0000, trust 0.1600 |
| P2 | positive affect without disclosure is the mirror | **held** — rapport 0.0600, trust 0.0000 |
| P3 | ≥ 90% correct on the 2×2 case set, errors skewing to false negatives | **held** — see below |
| P4 | rapport and trust end at different values, first diverging at the scripted disclosure turn | **held** — diverge at turn 3, the scripted disclosure turn |

## The divergence, demonstrated rather than asserted

Before this, "trust and rapport can differ" was a property of the *type signature* —
two floats, two parameters. A signature permits divergence; it does not exhibit it. The
scripted session below schedules warmth and disclosure on **different turns**, so any
divergence is attributable to a named turn rather than to drift:

```
turn  d_rapport  d_trust   rapport   trust   text
   0     0.0000   0.0000   0.0000  0.0000   'mm'
   1     0.0225   0.0000   0.0225  0.0000   "haha you're funny"
   2     0.0200   0.0000   0.0425  0.0000   "that's a good one"
   3     0.0000   0.0600   0.0425  0.0600   'i felt left out at school today'   <- diverge
   4     0.0000   0.0000   0.0425  0.0600   'okay'
   5     0.0000   0.0400   0.0425  0.1000   'my dad works nights'
```

Ending rapport 0.0425, trust 0.1000. The tier score `(r+t)/2 = 0.0713` is neither.

## Two accuracy numbers, and only one of them is evidence

**The 2×2 case set (n=38): 100%, 0 false positives.** This is a *specification*, not an
evaluation. The set was authored first, then run, and the three failures it exposed were
fixed — so the final number measures conformance to a spec, and is reported as such. The
three failures are worth recording because one was a real defect rather than a lexicon
gap: `_WORD` kept `"i'm"` whole, and since every route requires a first-person marker,
the contraction silently disabled *all three routes*. Children contract constantly, so
the detector would have looked calm while missing most of what it exists to catch. Fixed
by expanding contractions at token level rather than loosening the marker set.

**Held out (n=32, the E1 stimuli): κ = 0.739.** These utterances were authored months
earlier for a different experiment, stratified independently into
`question / disclosure / quiet / closing`, and were never seen during development.
Taking `stratum == "disclosure"` as gold:

| | value |
|---|---|
| precision | 0.857 |
| recall | 0.750 |
| specificity | 0.958 |
| accuracy | 0.906 |
| Cohen's κ | 0.739 |

Per stratum, the detector flagged **6/8 disclosure, 0/8 question, 0/8 quiet, 1/8 closing**.

The two misses are the same class — past-tense action facts whose verb is not in
`FACT_VERBS` ("I drew a picture of a dragon", "I fell over at playtime"). The single
false positive, "I'm getting a bit tired", is arguably not an error at all: `closing`
labels a *conversational function*, and that utterance does disclose an internal state.

**These were deliberately not fixed.** Patching `FACT_VERBS` against the held-out set
would convert it into a second training set and destroy the only unbiased number in this
report. The gap is recorded here instead, and closing it would require a fresh held-out
sample.

## Scope

Sarcasm, third-party framing ("my teacher says I'm lazy") and disclosure carried purely
by implicature are not detected. Errors run toward false **negatives** by construction,
since every route requires an explicit first-person marker — the safe direction: a
missed disclosure only slows trust accrual, whereas a false one would let trust drift up
on ordinary chat and re-collapse it onto rapport.
