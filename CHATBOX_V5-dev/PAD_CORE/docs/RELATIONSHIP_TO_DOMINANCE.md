# Relationship → Dominance: rapport, trust, tier

Slide-ready reference for how the knowledge graph drives the Dominance axis, and
what that changes in speech and movement. Every number here was produced by
running the code; the commands that regenerate them are in §9.

---

## 1. The one-line story

> The robot's **personality** decides where it starts, the person's **face** moves
> how it feels, and the **relationship** decides how much it leads.

Three influences that would otherwise fight are given **different axes of one
space**, so they add instead of overwriting each other.

| influence | source | axis it owns | timescale |
|---|---|---|---|
| personality | OCEAN traits | sets the baseline point | fixed |
| emotion | the camera | **Pleasure, Arousal** | per frame |
| relationship | the knowledge graph | **Dominance** | across sessions |

The face is deliberately given **no vote on Dominance**. Expressions carry
pleasure and arousal reliably; dominance reflects social standing between two
parties, so it has to come from recognising *who* the person is.

---

## 2. The chain, end to end

```
     ┌── camera ─────────────► valence, arousal ──┐
     │                                            ├──► PAD ──┬──► 3 words + directive ──► LLM ──► speech
OCEAN traits ──► baseline PAD ────────────────────┤          │
     │                                            │          └──► 5 style values ──────► ESP32 ──► movement
     └── knowledge graph ──► tier ──► D offset ────┘
```

**Only the middle box is new work.** Everything either side was already built.

---

## 3. What moves rapport and trust

They live on **one `InteractionNode` per (person, robot) pair**, alongside
`interaction_count`. Exactly three things write them:

| # | source | when | amount | can decrease? |
|---|---|---|---|---|
| 1 | **live tick** | every tick a face is visible **and** felt Pleasure > 0.05 | `+0.025 × P` to **rapport only** | ❌ |
| 2 | **session extraction** | once, at end of session | LLM proposes `rapport_delta`, `trust_delta`, clamped **±0.2** | ✅ **the only way down** |
| 3 | **manual (B key)** | operator | `+0.15` to both | ❌ |

All results clamped to `[0, 1]`.

**The LLM's entire brief for (2):** *"rapport_delta rises with warmth and positive
affect; trust_delta rises with the child sharing personal things."* No rubric —
the magnitude is the model's free judgment inside ±0.2.

**Why path (1) is rapport-only.** That delta is computed from felt Pleasure — the
warmth read off a face — and warmth is precisely what rapport is. Trust, by this
system's own definition above, is about what the child chose to disclose, and no
single frame carries that. So the only writer that can move trust is the one that
sees the transcript. Path (3) moves both because it is an operator override for
testing, not an inference from anything observed.

### Three consequences worth a slide

1. **The two axes are genuinely independent.** They used to move in lockstep —
   path (1) wrote the *same* delta to both, so `score = (rapport + trust) / 2`
   collapsed to `score = rapport` and trust was a decorative second copy. Now
   rapport tracks affect within a session and trust tracks disclosure across
   sessions.
2. **Nothing decays.** There is no time-based forgetting on rapport or trust.
   Closeness is effectively a ratchet. *(FAST edges — mood, attention — are
   separately session-scoped on read; that is a different mechanism and does not
   apply here.)*
3. **A sad face never advances closeness.** `P` stays ≤ 0.05, so nothing accrues;
   the tier can only climb via `interaction_count`, and can never reach `close`.

---

## 4. Tier — the step function

```
score = (rapport + trust) / 2

    score > 0.45          →  known
    interaction_count > 5 →  known      (familiarity without warmth)
    interaction_count > 0 →  visitor
    otherwise             →  unknown
```

`interaction_count` is **set from the transcript store** (`sessions.db`) on every
turn, not incremented in the graph.

> **A `family` tier existed and was removed.** It was unreachable: any score that
> would have been "family" already returns `known` at > 0.45. Dead configuration.

---

## 5. Tier → Dominance

```python
TIER_OFFSETS = {          # (dP, dAr, dD)
    "close":   (0.00, +0.10, +0.40),
    "known":   (0.00,  0.00,  0.00),   # the reference tier
    "visitor": (0.00,  0.00, -0.20),
    "unknown": (0.00,  0.00, -0.40),
}

D = clamp(baseline_D + offset,  -1, +1)
```

Applied **before** the body's `show` fraction — the style values read the *felt*
coordinate, so a tier applied afterwards would never reach a servo.

**Key property for the slide: Dominance is a 4-step ladder, not a continuous
signal.** rapport and trust move smoothly, but D does not move at all until a
threshold is crossed — then it jumps 0.20. The bearing changes in steps.

---

## 6. How it develops over time

Simulated with the real code, one tick per second, face continuously visible:

| | happy face | neutral face | sad face |
|---|---|---|---|
| → visitor | 1 s | 1 s | 1 s |
| → known | 6 s | 6 s | 6 s |
| → close | **not within a session** | not within a session | **never** |

`visitor` and `known` both arrive on `interaction_count` alone (`> 0` and `> 5`),
which is why they are the same for every face and why a sad face still reaches
`known`.

`close` needs `(rapport + trust) / 2 > 0.70`. The per-tick path moves rapport
only, so an unbroken hour of smiling saturates rapport at 1.0 and still scores
0.50 — `known`. Trust has to come from the end-of-session extractor at ±0.2, so
**three sessions is the floor**, and only if the model judges maximum disclosure
each time.

> **This is what the earlier version got wrong.** Both axes took the same delta,
> so `close` arrived after ~48 s of a happy face: the relationship, which the
> design calls the *slow* signal, was in practice the fastest-changing quantity in
> the system. It is now the slowest, which is what it was always supposed to be.
> Demos that need a specific rung on demand should use `tier_override`.

---

## 7. What Dominance changes

### 7.1 Movement — where most of the effect is

`amplitude = (D + 1) / 2` reads Dominance **alone** (Hagane & Venture 2022, Eq. 9).

CHATBOX, same `greeting` tag, neutral face:

| tier | D | amplitude | tempo | posture | RShoulder | step_ms |
|---|---|---|---|---|---|---|
| unknown | −1.000 | 0.300 | 0.698 | −0.67 | **145°** | 1289 |
| visitor | −0.843 | 0.355 | 0.721 | −0.56 | — | — |
| known | −0.643 | 0.425 | 0.759 | −0.42 | 151° | 1186 |
| close | −0.243 | 0.565 | 0.865 | −0.14 | **158°** | 1040 |

*Unstyled firmware would command 170° every time.* With someone it knows, the arm
lifts 13° higher, the head lifts, and each step runs 250 ms faster.

`posture` (0.70·D + 0.30·P) tracks D directly; `tempo` picks it up through `fv`.
**`droop` is the only parameter Dominance never touches** — it carries valence.

### 7.2 Speech — two fragments in the prompt

| tier | CHATBOX words | ELLEBOT words |
|---|---|---|
| unknown | even · calm · **withdrawn** | warm · lively · **even-handed** |
| visitor | even · calm · **retiring** | warm · lively · **forthright** |
| known | even · calm · **reserved** | warm · lively · **assertive** |
| close | even · calm · **even-handed** | warm · lively · **commanding** |

Emotion moves the first two words; the tier moves the third — the three-axis
split showing up directly in the output. Every tier gets its own word on both
robots. The bands are **absolute**, so CHATBOX@`close` and ELLEBOT@`unknown`
both say *even-handed*: they genuinely sit at nearly the same Dominance. That is
what keeps the word a reading of the coordinate rather than the tier relabelled.

Words read the **felt** coordinate, not the shown one. `show` is expressive
bandwidth for the *body*; scaling the words by it squashed twelve of CHATBOX's
sixteen emotion × tier cells onto "reserved" and every emotion onto "calm", and
contradicted the paper's own worked example — which only reproduces from felt.

```
• Your manner right now is warm, calm, reserved.            ← descriptor words
• With this person, follow their topic, and you may add      ← behavioural
  one follow-up question about it.                              directive
```

Plus one sentence opening the WHO block (*"You recognise this person…"*).
**No numbers ever reach the model** — rapport/trust in a prompt invite it to
narrate its own metrics.

| tier | D | CHATBOX directive | D | ELLEBOT directive |
|---|---|---|---|---|
| unknown | −1.00 | answer only what they ask | +0.02 | either of you may open a topic |
| visitor | −0.84 | ask about what they bring up | +0.22 | offer a topic if they do not |
| known | −0.64 | follow their topic, one follow-up | +0.42 | propose the next topic yourself |
| close | −0.24 | either of you may open a topic | +0.82 | open with something you know about them |

Each rung is phrased as **who introduces a topic**, on purpose. An earlier
version graded the *wording* ("let them lead" / "follow their lead" / "mostly
follow their lead") and three consecutive rungs came out as the same instruction
in different words — **0.60 of CHATBOX's 0.757 range, 79% of it**, producing
nothing anyone could observe. If a rater watching two clips cannot say which
directive was in force, the claim that this is the *measurable* half of the model
does not hold. *"Did the robot introduce a topic the person had not mentioned?"*
is a binary that can be coded straight off a transcript.

### The two robots never overlap — and that confounds any cross-robot comparison

`D = baseline + tier offset`, nothing else, so each robot has exactly four
possible values:

| | range | width |
|---|---|---|
| CHATBOX | [−1.000, −0.243] | 0.757 |
| ELLEBOT | [+0.021, +0.821] | 0.800 |

The gap between them is **0.264 — wider than the 0.200 a single tier step
moves**, so no amount of relationship history can bring them into contact. *A
maximally close CHATBOX is still more deferential than ELLEBOT meeting a total
stranger.* That is a real design result and worth stating.

It carries an evaluation cost, though: **persona and tier are fully confounded.**
Any cross-robot comparison is also a comparison across D ranges, so a difference
can never be attributed to one rather than the other. There is exactly one
exception — **CHATBOX at `close` (−0.243) and ELLEBOT at `unknown` (+0.021)**,
0.264 apart and sharing a directive rung. That is the only pair where D is nearly
matched and only the persona differs, which makes it the natural centrepiece of a
user study.

The **adjective** says what to *be*; the **directive** says what to *do*. Trait
labels must be interpreted before they change anything, and models interpret them
loosely — the directive is the half written to be observable, and therefore
measurable (hedge count, question rate, who proposes the topic).

---

## 8. Findings to be honest about

| finding | status |
|---|---|
| `family` tier unreachable | **fixed** — removed |
| rapport and trust move identically in live operation | **fixed** — the per-tick delta now moves rapport only, so trust reflects disclosure rather than duplicating warmth |
| `close` reachable in ~48 s, making the "slow" signal the fastest one | **fixed** — it is now cross-session and disclosure-gated, 3 sessions minimum |
| no decay: closeness is a ratchet | open — rapport and trust still never fall except by a negative extraction delta |
| a sad face can never reach `close` | open |
| a stale FAST mood edge survived a restart and blended into turn 1 | **fixed** — `pre_turn` ignores mood written before the current session |
| CHATBOX's `show = 0.30` compresses all three axes, so its *words* barely move (ELLEBOT's vary on every input) | **stale** — `show` no longer feeds the descriptor words at all; they read the *felt* coordinate, and all four CHATBOX tiers now yield distinct triplets |
| A/B of directive vs adjective alone | **no effect detected, and underpowered** — n=3/cell, keyword metrics counting 0–2 occurrences. Not evidence either way |
| memory is per-**person**, relationship per-**pair** — so a second robot holds full memory at tier `unknown` | **wording fixed**; the split is intentional |

---

## 9. Published vs ours

| | source |
|---|---|
| OCEAN → PAD | Mehrabian via ALMA. **Published, unchanged.** |
| Two axes for the face, and why D is excluded | Russell's circumplex. **Published.** |
| `amplitude = (D+1)/2` | Hagane & Venture (2022) Eq. 9. **Published, verbatim.** |
| `tempo = fv(P,Ar,D)` | Hagane & Venture (2022) Eq. A1. **Published**; test reproduces all four of their anchor values. |
| `DROOP_DEG` channels and signs | Coulson (2004). **Published.** |
| **`TIER_OFFSETS`** | **Proposal.** No published equation maps a relationship tier to a PAD offset. Direction is principled, magnitudes by eye. |
| **Tier thresholds** (0.45 / 0.70 / count > 5) | **Proposal.** |
| **`0.025 × P` per tick** | **Proposal.** |
| **The behavioural directive ladder** | **Proposal.** |
| `posture`, `idle`, `droop` gain, `empathy = 0.6`, `show` | **Proposal.** |

### Reproduce every number here

```bash
python3 tests/test_affect.py                     # the tier ladder, sections 17-21
python3 -m pad_core.adapter                      # tier ladder for both robots
python3 -m tools.pad_prompt_grid --no-llm        # prompts + descriptor matrix
python3 -m tools.e2e_pipeline --no-camera        # the whole chain, narrated
python3 -m modules.webui --llm --trace-prompt    # live, with prompt assembly
```
