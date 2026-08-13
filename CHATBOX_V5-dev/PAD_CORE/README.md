# pad_core

Persona, emotion and relationship fused into one PAD coordinate — and that
coordinate turned into both **movement** and **words**.

```
camera ──valence/arousal──┐
OCEAN persona ────────────┼──► PAD ──┬─► five style values ──► servo angles ──► robot
KG relationship tier ─────┘          └─► three words ────────► LLM prompt ────► reply
```

Three influences that would otherwise fight are given **different axes**, so they
add instead of overwriting each other:

| influence | wants to | axis it owns |
|---|---|---|
| the robot's **personality** | stay the same | sets the baseline position |
| the person's **emotion** | change moment to moment | Pleasure, Arousal |
| the **relationship** | change slowly | Dominance |

The face is deliberately given **no vote on Dominance**: expressions carry
pleasure and arousal reliably, while dominance reflects social standing between
two parties and has to come from recognising *who* the person is.

## Self-contained on purpose

`pad_core` imports **nothing but the standard library** — no camera, no torch, no
LLM, no knowledge graph. A test asserts it. That is what lets the whole model be
dropped into another project, and what lets the published equations be verified
on a laptop with `python3 tests/test_affect.py`.

The consuming application supplies two things and takes two back:

```python
from pad_core import PADPipelineAdapter

adapter = PADPipelineAdapter("chatbox")
out = adapter.process_turn(
    valence=-0.7, arousal=-0.4,       # in: the face
    relationship_tier="close",        # in: who they are (your graph decides this)
)

out["words"]   # ('even', 'calm', 'even-handed')      -> your system prompt
out["wire"]    # 'STYLE 0.56 0.89 -0.13 -0.18 0.55'   -> your robot
```

Nothing in here knows how `relationship_tier` was derived. In this repo it comes
from a knowledge graph (`CHATBOX_SERVER/modules/graph_relationship`), but any
five-valued familiarity signal works.

## Layout

| path | what |
|---|---|
| `pad_core/affect.py` | OCEAN→PAD, the face's pull, **relationship→Dominance**, descriptor bands, PAD→five style values |
| `pad_core/servo_style.py` | style values → per-servo angles, the `STYLE` wire format, move sets |
| `pad_core/prompt.py` | the one sentence the tier contributes to a system prompt |
| `pad_core/stream.py` | rolling-window smoothing of frame-to-frame V/A jitter |
| `pad_core/adapter.py` | `PADPipelineAdapter` — one call per conversation turn |
| `tests/` | `test_affect.py`, `test_servo_style.py`, `test_pad_core.py` — no camera, no hardware |
| `bench/` | `webcam_demo.py` (watch the coordinate move), `preview.py` (angle tables), `live_demo.py` (drive the robot) |
| `firmware/` | the ESP32 sketch that parses `STYLE` and applies it to every gesture |
| `docs/` | `CONCEPT.md` — the model from first principles, with an honest published-vs-proposal ledger |

## Run it

```bash
python3 tests/test_affect.py            # the affect maths, incl. the tier ladder
python3 tests/test_servo_style.py       # the angle maths
python3 tests/test_pad_core.py          # the package is consistent + self-contained
python3 -m pad_core.adapter             # the tier ladder for both robots, printed

python3 bench/webcam_demo.py            # needs a camera
python3 bench/preview.py greeting --emotion sad --wire
```

## Before citing anything from here

`docs/CONCEPT.md` §10 tracks exactly which coefficients are **published** work and
which are **proposals of ours**, because the two get cited very differently.
In short: OCEAN→PAD is Mehrabian via ALMA; `amplitude` and `tempo` are Hagane &
Venture (2022) Eq. 9 / A1, used verbatim and verified against their own anchor
values; `posture`, `idle`, `droop`, the empathy fraction, the `show` fractions and
the **tier offsets** are ours and tuned by eye.
