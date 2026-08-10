# Servo style

Turns an affective coordinate into **servo angles**, so the same gesture tag can
be performed brightly or sadly, and so two robots with different personalities
move by different amounts.

Nothing here modifies `CHATBOX_ARDUINO` or `CHATBOX_CLIENT`. The firmware
addition is a separate file plus a documented set of edits you apply when you are
ready — see [`firmware/INTEGRATION.md`](firmware/INTEGRATION.md).

## What the firmware does today

A tag arrives as a string (`"greeting"`), gets looked up in `listOfMoveSets`, and
a fixed five-step sequence plays. Each step holds a symbol per servo group — `D`,
`M`, `U` — and `setNeck` / `setShoulder` / `setEyes` / `setBrows` / `setEars` /
`setHand` turn that symbol into a hardcoded angle.

Identical every time, for every persona, in every mood. Which is exactly the
thing to change.

## What this adds

Five values reshape that same sequence without replacing any of it:

| | | range |
|---|---|---|
| **amplitude** | how much of the authored travel to perform | 0.30 … 1.00 |
| **droop** | signed valence tint — sad ↔ bright | −1 … +1 |
| **posture** | standing offset on neck and shoulders | −1 … +1 |
| **tempo** | playback speed (timing only, never angles) | 0.50 … 1.60 |
| **idle** | how often it stirs between gestures | 0.05 … 1.00 |

Per servo, per step:

```
1. symbol -> target angle        exactly as the firmware does now
2. scale distance from rest      by amplitude
3. add droop offset              signed by valence
4. add posture offset            neck and shoulders only
5. clamp to the stock pose range
```

### Why there are five, not four

The paper names four: amplitude, tempo, posture, idle frequency. **None of them
can make a greeting look sad.** Amplitude only makes it smaller; a small wave is
still a happy wave. Tempo only makes it slower.

Valence needs a *signed* offset of its own, which is `droop`: ears fall, brows
lower, lids drop, head dips, shoulders slump — added on top of whatever gesture is
playing. It is the parameter that answers the actual question, and it is an
addition to the paper rather than an implementation of it.

## The whole chain, live

```bash
cd SERVO_STYLE
python live_demo.py
```

Webcam on the left, affective state in the middle, gesture buttons on the right.
Look sad, click **greeting**, and the robot greets you sadly — small, slow, ears
down. The emotion comes from your face, the persona from the selected robot, and
the five style values plus the tag go straight down the USB serial link to the
ESP32.

It auto-detects the board (a CH340 or CP210x port) and falls back to preview-only
if there is nothing plugged in, so it is safe to run either way. `--dry-run`
forces preview, `--port COM8` picks the port by hand, `--list-ports` shows what
is available.

Buttons are greyed while a gesture is playing: the firmware blocks inside its
EXECUTE state, so a tag sent during one would just sit in the buffer and the
click would look ignored.

## Just the numbers

```bash
python preview.py greeting --emotion sad
```

```
tag: greeting    person looks: sad

  CHATBOX
    style        amp 0.34   tempo 0.63   posture -0.54   droop +0.29
    step timing  1437 ms  (stock 900 ms)
  ELLEBOT
    style        amp 0.82   tempo 0.89   posture +0.22   droop +0.35
    step timing  1007 ms  (stock 900 ms)

  servo      stock                 CHATBOX               ELLEBOT
  Ears       165 165 165 165 165   136 136 136 136 136   152 152 152 152 152  *
  RNeck       82  82  82  82  82    74  74  74  74  74    81  81  81  81  81  *
  RShoulder  170 170  50  50  50   142 142 102 102 102   162 162  64  64  64  *
```

Same tag. CHATBOX barely raises its arm (142 rather than 170), ears down 29°, head
dipped, all of it 60% slower. ELLEBOT performs most of the gesture but still
tinted. Neither has been given a different move set.

Other flags: `--emotion happy|angry|fear|neutral|…`, `--servo RShoulder` for one
row, `--wire` to see the message the Jetson would send, `--tags` to list what is
available.

```bash
python test_servo_style.py
```

## The safety property

Every styled angle is clamped to the span the stock move sets already reach, so
**styling can never command a pose the unstyled firmware would not have
commanded.** The test suite checks all 189 tag/style combinations for this.

That is also why amplitude stops at 1.00. Each servo has three symbols, so `D`
and `U` already *are* the ends of its range — asking for 1.3 of the way to an end
stop only saturates against the clamp. The authored gestures are treated as full
expression, which styling damps. If you want a robot to overshoot them, raise
`HEADROOM_DEG` in `servo_style.py`, but measure the mechanical travel first,
because nothing else will stop a servo grinding.

## The wire format

One new message. Tag messages are untouched, so the existing protocol still works.

```
STYLE 0.82 0.89 +0.22 +0.35 0.44
      amp  tempo posture droop idle
```

Values are clamped on arrival by the firmware as well as by the sender — a stray
digit should not be able to reach an end stop.

## Files

| | |
|---|---|
| `live_demo.py` | webcam → emotion → style → serial → the robot moves |
| `servo_style.py` | the angle model: move sets, symbol→angle tables, the styling |
| `preview.py` | CLI — per-robot angles for a tag and a mood |
| `test_servo_style.py` | 60-odd checks, no hardware needed |
| `firmware/ChatBoxPlus_Styled/` | the complete flashable sketch |
| `firmware/INTEGRATION.md` | what changed in it, and how to test |

`preview.py` imports `../AFFECT_LAB/affect.py` so PAD has one source of truth.
`servo_style.py` itself has no dependency on it — hand it a style dict and it
resolves angles.

## Numbers that are proposals, not findings

- **`DROOP_DEG`** — the per-servo degrees at full droop. Tuned by eye in the
  table, never on a real robot. Start here if a mood does not read right.
- **`POSTURE_DEG`** — same.
- **`TRAVEL`** — CHATBOX 0.65, ELLEBOT 1.00. Deliberately *not* the same numbers
  as `AFFECT_LAB`'s `show` (0.30 / 1.00): reusing those multiplies the restraint
  twice and lands CHATBOX at 0.19 amplitude, which does not read as reserved, it
  reads as broken.
- **The droop-from-Pleasure gain** (1.4, in `preview.py`) — chosen because the
  Pleasure axis rarely reaches ±1 in practice, so a straight copy would barely
  tint anything.

The symbol→angle tables, by contrast, are transcribed from `ServoControl.ino` and
the test suite asserts a neutral style reproduces the stock firmware exactly.
