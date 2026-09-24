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

Four parameters — amplitude, tempo, posture, idle frequency — **cannot make a
greeting look sad.** Amplitude only makes it smaller; a small wave is still a
happy wave. Tempo only makes it slower.

Valence needs a *signed* offset of its own, which is `droop`: ears fall, brows
lower, lids drop, head dips, shoulders slump — added on top of whatever gesture
is playing.

That parameter has a published name: it is **Sinking/Rising**, the vertical axis
of Laban Shape, and it is one of the three Shape dimensions rather than an
invention of this project. Its *gain* here (`−P × 1.4`) is still ours and still
chosen by eye — no published equation maps PAD to a signed vertical tint. See
`AFFECT_LAB/CONCEPT.md` §10.

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
    style        amp 0.42   tempo 0.75   posture -0.54   droop +0.44
    step timing  1202 ms  (stock 900 ms)
  ELLEBOT
    style        amp 0.80   tempo 1.03   posture +0.22   droop +0.35
    step timing  876 ms  (stock 900 ms)

  servo      stock                 CHATBOX               ELLEBOT
  Ears       165 165 165 165 165   136 136 136 136 136   151 151 151 151 151  *
  RNeck       82  82  82  82  82    73  73  73  73  73    81  81  81  81  81  *
  RShoulder  170 170  50  50  50   143 143  92  92  92   161 161  66  66  66  *
  RHand       90  90  90  90  90    90  90  90  90  90    90  90  90  90  90
```

Same tag. CHATBOX raises its arm to 143 where stock commands 170, ears down 29°,
head dipped, the whole gesture taking 6.0 s where ELLEBOT takes 4.4 s. ELLEBOT
performs most of the gesture but is still visibly tinted. Neither has been given
a different move set, and the hands stay put in both.

These are the values the design deck publishes on its "Same tag, two styles"
slide — amplitude 0.42 / 0.80 and droop +0.44 / +0.35 — so `preview.py` is the
check that the code and the deck have not drifted apart.

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

- **`DROOP_DEG` / `POSTURE_DEG` magnitudes** — the per-servo degrees at full
  deflection. The *channel set* and *signs* follow
  [Coulson (2004)](https://link.springer.com/article/10.1023/B:JONB.0000023655.25550.be)'s
  six postural joint rotations and
  [Wallbott (1998)](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-0992(1998110)28:6%3C879::AID-EJSP901%3E3.0.CO;2-W);
  the degrees are tuned by eye and are specific to this geometry. Start here if
  a mood does not read right.
- **Zero droop on the hands** — ours, and arguably wrong.
  [Xu et al. (2013)](https://ieeexplore.ieee.org/document/6628534/) found hand
  height among the four parameters that carried robot mood best. Worth testing.
**Removed:** there used to be a per-robot `TRAVEL` fraction here (CHATBOX 0.65,
ELLEBOT 1.00) multiplying amplitude and droop. It is gone. The design deck now
carries only one embodiment scaling — `shown = show_fraction × felt` in
`AFFECT_LAB` — and a second mechanical fraction on top no longer matched the
deck's worked numbers. It was also harmful: with travel, CHATBOX's amplitude
came out at 0.28 and saturated against the 0.30 clamp floor, so it played every
gesture at minimum size in every mood. The two robots still separate on
amplitude (0.42 vs 0.80) because `amplitude = (D+1)/2` already carries the
persona difference.

Of the five style values arriving over the wire, **`amplitude` and `tempo` are
now published** — [Hagane & Venture (2022)](https://doi.org/10.3390/machines10121118),
*Machines* 10(12):1118, Eq. 9 and A1, following
[Claret, Venture & Basañez (2017)](https://link.springer.com/article/10.1007/s12369-016-0387-2).
`posture`, `idle` and `droop` remain proposals; no published PAD equation for
them could be found. `AFFECT_LAB/CONCEPT.md` §10 keeps the tally.

The symbol→angle tables are transcribed from `ServoControl.ino`, and the test
suite asserts a neutral style reproduces the stock firmware exactly.
