# Wiring StyleControl.ino into the sketch

> **Already done for you.** [`ChatBoxPlus_Styled/`](ChatBoxPlus_Styled/) is a
> complete, flashable copy of the original sketch with every edit below applied.
> Open `ChatBoxPlus_Styled/ChatBoxPlus_Styled.ino` in the Arduino IDE and upload.
> The original `CHATBOX_ARDUINO` folder is untouched.
>
> This document is the record of what changed and why — read it if you want to
> apply the same edits to a future version of the firmware, or to check my work.

Four edits. None of them changes behaviour until a `STYLE` message arrives — the
style defaults are the identity, so an un-styled robot moves exactly as it does
today. That property is worth preserving: it means you can flash this and confirm
nothing regressed before sending a single style value.

## File ordering: the thing that will bite you

The IDE concatenates the `.ino` files **alphabetically, after the main sketch**.
So with files named `emotion_servo.ino` and `style_control.ino`, the servo code is
compiled *first* — and a `#define` is textual, so any weight defined in the style
`.ino` is "not declared in this scope" at every call site:

```
error: 'DROOP_SHOULDER' was not declared in this scope
note: the macro 'DROOP_SHOULDER' had not yet been defined
note: it was later defined here
```

That is why the constants and prototypes live in **`StyleControl.h`**, which all
three `.ino` files `#include`. A header is processed where it is included, so the
ordering stops mattering — and it keeps working if you rename the sketch files,
which changes the order.

Four files in the folder:

```
ChatBoxPlus_Styled.ino    the main sketch
ServoControl.ino          the servos, now styled
StyleControl.ino          the styling implementation
StyleControl.h            the weights, clamps, and prototypes
```

---

## Edit 1 — route the styling through the setX() functions

Each `setX()` currently computes an angle and assigns it. Wrap the assignment in
`styleAngle()`, passing that servo's rest pose, its stock range, and its droop and
posture weights.

`ServoControl.ino`, in `setShoulder()`:

```cpp
  if (side == 'R') {
    // was:  RShoulderDest = 90 + shoulderOffset;
    RShoulderDest = styleAngle(90 + shoulderOffset, 140, 50, 170,
                               DROOP_SHOULDER, POSTURE_SHOULDER);
  } else {
    LShoulderDest = styleAngle(90 + shoulderOffset, 40, 10, 130,
                               -DROOP_SHOULDER, -POSTURE_SHOULDER);
  }
```

The same shape for the rest. Arguments are `(target, rest, lo, hi, droopDeg,
posetureDeg)`:

| function | side | rest | lo | hi | droop | posture |
|---|---|---|---|---|---|---|
| `setEars` | — | 130 | 120 | 165 | `DROOP_EARS` | 0 |
| `setBrows` | R | 120 | 90 | 150 | `DROOP_BROW` | 0 |
| `setBrows` | L | 60 | 30 | 90 | `-DROOP_BROW` | 0 |
| `setEyes` | R | 110 | 90 | 130 | `DROOP_EYELID` | 0 |
| `setEyes` | L | 70 | 50 | 90 | `-DROOP_EYELID` | 0 |
| `setNeck` | R | 82 | 70 | 100 | `DROOP_NECK` | `POSTURE_NECK` |
| `setNeck` | L | 103 | 80 | 120 | `-DROOP_NECK` | `-POSTURE_NECK` |
| `setShoulder` | R | 140 | 50 | 170 | `DROOP_SHOULDER` | `POSTURE_SHOULDER` |
| `setShoulder` | L | 40 | 10 | 130 | `-DROOP_SHOULDER` | `-POSTURE_SHOULDER` |
| `setHand` | R | 90 | 50 | 150 | 0 | 0 |
| `setHand` | L | 90 | 30 | 130 | 0 | 0 |

The left-hand columns take the negated weight because both sides sit at 90 ± an
offset — a mood has to move them oppositely or the robot ends up lopsided.

`setNeck` is the one that needs a little care: it writes both sides from a single
symbol, so both assignments in each `case` get wrapped.

Hands get zero droop deliberately. A drooping hand reads as a failed servo, not
as sadness.

---

## Edit 2 — let tempo drive the step timer

`ServoControl.ino`, in `executeExpression()`:

```cpp
  // was:  if (millis() - timer > 900 || count == -1) {
  if (millis() - timer > styleStepMs(900) || count == -1) {
```

Tempo changes timing only, never an angle. That separation is worth keeping: it
means a tempo bug can make the robot sluggish but never make it reach somewhere
new.

---

## Edit 3 — accept the STYLE message

`ChatBoxPlus_ESP32.ino`, wherever an incoming TCP line is dispatched, before it
is treated as an expression name:

```cpp
  if (handleStyleCommand(line)) return;   // consumed a style update
  // ... existing expression handling ...
```

And the same in `updatePanTracking()` in `ServoControl.ino`, where non-numeric
serial lines are routed to `pendingCommand` — so the style can also be set from
the USB Serial Monitor while bench-testing:

```cpp
  if (handleStyleCommand(lineFromSerial)) return;
```

### A fifth thing, found while doing it

`PAN_MAX_INPUT_LEN` was **24**. A style line is 32 characters:

```
STYLE 1.00 1.00 -1.00 -1.00 1.00
```

So the serial reader truncated it mid-number and the parser silently saw four
values instead of five — no error, just a style that never applied. Raised to 48
in the styled copy. Worth remembering if you ever extend the message: the buffer
has to lead the format, not follow it.

---

## Edit 4 — optional: idle stirring

`styleIdleMs()` returns the interval to use. In `loop()`, outside the EXECUTE
state:

```cpp
  static unsigned long lastStir = 0;
  if (millis() - lastStir > styleIdleMs()) {
    lastStir = millis();
    pendingCommand = "idle";     // the existing subtle ears/eyes move set
  }
```

Leave this out on the first pass. It is the one edit that makes the robot move
without being asked, so it is the one most likely to surprise you during a demo.

---

## Testing it

**Before flashing**, from the `SERVO_STYLE` folder:

```bash
python test_servo_style.py                       # the angle maths
python preview.py greeting --emotion sad --wire  # the exact numbers, per robot
```

`preview.py` prints the same angles this firmware will produce, so you can check
a mood looks right in the table before it moves a real servo.

**After flashing**, from the Serial Monitor at 115200 — no Jetson needed:

```
STYLE 1.00 1.00 0.00 0.00 0.45     <- neutral, should look exactly as before
greeting
STYLE 0.65 0.75 -0.50 0.80 0.30    <- damped, slow, withdrawn, drooping
greeting                            <- same tag, visibly downcast
STYLE 1.00 1.20 0.50 -0.60 0.60    <- full, brisk, open, bright
greeting
```

The first block is the important one. If a neutral style does not look identical
to the current firmware, Edit 1 has a wrong rest or range value somewhere — check
that row against the table above before going further.
