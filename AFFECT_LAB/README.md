# Affect lab

A bench rig for the persona/emotion/embodiment model, running on a laptop
webcam. Nothing here imports or modifies the robot code — `CHATBOX_CLIENT` and
`CHATBOX_SERVER` are untouched.

**New here? Read [CONCEPT.md](CONCEPT.md) first** — it explains the whole model
from first principles, with worked numbers and an honest account of which parts
are published and which are proposals. This file is just how to run the thing.

What it does: reads your face, estimates **valence and arousal** with an
AffectNet-trained model, fuses that with a robot's OCEAN persona through the PAD
equations, and draws the result live on Russell's circumplex.

```
OCEAN traits  --Mehrabian/ALMA-->  baseline PAD
your face     --valence,arousal-->  pulls P and Ar   (never D)
the robot's body  ------------->    scales how much of it is shown
that coordinate ---------------->   3 descriptor words  -> the LLM prompt
                ---------------->   5 style values      -> servo angles
```

## Run it

```bash
cd AFFECT_LAB
pip install -r requirements.txt
python webcam_demo.py
```

First run downloads ~16 MB of model weights from GitHub; after that it works
offline.

**Click `CHATBOX` or `ELLEBOT`** at the top of the panel to swap persona. The
active one is filled in; the other is outlined.

| key | |
|---|---|
| `C` / `E` | same swap, from the keyboard |
| `[` `]` | empathy down / up |
| `S` | save a screenshot |
| `Q` or `Esc` | quit |

Useful flags: `--robot ELLEBOT`, `--camera 1`, `--categorical` (use the 8-class
model plus a circumplex lookup instead of the VA model), `--smoothing 0`.

## Check the maths without a camera

```bash
python test_affect.py
```

Verifies our Table I reproduces exactly, that the published CHATBOX coordinate
yields our own worked descriptors ("warm, calm, reserved"), that a detected face
never moves Dominance, and — §9 — that the published movement equation
reproduces all four anchor values its own paper states.

## Face detection

Three strategies, in preference order — the startup line tells you which one is
live, so you are never guessing:

1. **YuNet** — a 232 KB DNN detector, fetched automatically on first run. Much
   better than Haar at angles and through glasses. Needs OpenCV 4.5.4+.
2. **Haar cascade** — bundled with OpenCV 4.x, no download, but frontal-only.
   Absent on OpenCV 5, which removed `cv2.CascadeClassifier` entirely.
3. **Centre crop** — no detector. Fine for one person at a laptop, useless for a
   room.

`--no-download` skips step 1 and stays offline. `--yunet PATH` points at your own
copy of the weights.

If you ever fetch YuNet by hand, use the **media** URL, not the raw one —
`opencv_zoo` stores it in git-lfs, so `raw.githubusercontent.com` returns a
131-byte pointer that then fails to parse as ONNX:

```
https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## The emotion model

`enet_b0_8_va_mtl` from [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)
— EfficientNet-B0, trained on AffectNet, multi-task: 8 emotion classes *plus*
continuous valence and arousal. Chosen over EmoNet because it is small enough to
have a chance on a Jetson alongside YOLO-pose.

Two things the wrapper in `webcam_demo.py` handles that will bite you otherwise:

- The library calls `urllib.request.urlretrieve` but only does `import urllib`,
  so a fresh install raises `AttributeError` on the first download. Importing
  `urllib.request` first fixes it.
- Its preprocessing normalises with ImageNet statistics in **R, G, B** order, so
  crops must be converted out of OpenCV's BGR. Feeding it BGR degrades the
  estimate silently rather than failing.

## PAD → the five style values

`affect.gesture_style` turns a PAD coordinate into what the firmware consumes.
**Two of the five come from a published equation; three are ours.**

```
amplitude = (D + 1) / 2         PUBLISHED   clamp 0.30 … 1.00
tempo     = fv(P, Ar, D)        PUBLISHED   clamp 0.50 … 1.60
posture   = 0.70·D + 0.30·P     ours        clamp −1 … +1
idle      = 0.45 + 0.40·Ar      ours        clamp 0.05 … 1.00
droop     = −P × 1.4            ours        clamp −1 … +1
```

The published pair are **Eq. 9** and **Eq. A1** of
[Hagane & Venture (2022)](https://doi.org/10.3390/machines10121118), *Machines*
10(12):1118 (open access), following
[Claret, Venture & Basañez (2017)](https://link.springer.com/article/10.1007/s12369-016-0387-2),
*Int. J. Social Robotics* 9:277–292. Both map PAD to motion features by — their
words — "a simple linear formula". Their `Sp` (spatial extent, *"how large the
gestures of the hands and arms are"*) is amplitude; their `Ve` (velocity) is
tempo. Their third feature is jerkiness, dropped here because this firmware
plays a fixed five-step sequence and controls only `step_ms`.

`test_affect.py` §9 reproduces all four anchor values the paper states in its
Eq. 10 — that is the check that the transcription is faithful.

**Two consequences worth knowing before you cite this.** Both are recorded in
[CONCEPT.md](CONCEPT.md) §10:

- `Sp` uses **Dominance alone**, and the face never moves Dominance — so
  amplitude is fixed per persona (CHATBOX 0.42, ELLEBOT 0.80) no matter what
  expression is detected.
- `fv` gets **faster as Pleasure falls** (its last term is `4 − Pn`), so an
  unpleasant face speeds up rather than slowing down. Anger 0.88, happy 0.76,
  sad 0.75. The authors note the same clustering in their own §7.

## What is a proposal rather than published

- The **OCEAN → PAD** equations are Mehrabian's, via ALMA, and reproduce Table I
  exactly. Trustworthy, and untouched.
- **`posture`, `idle`, `droop`** — the three equations above. No published PAD
  equation for carriage, resting stir rate or a signed vertical tint could be
  found. `droop` as a *parameter* is Laban Sinking/Rising, but its 1.4 gain is
  by eye.
- **`EMPATHY = 0.6`** — how hard a face pulls the robot off its temperament. Not
  specified anywhere; tune it with `[` and `]`.
- **`show`: CHATBOX 0.30, ELLEBOT 1.00** — how much of its temperament each body
  can display. No published number for expressive bandwidth.
- The **descriptor bands** in `affect.py` were fitted so CHATBOX returns our own
  worked example, "warm, calm, reserved". The other words around them are mine.

## A quirk, partly explained

Neuroticism *raises* pleasure (`+0.19N`) and *lowers* arousal (`−0.57N`) in the
published regressions, so a high-N persona reads calm and faintly pleasant
rather than anxious.

A likely explanation turned up while evaluating a Laban-based alternative
(CONCEPT.md §7, "Superseded"): that model has `Time = … + 0.97N`, treating high
neuroticism as quick and agitated — the opposite of Mehrabian. The two measure
different things. Mehrabian's Arousal is **trait arousability**, a disposition;
Laban's Time is momentary agitation. Same word, different quantity.

It changes no equation here, since `fv` takes Arousal directly. Still worth a
sentence before anyone asks. The deployed config has `N = +0.3`, already pulling
arousal down by 0.17.
