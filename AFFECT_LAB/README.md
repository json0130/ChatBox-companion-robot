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

Verifies the paper's Table I reproduces exactly, that the published CHATBOX
coordinate yields the paper's own worked descriptors ("warm, calm, reserved"),
and that a detected face never moves Dominance.

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

## What is a proposal rather than published

- The **OCEAN → PAD** equations are Mehrabian's, via ALMA, and reproduce Table I
  exactly. Trustworthy.
- **`EMPATHY = 0.6`** — how hard a face pulls the robot off its temperament. Not
  specified in the paper; tune it with `[` and `]`.
- **`show`: CHATBOX 0.30, ELLEBOT 1.00** — how much of its temperament each body
  can display. The paper describes ELLEBOT having expressive channels "that a
  tabletop robot lacks" but puts no number on it.
- The **descriptor bands** in `affect.py` were fitted so CHATBOX returns the
  paper's own example, "warm, calm, reserved". The other words around them are
  mine.

## A quirk worth resolving

Neuroticism *raises* pleasure (`+0.19N`) and *lowers* arousal (`−0.57N`) in the
published regressions, so a high-N persona reads calm and faintly pleasant
rather than anxious. That is what the equations say, but it is worth confirming
against Mehrabian before anyone asks about it — your deployed config has
`N = +0.3`, which is already pulling arousal down by 0.17.
