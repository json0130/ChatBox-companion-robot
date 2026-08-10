"""
servo_style.py — turn a gesture tag plus an affective style into servo angles.

A faithful Python mirror of what ServoControl.ino does today, plus the styling
layer that does not exist there yet. Keeping it in Python first means the angle
tables can be inspected and tested before anything is flashed, and a bad number
shows up on screen rather than as a servo hitting its end stop.

Nothing here imports or modifies CHATBOX_ARDUINO — the constants below are copied
from it, and `test_servo_style.py` asserts they still agree.

--------------------------------------------------------------------------------
What the firmware does now
--------------------------------------------------------------------------------
A tag arrives ("greeting"), it is looked up in listOfMoveSets, and a fixed
five-step sequence plays. Each step holds a *symbol* per servo group — D (down),
M (middle), U (up) — and setNeck/setShoulder/setEyes/setBrows/setEars/setHand
convert that symbol into a hardcoded angle. Identical every time, for every
persona, in every mood.

--------------------------------------------------------------------------------
What this adds
--------------------------------------------------------------------------------
Five style values reshape that same sequence without replacing it:

  amplitude  scale on each servo's travel from its rest angle    0.30 … 1.30
  droop      signed valence tint: ears, brows, eyes, neck, arms  -1 … +1
  posture    standing offset on neck and shoulders               -1 … +1
  tempo      playback speed multiplier (timing, not angles)      0.50 … 1.60
  idle       how often it stirs between gestures                 0.05 … 1.00

`droop` is the one the paper does not name, and it is the one that answers "make
the greeting look sad". Amplitude cannot do that — a smaller wave is still a
happy wave, just smaller. Valence needs a *signed* offset that pushes the
expressive servos down when the coordinate is unpleasant and lifts them when it
is pleasant, applied on top of whatever gesture is playing.

Order of operations, per servo, per step:

    1. symbol -> target angle          (exactly as the firmware does now)
    2. scale the distance from rest    by amplitude
    3. add the droop offset            signed by valence
    4. add the posture offset          neck and shoulders only
    5. clamp to the pose range the unstyled firmware already reaches

Step 5 matters. The existing firmware never clamps expression servos, and an
amplitude of 1.3 on a shoulder would ask for 90 + 1.3*80 = 194 degrees. Clamping
to the span the original move sets already use makes styling provably unable to
drive a servo anywhere the stock firmware would not have driven it anyway.
"""

from typing import Dict, List, Tuple

# ── Symbols, as ServoControl.ino defines them ───────────────────────────────
D, M, U, R, L, C = 0, 1, 2, 3, 4, 6
SYMBOL_NAME = {D: "D", M: "M", U: "U", R: "R", L: "L", C: "C"}

# ── Angle tables, transcribed from the setX() functions ─────────────────────
# Every entry is the exact angle that firmware writes today. `rest` is the M
# pose — the robot's resting position, and what amplitude scales away from.
SERVOS: Dict[str, Dict] = {
    "Ears":       {"rest": 130, "angles": {D: 120, M: 130, U: 165}},
    # setBrows: R = 90 + off, L = 90 - off, off: U->0, M->30, D->60
    "RBrow":      {"rest": 120, "angles": {U: 90, M: 120, D: 150}},
    "LBrow":      {"rest": 60,  "angles": {U: 90, M: 60,  D: 30}},
    # setEyes: off U->40, M->20, D->0
    "REyelid":    {"rest": 110, "angles": {D: 90, M: 110, U: 130}},
    "LEyelid":    {"rest": 70,  "angles": {D: 90, M: 70,  U: 50}},
    # setNeck writes both sides from one symbol
    "RNeck":      {"rest": 82,  "angles": {D: 70, M: 82, U: 100, R: 75, L: 100}},
    "LNeck":      {"rest": 103, "angles": {D: 110, M: 103, U: 80, R: 85, L: 120}},
    # setShoulder: off U->80, M->50, D->-40
    "RShoulder":  {"rest": 140, "angles": {D: 50, M: 140, U: 170}},
    "LShoulder":  {"rest": 40,  "angles": {D: 130, M: 40,  U: 10}},
    # setHand: off D->-40, M->0, U->60
    "RHand":      {"rest": 90,  "angles": {D: 50, M: 90, U: 150}},
    "LHand":      {"rest": 90,  "angles": {D: 130, M: 90, U: 30}},
}

# Which move-set column drives which servo, and whether it is per-step.
# Brows are a single symbol for the whole gesture, everything else is 5 steps.
CHANNELS: List[Tuple[str, str, bool]] = [
    ("Ears", "ears", True),
    ("RBrow", "rbrow", False),
    ("LBrow", "lbrow", False),
    ("REyelid", "reye", True),
    ("LEyelid", "leye", True),
    ("RNeck", "neck", True),
    ("LNeck", "neck", True),
    ("RShoulder", "rshoulder", True),
    ("LShoulder", "lshoulder", True),
    ("RHand", "rhand", True),
    ("LHand", "lhand", True),
]

# ── Move sets, transcribed from ServoControl.ino ────────────────────────────
# A subset — enough to see the styling work. Add more by copying the arrays
# across in the same column order the struct declares them.
MOVE_SETS: Dict[str, Dict[str, object]] = {
    "greeting": {
        "ears":  [U, U, U, U, U], "rbrow": M, "lbrow": M,
        "reye":  [U, D, D, D, D], "leye": [U, D, D, D, D],
        "neck":  [M, M, M, M, M],
        "rshoulder": [U, U, D, D, D], "lshoulder": [D, D, D, D, D],
        "rhand": [M, M, M, M, M], "lhand": [M, M, M, M, M],
    },
    "wave": {
        "ears":  [U, U, U, U, U], "rbrow": M, "lbrow": M,
        "reye":  [U, U, D, U, U], "leye": [U, U, D, U, U],
        "neck":  [U, U, M, M, M],
        "rshoulder": [U, U, U, U, U], "lshoulder": [D, D, D, D, D],
        "rhand": [M, U, M, U, M], "lhand": [M, M, M, M, M],
    },
    "sad": {
        "ears":  [M, M, D, D, D], "rbrow": M, "lbrow": M,
        "reye":  [M, M, D, D, M], "leye": [M, M, D, D, M],
        "neck":  [M, D, D, M, M],
        "rshoulder": [M, M, D, M, M], "lshoulder": [M, M, D, M, M],
        "rhand": [M, M, M, M, M], "lhand": [M, M, M, M, M],
    },
    "angry": {
        "ears":  [U, M, U, M, M], "rbrow": U, "lbrow": U,
        "reye":  [U, M, U, M, M], "leye": [U, M, U, M, M],
        "neck":  [M, M, M, M, M],
        "rshoulder": [M, U, M, M, M], "lshoulder": [M, M, U, M, M],
        "rhand": [M, M, M, M, M], "lhand": [M, M, M, M, M],
    },
    "confused": {
        "ears":  [U, U, M, M, M], "rbrow": U, "lbrow": D,
        "reye":  [U, M, U, M, M], "leye": [U, M, U, M, M],
        "neck":  [M, R, M, L, M],
        "rshoulder": [M, M, U, M, M], "lshoulder": [M, M, M, M, M],
        "rhand": [M, M, M, M, M], "lhand": [M, M, M, M, M],
    },
    "shrug": {
        "ears":  [M, M, M, M, M], "rbrow": U, "lbrow": U,
        "reye":  [M, M, D, M, M], "leye": [M, M, D, M, M],
        "neck":  [M, M, M, M, M],
        "rshoulder": [U, U, U, U, U], "lshoulder": [U, U, U, U, U],
        "rhand": [M, U, U, U, U], "lhand": [M, U, U, U, U],
    },
    "default": {
        "ears":  [U, U, U, U, C], "rbrow": M, "lbrow": M,
        "reye":  [U, U, U, U, C], "leye": [U, U, U, U, C],
        "neck":  [M, M, M, M, C],
        "rshoulder": [D, D, D, D, C], "lshoulder": [D, D, D, D, C],
        "rhand": [M, M, M, M, C], "lhand": [M, M, M, M, C],
    },
}

STEPS = 5

# Poses whose job is to *be* a specific position rather than to express
# something. Amplitude scales a servo toward its rest angle, which on a home pose
# undoes the pose: 'default' puts the arms down at 50 degrees, and an amplitude of
# 0.34 drags them back up to 102 — half-raised, which is neither the home pose nor
# a gesture. So amplitude is forced to 1.0 for these.
#
# Droop and posture still apply, on purpose. Those are resting carriage, and a
# robot that has just greeted you sadly should keep looking sad while it waits
# rather than snapping back to a bright neutral the instant the gesture ends.
HOME_TAGS = {"default", "sleep"}

# ...with one exception. On a home pose the neck returns level, because droop and
# posture both push it the same way for a bright, dominant robot and together they
# consume 12 of its 30 degrees of travel — the head parks visibly off-centre and
# stays there. A slumped shoulder or a drooped ear reads as mood; a permanently
# tilted head reads as a fault, or as looking at something else.
#
# Gestures are untouched: the neck still moves with the mood while one plays.
HOME_LEVEL_SERVOS = {"RNeck", "LNeck"}

# ── How droop and posture reach each servo ──────────────────────────────────
# Degrees at full deflection (droop or posture = 1.0), signed so that a positive
# droop reads as "sadder" and a positive posture as "more open". Ears carry the
# most because they are the highest-impact channel on this build; hands carry
# none, because a drooping hand just looks like a broken servo.
#
# These weights are the tuning knobs. Start here if a mood does not read right.
DROOP_DEG = {
    "Ears": -20,        # ears fall
    "RBrow": +12, "LBrow": -12,      # brows down (R increases toward D=150)
    "REyelid": -10, "LEyelid": +10,  # lids lower (R decreases toward D=90)
    "RNeck": -8, "LNeck": +8,        # head dips
    "RShoulder": -12, "LShoulder": +12,   # shoulders slump
    "RHand": 0, "LHand": 0,
}

POSTURE_DEG = {
    "RNeck": +10, "LNeck": -10,           # chin up and back
    "RShoulder": +8, "LShoulder": -8,     # shoulders open
}

# Amplitude tops out at 1.0 on purpose. Each servo only has three symbols, so
# D and U *are* the ends of its stock range — asking for 1.3 of the way to an end
# stop just saturates against the clamp and changes nothing. Treat the stock move
# sets as full expression, which styling can damp but not exceed. Then ELLEBOT is
# simply "all of it" and CHATBOX "a third of it", which is the right reading
# anyway: the gestures were authored at full size.
#
# If you do want a robot to overshoot the authored poses, raise HEADROOM_DEG
# below — but measure the mechanical travel first, because nothing else will stop
# a servo grinding against its end stop.
HEADROOM_DEG = 0

STYLE_LIMITS = {
    "amplitude": (0.30, 1.00),
    "tempo":     (0.50, 1.60),
    "posture":   (-1.00, 1.00),
    "droop":     (-1.00, 1.00),
    "idle":      (0.05, 1.00),
}

NEUTRAL_STYLE = {"amplitude": 1.0, "tempo": 1.0, "posture": 0.0,
                 "droop": 0.0, "idle": 0.45}

# How much of an authored gesture each body physically performs. Deliberately
# NOT the same number as AFFECT_LAB's `show`: that one scales a PAD coordinate,
# and reusing it here multiplies the restraint twice — CHATBOX ends up at 0.19
# amplitude, which does not read as reserved, it reads as broken. These are
# mechanical fractions, chosen so the damping is clearly visible without the
# robot looking faulty.
TRAVEL = {
    "CHATBOX": 0.65,   # tabletop: the same gestures, kept small
    "ELLEBOT": 1.00,   # mobile: performs them at authored size
}

# Playback timing in the firmware today: 900 ms per step.
BASE_STEP_MS = 900


def servo_range(name: str) -> Tuple[int, int]:
    """The angle span styling is allowed to use.

    By default this is exactly the span the stock firmware already reaches, so
    styling can never command a pose the unstyled robot would not have commanded
    anyway. HEADROOM_DEG widens it if you have measured the real travel; the
    result is still clipped to a servo's physical 0-180.
    """
    angles = SERVOS[name]["angles"].values()
    return (max(0, min(angles) - HEADROOM_DEG),
            min(180, max(angles) + HEADROOM_DEG))


def clamp_style(style: Dict[str, float]) -> Dict[str, float]:
    """Fill in missing keys and clamp the rest. Never trust the wire."""
    out = dict(NEUTRAL_STYLE)
    out.update({k: v for k, v in (style or {}).items() if k in STYLE_LIMITS})
    for k, (lo, hi) in STYLE_LIMITS.items():
        out[k] = max(lo, min(hi, float(out[k])))
    return out


def resolve_servo(name: str, symbol: int, style: Dict[str, float],
                  home: bool = False) -> int:
    """One servo, one step: symbol -> styled angle."""
    spec = SERVOS[name]
    rest = spec["rest"]

    # C means "cancel" — the firmware treats it as end-of-sequence, so hold rest.
    target = spec["angles"].get(symbol, rest)

    # 1+2. scale the excursion from rest
    angle = rest + style["amplitude"] * (target - rest)

    # 3+4. mood: valence tint, then carriage. Skipped for the neck on a home pose
    # so the head comes back level — see HOME_LEVEL_SERVOS.
    if not (home and name in HOME_LEVEL_SERVOS):
        angle += style["droop"] * DROOP_DEG.get(name, 0)
        angle += style["posture"] * POSTURE_DEG.get(name, 0)

    # 5. never leave the stock pose range
    lo, hi = servo_range(name)
    return int(round(max(lo, min(hi, angle))))


def resolve_gesture(tag: str, style: Dict[str, float] = None
                    ) -> Dict[str, object]:
    """A whole tag under one style: per-step angles for every servo.

    Returns {"tag", "style", "step_ms", "angles": {servo: [5 ints]}}.
    """
    if tag not in MOVE_SETS:
        raise KeyError(f"unknown tag {tag!r}; have {sorted(MOVE_SETS)}")
    st = clamp_style(style)
    home = tag in HOME_TAGS
    if home:
        # Reach the home pose fully; keep the mood in droop and posture, except
        # on the neck, which comes back level.
        st = dict(st, amplitude=1.0)
    move = MOVE_SETS[tag]

    angles: Dict[str, List[int]] = {}
    for servo, column, per_step in CHANNELS:
        raw = move[column]
        seq = raw if per_step else [raw] * STEPS
        angles[servo] = [resolve_servo(servo, seq[i], st, home)
                         for i in range(STEPS)]

    return {
        "tag": tag,
        "style": st,
        # Tempo changes timing, never angles. Higher tempo = shorter step.
        "step_ms": int(round(BASE_STEP_MS / st["tempo"])),
        "angles": angles,
    }


def wire_message(style: Dict[str, float]) -> str:
    """The line the Jetson would send ahead of a tag."""
    s = clamp_style(style)
    return ("STYLE {amplitude:.2f} {tempo:.2f} {posture:+.2f} "
            "{droop:+.2f} {idle:.2f}").format(**s)


def parse_wire(line: str) -> Dict[str, float]:
    """Inverse of wire_message, matching the firmware's parser."""
    parts = line.strip().split()
    if not parts or parts[0].upper() != "STYLE" or len(parts) != 6:
        raise ValueError(f"malformed style line: {line!r}")
    keys = ("amplitude", "tempo", "posture", "droop", "idle")
    return clamp_style(dict(zip(keys, (float(p) for p in parts[1:]))))
