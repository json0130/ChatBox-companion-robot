"""
affect.py — persona, emotion and embodiment fused into one PAD coordinate.

Pure functions only: no camera, no model, no window. Everything here can be
tested from a plain Python prompt, which is why the numbers are trustworthy
before any hardware is involved.

Three influences, three sources, one coordinate:

    OCEAN persona   -> baseline PAD           (Mehrabian regressions, via ALMA)
    detected face   -> pulls P and Ar         (Russell's circumplex axes)
    the embodiment  -> scales what is shown   (how many channels it has to say it)

Dominance is deliberately untouched by the face. What a facial expression
reliably carries is pleasure and arousal — the two dimensions of core affect —
while dominance reflects social standing between two parties and has to come
from recognising *who* the person is, not how they look.
"""

from typing import Dict, Tuple

# ── Persona -> baseline temperament ─────────────────────────────────────────
# Mehrabian's temperament regressions as used by the ALMA model. Traits are on
# [-1, +1] with 0 as the population mean; a config storing them on [0, 1] must
# be remapped with (2v - 1) first or every coordinate lands too positive.
WEIGHTS = {
    "P":  {"E": 0.21, "A":  0.59, "N":  0.19},
    "Ar": {"O": 0.15, "A":  0.30, "N": -0.57},
    "D":  {"O": 0.25, "C":  0.17, "E":  0.60, "A": -0.32},
}

TRAIT_KEYS = ("O", "C", "E", "A", "N")


def to_pad(traits: Dict[str, float]) -> Dict[str, float]:
    """OCEAN vector -> baseline (P, Ar, D)."""
    return {axis: sum(w * traits[t] for t, w in terms.items())
            for axis, terms in WEIGHTS.items()}


# ── The two robots ─────────────────────────────────────────────────────────
# `ocean` is the published persona for each; `show` is how much of its internal
# temperament the body can actually put on display. ELLEBOT adds a wheeled base
# that can approach and turn plus large fan ears devoted to amplifying valence,
# where CHATBOX is a fixed tabletop unit — so the same temperament reads muted
# on one and vivid on the other.
#
# The `show` values are a modelling proposal. The paper describes the asymmetry
# in expressive channels but puts no number on it, so treat them as tunable.
ROBOTS = {
    "CHATBOX": {
        "ocean": {"O": -0.5, "C": 0.2, "E": -0.6, "A": 0.6, "N":  0.2},
        "show": 0.30,
        "body": "fixed tabletop, 12-DOF upper face",
    },
    "ELLEBOT": {
        "ocean": {"O":  0.5, "C": 0.4, "E":  0.7, "A": 0.6, "N": -0.4},
        "show": 1.00,
        "body": "wheeled base, fan ears, 12-DOF upper face",
    },
}

# How hard the person's expression pulls the robot off its own temperament.
# Held equal across bodies on purpose: empathy is a property of the software,
# not of how many servos are available to express it with.
EMPATHY = 0.6


def feel(baseline: Dict[str, float], valence: float, arousal: float,
         empathy: float = EMPATHY) -> Dict[str, float]:
    """Displace the baseline toward what was detected in the person's face.

    Only P and Ar move. Dominance is carried through untouched — the face has
    no reliable say in it.
    """
    return {
        "P":  baseline["P"] + empathy * (valence - baseline["P"]),
        "Ar": baseline["Ar"] + empathy * (arousal - baseline["Ar"]),
        "D":  baseline["D"],
    }


def show(coord: Dict[str, float], fraction: float) -> Dict[str, float]:
    """Scale an internal coordinate down to what a given body can display."""
    return {k: fraction * v for k, v in coord.items()}


# ── Naming a coordinate ────────────────────────────────────────────────────
# Russell's octants, read off the angle in valence-arousal space: 0 degrees is
# pleasant-and-unaroused, then anticlockwise one name per 45 degrees.
SECTORS = ("pleased", "elated", "alert", "tense",
           "unhappy", "dejected", "drowsy", "serene")

# Per-axis descriptor bands — the words that would go into the LLM system
# prompt. Checked against the paper's worked example: CHATBOX's published
# coordinate returns "warm, calm, reserved".
BANDS = {
    "P":  ((0.50, "affectionate"), (0.15, "warm"),      (-0.15, "even"),
           (-0.50, "cool"),        (-9.0, "cold")),
    "Ar": ((0.50, "excitable"),    (0.15, "lively"),     (-0.15, "calm"),
           (-0.50, "placid"),      (-9.0, "languid")),
    "D":  ((0.50, "commanding"),   (0.15, "assertive"),  (-0.15, "even-handed"),
           (-0.70, "reserved"),    (-9.0, "retiring")),
}


def band(axis: str, value: float) -> str:
    for edge, word in BANDS[axis]:
        if value >= edge:
            return word
    return BANDS[axis][-1][1]


def descriptors(coord: Dict[str, float]) -> Tuple[str, str, str]:
    """The three words that describe this coordinate, for the LLM prompt."""
    return band("P", coord["P"]), band("Ar", coord["Ar"]), band("D", coord["D"])


def affect_name(pleasure: float, arousal: float) -> str:
    """Nearest named region of the circumplex, with an intensity qualifier."""
    import math
    radius = math.hypot(pleasure, arousal)
    # Near the origin the angle is all noise and no octant is meaningful.
    if radius < 0.12:
        return "neutral"
    degrees = math.degrees(math.atan2(arousal, pleasure)) % 360
    name = SECTORS[round(degrees / 45) % 8]
    if radius < 0.35:
        return "mildly " + name
    if radius > 0.72:
        return "strongly " + name
    return name


# ── Circumplex fallback for categorical models ─────────────────────────────
# Only needed when running a classifier rather than a valence-arousal model.
# These are estimates from the circumplex literature, not measurements — a real
# VA model reports the position directly and makes this table redundant.
CATEGORY_VA = {
    "anger":     (-0.70,  0.65),
    "angry":     (-0.70,  0.65),
    "contempt":  (-0.55,  0.20),
    "disgust":   (-0.70,  0.30),
    "fear":      (-0.65,  0.72),
    "happiness": (0.80,  0.50),
    "happy":     (0.80,  0.50),
    "neutral":   (0.00,  0.00),
    "sadness":   (-0.70, -0.38),
    "sad":       (-0.70, -0.38),
    "surprise":  (0.20,  0.80),
}


def category_to_va(label: str) -> Tuple[float, float]:
    """Map a discrete emotion label onto the circumplex. Unknown -> neutral."""
    return CATEGORY_VA.get(label.strip().lower(), (0.0, 0.0))


# ── PAD -> gesture style ───────────────────────────────────────────────────
# The paper names four movement parameters — amplitude, tempo, posture, idle
# frequency — and says the PAD coordinate maps to them, but gives no equations.
# These are that missing step, and they are a proposal: the coefficients below
# are chosen so the two published personas reproduce the paper's own qualitative
# claim ("a broad, brisk wave from ELLEBOT, a slow, gentle one from CHATBOX").
#
# Note which axes feed what, and which does not. Pleasure is absent from
# amplitude and tempo on purpose: valence decides *which* gesture plays — a wave
# rather than a slump — while Arousal and Dominance decide *how* it is performed.
# Arousal carries energy (speed, size, restlessness) and Dominance carries
# expansiveness, which is also what the nonverbal literature reports.
STYLE_LIMITS = {
    "amplitude": (0.30, 1.30),   # scale on each servo's travel from neutral
    "tempo":     (0.50, 1.60),   # multiplier on playback speed
    "posture":   (-1.00, 1.00),  # neck/shoulder carriage, withdrawn -> open
    "idle":      (0.05, 1.00),   # how often it stirs between gestures
}


def gesture_style(coord: Dict[str, float]) -> Dict[str, float]:
    """PAD -> the four movement parameters the firmware would consume."""
    raw = {
        "amplitude": 0.75 + 0.45 * coord["Ar"] + 0.20 * coord["D"],
        "tempo":     0.85 + 0.55 * coord["Ar"] + 0.15 * coord["D"],
        "posture":   0.70 * coord["D"] + 0.30 * coord["P"],
        "idle":      0.45 + 0.40 * coord["Ar"],
    }
    return {k: max(lo, min(hi, raw[k]))
            for k, (lo, hi) in STYLE_LIMITS.items()}


def pipeline(traits: Dict[str, float], valence: float, arousal: float,
             robot: str, empathy: float = EMPATHY) -> Dict[str, object]:
    """The whole chain, in one call — what the demo overlay renders."""
    baseline = to_pad(traits)
    felt = feel(baseline, valence, arousal, empathy)
    shown = show(felt, ROBOTS[robot]["show"])
    return {
        "baseline": baseline,
        "felt": felt,
        "shown": shown,
        "name": affect_name(shown["P"], shown["Ar"]),
        "words": descriptors(shown),
        # Derived from `felt`, not `shown`. The body's display fraction and
        # amplitude are two descriptions of the same restraint, so multiplying
        # them would apply it twice and flatten the contrast the paper claims.
        "style": gesture_style(felt),
    }
