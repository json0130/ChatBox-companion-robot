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

One coordinate, two outputs:

    PAD -> three descriptor words   -> the LLM prompt
    PAD -> five style values        -> servo angles

Of the five style values, `amplitude` and `tempo` come from a published
equation -- Hagane & Venture, Machines 10(12):1118, 2022, following Claret,
Venture & Basanez, Int. J. Social Robotics 9:277-292, 2017 -- which maps PAD
directly to motion features. `posture`, `idle` and `droop` are this project's
own proposals; no published PAD equation for them could be found. Every one is
marked at its definition, and CONCEPT.md section 10 keeps the tally.
"""

import math
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


# ── Relationship -> Dominance ──────────────────────────────────────────────
# The third influence, and the one the face is deliberately silent on (see
# `feel`). Dominance reflects social standing between two parties, so it has to
# come from recognising *who* the person is rather than how they look. That
# recognition lives in the knowledge graph: a person and a robot share one
# InteractionNode carrying rapport, trust and a turn count, and the graph's
# `kg_bridge.derive_tier` reduces those to one of these five tiers. This table
# is the only thing that crosses back, which is what keeps this module free of
# any graph dependency.
#
# PROPOSAL. No published equation maps a relationship tier onto a PAD offset.
# The DIRECTION is not a guess -- a familiar partner licenses more assertive
# behaviour and a stranger less -- but the magnitudes are tuned by eye.
TIER_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "close":   (0.00, +0.10, +0.40),
    "family":  (0.00, +0.05, +0.20),
    "known":   (0.00,  0.00,  0.00),   # reference tier: no displacement at all
    "visitor": (0.00,  0.00, -0.20),
    "unknown": (0.00,  0.00, -0.40),
}

TIERS = tuple(TIER_OFFSETS)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def tier_offset(tier: str) -> Tuple[float, float, float]:
    """(dP, dAr, dD) for a relationship tier. Unrecognised -> 'unknown'."""
    return TIER_OFFSETS.get(tier, TIER_OFFSETS["unknown"])


def feel_with_relationship(baseline: Dict[str, float],
                           valence: float, arousal: float, tier: str,
                           empathy: float = EMPATHY) -> Dict[str, float]:
    """All three influences in one coordinate.

    `feel` moves P and Ar toward the detected face; the tier then displaces
    Dominance (and nudges Arousal at the two closest tiers). They never fight,
    because each pushes along its own axis.

    Applied BEFORE `show`, deliberately: `gesture_style` reads the *felt*
    coordinate, not the shown one, so a tier applied after `show` would never
    reach a servo at all. The design also admits exactly one embodiment scaling.

    Clamped to [-1, +1] -- CHATBOX's baseline D of -0.643 plus the 'unknown'
    offset lands at -1.043, outside the space PAD is defined on and outside the
    domain the published Eq. 9 / Eq. A1 assume.
    """
    felt = feel(baseline, valence, arousal, empathy)
    dP, dAr, dD = tier_offset(tier)
    return {"P":  _clamp(felt["P"] + dP),
            "Ar": _clamp(felt["Ar"] + dAr),
            "D":  _clamp(felt["D"] + dD)}


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
# Five values, each computed straight from the PAD coordinate and clamped. Two
# of them come from a published equation; three are this project's own, and the
# split is marked on every one so it cannot be misread.
#
#   amplitude   PUBLISHED   Hagane & Venture (2022), Eq. 9    "spatial extent"
#   tempo       PUBLISHED   Hagane & Venture (2022), Eq. A1   "velocity"
#   posture     OURS        the deck, p.15
#   idle        OURS        the deck, p.15
#   droop       OURS        the deck, p.16
#
# --- The published half -----------------------------------------------------
# S. Hagane and G. Venture, "Robotic Manipulator's Expressive Movements Control
# Using Kinematic Redundancy", Machines 10(12):1118, 2022, following
# J.-A. Claret, G. Venture and L. Basanez, "Exploiting the Robot Kinematic
# Redundancy for Emotion Conveyance to Humans as a Lower Priority Task",
# Int. J. Social Robotics 9:277-292, 2017.
#
# Both map PAD to motion features by, in their words, "a simple linear formula".
# Their Eq. 9 defines three features on [0,1] from a PAD point on [-1,1]^3:
#
#     Jr (jerkiness)      = (1 - P) / 2
#     Ve (velocity)       = fv(P, A, D)          <- Eq. A1, below
#     Sp (spatial extent) = (D + 1) / 2
#
# `Sp` is amplitude: the paper defines Extent as "how large the gestures of the
# hands and arms are". `Ve` is tempo. `Jr` is dropped because this firmware
# plays a fixed five-step sequence and controls only step_ms -- there is no
# actuator for jerk. Claret's third feature is gaze, which Hagane replaced with
# Sp because a robot arm has no eyes; this robot has no gaze servos either.
#
# Two properties of Sp worth knowing before reading the output:
#   - It depends on Dominance ALONE. The face never moves Dominance here, so
#     amplitude is fixed per persona and a detected emotion cannot resize a
#     gesture. That follows this project's own design, but it is a real
#     consequence rather than an accident.
#   - The authors flag their own limitation on Ve (their sec. 7): the values
#     "were not scattered between 0 and 1 but clustered between 0 and 0.5,
#     resulting in similar movements ... except for the case of hostile".
#
# The one unavoidable step: Sp and Ve are dimensionless indices on [0,1], while
# amplitude is a fraction of authored servo travel and tempo is a multiplier on
# playback speed. Each is mapped affinely onto the range this project already
# documents in STYLE_LIMITS. That is a unit conversion onto our own published
# range, not a new equation -- no coefficient is chosen.
#
# --- The unpublished half ---------------------------------------------------
# posture, idle and droop are kept exactly as the deck states them. No published
# PAD equation for any of the three could be found: Claret uses gaze for
# Dominance, Hagane uses spatial extent, and neither models carriage, resting
# stir rate, or a signed vertical tint. These three remain proposals, and are
# labelled as such in CONCEPT.md section 10.

_HALF_ROOT_TWO_GAP = 2 - math.sqrt(2)


def _velocity(coord: Dict[str, float]) -> float:
    """Hagane & Venture (2022) Eq. A1 — the fv mapping, on [0, 1].

    Verified against the four anchor values the paper states in its Eq. 10:
    Hostile (-1,1,1) -> 1.0, Exuberant (1,1,1) -> 0.5,
    Anxious (-1,1,-1) -> 0.5, Bored (-1,-1,-1) -> 0.0.
    """
    Pn = coord["P"] + 1
    An = coord["Ar"] + 1
    Dn = coord["D"] + 1
    r = math.hypot(An, Dn)
    if r == 0:                      # Bored corner; the paper's fv is 0 there
        return 0.0
    beta = math.acos(max(-1.0, min(1.0, Dn / r)))
    S = 2 + _HALF_ROOT_TWO_GAP * math.sin(2 * beta + math.pi)
    return 0.125 * r / S * (4 - Pn)


def _spatial_extent(coord: Dict[str, float]) -> float:
    """Hagane & Venture (2022) Eq. 9 — Sp, on [0, 1]. Dominance only."""
    return (coord["D"] + 1) / 2


STYLE_LIMITS = {
    "amplitude": (0.30, 1.00),   # scale on each servo's travel from neutral
    "tempo":     (0.50, 1.60),   # multiplier on playback speed
    "posture":   (-1.00, 1.00),  # neck/shoulder carriage, withdrawn -> open
    "droop":     (-1.00, 1.00),  # signed valence tint, sagging -> lifted
    "idle":      (0.05, 1.00),   # how often it stirs between gestures
}

# The deck's own coefficients, unchanged. See CONCEPT.md section 10 -- these are
# proposals, not findings, and nobody has published an equation for them.
DROOP_GAIN = 1.4


def _onto(value: float, name: str) -> float:
    """Map a published [0,1] index onto the range this firmware documents."""
    lo, hi = STYLE_LIMITS[name]
    return lo + value * (hi - lo)


def gesture_style(coord: Dict[str, float]) -> Dict[str, float]:
    """PAD -> the five movement parameters the firmware consumes."""
    raw = {
        # published — Hagane & Venture (2022)
        "amplitude": _onto(_spatial_extent(coord), "amplitude"),
        "tempo":     _onto(_velocity(coord), "tempo"),
        # ours — the deck, p.15 and p.16
        "posture":   0.70 * coord["D"] + 0.30 * coord["P"],
        "idle":      0.45 + 0.40 * coord["Ar"],
        "droop":     -coord["P"] * DROOP_GAIN,
    }
    return {k: max(lo, min(hi, raw[k]))
            for k, (lo, hi) in STYLE_LIMITS.items()}


def pipeline(traits: Dict[str, float], valence: float, arousal: float,
             robot: str, empathy: float = EMPATHY,
             tier: str = "known") -> Dict[str, object]:
    """The whole chain, in one call — what the demo overlay renders.

    `tier` defaults to "known", whose offset is (0, 0, 0), so a call that omits
    it is bit-identical to the pre-relationship pipeline. That identity is what
    makes the relationship path a strict superset rather than a change, and it
    is asserted in the test suite.
    """
    baseline = to_pad(traits)
    felt = feel_with_relationship(baseline, valence, arousal, tier, empathy)
    shown = show(felt, ROBOTS[robot]["show"])
    return {
        "baseline": baseline,
        "felt": felt,
        "shown": shown,
        "tier": tier,
        "name": affect_name(shown["P"], shown["Ar"]),
        "words": descriptors(shown),
        # Derived from `felt`, not `shown`. The body's display fraction and
        # amplitude are two descriptions of the same restraint, so multiplying
        # them would apply it twice and flatten the contrast between the robots.
        "style": gesture_style(felt),
    }
