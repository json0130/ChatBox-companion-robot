"""
Which source writes which PAD axis — the object E3 permutes.

The system's central architectural claim is that three influences at three
timescales each own a DIFFERENT axis, so they add instead of overwriting:

    OCEAN persona  -> the baseline point   (fixed)
    face valence   -> P                    (per frame)
    face arousal   -> Ar                   (per frame)
    relationship   -> D                    (across sessions)

That claim is only interesting if the assignment could have been otherwise. This
module makes the routing a parameter so the alternatives can be measured rather
than asserted.

WHAT IS PERMUTED, AND WHAT IS NOT
---------------------------------
Only the SOURCE -> AXIS routing moves. The effector bindings are held fixed: the
behavioural directive keeps reading axis D, the descriptor words keep reading
(P, Ar, D) in that order, and `gesture_style` is untouched. Permuting the
effectors as well would give a trivially different and far less interesting
result — it would merely relabel the axes, and nothing could be learned from it.

`affect.py` IS NOT MODIFIED
--------------------------
`compose` calls `affect.to_pad` for the baseline and then routes the deltas
itself. Under IDENTITY it must reproduce `affect.feel_with_relationship` bit for
bit; `tests/test_axes.py` asserts that over the full grid plus 1000 random
points. That mirrors the guarantee `affect.pipeline` already makes about
`tier="known"` (affect.py:365-369) and is what makes this machinery a strict
superset rather than a reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from modules.affect_bridge import affect

AXES = ("P", "Ar", "D")


@dataclass(frozen=True)
class AxisAssignment:
    """A routing of the three sources onto the three PAD axes.

    valence_axis / arousal_axis: which axis each half of the face writes, via
        the empathy fraction-of-the-gap in `affect.feel`.
    tier_axis: which axis the relationship tier's principal displacement writes
        (the +/-0.40 / -0.20 term, i.e. index 2 of the TIER_OFFSETS tuple).
    collapse: all three sources onto `tier_axis`, the degenerate assignment.
    """
    name: str
    valence_axis: str = "P"
    arousal_axis: str = "Ar"
    tier_axis: str = "D"
    collapse: bool = False

    def __post_init__(self) -> None:
        for field in ("valence_axis", "arousal_axis", "tier_axis"):
            value = getattr(self, field)
            if value not in AXES:
                raise ValueError(f"{field}={value!r} is not one of {AXES}")


# The deployed system.
IDENTITY = AxisAssignment("identity")

# Emotion and relationship swap axes: the face drives Dominance (and therefore
# the behavioural directive), the relationship drives Pleasure. This is the
# assignment that should produce directive chattering — a per-frame signal
# driving a per-session effector.
PERM_EMOTION_D = AxisAssignment("perm_emotion_D",
                                valence_axis="D", arousal_axis="Ar",
                                tier_axis="P")

# Arousal and relationship swap.
PERM_AROUSAL_D = AxisAssignment("perm_arousal_D",
                                valence_axis="P", arousal_axis="D",
                                tier_axis="Ar")

# Every source onto one axis. The sources become unrecoverable in principle,
# not merely in practice — E2's Jacobian loses rank in the source subspace.
COLLAPSE_D = AxisAssignment("collapse_D",
                            valence_axis="D", arousal_axis="D", tier_axis="D",
                            collapse=True)

ASSIGNMENTS: Dict[str, AxisAssignment] = {
    a.name: a for a in (IDENTITY, PERM_EMOTION_D, PERM_AROUSAL_D, COLLAPSE_D)
}


def _clamp(v: float) -> float:
    """Identical to affect._clamp. Duplicated rather than imported because it is
    private there, and the bit-identity test would silently pass if a future
    change to one were mirrored into the other by accident."""
    return max(-1.0, min(1.0, v))


def route_offsets(assignment: AxisAssignment,
                  offsets: Tuple[float, float, float]) -> Dict[str, float]:
    """Route an explicit (dP, dAr, dD) offset triple onto axes.

    `affect.TIER_OFFSETS` is authored in AXIS space, not source space: index 2 is
    the tier's principal displacement and indices 0/1 are deliberate secondary
    nudges onto whichever axes the face owns. Today only one is non-zero — the
    `close` tier's +0.10 on arousal (affect.py:110) — which is a real violation
    of the disjoint-axis claim and is kept so E2 can measure it.

    Routing follows the SEMANTICS rather than the index: the principal term goes
    to `tier_axis`, the pleasure-nudge follows whichever axis valence owns, and
    the arousal-nudge follows whichever axis arousal owns. Under IDENTITY this is
    the identity mapping.

    Taking the triple explicitly is what lets E2 run the leak/no-leak pair
    without monkey-patching `affect.TIER_OFFSETS` — the ablation is a caller's
    argument, not a mutation of the shipped model.
    """
    d_p, d_ar, d_d = offsets
    out = {"P": 0.0, "Ar": 0.0, "D": 0.0}
    out[assignment.tier_axis] += d_d
    out[assignment.valence_axis] += d_p
    out[assignment.arousal_axis] += d_ar
    return out


def tier_deltas(assignment: AxisAssignment, tier: str) -> Dict[str, float]:
    """Route a named tier's offset onto axes. See `route_offsets`."""
    return route_offsets(assignment, affect.tier_offset(tier))


def compose(assignment: AxisAssignment,
            baseline: Dict[str, float],
            valence: float,
            arousal: float,
            tier: str,
            empathy: float = affect.EMPATHY,
            ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Baseline + face + relationship -> one felt coordinate, under `assignment`.

    Returns (felt, clamp_flags). `clamp_flags[axis]` is True when that axis was
    driven outside [-1, +1] and had to be clipped — recorded per trial because
    clamping destroys information (CHATBOX at `unknown` wants D=-1.043) and fires
    far more often under permutation, which makes its rate a measure of
    interference rather than a nuisance.
    """
    return compose_offsets(assignment, baseline, valence, arousal,
                           affect.tier_offset(tier), empathy)


def compose_offsets(assignment: AxisAssignment,
                    baseline: Dict[str, float],
                    valence: float,
                    arousal: float,
                    offsets: Tuple[float, float, float],
                    empathy: float = affect.EMPATHY,
                    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """`compose` with the relationship supplied as an explicit offset triple.

    Lets E2 sweep the relationship continuously (a tier is a discrete factor, so
    a derivative with respect to it is otherwise undefined) and run the
    leak/no-leak ablation without mutating the shipped table.
    """
    felt = dict(baseline)

    if assignment.collapse:
        # One axis carries everything. The two face signals are averaged and
        # fused once, rather than fused twice, which would double-count the
        # baseline and make the collapse look artificially violent.
        axis = assignment.tier_axis
        face = (valence + arousal) / 2.0
        felt[axis] = baseline[axis] + empathy * (face - baseline[axis])
    else:
        # Each face signal displaces its own axis by a fraction of the gap to
        # that axis's baseline — exactly affect.feel, but routed.
        felt[assignment.valence_axis] = (
            baseline[assignment.valence_axis]
            + empathy * (valence - baseline[assignment.valence_axis])
        )
        felt[assignment.arousal_axis] = (
            baseline[assignment.arousal_axis]
            + empathy * (arousal - baseline[assignment.arousal_axis])
        )

    deltas = route_offsets(assignment, offsets)
    raw = {axis: felt[axis] + deltas[axis] for axis in AXES}
    clamped = {axis: _clamp(raw[axis]) for axis in AXES}
    flags = {axis: raw[axis] != clamped[axis] for axis in AXES}
    return clamped, flags
