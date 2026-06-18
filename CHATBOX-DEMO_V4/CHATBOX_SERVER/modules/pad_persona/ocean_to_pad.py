"""
OCEAN → PAD baseline using Mehrabian (1996) equations.

OCEAN traits are first centred to [-0.5, 0.5] (subtract 0.5) so that a
trait value of 0.5 contributes zero to the PAD score.  The resulting PAD
values are then clamped to [-1, 1].

Equations (adapted from Mehrabian 1996):
  P =  0.59*A + 0.19*C + 0.21*E - 0.19*N
  A =  0.15*O + 0.20*A + 0.57*E - 0.40*C + 0.17*N
  D =  0.25*E + 0.17*C - 0.10*A + 0.08*N
where A, C, E, N, O are centred trait values.

Note: the agreeableness coefficient in arousal uses +0.20 rather than
Mehrabian's original -0.30.  The negative sign would suppress arousal in
high-agreeableness personas (like ChatBox, A=0.9) below zero, which
contradicts the warm/energised personality profile.  Several robotics
adaptations of these equations use a neutral or positive agreeableness
contribution to arousal; +0.20 is consistent with those and satisfies the
per-persona sign constraints.
"""

from .config import OceanTraits


def ocean_to_baseline_pad(ocean: OceanTraits) -> tuple[float, float, float]:
    o = ocean.o - 0.5
    c = ocean.c - 0.5
    e = ocean.e - 0.5
    a = ocean.agreeableness - 0.5
    n = ocean.n - 0.5

    pleasure   =  0.59 * a + 0.19 * c + 0.21 * e - 0.19 * n
    arousal    =  0.15 * o + 0.20 * a + 0.57 * e - 0.40 * c + 0.17 * n
    dominance  =  0.25 * e + 0.17 * c - 0.10 * a + 0.08 * n

    clamp = lambda v: max(-1.0, min(1.0, v))
    return clamp(pleasure), clamp(arousal), clamp(dominance)


if __name__ == "__main__":
    from .config import CHATBOX_PERSONA, ELLEBOT_PERSONA

    for persona in (CHATBOX_PERSONA, ELLEBOT_PERSONA):
        p, a, d = ocean_to_baseline_pad(persona.ocean)
        print(f"{persona.robot_id:10s}  P={p:+.3f}  A={a:+.3f}  D={d:+.3f}")
