"""
pad_core — persona, emotion and relationship fused into one PAD coordinate, and
that coordinate turned into both movement and words.

Self-contained and dependency-free: `affect`, `servo_style` and `prompt` import
nothing but the standard library, so the whole model can be dropped into another
project, exercised with no camera, no LLM and no robot, and cited from its tests.

    camera ──valence/arousal─┐
    OCEAN persona ──────────►├─► PAD ──► five style values ──► servo angles
    KG relationship tier ────┘     └───► three words ────────► LLM prompt

The three influences occupy different axes on purpose — the face moves Pleasure
and Arousal, the relationship moves Dominance, the persona sets where you start —
so they add instead of overwriting each other.

Typical use, one call per conversation turn:

    from pad_core import PADPipelineAdapter

    adapter = PADPipelineAdapter("chatbox")
    out = adapter.process_turn(valence=-0.7, arousal=-0.4, relationship_tier="close")

    out["words"]  # ('even', 'calm', 'even-handed')  -> the system prompt
    out["wire"]   # 'STYLE 0.56 0.89 -0.13 -0.18 0.55' -> the ESP32
    out["style"]  # the five parameters, if you want them directly

Lower-level pieces are re-exported for the bench rigs and the tests:

    from pad_core import affect, servo_style
    affect.to_pad(traits)                                    OCEAN -> baseline PAD
    affect.feel_with_relationship(base, v, a, tier)          all three influences
    affect.descriptors(coord)                                -> three words
    affect.gesture_style(coord)                              -> five parameters
    servo_style.resolve_gesture(tag, style)                  -> per-servo angles
    servo_style.wire_message(style)                          -> the STYLE line

What is published and what is a proposal is tracked in docs/ — read that before
citing any coefficient.
"""

from . import affect, prompt, servo_style
from .adapter import NullPADAdapter, PADPipelineAdapter
from .stream import AffectStream

__version__ = "1.0.0"

__all__ = [
    "affect", "servo_style", "prompt",
    "PADPipelineAdapter", "NullPADAdapter", "AffectStream",
    "__version__",
]
