"""
The server's PAD path is exactly AFFECT_LAB's, with nothing added in between.

`AFFECT_LAB/affect.py` is the single source of truth for the affect maths, and
the server reaches it through `modules.affect_bridge`. These tests pin that:
the adapter must reproduce `affect.pipeline()` term for term, the bench module
must be the one actually imported, and the V/A tables scattered across the
codebase must agree with each other.

The failure this guards against is silent. An earlier `AffectStream` scaled
valence by 0.3 before it reached the engine, which multiplied with the empathy
fraction (0.60) to an effective 0.18 — PAD "worked", it just barely moved.
No exception, no wrong type; only an equality test catches that.

No camera, no LLM, no hardware. Run directly or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.affect_bridge import REPO_ROOT, affect, servo_style   # noqa: E402
from modules.pad_persona.pipeline_adapter import (                 # noqa: E402
    EMOTION_VA, NullPADAdapter, PADPipelineAdapter,
)

_TIERS = ("unknown", "visitor", "known", "family", "close")
_EMOTIONS = ("happy", "sad", "angry", "neutral", "surprise")


def test_affect_resolves_to_the_bench():
    """The server must import AFFECT_LAB's module, not a stray copy — `affect`
    and `servo_style` are bare top-level names and could collide on sys.path."""
    assert affect.__file__.startswith(os.path.join(REPO_ROOT, "AFFECT_LAB"))
    assert servo_style.__file__.startswith(os.path.join(REPO_ROOT, "SERVO_STYLE"))
    print("1. affect/servo_style resolve to the bench directories ✓")


def test_adapter_equals_affect_pipeline():
    """The adapter adds no maths of its own: for every robot, tier and emotion
    its PAD state and style must equal affect.pipeline() exactly."""
    for robot, key in (("chatbox", "CHATBOX"), ("ellebot", "ELLEBOT")):
        ad = PADPipelineAdapter(robot)
        traits = affect.ROBOTS[key]["ocean"]
        for tier in _TIERS:
            for emo in _EMOTIONS:
                v, a = affect.category_to_va(emo)
                # a fresh adapter each time: AffectStream carries a window
                got = PADPipelineAdapter(robot).process_turn(v, a, tier)
                want = affect.pipeline(traits, v, a, key, tier=tier)
                for i, axis in enumerate(("P", "Ar", "D")):
                    assert round(got["pad_state"][i], 12) == round(want["felt"][axis], 12), \
                        f"{robot}/{tier}/{emo} felt {axis}"
                    assert round(got["shown"][i], 12) == round(want["shown"][axis], 12), \
                        f"{robot}/{tier}/{emo} shown {axis}"
                assert got["words"] == want["words"], f"{robot}/{tier}/{emo} words"
                for k, v_ in want["style"].items():
                    assert round(got["style"][k], 12) == round(v_, 12), \
                        f"{robot}/{tier}/{emo} style {k}"
    print(f"2. adapter == affect.pipeline() over {2*len(_TIERS)*len(_EMOTIONS)} "
          "robot/tier/emotion combinations ✓")


def test_no_hidden_attenuation():
    """A gain anywhere between the camera and affect.feel would silently shrink
    the response. With a single frame the stream must be transparent."""
    ad = PADPipelineAdapter("ellebot")
    v, a = 0.8, 0.6
    got = ad.process_turn(v, a, "known")
    want = affect.pipeline(affect.ROBOTS["ELLEBOT"]["ocean"], v, a, "ELLEBOT",
                           tier="known")
    assert round(got["pad_state"][0], 12) == round(want["felt"]["P"], 12)
    assert round(got["pad_state"][1], 12) == round(want["felt"]["Ar"], 12)
    print("3. no attenuation between the camera and affect.feel ✓")


def test_tier_moves_dominance_only():
    """The relationship's axis is D. Changing tier must not touch P or Ar,
    except at the two closest tiers, which deliberately nudge arousal."""
    ad_key = "ELLEBOT"
    traits = affect.ROBOTS[ad_key]["ocean"]
    base = affect.pipeline(traits, 0.0, 0.0, ad_key, tier="known")
    for tier in ("unknown", "visitor"):
        alt = affect.pipeline(traits, 0.0, 0.0, ad_key, tier=tier)
        assert round(alt["felt"]["P"], 12) == round(base["felt"]["P"], 12)
        assert round(alt["felt"]["Ar"], 12) == round(base["felt"]["Ar"], 12)
        assert alt["felt"]["D"] < base["felt"]["D"]
    for tier in ("family", "close"):
        alt = affect.pipeline(traits, 0.0, 0.0, ad_key, tier=tier)
        assert round(alt["felt"]["P"], 12) == round(base["felt"]["P"], 12)
        assert alt["felt"]["D"] > base["felt"]["D"]
    print("4. tier moves Dominance (and only arousal at family/close) ✓")


def test_va_tables_agree():
    """Three V/A tables exist — affect.CATEGORY_VA, the adapter's EMOTION_VA and
    kg_bridge's deliberate private copy (kept so graph_relationship imports
    nothing). They must not drift apart on the labels they share."""
    from modules.graph_relationship.kg_bridge import _EMOTION_VA
    for label, (v, a) in _EMOTION_VA.items():
        if label in affect.CATEGORY_VA:
            cv, ca = affect.CATEGORY_VA[label]
            assert abs(cv - v) < 1e-9 and abs(ca - a) < 1e-9, \
                f"kg_bridge and affect disagree on {label!r}: {(v,a)} vs {(cv,ca)}"
    for label in EMOTION_VA:
        assert EMOTION_VA[label] == affect.CATEGORY_VA[label], f"adapter drift: {label}"
    print(f"5. the V/A tables agree across {len(_EMOTION_VA)} shared labels ✓")


def test_null_adapter_same_shape():
    """NullPADAdapter must be swappable without the caller noticing."""
    live = PADPipelineAdapter("chatbox").process_turn(0.5, 0.3, "known")
    null = NullPADAdapter().process_turn(0.5, 0.3, "known")
    assert set(live) == set(null), set(live) ^ set(null)
    assert null["system_prompt"] is None and live["system_prompt"] is None
    print("6. NullPADAdapter returns the same keys; system_prompt is always None ✓")


def test_wire_roundtrips():
    """What the document prints and what crosses the wire are the same numbers."""
    for robot in ("chatbox", "ellebot"):
        for tier in _TIERS:
            r = PADPipelineAdapter(robot).process_turn(0.0, 0.0, tier)
            back = servo_style.parse_wire(r["wire"])
            for k, v in servo_style.clamp_style(r["style"]).items():
                assert abs(back[k] - v) < 0.005, f"{robot}/{tier} {k}"
    print("7. parse_wire(wire_message(style)) round-trips over the tier ladder ✓")


if __name__ == "__main__":
    test_affect_resolves_to_the_bench()
    test_adapter_equals_affect_pipeline()
    test_no_hidden_attenuation()
    test_tier_moves_dominance_only()
    test_va_tables_agree()
    test_null_adapter_same_shape()
    test_wire_roundtrips()
    print("\nThe server's PAD path is AFFECT_LAB's, unmodified.")
