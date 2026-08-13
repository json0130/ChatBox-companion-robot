"""
pad_core is internally consistent: the adapter adds no maths of its own.

`pad_core.affect` is the single source of truth. These tests pin that the
adapter reproduces `affect.pipeline()` term for term, that the package is
self-contained, and that the V/A tables in the tree agree with each other.

The failure this guards against is silent. An earlier `AffectStream` scaled
valence by 0.3 before it reached the engine, which multiplied with the empathy
fraction (0.60) to an effective 0.18 — PAD "worked", it just barely moved.
No exception, no wrong type; only an equality test catches that.

No camera, no LLM, no hardware. Run directly or under pytest.
"""

import os
import sys

# tests/ -> PAD_CORE/ on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pad_core import affect, servo_style                           # noqa: E402
from pad_core.adapter import (                                     # noqa: E402
    EMOTION_VA, NullPADAdapter, PADPipelineAdapter,
)

_TIERS = ("unknown", "visitor", "known", "family", "close")
_EMOTIONS = ("happy", "sad", "angry", "neutral", "surprise")


def test_package_is_self_contained():
    """pad_core must import with nothing but the standard library on the path —
    that is what lets it be dropped into another project, and what lets the test
    suite verify the published equations with no camera, torch or server."""
    import ast
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    allowed = {"math", "typing", "collections", "os", "sys", "pad_core"}
    for mod in ("affect", "servo_style", "prompt", "stream", "adapter"):
        tree = ast.parse(open(os.path.join(here, "pad_core", f"{mod}.py")).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:            # `from . import x` — inside the package
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for n in names:
                assert n in allowed, f"pad_core.{mod} imports {n!r} — not stdlib"
    print("1. pad_core imports nothing outside the standard library ✓")


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
    """The adapter re-exports affect.CATEGORY_VA; they must not drift apart.
    (The consuming server keeps a third, deliberate copy in kg_bridge so its
    graph package imports nothing — test_pad_integration checks that one.)"""
    for label in EMOTION_VA:
        assert EMOTION_VA[label] == affect.CATEGORY_VA[label], f"adapter drift: {label}"
    print(f"5. adapter and affect agree on all {len(EMOTION_VA)} V/A labels ✓")


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
    test_package_is_self_contained()
    test_adapter_equals_affect_pipeline()
    test_no_hidden_attenuation()
    test_tier_moves_dominance_only()
    test_va_tables_agree()
    test_null_adapter_same_shape()
    test_wire_roundtrips()
    print("\npad_core is self-contained and internally consistent.")
