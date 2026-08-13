"""
PAD pipeline adapter — one conversation turn in, one affect state out.

Thin wrapper over `AFFECT_LAB/affect.py`, which is the single source of truth for
the affect maths (OCEAN -> PAD, the face's pull on P/Ar, the relationship tier's
pull on D, the body's `show` fraction, the descriptor bands, and the five style
values). This module owns no equations of its own; it only threads the graph's
view of a turn into that pipeline and shapes the result for the host loop.

Two interchangeable classes, same interface:
    PADPipelineAdapter  — live PAD-driven affect state
    NullPADAdapter      — neutral fallback

`system_prompt` is deliberately always None. An earlier version built a whole
system prompt here, and the host loop used it INSTEAD of its own — which silently
dropped the identity block, the KG memory, the RAG hits and the anti-hallucination
rules whenever PAD was enabled. Prompt assembly belongs to the loop, which is the
only place that has all of those; this adapter supplies the three descriptor
words and the loop decides where they go.
"""

from . import affect, servo_style
from .stream import AffectStream

# ---------------------------------------------------------------------------
# Emotion label -> (valence, arousal)
# ---------------------------------------------------------------------------
# Delegates to affect.CATEGORY_VA so the server and the bench cannot disagree
# about what "sad" means. Kept as a module attribute because callers import it.
EMOTION_VA: dict[str, tuple[float, float]] = dict(affect.CATEGORY_VA)

# robot_id -> (key into affect.ROBOTS, display name)
_PERSONA_REGISTRY = {
    "chatbox": ("CHATBOX", "ChatBox"),
    "ellebot": ("ELLEBOT", "ElleBot"),
}

_NEUTRAL_DESCRIPTORS: dict[str, str] = {
    "pleasure": "even", "arousal": "calm", "dominance": "even-handed",
}


class PADPipelineAdapter:
    """One instance per connected robot. Call process_turn() once per turn."""

    def __init__(self, robot_id: str, affect_stream: AffectStream | None = None,
                 show: float | None = None):
        entry = _PERSONA_REGISTRY.get(robot_id.lower())
        if entry is None:
            raise ValueError(f"Unknown robot_id {robot_id!r}. "
                             f"Available: {list(_PERSONA_REGISTRY)}")
        self._key, self._display_name = entry
        self._traits = affect.ROBOTS[self._key]["ocean"]
        # `show` is overridable purely as a diagnostic: CHATBOX's 0.30 compresses
        # Dominance so hard that the descriptor words barely move with tier, and
        # being able to raise it separates "PAD isn't reaching the words" from
        # "the body is muting it". Not a runtime feature.
        self._show = affect.ROBOTS[self._key]["show"] if show is None else show
        self._stream = affect_stream or AffectStream()

    def process_turn(
        self,
        valence: float,
        arousal: float,
        relationship_tier: str,
        memory_context: str = "",      # accepted for interface stability; unused
        rapport: float = 0.0,
        trust: float = 0.0,
        interaction_count: int = 0,
    ) -> dict:
        """Run the affect pipeline for one turn.

        Args:
            valence/arousal:   the person's face, [-1, 1]. The ONLY inputs the
                               face is allowed to influence (P and Ar).
            relationship_tier: from kg_bridge.derive_tier — moves Dominance.
            memory_context/rapport/trust/interaction_count: carried for the
                               caller's convenience; prompt assembly is the
                               loop's job, so nothing here consumes them.

        Returns:
            pad_state    (P, Ar, D) FELT — the internal state, not display-scaled.
                         kg_bridge.post_turn persists P/Ar from this.
            shown        (P, Ar, D) after the body's `show` fraction.
            words        ordered (pleasure, arousal, dominance) descriptors.
            descriptors  the same three, keyed — what the prompt consumes.
            name         nearest named circumplex region, e.g. "mildly dejected".
            style        the five servo parameters, derived from FELT.
            wire         the STYLE line for the ESP32.
            tier         echoed back for logging.
            system_prompt  always None — see the module docstring.
        """
        v, a = self._stream.update(valence, arousal)
        baseline = affect.to_pad(self._traits)
        felt = affect.feel_with_relationship(baseline, v, a, relationship_tier)
        shown = affect.show(felt, self._show)

        words = affect.descriptors(shown)
        # Derived from FELT, not shown: the body's display fraction and amplitude
        # are two descriptions of the same restraint, so applying both would
        # double-count it and flatten the contrast between the two robots.
        style = affect.gesture_style(felt)

        return {
            "pad_state":   (felt["P"], felt["Ar"], felt["D"]),
            "shown":       (shown["P"], shown["Ar"], shown["D"]),
            "words":       words,
            "descriptors": {"pleasure": words[0], "arousal": words[1],
                            "dominance": words[2]},
            "name":        affect.affect_name(shown["P"], shown["Ar"]),
            "style":       style,
            "wire":        servo_style.wire_message(style),
            "tier":        relationship_tier,
            "system_prompt": None,
        }

    def enrich_hardware_command(self, tag: str, style: dict) -> dict:
        """Bundle a gesture tag with the style values that should precede it."""
        return {
            "command":  tag.lower(),
            "style":    style,
            "wire":     servo_style.wire_message(style),
            "raw_tag":  tag.upper(),
        }

    @staticmethod
    def emotion_label_to_va(emotion_label: str) -> tuple[float, float]:
        """Emotion label -> (valence, arousal). Unknown -> neutral."""
        return affect.category_to_va(emotion_label)


class NullPADAdapter:
    """Neutral drop-in: same keys, no affect. Disables PAD without other edits."""

    def process_turn(self, valence: float = 0.0, arousal: float = 0.0,
                     relationship_tier: str = "unknown", memory_context: str = "",
                     rapport: float = 0.0, trust: float = 0.0,
                     interaction_count: int = 0) -> dict:
        neutral = {"amplitude": 1.0, "tempo": 1.0, "posture": 0.0,
                   "droop": 0.0, "idle": 0.45}
        return {
            "pad_state": (0.0, 0.0, 0.0), "shown": (0.0, 0.0, 0.0),
            "words": ("even", "calm", "even-handed"),
            "descriptors": dict(_NEUTRAL_DESCRIPTORS),
            "name": "neutral", "style": neutral,
            "wire": servo_style.wire_message(neutral),
            "tier": relationship_tier, "system_prompt": None,
        }

    def enrich_hardware_command(self, tag: str, style: dict) -> dict:
        return {"command": tag.lower(), "style": style,
                "wire": servo_style.wire_message(style), "raw_tag": tag.upper()}

    @staticmethod
    def emotion_label_to_va(emotion_label: str) -> tuple[float, float]:
        return affect.category_to_va(emotion_label)


if __name__ == "__main__":
    print("=" * 74)
    print("PAD pipeline adapter — the relationship ladder at a neutral face")
    print("=" * 74)
    for robot in ("chatbox", "ellebot"):
        ad = PADPipelineAdapter(robot)
        print(f"\n{robot.upper()}  (show={ad._show})")
        for tier in affect.TIERS:
            r = ad.process_turn(0.0, 0.0, tier)
            p, a, d = r["pad_state"]
            print(f"  {tier:8s} PAD=({p:+.3f},{a:+.3f},{d:+.3f})  "
                  f"{'/'.join(r['words']):32s}  {r['wire']}")
