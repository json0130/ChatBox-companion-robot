"""Step 0 checks. Run:  python -m unittest test_contracts -v   (no pytest needed)"""
import json
import unittest

from interpreter.contracts import (Concept, EmotionTarget, EyeColour, InterpretationFrame, LexiconEntry,
                       NuanceTag, Pause, Prosody, SignPlan, SignUnit, SpeechPlan, Voice, Emotion)


def frame_d1() -> dict:
    """Direction 1 example: Aroha signs, Pepper will speak."""
    return {
        "frame_id": "f-0001", "direction": "sign_to_speech",
        "speaker_id": "person:aroha", "addressee_id": "person:visitor_01",
        "gloss": "TOMORROW HUI MY WHANAU COME",
        "text_en": "My family is coming to the meeting tomorrow.",
        "emotion": {"label": "excited", "valence": 0.7, "arousal": 0.6, "confidence": 0.82},
        "nuance": [{"kind": "statement"}, {"kind": "emphasis", "target": "family"}],
        "concepts": [{"slug": "ck:whanau", "english": "family", "te_reo": "whānau"},
                     {"slug": "ck:hui", "english": "meeting", "te_reo": "hui"}],
        "input_confidence": 0.88, "setting": "lab_open_day",
    }


def frame_d2() -> dict:
    """Direction 2 example: visitor speaks, Aroha will receive sign."""
    return {
        "frame_id": "f-0002", "direction": "speech_to_sign",
        "speaker_id": "person:visitor_01", "addressee_id": "person:aroha",
        "text_en": "Ah sorry, we'll have to push the meeting back a week, the room's double-booked.",
        "emotion": {"label": "apologetic", "valence": -0.3, "arousal": 0.2, "confidence": 0.71},
        "nuance": [{"kind": "apology"},
                   {"kind": "idiom", "target": "push the meeting back", "meaning": "postpone the meeting"},
                   {"kind": "idiom", "target": "double-booked", "meaning": "already booked for something else"}],
        "concepts": [{"slug": "ck:hui", "english": "meeting", "te_reo": "hui"}],
        "input_confidence": 0.93,
    }


class FrameTests(unittest.TestCase):
    def test_roundtrip_both_directions(self):
        for d in (frame_d1(), frame_d2()):
            f = InterpretationFrame.from_dict(d)
            f2 = InterpretationFrame.from_dict(json.loads(f.to_json()))
            self.assertEqual(f, f2)

    def test_gloss_only_for_sign_to_speech(self):
        d = frame_d2(); d["gloss"] = "X"
        with self.assertRaises(ValueError):
            InterpretationFrame.from_dict(d)

    def test_bad_values_rejected(self):
        d = frame_d1(); d["emotion"]["valence"] = 1.5
        with self.assertRaises(ValueError): InterpretationFrame.from_dict(d)
        d = frame_d1(); d["emotion"]["label"] = "ecstatic"
        with self.assertRaises(ValueError): InterpretationFrame.from_dict(d)
        d = frame_d1(); d["input_confidence"] = 1.2
        with self.assertRaises(ValueError): InterpretationFrame.from_dict(d)
        d = frame_d1(); d["direction"] = "sideways"
        with self.assertRaises(ValueError): InterpretationFrame.from_dict(d)

    def test_concept_slug_namespace(self):
        with self.assertRaises(ValueError): Concept(slug="topic:family", english="family")

    def test_idiom_needs_meaning_and_emphasis_needs_target(self):
        with self.assertRaises(ValueError): NuanceTag(kind="idiom", target="push back")
        with self.assertRaises(ValueError): NuanceTag(kind="emphasis")


class SpeechPlanTests(unittest.TestCase):
    def plan(self, **kw) -> SpeechPlan:
        base = dict(
            frame_id="f-0001", mode="interpret",
            utterance="My whānau are coming to the hui tomorrow!", emotion_label="excited",
            prosody=Prosody(rate=1.10, pitch=1.08, volume=0.9, emphasis=["whānau"],
                            pauses=[Pause(after_word_index=5, ms=150)]),
            eye=EyeColour(rgb=(255, 190, 60), intensity=0.9, pattern="solid", transition_ms=250),
            lexicon=[LexiconEntry("whānau", "FAH-no"), LexiconEntry("hui", "HOO-ee")])
        base.update(kw)
        return SpeechPlan(**base)

    def test_roundtrip(self):
        p = self.plan()
        p2 = SpeechPlan.from_dict(json.loads(p.to_json()))
        self.assertEqual(p, p2)
        self.assertIsInstance(p2.eye.rgb, tuple)

    def test_clarify_must_be_robot_voice(self):
        with self.assertRaises(ValueError):
            self.plan(mode="clarify", utterance="Sorry, could you say that again?", emotion_label="neutral",
                      prosody=Prosody(), lexicon=[])
        ok = self.plan(mode="clarify", utterance="Sorry, could you say that again?", emotion_label="neutral",
                       voice=Voice(persona="robot"), prosody=Prosody(), lexicon=[])
        self.assertEqual(ok.mode, "clarify")

    def test_interpret_must_be_interpreter_voice(self):
        with self.assertRaises(ValueError): self.plan(voice=Voice(persona="robot"))

    def test_emphasis_word_must_exist(self):
        with self.assertRaises(ValueError): self.plan(prosody=Prosody(emphasis=["kaitiaki"]))

    def test_pause_index_in_range(self):
        with self.assertRaises(ValueError): self.plan(prosody=Prosody(pauses=[Pause(99, 100)]))

    def test_ranges(self):
        with self.assertRaises(ValueError): Prosody(rate=3.0)
        with self.assertRaises(ValueError): EyeColour(rgb=(300, 0, 0))
        with self.assertRaises(ValueError): EyeColour(rgb=(1, 2))
        with self.assertRaises(ValueError): Voice(language="mi")


class SignPlanTests(unittest.TestCase):
    def plan(self, **kw) -> SignPlan:
        base = dict(
            frame_id="f-0002",
            meaning="Sorry. The hui is postponed by one week. The room is already booked.",
            units=[SignUnit("apology"),
                   SignUnit("ck:hui", surface="meeting", prefer_sign="maori_lexicon:HUI", role="topic"),
                   SignUnit("postpone", time="+1 week"),
                   SignUnit("room_already_booked")],
            emotion_target=EmotionTarget("apologetic", -0.3, 0.2, 0.5),
            register="respectful", nonmanual_hints=["brows_down", "head_tilt", "slow_pace"], confidence=0.86)
        base.update(kw)
        return SignPlan(**base)

    def test_roundtrip(self):
        p = self.plan()
        self.assertEqual(p, SignPlan.from_dict(json.loads(p.to_json())))

    def test_rejects_empty_units_and_unknown_hint(self):
        with self.assertRaises(ValueError): self.plan(units=[])
        with self.assertRaises(ValueError): self.plan(nonmanual_hints=["wink"])
        with self.assertRaises(ValueError): self.plan(register="chatty")


class EndToEndShape(unittest.TestCase):
    def test_frames_build_expected_plans(self):
        """Both worked examples can be expressed in the contracts end to end."""
        f1 = InterpretationFrame.from_dict(frame_d1())
        sp = SpeechPlan(frame_id=f1.frame_id, mode="interpret",
                        utterance="My whānau are coming to the hui tomorrow!", emotion_label=f1.emotion.label)
        self.assertEqual(sp.frame_id, f1.frame_id)
        f2 = InterpretationFrame.from_dict(frame_d2())
        sg = SignPlan(frame_id=f2.frame_id, meaning="Sorry, the hui is postponed a week.",
                      units=[SignUnit("apology")],
                      emotion_target=EmotionTarget(f2.emotion.label, f2.emotion.valence, f2.emotion.arousal))
        self.assertEqual(sg.frame_id, f2.frame_id)


if __name__ == "__main__":
    unittest.main()
