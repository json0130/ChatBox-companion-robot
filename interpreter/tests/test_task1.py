"""Step 2a checks. Run from repo root:  python3 -m unittest discover -s interpreter/tests -t . -v"""
import copy
import json
import unittest
from pathlib import Path

from interpreter.context import StaticContext
from interpreter.contracts import EMOTIONS, InterpretationFrame, SpeechPlan
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import EYE_PALETTE, plan_speech

FRAMES = Path(__file__).resolve().parents[1] / "mocks" / "frames"
BY_ID = {f.frame_id: f for f in load_frames(FRAMES)}
CTX = StaticContext()


def f(frame_id: str) -> InterpretationFrame:
    return copy.deepcopy(BY_ID[frame_id])


class RuleTests(unittest.TestCase):
    def test_f001_word_choice_emphasis_prosody_eye_lexicon(self):
        p = plan_speech(f("f-001"), CTX)
        self.assertEqual(p.mode, "interpret")
        self.assertEqual(p.utterance, "My whānau is coming to the hui tomorrow.")
        self.assertEqual(p.prosody.emphasis, ["whānau"])
        self.assertGreater(p.prosody.rate, 1)
        self.assertGreater(p.prosody.pitch, 1)
        self.assertEqual(p.eye.pattern, "pulse")
        self.assertEqual({e.word for e in p.lexicon}, {"whānau", "hui"})

    def test_no_te_reo_preference_leaves_text_alone(self):
        p = plan_speech(f("f-001"), StaticContext(default=False))
        self.assertEqual(p.utterance, "My family is coming to the meeting tomorrow.")
        self.assertEqual(p.prosody.emphasis, ["family"])
        self.assertEqual(p.lexicon, [])

    def test_per_person_override_and_leading_capital(self):
        fr = f("f-001"); fr.text_en = "Family is coming."
        self.assertEqual(plan_speech(fr, CTX).utterance, "Whānau is coming.")
        ctx = StaticContext({("person:aroha", "ck:whanau"): False}, default=True)
        self.assertIn("family", plan_speech(f("f-001"), ctx).utterance)
        self.assertIn("hui", plan_speech(f("f-001"), ctx).utterance)

    def test_sad_is_slow_quiet_dim_blue(self):
        sad, happy = plan_speech(f("f-003"), CTX), plan_speech(f("f-001"), CTX)
        self.assertLess(sad.prosody.rate, 1)
        self.assertLess(sad.prosody.volume, 1)
        self.assertEqual(sad.eye.rgb, EYE_PALETTE["sad"])
        self.assertLess(sad.eye.intensity, happy.eye.intensity)

    def test_angry_is_fast_and_loud(self):
        p = plan_speech(f("f-004"), CTX)
        self.assertGreater(p.prosody.rate, 1)
        self.assertGreater(p.prosody.volume, 1)
        self.assertEqual(p.prosody.emphasis, ["nobody"])

    def test_pauses_follow_commas(self):
        fr = f("f-002")                                   # "Hello, nice to meet you.", arousal .3
        self.assertEqual([(x.after_word_index, x.ms) for x in plan_speech(fr, CTX).prosody.pauses], [(0, 150)])
        fr = f("f-003"); fr.text_en = "I miss, you know, my family."   # arousal -.3
        self.assertEqual([x.ms for x in plan_speech(fr, CTX).prosody.pauses], [300, 300])

    def test_low_input_confidence_clarifies(self):
        p = plan_speech(f("f-005"), CTX)
        self.assertEqual((p.mode, p.voice.persona, p.emotion_label), ("clarify", "robot", "neutral"))
        self.assertEqual((p.eye.rgb, p.eye.pattern), ((255, 170, 0), "pulse"))
        self.assertEqual(SpeechPlan.from_dict(json.loads(p.to_json())), p)

    def test_lower_emotion_confidence_moves_toward_neutral(self):
        hi, lo = f("f-001"), f("f-001")
        lo.emotion.confidence = 0.3
        a, b = plan_speech(hi, CTX).prosody, plan_speech(lo, CTX).prosody
        for n in ("rate", "pitch", "volume"):
            self.assertLess(abs(getattr(b, n) - 1), abs(getattr(a, n) - 1), n)

    def test_speech_to_sign_rejected(self):
        with self.assertRaises(ValueError):
            plan_speech(f("f-006"), CTX)

    def test_grid_always_valid(self):
        base = f("f-001").to_dict()
        base["text_en"] = "My family, sorry, is coming to the meeting tomorrow."
        vals = (-1, -.5, 0, .5, 1)
        for label in EMOTIONS:
            for v in vals:
                for a in vals:
                    for conf in (0, 1):
                        d = copy.deepcopy(base)
                        d["emotion"] = {"label": label, "valence": v, "arousal": a, "confidence": conf}
                        plan = plan_speech(InterpretationFrame.from_dict(d), CTX)
                        self.assertEqual(plan.emotion_label, label)

    def test_all_plans_roundtrip(self):
        for fr in BY_ID.values():
            if fr.direction == "sign_to_speech":
                p = plan_speech(fr, CTX)
                self.assertEqual(SpeechPlan.from_dict(json.loads(p.to_json())), p)

    def test_task1_does_not_mutate_frame(self):
        fr = f("f-001"); before = fr.to_dict()
        plan_speech(fr, CTX)
        self.assertEqual(fr.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
