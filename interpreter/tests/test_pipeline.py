"""Step 1 checks. Run from repo root:  python3 -m unittest discover -s interpreter/tests -t . -v"""
import json
import tempfile
import unittest
from pathlib import Path

from interpreter.baseline import baseline_task1, baseline_task2
from interpreter.contracts import InterpretationFrame, SignPlan, SpeechPlan
from interpreter.mocks.mock_pepper import MockPepper
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.pipeline import Pipeline

FRAMES = Path(__file__).resolve().parents[1] / "mocks" / "frames"


class LoadTests(unittest.TestCase):
    def test_all_ten_load_five_per_direction(self):
        frames = load_frames(FRAMES)
        self.assertEqual(len(frames), 10)
        self.assertEqual(len({f.frame_id for f in frames}), 10)
        self.assertEqual(len(load_frames(FRAMES, "sign_to_speech")), 5)
        self.assertEqual(len(load_frames(FRAMES, "speech_to_sign")), 5)
        self.assertTrue(all(f.direction == "sign_to_speech" for f in load_frames(FRAMES, "sign_to_speech")))
        with self.assertRaises(ValueError):
            load_frames(FRAMES, "sideways")

    def test_duplicate_and_invalid_rejected_with_file_named(self):
        src = json.loads((FRAMES / "f-002-happy-greeting.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as t:
            Path(t, "a.json").write_text(json.dumps(src))
            Path(t, "b.json").write_text(json.dumps(src))
            with self.assertRaisesRegex(ValueError, r"b\.json.*duplicate"):
                load_frames(t)
        with tempfile.TemporaryDirectory() as t:
            Path(t, "broken.json").write_text("{not json")
            with self.assertRaisesRegex(ValueError, r"broken\.json"):
                load_frames(t)
        with tempfile.TemporaryDirectory() as t:
            src["emotion"]["valence"] = 5
            Path(t, "badvalue.json").write_text(json.dumps(src))
            with self.assertRaisesRegex(ValueError, r"badvalue\.json"):
                load_frames(t)

    def test_f005_low_confidence(self):
        f5 = next(f for f in load_frames(FRAMES) if f.frame_id == "f-005")
        self.assertLess(f5.input_confidence, 0.5)   # baseline ignores this for now


class PipelineTests(unittest.TestCase):
    def test_one_plan_per_frame_to_right_sink(self):
        frames = load_frames(FRAMES)
        pepper, signgen = MockPepper(), StubSignGenerator()
        pipe = Pipeline(baseline_task1, baseline_task2, pepper, signgen)
        for f in frames:
            pipe.process(f)
        self.assertEqual([p.frame_id for p in pepper.received],
                         [f.frame_id for f in frames if f.direction == "sign_to_speech"])
        self.assertEqual([p.frame_id for p in signgen.received],
                         [f.frame_id for f in frames if f.direction == "speech_to_sign"])
        self.assertTrue(all(isinstance(p, SpeechPlan) for p in pepper.received))
        self.assertTrue(all(isinstance(p, SignPlan) for p in signgen.received))

    def test_wrong_plan_type_or_frame_id_raises(self):
        frames = load_frames(FRAMES)
        d1 = next(f for f in frames if f.direction == "sign_to_speech")
        d2 = next(f for f in frames if f.direction == "speech_to_sign")
        pepper, signgen = MockPepper(), StubSignGenerator()

        with self.assertRaises(TypeError):
            Pipeline(baseline_task2, baseline_task2, pepper, signgen).process(d1)
        with self.assertRaises(TypeError):
            Pipeline(baseline_task1, baseline_task1, pepper, signgen).process(d2)

        def wrong_id1(f):
            p = baseline_task1(f); p.frame_id = "f-999"; return p

        def wrong_id2(f):
            p = baseline_task2(f); p.frame_id = "f-999"; return p
        with self.assertRaises(ValueError):
            Pipeline(wrong_id1, baseline_task2, pepper, signgen).process(d1)
        with self.assertRaises(ValueError):
            Pipeline(baseline_task1, wrong_id2, pepper, signgen).process(d2)
        self.assertEqual(pepper.received + signgen.received, [])   # nothing delivered on failure


class BaselineTests(unittest.TestCase):
    def test_baseline_plans_validate_and_roundtrip(self):
        for f in load_frames(FRAMES, "sign_to_speech"):
            p = baseline_task1(f)
            self.assertEqual(p, SpeechPlan.from_dict(json.loads(p.to_json())))
            self.assertEqual((p.mode, p.voice.persona, p.utterance), ("interpret", "interpreter", f.text_en))
        for f in load_frames(FRAMES, "speech_to_sign"):
            p = baseline_task2(f)
            self.assertEqual(p, SignPlan.from_dict(json.loads(p.to_json())))
            self.assertEqual((p.meaning, p.register, p.confidence), (f.text_en, "neutral", f.input_confidence))


if __name__ == "__main__":
    unittest.main()
