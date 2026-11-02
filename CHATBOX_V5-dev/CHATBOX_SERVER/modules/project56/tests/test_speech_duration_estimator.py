"""Tests for the selected fixed runtime speech-duration estimator."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.co_speech import (
    LinearSpeechDurationProfile,
    NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
    SpeechDurationEstimator,
    SpeechSegment,
    SpeechSegmenter,
)
from modules.project56.co_speech.text_metrics import count_spoken_words


class SpeechDurationEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = SpeechDurationEstimator()

    def test_uses_exact_selected_linear_formula(self):
        expected = 0.5379889576211432 + 0.32360131854586377 * 7

        self.assertAlmostEqual(
            self.estimator.estimate("one two three four five six seven"),
            expected,
        )

    @unittest.skip("needs the 690 KB evaluation artifact, which stays in the Navel repo")
    def test_selected_profile_matches_preserved_comparison_artifact(self):
        artifact_path = (
            SERVER_DIR
            / "evaluation"
            / "results_duration_estimators"
            / "duration_estimator_comparison.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        fitted = artifact["harvard_external_validation"][
            "fitted_parameters_from_all_lab_tour_sentences"
        ]["linear_regression_with_intercept"]

        self.assertEqual(
            NAVEL_LINEAR_SPEECH_DURATION_PROFILE.intercept_seconds,
            fitted["intercept_seconds"],
        )
        self.assertEqual(
            NAVEL_LINEAR_SPEECH_DURATION_PROFILE.seconds_per_word,
            fitted["slope_seconds_per_word"],
        )

    def test_word_count_matches_frozen_segmenter_and_calibration_examples(self):
        frozen_segmenter = SpeechSegmenter()
        examples = {
            "Hello!": 1,
            "Of course.": 2,
            "Yes, that is correct.": 4,
            "Hello! I'm Navel robot.": 4,
        }

        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(count_spoken_words(text), expected)
                self.assertEqual(
                    count_spoken_words(text),
                    frozen_segmenter.count_words(text),
                )

    def test_known_short_utterance_predictions(self):
        self.assertAlmostEqual(self.estimator.estimate("Hello!"), 0.861590276167007)
        self.assertAlmostEqual(self.estimator.estimate("Of course."), 1.1851915947128707)
        self.assertAlmostEqual(
            self.estimator.estimate("Yes, that is correct."),
            1.8323942318045983,
        )

    def test_multiple_segments_are_annotated_independently(self):
        original = [
            SpeechSegment(1, "Hello!", 1, 0.391),
            SpeechSegment(2, "Yes, that is correct.", 4, 1.565),
        ]

        annotated = self.estimator.annotate_segments(original)

        self.assertEqual(len(annotated), 2)
        self.assertAlmostEqual(annotated[0].estimated_duration, 0.861590276167007)
        self.assertAlmostEqual(annotated[1].estimated_duration, 1.8323942318045983)
        self.assertNotEqual(
            annotated[0].estimated_duration,
            annotated[1].estimated_duration,
        )

    def test_annotation_preserves_text_ids_and_original_objects(self):
        original = SpeechSegment(3, "  Keep this exact text!  ", 4, 99.0)

        annotated = self.estimator.annotate_segments([original])

        self.assertEqual(annotated[0].segment_id, 3)
        self.assertEqual(annotated[0].text, original.text)
        self.assertEqual(original.estimated_duration, 99.0)
        self.assertIsNot(annotated[0], original)

    def test_empty_and_whitespace_input_fail_clearly(self):
        for text in ("", "   \n"):
            with self.subTest(text=repr(text)):
                with self.assertRaisesRegex(ValueError, "at least one word"):
                    self.estimator.estimate(text)

    def test_valid_text_estimates_are_positive(self):
        self.assertGreater(self.estimator.estimate("Hello!"), 0.0)

    def test_annotation_does_not_change_frozen_segment_boundaries(self):
        text = "Hello! I am Navel robot. How can I help you today?"
        frozen_segmenter = SpeechSegmenter()
        raw_segments = frozen_segmenter.segment(text)
        raw_snapshot = [segment.to_dict() for segment in raw_segments]

        annotated = self.estimator.annotate_segments(raw_segments)

        self.assertEqual(
            [(segment.segment_id, segment.text) for segment in annotated],
            [(segment.segment_id, segment.text) for segment in raw_segments],
        )
        self.assertEqual(
            [segment.to_dict() for segment in raw_segments],
            raw_snapshot,
        )

    def test_runtime_estimation_performs_no_file_or_fitting_access(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected I/O")):
            result = self.estimator.estimate("Runtime estimation stays local.")

        self.assertGreater(result, 0.0)

    def test_invalid_profiles_fail_during_configuration(self):
        with self.assertRaisesRegex(ValueError, "intercept"):
            LinearSpeechDurationProfile(-0.1, 0.3)
        with self.assertRaisesRegex(ValueError, "seconds_per_word"):
            LinearSpeechDurationProfile(0.5, 0.0)
        with self.assertRaisesRegex(ValueError, "model"):
            LinearSpeechDurationProfile(0.5, 0.3, model="global_wps")
        with self.assertRaisesRegex(ValueError, "intercept"):
            LinearSpeechDurationProfile(float("nan"), 0.3)
        with self.assertRaisesRegex(ValueError, "seconds_per_word"):
            LinearSpeechDurationProfile(0.5, float("inf"))

    def test_empty_segment_collection_returns_empty_list(self):
        self.assertEqual(self.estimator.annotate_segments([]), [])


if __name__ == "__main__":
    unittest.main()
