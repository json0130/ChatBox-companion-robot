"""DurationFilter tests against the current Method B domain contracts."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector import DurationFilter
from modules.project56.co_speech import SpeechSegment
from modules.project56.robot_profile import PlaceholderRobotProfileProvider


class DurationFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = PlaceholderRobotProfileProvider().get_profile("Navel_001")

    @staticmethod
    def _segment(duration: float, *, text: str = "Hello.", words: int = 1):
        return SpeechSegment(
            segment_id=1,
            text=text,
            word_count=words,
            estimated_duration=duration,
        )

    def test_real_speech_segment_and_robot_profile_actions_are_compatible(self):
        segment = self._segment(2.0)

        result = DurationFilter().filter(
            segment,
            ["AGREE", "POINT"],
            self.profile.actions,
        )

        self.assertEqual(result.eligible_tags, ("AGREE",))
        self.assertEqual(result.rejected["POINT"], "exceeds_segment_duration")

    def test_default_uses_current_profiled_duration_like_any_other_action(self):
        default = self.profile.actions["DEFAULT"]
        self.assertTrue(default.profiled)
        self.assertEqual(default.duration_seconds, 2.402)

        too_short = self._segment(default.duration_seconds - 0.001)
        exact_fit = self._segment(default.duration_seconds)

        rejected = DurationFilter().filter(
            too_short,
            ["DEFAULT"],
            self.profile.actions,
        )
        retained = DurationFilter().filter(
            exact_fit,
            ["DEFAULT"],
            self.profile.actions,
        )

        self.assertEqual(rejected.eligible_tags, ())
        self.assertEqual(
            rejected.rejected["DEFAULT"],
            "exceeds_segment_duration",
        )
        self.assertEqual(retained.eligible_tags, ("DEFAULT",))

    def test_safety_margin_is_included_in_equality_boundary(self):
        agree_duration = self.profile.actions["AGREE"].duration_seconds
        segment = self._segment(agree_duration + 0.25)

        exact = DurationFilter(margin_seconds=0.25).filter(
            segment,
            ["AGREE"],
            self.profile.actions,
        )
        over = DurationFilter(margin_seconds=0.251).filter(
            segment,
            ["AGREE"],
            self.profile.actions,
        )

        self.assertEqual(exact.eligible_tags, ("AGREE",))
        self.assertEqual(over.eligible_tags, ())

    def test_zero_overrun_allowance_reproduces_current_rejection(self):
        point_duration = self.profile.actions["POINT"].duration_seconds
        segment = self._segment(point_duration - 0.001)

        default = DurationFilter().filter(segment, ["POINT"], self.profile.actions)
        explicit_zero = DurationFilter(overrun_allowance_seconds=0.0).filter(
            segment, ["POINT"], self.profile.actions,
        )

        self.assertEqual(default, explicit_zero)
        self.assertEqual(default.rejected["POINT"], "exceeds_segment_duration")

    def test_overrun_allowance_includes_exact_boundary_only(self):
        point_duration = self.profile.actions["POINT"].duration_seconds
        segment = self._segment(point_duration - 0.5)

        exact = DurationFilter(overrun_allowance_seconds=0.5).filter(
            segment, ["POINT"], self.profile.actions,
        )
        below = DurationFilter(overrun_allowance_seconds=0.499).filter(
            segment, ["POINT"], self.profile.actions,
        )

        self.assertEqual(exact.eligible_tags, ("POINT",))
        self.assertEqual(below.rejected["POINT"], "exceeds_segment_duration")

    def test_missing_action_profile_is_rejected_without_fallback(self):
        result = DurationFilter().filter(
            self._segment(10.0),
            ["UNKNOWN"],
            self.profile.actions,
        )

        self.assertEqual(result.eligible_tags, ())
        self.assertEqual(result.rejected, {"UNKNOWN": "missing_action_profile"})

    def test_unprofiled_or_missing_duration_is_rejected(self):
        fixtures = (
            SimpleNamespace(profiled=False, duration_seconds=None),
            SimpleNamespace(profiled=True, duration_seconds=None),
        )
        for fixture in fixtures:
            with self.subTest(profiled=fixture.profiled):
                result = DurationFilter().filter(
                    self._segment(10.0),
                    ["TEST"],
                    {"TEST": fixture},
                )
                self.assertEqual(result.eligible_tags, ())
                self.assertEqual(result.rejected, {"TEST": "unknown_duration"})

    def test_invalid_profiled_action_durations_fail_closed(self):
        invalid = (True, False, 0, -1, math.inf, -math.inf, math.nan, "2")
        for duration in invalid:
            with self.subTest(duration=duration):
                result = DurationFilter().filter(
                    self._segment(10.0),
                    ["BAD"],
                    {
                        "BAD": SimpleNamespace(
                            profiled=True,
                            duration_seconds=duration,
                        )
                    },
                )
                self.assertEqual(result.eligible_tags, ())
                self.assertEqual(
                    result.rejected,
                    {"BAD": "invalid_action_duration"},
                )

    def test_already_annotated_duration_wins_over_word_count(self):
        # One word would be much too short under either project estimator, but
        # the supplied runtime annotation deliberately gives POINT enough time.
        segment = self._segment(
            5.0,
            text="Brief.",
            words=1,
        )

        result = DurationFilter().filter(
            segment,
            ["POINT"],
            self.profile.actions,
        )

        self.assertEqual(result.eligible_tags, ("POINT",))
        self.assertEqual(segment.word_count, 1)
        self.assertEqual(segment.estimated_duration, 5.0)

    def test_empty_allowed_tags_produce_empty_result(self):
        result = DurationFilter().filter(
            self._segment(2.0),
            [],
            self.profile.actions,
        )

        self.assertEqual(result.eligible_tags, ())
        self.assertEqual(result.rejected, {})

    def test_invalid_segment_duration_raises_clearly(self):
        for duration in (True, 0, -1, math.inf, -math.inf, math.nan, "2", None):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(
                    ValueError,
                    "Segment duration must be finite and positive",
                ):
                    DurationFilter().filter(
                        self._segment(duration),
                        [],
                        self.profile.actions,
                    )

    def test_invalid_margin_raises_clearly(self):
        for margin in (True, -1, math.inf, math.nan, "0", None):
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(
                    ValueError,
                    "margin_seconds must be finite and non-negative",
                ):
                    DurationFilter(margin)

    def test_invalid_overrun_allowance_raises_clearly(self):
        for allowance in (True, -1, math.inf, math.nan, "0", None):
            with self.subTest(allowance=allowance):
                with self.assertRaisesRegex(
                    ValueError,
                    "overrun_allowance_seconds must be finite and non-negative",
                ):
                    DurationFilter(overrun_allowance_seconds=allowance)

    def test_tags_and_real_profile_mapping_are_not_mutated(self):
        tags = [" [agree] ", "AGREE", "DEFAULT", ""]
        original_tags = list(tags)
        original_actions = dict(self.profile.actions)

        result = DurationFilter().filter(
            self._segment(3.0),
            tags,
            self.profile.actions,
        )

        self.assertEqual(result.eligible_tags, ("AGREE", "DEFAULT"))
        self.assertEqual(tags, original_tags)
        self.assertEqual(dict(self.profile.actions), original_actions)


if __name__ == "__main__":
    unittest.main()
