"""Tests for the standalone Method B RepetitionFilter V1 policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector import RepetitionFilter
from modules.project56.robot_profile import PlaceholderRobotProfileProvider


class RepetitionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repetition_filter = RepetitionFilter()
        self.profile = PlaceholderRobotProfileProvider().get_profile("Navel_001")

    def test_first_segment_preserves_all_candidates(self):
        candidates = ["AGREE", "POINT", "DEFAULT"]

        result = self.repetition_filter.filter(candidates, previous_tag=None)

        self.assertEqual(result, candidates)

    def test_repeated_expressive_action_is_removed(self):
        result = self.repetition_filter.filter(
            ["AGREE", "POINT"],
            previous_tag="AGREE",
        )

        self.assertEqual(result, ["POINT"])

    def test_unrelated_previous_expressive_action_changes_nothing(self):
        candidates = ["AGREE", "POINT"]

        result = self.repetition_filter.filter(
            candidates,
            previous_tag="DENY",
        )

        self.assertEqual(result, candidates)

    def test_repeated_default_remains_available_under_v1_policy(self):
        default = self.profile.actions["DEFAULT"]
        self.assertTrue(default.profiled)
        self.assertEqual(default.kind, "generic_speaking")

        result = self.repetition_filter.filter(
            ["DEFAULT", "AGREE"],
            previous_tag="DEFAULT",
        )

        self.assertEqual(result, ["DEFAULT", "AGREE"])

    def test_default_remains_when_repeated_expressive_action_is_removed(self):
        result = self.repetition_filter.filter(
            ["DEFAULT", "AGREE", "POINT"],
            previous_tag="AGREE",
        )

        self.assertEqual(result, ["DEFAULT", "POINT"])

    def test_empty_candidates_return_new_empty_list(self):
        candidates: list[str] = []

        result = self.repetition_filter.filter(candidates, "AGREE")

        self.assertEqual(result, [])
        self.assertIsNot(result, candidates)

    def test_previous_none_after_no_action_removes_nothing(self):
        candidates = ["AGREE", "POINT"]

        result = self.repetition_filter.filter(candidates, previous_tag=None)

        self.assertEqual(result, candidates)
        self.assertNotIn(None, result)

    def test_candidate_order_is_preserved_while_duplicates_are_removed(self):
        result = self.repetition_filter.filter(
            ["POINT", "AGREE", "POINT", "DEFAULT", "HAPPY"],
            previous_tag="AGREE",
        )

        self.assertEqual(result, ["POINT", "DEFAULT", "HAPPY"])

    def test_input_list_and_tuple_are_not_mutated(self):
        candidate_list = ["AGREE", "POINT", "DEFAULT"]
        candidate_tuple = ("AGREE", "POINT", "DEFAULT")
        original_list = list(candidate_list)
        original_tuple = tuple(candidate_tuple)

        self.repetition_filter.filter(candidate_list, "AGREE")
        self.repetition_filter.filter(candidate_tuple, "AGREE")

        self.assertEqual(candidate_list, original_list)
        self.assertEqual(candidate_tuple, original_tuple)

    def test_no_no_action_or_none_candidate_is_introduced(self):
        result = self.repetition_filter.filter(
            [None, "AGREE"],  # type: ignore[list-item]
            previous_tag=None,
        )

        self.assertEqual(result, ["AGREE"])
        self.assertNotIn("NO_ACTION", result)
        self.assertNotIn(None, result)
        self.assertNotIn("NO_ACTION", self.profile.actions)

    def test_tags_follow_current_canonical_format(self):
        result = self.repetition_filter.filter(
            ["agree", "[POINT]", " [default] ", "AGREE"],
            previous_tag="[agree]",
        )

        self.assertEqual(result, ["POINT", "DEFAULT"])

    def test_filter_instance_stores_no_previous_action_state(self):
        first = self.repetition_filter.filter(["AGREE", "POINT"], "AGREE")
        second = self.repetition_filter.filter(["AGREE", "POINT"], None)

        self.assertEqual(first, ["POINT"])
        self.assertEqual(second, ["AGREE", "POINT"])
        self.assertNotIn("previous_tag", vars(self.repetition_filter))


if __name__ == "__main__":
    unittest.main()
