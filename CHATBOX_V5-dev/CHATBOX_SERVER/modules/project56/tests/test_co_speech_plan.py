"""Tests for temporary response-scoped co-speech plan state."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.co_speech import (
    CoSpeechPlan,
    CoSpeechPlanBuilder,
    PlannedSegment,
    SelectionStatus,
    SpeechSegment,
)


def segment(
    segment_id: int,
    text: str,
    word_count: int,
    duration: float,
) -> SpeechSegment:
    return SpeechSegment(segment_id, text, word_count, duration)


class CoSpeechPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = (
            segment(4, "Yes, that is correct.", 4, 1.8323942318045983),
            segment(
                9,
                "Let me show you the next robot.",
                7,
                2.8031981874421895,
            ),
            segment(12, "Please stand beside the display.", 5, 2.155995550350462),
        )

    def test_one_real_segment_starts_pending_without_duplication(self):
        original = self.segments[0]

        plan = CoSpeechPlanBuilder([original]).plan
        planned = plan.segments[0]

        self.assertIs(planned.speech_segment, original)
        self.assertEqual(planned.selection_status, SelectionStatus.PENDING)
        self.assertIsNone(planned.selected_tag)

    def test_multiple_segments_preserve_order_ids_and_values(self):
        plan = CoSpeechPlanBuilder(self.segments).plan

        self.assertEqual(
            [item.speech_segment.segment_id for item in plan.segments],
            [4, 9, 12],
        )
        self.assertEqual(
            [item.speech_segment for item in plan.segments],
            list(self.segments),
        )
        self.assertTrue(
            all(
                item.selection_status is SelectionStatus.PENDING
                and item.selected_tag is None
                for item in plan.segments
            )
        )

    def test_record_action_creates_selected_state(self):
        builder = CoSpeechPlanBuilder(self.segments)

        resolved = builder.record_selection(4, "AGREE")

        self.assertEqual(resolved.selection_status, SelectionStatus.SELECTED)
        self.assertEqual(resolved.selected_tag, "AGREE")
        self.assertEqual(builder.plan.segments[0], resolved)

    def test_record_none_creates_distinct_no_action_state(self):
        builder = CoSpeechPlanBuilder(self.segments)

        resolved = builder.record_selection(4, None)

        self.assertEqual(resolved.selection_status, SelectionStatus.NO_ACTION)
        self.assertIsNone(resolved.selected_tag)
        self.assertNotEqual(
            resolved.selection_status,
            builder.plan.segments[1].selection_status,
        )

    def test_default_is_a_normal_selected_action(self):
        builder = CoSpeechPlanBuilder(self.segments)

        resolved = builder.record_selection(4, "DEFAULT")

        self.assertEqual(resolved.selection_status, SelectionStatus.SELECTED)
        self.assertEqual(resolved.selected_tag, "DEFAULT")

    def test_invalid_status_and_tag_combinations_are_rejected(self):
        speech = self.segments[0]
        invalid = (
            (SelectionStatus.PENDING, "AGREE"),
            (SelectionStatus.NO_ACTION, "AGREE"),
            (SelectionStatus.SELECTED, None),
            (SelectionStatus.SELECTED, "agree"),
            (SelectionStatus.SELECTED, "[AGREE]"),
            (SelectionStatus.SELECTED, "NO ACTION"),
            (SelectionStatus.SELECTED, "NO_ACTION"),
        )
        for status, selected_tag in invalid:
            with self.subTest(status=status, selected_tag=selected_tag):
                with self.assertRaises(ValueError):
                    PlannedSegment(speech, status, selected_tag)

    def test_duplicate_selection_attempt_fails_clearly(self):
        builder = CoSpeechPlanBuilder(self.segments)
        builder.record_selection(4, "AGREE")

        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            builder.record_selection(4, None)

    def test_unknown_segment_id_fails_clearly(self):
        builder = CoSpeechPlanBuilder(self.segments)

        with self.assertRaisesRegex(KeyError, "Unknown speech segment ID"):
            builder.record_selection(999, "AGREE")
        with self.assertRaisesRegex(KeyError, "Unknown speech segment ID"):
            builder.plan.previous_selected_tag(999)

    def test_duplicate_input_ids_are_rejected_without_renumbering(self):
        duplicate = segment(4, "Different text.", 2, 1.0)

        with self.assertRaisesRegex(ValueError, "Duplicate speech segment ID: 4"):
            CoSpeechPlanBuilder([self.segments[0], duplicate])

    def test_invalid_input_sequence_and_items_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a sequence"):
            CoSpeechPlanBuilder(iter(self.segments))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "must contain SpeechSegment"):
            CoSpeechPlanBuilder([object()])  # type: ignore[list-item]

    def test_original_speech_segments_are_never_mutated(self):
        original = self.segments[0]
        original_values = original.to_dict()
        builder = CoSpeechPlanBuilder([original])

        builder.record_selection(original.segment_id, "AGREE")

        self.assertIs(builder.plan.segments[0].speech_segment, original)
        self.assertEqual(original.to_dict(), original_values)

    def test_separate_builders_do_not_share_mutable_selection_state(self):
        first = CoSpeechPlanBuilder(self.segments)
        second = CoSpeechPlanBuilder(self.segments)
        initial_second = second.plan

        first.record_selection(4, "AGREE")

        self.assertEqual(
            first.plan.segments[0].selection_status,
            SelectionStatus.SELECTED,
        )
        self.assertEqual(
            second.plan.segments[0].selection_status,
            SelectionStatus.PENDING,
        )
        self.assertEqual(second.plan, initial_second)

    def test_plan_snapshots_do_not_change_after_later_builder_updates(self):
        builder = CoSpeechPlanBuilder(self.segments)
        pending_snapshot = builder.plan

        builder.record_selection(4, "AGREE")

        self.assertEqual(
            pending_snapshot.segments[0].selection_status,
            SelectionStatus.PENDING,
        )
        self.assertEqual(
            builder.plan.segments[0].selection_status,
            SelectionStatus.SELECTED,
        )

    def test_completion_requires_every_segment_to_be_resolved(self):
        builder = CoSpeechPlanBuilder(self.segments)
        self.assertFalse(builder.plan.is_complete)

        builder.record_selection(4, "AGREE")
        builder.record_selection(9, None)
        self.assertFalse(builder.plan.is_complete)

        builder.record_selection(12, "DEFAULT")
        self.assertTrue(builder.plan.is_complete)

    def test_empty_plan_is_explicitly_complete(self):
        plan = CoSpeechPlanBuilder([]).plan

        self.assertEqual(plan.segments, ())
        self.assertTrue(plan.is_complete)
        self.assertEqual(plan.to_dict(), {"segments": []})

    def test_previous_selection_is_derived_from_resolved_predecessors(self):
        builder = CoSpeechPlanBuilder(self.segments)
        self.assertIsNone(builder.plan.previous_selected_tag(4))

        builder.record_selection(4, "AGREE")
        self.assertEqual(builder.plan.previous_selected_tag(9), "AGREE")

        builder.record_selection(9, None)
        self.assertIsNone(builder.plan.previous_selected_tag(12))

    def test_debug_output_preserves_order_and_no_action_null(self):
        builder = CoSpeechPlanBuilder(self.segments)
        builder.record_selection(4, "AGREE")
        builder.record_selection(9, None)

        payload = builder.plan.to_dict()

        self.assertEqual(
            [item["segment_id"] for item in payload["segments"]],
            [4, 9, 12],
        )
        self.assertEqual(payload["segments"][0]["text"], self.segments[0].text)
        self.assertEqual(payload["segments"][0]["word_count"], 4)
        self.assertEqual(
            payload["segments"][0]["estimated_duration"],
            self.segments[0].estimated_duration,
        )
        self.assertEqual(payload["segments"][0]["selection_status"], "selected")
        self.assertEqual(payload["segments"][0]["selected_tag"], "AGREE")
        self.assertEqual(payload["segments"][1]["selection_status"], "no_action")
        self.assertIsNone(payload["segments"][1]["selected_tag"])
        self.assertEqual(payload["segments"][2]["selection_status"], "pending")

    def test_builder_does_not_call_selection_filters_or_execution(self):
        with (
            patch(
                "modules.project56.action_selector.action_selector_module."
                "ActionSelectorModule.select_method_b"
            ) as selector,
            patch(
                "modules.project56.action_selector.duration_filter.DurationFilter.filter"
            ) as duration_filter,
            patch(
                "modules.project56.action_selector.repetition_filter."
                "RepetitionFilter.filter"
            ) as repetition_filter,
        ):
            builder = CoSpeechPlanBuilder(self.segments)
            builder.record_selection(4, "AGREE")
            builder.record_selection(9, None)

        selector.assert_not_called()
        duration_filter.assert_not_called()
        repetition_filter.assert_not_called()

    def test_plan_constructor_validates_tuple_and_duplicate_ids(self):
        pending = PlannedSegment(self.segments[0])
        duplicate = PlannedSegment(
            segment(4, "Different text.", 2, 1.0)
        )

        with self.assertRaisesRegex(TypeError, "segments must be a tuple"):
            CoSpeechPlan([pending])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Duplicate speech segment ID"):
            CoSpeechPlan((pending, duplicate))


if __name__ == "__main__":
    unittest.main()
