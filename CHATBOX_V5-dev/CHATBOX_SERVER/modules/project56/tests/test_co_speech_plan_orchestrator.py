"""Tests for sequential response-scoped Method B plan orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector import ActionContext, ActionSelection, ActionSelectorModule
from modules.project56.co_speech import (
    CoSpeechPlanOrchestrator,
    SelectionStatus,
    SpeechSegment,
)
from modules.project56.robot_profile import PlaceholderRobotProfileProvider


def segment(segment_id: int, text: str, duration: float = 3.0) -> SpeechSegment:
    return SpeechSegment(
        segment_id=segment_id,
        text=text,
        word_count=len(text.split()),
        estimated_duration=duration,
    )


class FakeSelector:
    def __init__(
        self,
        selected_tags: list[str | None],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._selected_tags = selected_tags
        self._fail_on_call = fail_on_call
        self.calls: list[dict[str, object]] = []
        self.events: list[str] = []

    def select_method_b(
        self,
        allowed_tags,
        context,
        segment,
        *,
        previous_tag=None,
    ) -> ActionSelection:
        call_number = len(self.calls) + 1
        self.events.append(f"start:{segment.segment_id}")
        self.calls.append({
            "allowed_tags": list(allowed_tags),
            "context": context,
            "segment": segment,
            "previous_tag": previous_tag,
        })
        if self._fail_on_call == call_number:
            raise RuntimeError(f"selection failed for segment {segment.segment_id}")

        selected_tag = self._selected_tags[call_number - 1]
        self.events.append(f"end:{segment.segment_id}")
        return ActionSelection(selected_tag=selected_tag, scores=())


class CapturingLanguageScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, str | None]] = []

    def score_actions(
        self,
        tags,
        context,
        *,
        no_action_on_failure=False,
    ):
        self.calls.append((list(tags), context.dialogue, no_action_on_failure))
        return {
            tag: 1.0 if tag == "AGREE" else 0.5
            for tag in tags
        }


class CoSpeechPlanOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ActionContext(
            robot_id="Navel_001",
            user_message="Please explain the laboratory.",
            dialogue="Full response before segmentation.",
            robot_state={"gesture_module_ready": True},
        )
        self.segments = (
            segment(1, "Hello, welcome to the laboratory."),
            segment(2, "This robot supports interaction research."),
            segment(3, "You may ask another question."),
        )

    def _build_with_fake(self, selected_tags, segments=None):
        selector = FakeSelector(selected_tags)
        plan = CoSpeechPlanOrchestrator(selector).build_selected_plan(
            segments or self.segments,
            allowed_tags=["GREET", "DEFAULT", "AGREE"],
            context=self.context,
        )
        return plan, selector

    def test_one_selected_action_produces_complete_selected_plan(self):
        plan, selector = self._build_with_fake(
            ["AGREE"],
            (self.segments[0],),
        )

        self.assertTrue(plan.is_complete)
        self.assertEqual(plan.segments[0].selection_status, SelectionStatus.SELECTED)
        self.assertEqual(plan.segments[0].selected_tag, "AGREE")
        self.assertEqual(len(selector.calls), 1)

    def test_one_none_result_produces_complete_no_action_plan(self):
        plan, _ = self._build_with_fake([None], (self.segments[0],))

        self.assertTrue(plan.is_complete)
        self.assertEqual(plan.segments[0].selection_status, SelectionStatus.NO_ACTION)
        self.assertIsNone(plan.segments[0].selected_tag)

    def test_multiple_results_preserve_segment_and_selection_order(self):
        plan, selector = self._build_with_fake(["GREET", "DEFAULT", None])

        self.assertEqual(
            [item.speech_segment.segment_id for item in plan.segments],
            [1, 2, 3],
        )
        self.assertEqual(
            [item.selected_tag for item in plan.segments],
            ["GREET", "DEFAULT", None],
        )
        self.assertEqual(
            [item.selection_status for item in plan.segments],
            [
                SelectionStatus.SELECTED,
                SelectionStatus.SELECTED,
                SelectionStatus.NO_ACTION,
            ],
        )
        self.assertEqual(len(selector.calls), 3)

    def test_previous_selected_tag_is_passed_to_next_segment(self):
        _, selector = self._build_with_fake(["AGREE", "DEFAULT", None])

        self.assertEqual(
            [call["previous_tag"] for call in selector.calls],
            [None, "AGREE", "DEFAULT"],
        )

    def test_previous_no_action_passes_none_and_does_not_stop_processing(self):
        plan, selector = self._build_with_fake([None, "AGREE", "DEFAULT"])

        self.assertEqual(
            [call["previous_tag"] for call in selector.calls],
            [None, None, "AGREE"],
        )
        self.assertTrue(plan.is_complete)
        self.assertEqual(plan.segments[1].selected_tag, "AGREE")

    def test_selection_calls_are_strictly_sequential(self):
        _, selector = self._build_with_fake(["GREET", "DEFAULT", None])

        self.assertEqual(
            selector.events,
            ["start:1", "end:1", "start:2", "end:2", "start:3", "end:3"],
        )

    def test_segment_context_uses_exact_current_segment_text(self):
        _, selector = self._build_with_fake(["GREET", "DEFAULT", None])

        self.assertEqual(
            [call["context"].dialogue for call in selector.calls],
            [item.text for item in self.segments],
        )
        self.assertEqual(self.context.dialogue, "Full response before segmentation.")

    def test_selector_error_propagates_and_later_segments_are_not_processed(self):
        selector = FakeSelector(
            ["GREET", "DEFAULT", "AGREE"],
            fail_on_call=2,
        )
        orchestrator = CoSpeechPlanOrchestrator(selector)

        with self.assertRaisesRegex(RuntimeError, "segment 2"):
            orchestrator.build_selected_plan(
                self.segments,
                allowed_tags=["GREET", "DEFAULT", "AGREE"],
                context=self.context,
            )

        self.assertEqual(
            [call["segment"].segment_id for call in selector.calls],
            [1, 2],
        )

    def test_original_segments_and_allowed_tags_are_not_mutated(self):
        original_segments = tuple(self.segments)
        original_values = [item.to_dict() for item in self.segments]
        allowed_tags = ["GREET", "DEFAULT", "AGREE"]
        original_tags = list(allowed_tags)
        selector = FakeSelector(["GREET", "DEFAULT", None])

        CoSpeechPlanOrchestrator(selector).build_selected_plan(
            self.segments,
            allowed_tags=allowed_tags,
            context=self.context,
        )

        self.assertEqual(self.segments, original_segments)
        self.assertEqual(
            [item.to_dict() for item in self.segments],
            original_values,
        )
        self.assertEqual(allowed_tags, original_tags)

    def test_no_action_tag_is_never_introduced(self):
        plan, selector = self._build_with_fake([None, "DEFAULT", None])

        self.assertNotIn(
            "NO_ACTION",
            [item.selected_tag for item in plan.segments],
        )
        self.assertTrue(
            all(
                "NO_ACTION" not in call["allowed_tags"]
                for call in selector.calls
            )
        )

    def test_independent_calls_do_not_share_plan_or_previous_tag(self):
        selector = FakeSelector(["AGREE", "DEFAULT"])
        orchestrator = CoSpeechPlanOrchestrator(selector)
        first = orchestrator.build_selected_plan(
            (self.segments[0],),
            allowed_tags=["AGREE"],
            context=self.context,
        )

        second = orchestrator.build_selected_plan(
            (self.segments[1],),
            allowed_tags=["DEFAULT"],
            context=self.context,
        )

        self.assertEqual(first.segments[0].selected_tag, "AGREE")
        self.assertEqual(second.segments[0].selected_tag, "DEFAULT")
        self.assertIsNone(selector.calls[1]["previous_tag"])

    def test_real_method_b_components_compose_into_complete_plan(self):
        profile = PlaceholderRobotProfileProvider().get_profile("Navel_001")
        selector = ActionSelectorModule(llm=None, robot_profile=profile)
        scorer = CapturingLanguageScorer()
        selector._language_scorer = scorer
        segments = (
            segment(1, "That is correct.", 3.0),
            segment(2, "The robot continues explaining.", 3.0),
            segment(3, "This is the final detail.", 3.0),
        )

        plan = CoSpeechPlanOrchestrator(selector).build_selected_plan(
            segments,
            allowed_tags=["AGREE", "DEFAULT", "POINT"],
            context=self.context,
        )

        self.assertTrue(plan.is_complete)
        self.assertEqual(
            [item.selected_tag for item in plan.segments],
            ["AGREE", "DEFAULT", "AGREE"],
        )
        self.assertEqual(
            [call[0] for call in scorer.calls],
            [
                ["AGREE", "DEFAULT"],
                ["DEFAULT"],
                ["AGREE", "DEFAULT"],
            ],
        )
        self.assertEqual(
            [call[1] for call in scorer.calls],
            [item.text for item in segments],
        )
        self.assertTrue(all(call[2] for call in scorer.calls))

    def test_debug_output_contains_complete_ordered_plan(self):
        plan, _ = self._build_with_fake(["GREET", "DEFAULT", None])

        payload = plan.to_dict()

        self.assertEqual(
            [item["segment_id"] for item in payload["segments"]],
            [1, 2, 3],
        )
        self.assertEqual(
            [item["selection_status"] for item in payload["segments"]],
            ["selected", "selected", "no_action"],
        )
        self.assertEqual(
            [item["selected_tag"] for item in payload["segments"]],
            ["GREET", "DEFAULT", None],
        )


if __name__ == "__main__":
    unittest.main()
