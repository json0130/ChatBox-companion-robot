"""Tests for one-segment-lookahead Method B scheduling."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import MappingProxyType


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector import (
    ActionContext,
    ActionSelection,
    ActionSelectorModule,
)
from modules.project56.action_selector.duration_filter import DurationFilterResult
from modules.project56.co_speech import (
    OneSegmentLookaheadCoordinator,
    SelectionStatus,
    SpeechSegment,
)
from modules.project56.robot_profile import PlaceholderRobotProfileProvider


def speech_segment(segment_id: int, duration: float = 3.0) -> SpeechSegment:
    text = f"Segment {segment_id} explanatory speech."
    return SpeechSegment(segment_id, text, len(text.split()), duration)


class ControlledPreparedSelector:
    def __init__(
        self,
        selections: dict[int, str | None],
        *,
        gated_segments: set[int] | None = None,
        failing_segments: set[int] | None = None,
    ) -> None:
        self.selections = selections
        self.gated_segments = gated_segments or set()
        self.failing_segments = failing_segments or set()
        self.prepare_calls: list[int] = []
        self.select_calls: list[tuple[int, str | None]] = []
        self.select_started = {
            segment_id: asyncio.Event() for segment_id in selections
        }
        self.select_release = {
            segment_id: asyncio.Event() for segment_id in selections
        }
        self.select_cancelled = {
            segment_id: asyncio.Event() for segment_id in selections
        }

    def prepare_method_b_candidates(self, allowed_tags, segment):
        self.prepare_calls.append(segment.segment_id)
        return DurationFilterResult(
            eligible_tags=tuple(allowed_tags),
            rejected=MappingProxyType({}),
        )

    async def select_method_b_prepared(
        self,
        duration_result,
        context,
        *,
        previous_tag=None,
    ):
        segment_id = int(context.dialogue.split()[1])
        self.select_calls.append((segment_id, previous_tag))
        self.select_started[segment_id].set()
        try:
            if segment_id in self.gated_segments:
                await self.select_release[segment_id].wait()
            if segment_id in self.failing_segments:
                raise RuntimeError(f"planning failed for segment {segment_id}")
            return ActionSelection(
                selected_tag=self.selections[segment_id],
                scores=(),
            )
        except asyncio.CancelledError:
            self.select_cancelled[segment_id].set()
            raise


class ControlledExecutor:
    def __init__(
        self,
        segment_ids: tuple[int, ...],
        *,
        failing_segments: set[int] | None = None,
    ) -> None:
        self.failing_segments = failing_segments or set()
        self.started = {segment_id: asyncio.Event() for segment_id in segment_ids}
        self.release = {segment_id: asyncio.Event() for segment_id in segment_ids}
        self.finished = {segment_id: asyncio.Event() for segment_id in segment_ids}
        self.calls = []

    async def execute(self, segment):
        segment_id = segment.speech_segment.segment_id
        self.calls.append(segment)
        self.started[segment_id].set()
        await self.release[segment_id].wait()
        if segment_id in self.failing_segments:
            raise RuntimeError(f"execution failed for segment {segment_id}")
        self.finished[segment_id].set()


class DefaultPreferringScorer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score_actions(self, tags, context, *, no_action_on_failure=False):
        self.calls.append(list(tags))
        return {tag: 1.0 if tag == "DEFAULT" else 0.5 for tag in tags}


class OneSegmentLookaheadCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.segments = tuple(speech_segment(index) for index in (1, 2, 3))
        self.context = ActionContext(
            robot_id="Navel_001",
            user_message="Tell me more.",
            dialogue="Unsegmented response.",
            robot_state={"gesture_module_ready": True},
        )

    def coordinator(self, selector, executor):
        return OneSegmentLookaheadCoordinator(selector, executor)

    async def test_duration_prepares_all_segments_before_first_selection(self):
        selector = ControlledPreparedSelector({1: "AGREE", 2: None, 3: "DEFAULT"})
        executor = ControlledExecutor((1, 2, 3))
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments,
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        self.assertEqual(selector.prepare_calls, [1, 2, 3])
        self.assertEqual(selector.select_calls[0], (1, None))
        for event in executor.release.values():
            event.set()
        await task

    async def test_only_one_segment_is_planned_ahead(self):
        selector = ControlledPreparedSelector(
            {1: "AGREE", 2: "DEFAULT", 3: None},
            gated_segments={2, 3},
        )
        executor = ControlledExecutor((1, 2, 3))
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments,
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        await selector.select_started[2].wait()
        self.assertFalse(selector.select_started[3].is_set())
        selector.select_release[2].set()
        await asyncio.sleep(0)
        self.assertFalse(selector.select_started[3].is_set())

        executor.release[1].set()
        await executor.started[2].wait()
        await selector.select_started[3].wait()
        selector.select_release[3].set()
        executor.release[2].set()
        await executor.started[3].wait()
        executor.release[3].set()
        result = await task

        self.assertTrue(result.plan.is_complete)
        self.assertEqual(
            selector.select_calls,
            [(1, None), (2, "AGREE"), (3, "DEFAULT")],
        )

    async def test_ready_next_segment_starts_after_current_finishes(self):
        selector = ControlledPreparedSelector({1: "AGREE", 2: "DEFAULT"})
        executor = ControlledExecutor((1, 2))
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        await selector.select_started[2].wait()
        self.assertFalse(executor.started[2].is_set())
        executor.release[1].set()
        await executor.started[2].wait()
        executor.release[2].set()
        result = await task

        timing = result.planning_timings[0]
        self.assertTrue(timing.ready_before_current_finished)
        self.assertEqual(result.lookahead_success_count, 1)
        self.assertEqual(result.lookahead_success_rate, 1.0)
        self.assertEqual(timing.overrun_seconds, 0.0)

    async def test_late_planning_records_overrun_and_delays_next_only(self):
        selector = ControlledPreparedSelector(
            {1: "AGREE", 2: "DEFAULT"},
            gated_segments={2},
        )
        executor = ControlledExecutor((1, 2))
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        await selector.select_started[2].wait()
        executor.release[1].set()
        await executor.finished[1].wait()
        self.assertFalse(executor.started[2].is_set())
        await asyncio.sleep(0.001)
        selector.select_release[2].set()
        await executor.started[2].wait()
        executor.release[2].set()
        result = await task

        timing = result.planning_timings[0]
        self.assertFalse(timing.ready_before_current_finished)
        self.assertGreater(timing.overrun_seconds, 0.0)
        self.assertEqual(result.lookahead_success_count, 0)
        self.assertEqual(result.total_overrun_seconds, timing.overrun_seconds)

    async def test_none_is_executed_as_speech_and_propagated(self):
        selector = ControlledPreparedSelector({1: None, 2: "AGREE"})
        executor = ControlledExecutor((1, 2))
        for event in executor.release.values():
            event.set()

        result = await self.coordinator(selector, executor).run_with_lookahead(
            self.segments[:2],
            allowed_tags=["AGREE", "DEFAULT"],
            context=self.context,
        )

        self.assertEqual(selector.select_calls, [(1, None), (2, None)])
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(
            executor.calls[0].selection_status,
            SelectionStatus.NO_ACTION,
        )
        self.assertIsNone(executor.calls[0].selected_tag)

    async def test_default_is_selected_and_propagated_normally(self):
        selector = ControlledPreparedSelector({1: "DEFAULT", 2: "AGREE"})
        executor = ControlledExecutor((1, 2))
        for event in executor.release.values():
            event.set()

        result = await self.coordinator(selector, executor).run_with_lookahead(
            self.segments[:2],
            allowed_tags=["AGREE", "DEFAULT"],
            context=self.context,
        )

        self.assertEqual(selector.select_calls, [(1, None), (2, "DEFAULT")])
        self.assertEqual(
            result.plan.segments[0].selection_status,
            SelectionStatus.SELECTED,
        )
        self.assertEqual(result.plan.segments[0].selected_tag, "DEFAULT")

    async def test_planning_failure_waits_for_current_execution_then_propagates(self):
        selector = ControlledPreparedSelector(
            {1: "AGREE", 2: "DEFAULT"},
            failing_segments={2},
        )
        executor = ControlledExecutor((1, 2))
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        await selector.select_started[2].wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        executor.release[1].set()
        with self.assertRaisesRegex(RuntimeError, "planning failed for segment 2"):
            await task
        self.assertEqual(
            [call.speech_segment.segment_id for call in executor.calls],
            [1],
        )

    async def test_execution_failure_cancels_pending_planning(self):
        selector = ControlledPreparedSelector(
            {1: "AGREE", 2: "DEFAULT"},
            gated_segments={2},
        )
        executor = ControlledExecutor((1, 2), failing_segments={1})
        task = asyncio.create_task(
            self.coordinator(selector, executor).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            )
        )

        await executor.started[1].wait()
        await selector.select_started[2].wait()
        executor.release[1].set()
        with self.assertRaisesRegex(RuntimeError, "execution failed for segment 1"):
            await task
        self.assertTrue(selector.select_cancelled[2].is_set())
        self.assertEqual(
            [call.speech_segment.segment_id for call in executor.calls],
            [1],
        )

    async def test_final_segment_has_no_extra_planning_call(self):
        selector = ControlledPreparedSelector({1: "AGREE"})
        executor = ControlledExecutor((1,))
        executor.release[1].set()

        result = await self.coordinator(selector, executor).run_with_lookahead(
            self.segments[:1],
            allowed_tags=["AGREE"],
            context=self.context,
        )

        self.assertEqual(selector.select_calls, [(1, None)])
        self.assertEqual(result.planning_timings, ())
        self.assertGreaterEqual(result.initial_planning_latency_seconds, 0.0)
        self.assertGreaterEqual(
            result.initial_planning_latency_seconds,
            result.duration_preparation_latency_seconds,
        )

    async def test_two_concurrent_responses_share_no_selection_state(self):
        selector_a = ControlledPreparedSelector({1: "AGREE", 2: None})
        selector_b = ControlledPreparedSelector({1: "DEFAULT", 2: "AGREE"})
        executor_a = ControlledExecutor((1, 2))
        executor_b = ControlledExecutor((1, 2))
        for executor in (executor_a, executor_b):
            for event in executor.release.values():
                event.set()

        result_a, result_b = await asyncio.gather(
            self.coordinator(selector_a, executor_a).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            ),
            self.coordinator(selector_b, executor_b).run_with_lookahead(
                self.segments[:2],
                allowed_tags=["AGREE", "DEFAULT"],
                context=self.context,
            ),
        )

        self.assertEqual(selector_a.select_calls, [(1, None), (2, "AGREE")])
        self.assertEqual(selector_b.select_calls, [(1, None), (2, "DEFAULT")])
        self.assertEqual(
            [item.selected_tag for item in result_a.plan.segments],
            ["AGREE", None],
        )
        self.assertEqual(
            [item.selected_tag for item in result_b.plan.segments],
            ["DEFAULT", "AGREE"],
        )

    async def test_real_selector_profile_and_filters_compose_with_lookahead(self):
        profile = PlaceholderRobotProfileProvider().get_profile("Navel_001")
        selector = ActionSelectorModule(llm=None, robot_profile=profile)
        scorer = DefaultPreferringScorer()
        selector._language_scorer = scorer
        executor = ControlledExecutor((1, 2))
        executor.release[1].set()
        executor.release[2].set()
        segments = self.segments[:2]
        original_payloads = [segment.to_dict() for segment in segments]

        result = await self.coordinator(selector, executor).run_with_lookahead(
            segments,
            allowed_tags=["DEFAULT", "AGREE", "POINT"],
            context=self.context,
        )

        self.assertEqual(
            [
                prepared.duration_filter_result.eligible_tags
                for prepared in result.prepared_segments
            ],
            [("DEFAULT", "AGREE"), ("DEFAULT", "AGREE")],
        )
        self.assertEqual(
            scorer.calls,
            [["DEFAULT", "AGREE"], ["DEFAULT", "AGREE"]],
        )
        self.assertEqual(
            [planned.selected_tag for planned in result.plan.segments],
            ["DEFAULT", "DEFAULT"],
        )
        self.assertEqual([segment.to_dict() for segment in segments], original_payloads)


if __name__ == "__main__":
    unittest.main()
