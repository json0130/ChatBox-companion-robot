"""Hardware-free tests for the real Navel Method B execution boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.co_speech import (  # noqa: E402
    NavelSegmentExecutor,
    PlannedSegment,
    SelectionStatus,
    SpeechSegment,
)


def planned(selected_tag: str | None, segment_id: int = 1) -> PlannedSegment:
    speech = SpeechSegment(segment_id, "That is correct.", 3, 1.5)
    return PlannedSegment(
        speech_segment=speech,
        selection_status=(
            SelectionStatus.NO_ACTION
            if selected_tag is None
            else SelectionStatus.SELECTED
        ),
        selected_tag=selected_tag,
    )


class EchoTransport:
    def __init__(self, *, error: str | None = None) -> None:
        self.payloads = []
        self.error = error

    async def execute_segment(self, payload, *, timeout_seconds):
        self.payloads.append((dict(payload), timeout_seconds))
        response = {
            "request_id": payload["request_id"],
            "segment_id": payload["segment_id"],
            "selected_tag": payload["selected_tag"],
            "default_variant": payload.get("default_variant"),
            "ok": self.error is None,
            "metrics": {"actual_speech_duration_seconds": 1.25},
        }
        if self.error is not None:
            response.update({"error_type": "RuntimeError", "error": self.error})
        return response


class NavelSegmentExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_expressive_action_sends_exactly_one_selected_tag(self):
        transport = EchoTransport()
        executor = NavelSegmentExecutor(transport)

        await executor.execute(planned("AGREE"))

        payload, timeout = transport.payloads[0]
        self.assertEqual(payload["event"], "method_b_segment")
        self.assertEqual(payload["selected_tag"], "AGREE")
        self.assertEqual(payload["text"], "That is correct.")
        self.assertEqual(payload["predicted_speech_duration_seconds"], 1.5)
        self.assertNotIn("default_variant", payload)
        self.assertEqual(len(executor.records), 1)
        self.assertIsNone(executor.records[0].default_variant)
        self.assertEqual(timeout, 120.0)

    async def test_none_remains_json_null_and_is_not_default(self):
        transport = EchoTransport()
        executor = NavelSegmentExecutor(transport)

        await executor.execute(planned(None))

        payload, _ = transport.payloads[0]
        self.assertIsNone(payload["selected_tag"])
        self.assertNotEqual(payload["selected_tag"], "DEFAULT")
        self.assertNotIn("NO_ACTION", payload.values())
        self.assertNotIn("default_variant", payload)
        self.assertIsNone(executor.records[0].default_variant)

    async def test_default_uses_explicit_internal_variant(self):
        transport = EchoTransport()
        executor = NavelSegmentExecutor(transport, default_variant="both")

        await executor.execute(planned("DEFAULT"))

        payload, _ = transport.payloads[0]
        self.assertEqual(payload["selected_tag"], "DEFAULT")
        self.assertEqual(payload["default_variant"], "both")
        self.assertEqual(executor.records[0].default_variant, "both")
        self.assertEqual(executor.records[0].to_dict()["default_variant"], "both")

    async def test_auto_selector_advances_only_for_default(self):
        class SequenceSelector:
            def __init__(self):
                self.values = iter(("right", "both"))
                self.calls = 0

            def next_variant(self):
                self.calls += 1
                return next(self.values)

        transport = EchoTransport()
        selector = SequenceSelector()
        executor = NavelSegmentExecutor(
            transport,
            default_variant_selector=selector,
        )

        await executor.execute(planned("DEFAULT", 1))
        await executor.execute(planned("AGREE", 2))
        await executor.execute(planned(None, 3))
        await executor.execute(planned("DEFAULT", 4))

        payloads = [payload for payload, _timeout in transport.payloads]
        self.assertEqual(selector.calls, 2)
        self.assertEqual(payloads[0]["default_variant"], "right")
        self.assertNotIn("default_variant", payloads[1])
        self.assertNotIn("default_variant", payloads[2])
        self.assertEqual(payloads[3]["default_variant"], "both")

    async def test_explicit_left_override_always_sends_left(self):
        await self._assert_fixed_override("left")

    async def test_explicit_right_override_always_sends_right(self):
        await self._assert_fixed_override("right")

    async def test_explicit_both_override_always_sends_both(self):
        await self._assert_fixed_override("both")

    async def _assert_fixed_override(self, variant):
        class FailIfUsedSelector:
            def next_variant(self):
                raise AssertionError("manual override advanced automatic selector")

        transport = EchoTransport()
        executor = NavelSegmentExecutor(
            transport,
            default_variant=variant,
            default_variant_selector=FailIfUsedSelector(),
        )

        await executor.execute(planned("DEFAULT", 1))
        await executor.execute(planned("DEFAULT", 2))

        self.assertEqual(
            [payload["default_variant"] for payload, _ in transport.payloads],
            [variant, variant],
        )

    async def test_pending_segment_is_rejected_without_transport_call(self):
        transport = EchoTransport()
        executor = NavelSegmentExecutor(transport)
        pending = PlannedSegment(
            speech_segment=SpeechSegment(1, "Pending.", 1, 1.0),
        )

        with self.assertRaisesRegex(ValueError, "PENDING"):
            await executor.execute(pending)
        self.assertEqual(transport.payloads, [])

    async def test_robot_failure_propagates_and_is_not_recorded_as_no_action(self):
        transport = EchoTransport(error="gesture failed")
        executor = NavelSegmentExecutor(transport)

        with self.assertRaisesRegex(RuntimeError, "gesture failed"):
            await executor.execute(planned("AGREE"))
        self.assertEqual(executor.records, [])

    async def test_mismatched_default_variant_ack_is_rejected(self):
        class MismatchedVariantTransport(EchoTransport):
            async def execute_segment(self, payload, *, timeout_seconds):
                response = await super().execute_segment(
                    payload,
                    timeout_seconds=timeout_seconds,
                )
                response["default_variant"] = "right"
                return response

        executor = NavelSegmentExecutor(
            MismatchedVariantTransport(),
            default_variant="left",
        )

        with self.assertRaisesRegex(RuntimeError, "different DEFAULT variant"):
            await executor.execute(planned("DEFAULT"))
        self.assertEqual(executor.records, [])

    def test_invalid_default_variant_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "DEFAULT variant"):
            NavelSegmentExecutor(EchoTransport(), default_variant="random")


if __name__ == "__main__":
    unittest.main()
