"""Focused tests for the independent boundary-ID LLM Speech Segmenter."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.co_speech import (
    LLMSpeechSegmenter,
    SpeechSegmenter,
    SpeechTimingProfile,
)
from modules.project56.co_speech.llm_speech_segmenter import OUTPUT_CONTRACT_VERSION
from modules.project56.co_speech.llm_speech_segmenter import SEGMENTATION_POLICY_VERSION
from modules.project56.llm.llm_provider import LLMResponse, parse_response


class FakeLLM:
    def __init__(self, payload=None, raw=None, available=True):
        self.available = available
        self.raw = raw if raw is not None else json.dumps(payload)
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return self.available

    def generate(
        self,
        system_prompt,
        user_message,
        temperature=None,
        response_format=None,
    ):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": temperature,
            "response_format": response_format,
        })
        return LLMResponse(text=self.raw, emotion_tag="", clean_text=self.raw)


def boundary_payload(*unit_ids: int) -> dict:
    return {"boundaries_after": list(unit_ids)}


class LLMSpeechSegmenterTests(unittest.TestCase):
    def test_one_atomic_unit_with_empty_boundaries_returns_one_segment(self):
        text = "Welcome to our lab."
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload()))

        segments = segmenter.segment(text)

        self.assertEqual([segment.text for segment in segments], [text])
        self.assertIsNone(segmenter.last_error)

    def test_three_units_boundary_after_one_groups_remaining_units(self):
        text = (
            "That is correct. There is still one limitation to consider. "
            "The robot cannot perform every movement at the same speed."
        )
        segments = LLMSpeechSegmenter(
            FakeLLM(boundary_payload(1))
        ).segment(text)

        self.assertEqual(
            [segment.text for segment in segments],
            [
                "That is correct.",
                (
                    "There is still one limitation to consider. "
                    "The robot cannot perform every movement at the same speed."
                ),
            ],
        )

    def test_four_units_with_two_boundaries_returns_three_segments(self):
        text = "First idea. Second idea. Third idea. Fourth idea."
        segments = LLMSpeechSegmenter(
            FakeLLM(boundary_payload(1, 3))
        ).segment(text)

        self.assertEqual([segment.segment_id for segment in segments], [1, 2, 3])
        self.assertEqual(
            [segment.text for segment in segments],
            ["First idea.", "Second idea. Third idea.", "Fourth idea."],
        )

    def test_empty_boundaries_group_all_sentence_units(self):
        text = "Welcome to our lab. Today we will introduce Navel."
        segments = LLMSpeechSegmenter(
            FakeLLM(boundary_payload())
        ).segment(text)

        self.assertEqual([segment.text for segment in segments], [text])

    def test_original_wording_punctuation_and_internal_whitespace_are_lossless(self):
        text = "Hello, visitor!  Navel's ready—are you?"
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload()))

        segments = segmenter.segment(text)

        self.assertEqual(segments[0].text, text)
        self.assertEqual(
            "".join(
                unit.text for unit in segmenter.last_diagnostics.atomic_units
            ),
            text,
        )

    def test_word_counts_match_shared_rule_metric(self):
        text = "Hello! I'm Navel robot."
        llm_segment = LLMSpeechSegmenter(
            FakeLLM(boundary_payload())
        ).segment(text)[0]
        rule_metrics = SpeechSegmenter()

        self.assertEqual(llm_segment.word_count, 4)
        self.assertEqual(llm_segment.word_count, rule_metrics.count_words(text))

    def test_duration_uses_same_injected_profile_and_shared_logic(self):
        text = "One two three four."
        profile = SpeechTimingProfile(words_per_second=2.0)
        segment = LLMSpeechSegmenter(
            FakeLLM(boundary_payload()),
            profile=profile,
        ).segment(text)[0]

        self.assertEqual(segment.word_count, 4)
        self.assertEqual(segment.estimated_duration, 2.0)
        self.assertAlmostEqual(
            segment.estimated_duration,
            SpeechSegmenter(profile).estimate_duration(text),
            places=3,
        )

    def test_invalid_json_triggers_safe_fallback_and_diagnostics(self):
        text = "Welcome to our lab. Please come inside."
        segmenter = LLMSpeechSegmenter(FakeLLM(raw="not valid JSON"))

        segments = segmenter.segment(text)

        self.assertEqual([segment.text for segment in segments], [text])
        self.assertIn("malformed", segmenter.last_error)
        self.assertEqual(
            segmenter.last_diagnostics.raw_llm_response,
            "not valid JSON",
        )
        self.assertEqual(
            segmenter.last_diagnostics.validation_error,
            segmenter.last_error,
        )

    def test_wrong_schema_triggers_safe_fallback(self):
        segmenter = LLMSpeechSegmenter(
            FakeLLM({"segments": [{"text": "Welcome."}]})
        )

        segments = segmenter.segment("Welcome.")

        self.assertEqual([segment.text for segment in segments], ["Welcome."])
        self.assertIn("only 'boundaries_after'", segmenter.last_error)

    def test_additional_schema_key_is_rejected(self):
        segmenter = LLMSpeechSegmenter(
            FakeLLM({
                "boundaries_after": [],
                "explanation": "No boundary needed.",
            })
        )

        segmenter.segment("Welcome.")

        self.assertIn("only 'boundaries_after'", segmenter.last_error)

    def test_non_integer_boundary_is_rejected(self):
        segmenter = LLMSpeechSegmenter(
            FakeLLM({"boundaries_after": ["1"]})
        )

        segmenter.segment("First. Second.")

        self.assertIn("integer", segmenter.last_error)
        self.assertEqual(segmenter.last_diagnostics.parsed_boundaries_after, ["1"])

    def test_boolean_boundary_is_rejected_as_non_integer(self):
        segmenter = LLMSpeechSegmenter(
            FakeLLM({"boundaries_after": [True]})
        )

        segmenter.segment("First. Second.")

        self.assertIn("integer", segmenter.last_error)

    def test_boundary_below_range_is_rejected(self):
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload(0)))

        segmenter.segment("First. Second.")

        self.assertIn("outside", segmenter.last_error)

    def test_boundary_above_range_is_rejected(self):
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload(3)))

        segmenter.segment("First. Second.")

        self.assertIn("outside", segmenter.last_error)

    def test_final_atomic_unit_boundary_is_rejected(self):
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload(2)))

        segmenter.segment("First. Second.")

        self.assertIn("final", segmenter.last_error)

    def test_duplicate_boundary_ids_are_rejected(self):
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload(1, 1)))

        segmenter.segment("First. Second. Third.")

        self.assertIn("unique", segmenter.last_error)

    def test_unsorted_boundary_ids_are_rejected(self):
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload(2, 1)))

        segmenter.segment("First. Second. Third.")

        self.assertIn("strictly increasing", segmenter.last_error)

    def test_comma_connector_candidates_can_be_recombined(self):
        text = (
            "The robot can turn its head, but the movement is deliberately "
            "kept slow so that it remains comfortable to watch."
        )
        segmenter = LLMSpeechSegmenter(FakeLLM(boundary_payload()))

        segments = segmenter.segment(text)

        self.assertGreaterEqual(len(segmenter.last_diagnostics.atomic_units), 2)
        self.assertEqual([segment.text for segment in segments], [text])

    def test_sentences_can_be_grouped_across_full_stops(self):
        text = (
            "Welcome to our lab. Today I'll introduce our Navel robot. "
            "It supports our human-robot interaction research."
        )

        segments = LLMSpeechSegmenter(
            FakeLLM(boundary_payload())
        ).segment(text)

        self.assertEqual([segment.text for segment in segments], [text])

    def test_clean_text_excludes_method_a_action_tag(self):
        response = parse_response("[HAPPY] Hello! I'm Navel robot.")
        llm = FakeLLM(boundary_payload())

        segments = LLMSpeechSegmenter(llm).segment(response.clean_text)
        supplied = json.loads(llm.calls[0]["user_message"])

        self.assertEqual(segments[0].text, "Hello! I'm Navel robot.")
        self.assertNotIn("[HAPPY]", llm.calls[0]["user_message"])
        self.assertIn("atomic_units", supplied)

    def test_request_uses_temperature_zero_and_native_json_schema(self):
        llm = FakeLLM(boundary_payload())

        LLMSpeechSegmenter(llm).segment("First. Second. Third.")

        self.assertEqual(llm.calls[0]["temperature"], 0.0)
        response_format = llm.calls[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])
        boundary_schema = schema["properties"]["boundaries_after"]
        self.assertEqual(boundary_schema["items"]["minimum"], 1)
        self.assertEqual(boundary_schema["items"]["maximum"], 2)
        self.assertTrue(boundary_schema["uniqueItems"])

    def test_request_uses_semantic_timing_v2_policy_and_local_durations(self):
        llm = FakeLLM(boundary_payload())

        LLMSpeechSegmenter(llm).segment("That is correct. Another explanation follows.")

        call = llm.calls[0]
        prompt = call["system_prompt"]
        supplied = json.loads(call["user_message"])
        self.assertEqual(SEGMENTATION_POLICY_VERSION, "semantic_timing_v2")
        self.assertIn("2 to 6 seconds", prompt)
        self.assertIn("may exceed 6 seconds", prompt)
        self.assertIn("Short standalone acts", prompt)
        self.assertIn("Do not create a boundary solely", prompt)
        self.assertEqual(
            supplied["timing_policy"],
            {
                "target_min_seconds": 2.0,
                "target_max_seconds": 6.0,
                "substantially_long_seconds": 8.0,
                "speech_rate_wps": 2.556228887,
            },
        )
        self.assertAlmostEqual(
            supplied["atomic_units"][0]["estimated_duration_seconds"],
            3 / 2.556228887,
            places=3,
        )

    def test_single_unit_native_schema_allows_only_empty_boundary_list(self):
        llm = FakeLLM(boundary_payload())

        LLMSpeechSegmenter(llm).segment("Welcome.")

        response_format = llm.calls[0]["response_format"]
        boundary_schema = response_format["json_schema"]["schema"][
            "properties"
        ]["boundaries_after"]
        self.assertEqual(boundary_schema["maxItems"], 0)
        self.assertNotIn("maximum", boundary_schema["items"])

    def test_success_diagnostics_capture_contract_units_raw_ids_and_segments(self):
        raw = json.dumps(boundary_payload(1))
        segmenter = LLMSpeechSegmenter(FakeLLM(raw=raw))

        segmenter.segment("First. Second.")
        diagnostics = segmenter.last_diagnostics.to_dict()

        self.assertEqual(
            diagnostics["output_contract_version"],
            OUTPUT_CONTRACT_VERSION,
        )
        self.assertEqual(
            diagnostics["segmentation_policy_version"],
            SEGMENTATION_POLICY_VERSION,
        )
        self.assertEqual(diagnostics["raw_llm_response"], raw)
        self.assertEqual(diagnostics["parsed_boundaries_after"], [1])
        self.assertEqual(
            diagnostics["reconstructed_segments"],
            ["First.", "Second."],
        )
        self.assertEqual(len(diagnostics["atomic_units"]), 2)
        self.assertIsNone(diagnostics["validation_error"])

    def test_unavailable_provider_uses_safe_full_text_fallback(self):
        text = "Welcome to our robotics lab."
        llm = FakeLLM(available=False)
        segmenter = LLMSpeechSegmenter(llm)

        segments = segmenter.segment(text)

        self.assertEqual([segment.text for segment in segments], [text])
        self.assertIn("not available", segmenter.last_error)
        self.assertEqual(llm.calls, [])

    def test_empty_or_whitespace_input_does_not_call_llm(self):
        for text in ("", "  ", "\n\t"):
            with self.subTest(text=text):
                llm = FakeLLM(boundary_payload())
                self.assertEqual(LLMSpeechSegmenter(llm).segment(text), [])
                self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
