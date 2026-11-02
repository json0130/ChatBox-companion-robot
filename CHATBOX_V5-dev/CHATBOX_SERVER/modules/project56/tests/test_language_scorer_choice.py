"""Choice-mode tests; legacy scoring remains the default production mode."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector import ActionContext, ActionSelectorModule
from modules.project56.action_selector.language_scorer import LanguageScorer
from modules.project56.action_selector.action_catalog import (
    V2_ACTION_DESCRIPTIONS,
    V2_1_ACTION_DESCRIPTIONS,
)
from modules.project56.co_speech import SpeechSegment
from modules.project56.robot_profile import PlaceholderRobotProfileProvider


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def is_available(self):
        return True

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        raw = self.replies.pop(0)
        return SimpleNamespace(text=raw, clean_text=raw)


class LanguageScorerChoiceTests(unittest.TestCase):
    def setUp(self):
        self.context = ActionContext(
            robot_id="Navel_001", user_message="Explain this.",
            dialogue="The camera detects visitors.",
        )

    def test_choice_parsing_maps_one_choice_to_existing_scores(self):
        llm = FakeLLM(['{"tag": "POINT", "confidence": 0.75}'])
        scores = LanguageScorer(llm, scoring_mode="choice").score_actions(
            ["POINT", "DEFAULT", "AGREE"], self.context,
        )
        self.assertEqual(scores, {"POINT": 0.75, "DEFAULT": 0.0, "AGREE": 0.0})

    def test_invalid_tag_retries_once_with_correction(self):
        llm = FakeLLM([
            '{"tag": "NOT_A_CANDIDATE", "confidence": 0.8}',
            '{"tag": "AGREE", "confidence": 0.6}',
        ])
        scores = LanguageScorer(llm, scoring_mode="choice").score_actions(
            ["AGREE", "DEFAULT"], self.context,
        )
        self.assertEqual(scores, {"AGREE": 0.6, "DEFAULT": 0.0})
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Correction:", llm.calls[1][1])

    def test_invalid_choice_falls_back_to_default_then_no_action(self):
        scorer = LanguageScorer(FakeLLM(["{}", "{}"]), scoring_mode="choice")
        traces = []
        scores = scorer.score_actions(["POINT", "DEFAULT"], self.context,
                                      diagnostic_callback=traces.append)
        self.assertEqual(scores, {"POINT": 0.0, "DEFAULT": 1.0})
        self.assertEqual(traces[0].fallback_reason, "choice_invalid_after_retry_default")

        scorer = LanguageScorer(FakeLLM(["{}", "{}"]), scoring_mode="choice")
        scores = scorer.score_actions(["POINT"], self.context)
        self.assertEqual(scores, {"POINT": 0.0})

        profile = PlaceholderRobotProfileProvider().get_profile("Navel_001")
        selector = ActionSelectorModule(
            FakeLLM(["{}", "{}"]), robot_profile=profile,
            language_scoring_mode="choice",
        )
        selection = selector.select_method_b(
            ["POINT"], self.context,
            SpeechSegment(1, "A long enough segment.", 4, 10.0),
        )
        self.assertIsNone(selection.selected_tag)

    def test_legacy_prompts_remain_byte_for_byte(self):
        llm = FakeLLM(['{"POINT": 0.8, "DEFAULT": 0.2}'])
        LanguageScorer(llm, scorer_context="none").score_actions(
            ["POINT", "DEFAULT"], self.context, no_action_on_failure=True,
        )
        system, user, kwargs = llm.calls[0]
        self.assertEqual(kwargs, {})
        self.assertEqual(system, '''You score how appropriately robot actions accompany generated dialogue.

Each candidate includes a tag and a description of what the action means.

For every candidate, return a relevance score from 0.0 to 1.0.

Rules:
- Score how naturally the action accompanies the generated dialogue.
- Give an expressive action a high score when it clearly matches.
- Do not assess physical availability; it is evaluated separately.
- Only return scores for the supplied candidate actions.
- The caller handles the valid outcome where no gesture is selected.

Return exactly one valid JSON object mapping each tag to its score.
Do not return Markdown, explanations, or additional keys.

Example:
{"POINT": 0.8, "AGREE": 0.2, "HAPPY": 0.1}''')
        self.assertEqual(user, '''User message:
Explain this.

Generated robot dialogue:
The camera detects visitors.

Detected user emotion:
unknown

Candidate actions and descriptions:
{
  "POINT": "Point towards a person, object, display, or direction being discussed.",
  "DEFAULT": "Remain in a neutral pose. Use only when no expressive action fits the dialogue."
}''')

    def test_v2_description_text_reaches_the_prompt(self):
        for mode, reply in (
            ("legacy", '{"POINT": 0.8, "DEFAULT": 0.2}'),
            ("choice", '{"tag": "POINT", "confidence": 0.8}'),
        ):
            llm = FakeLLM([reply])
            LanguageScorer(
                llm, scoring_mode=mode, action_description_set="v2",
            ).score_actions(["POINT", "DEFAULT"], self.context,
                            no_action_on_failure=True)
            prompt = llm.calls[0][1]
            self.assertIn("Use ONLY when the sentence refers to a specific physical thing", prompt)
            self.assertIn("Baseline speaking gesture. Use for ordinary statements", prompt)

    def test_v2_1_differs_from_v2_only_in_point(self):
        self.assertEqual(set(V2_1_ACTION_DESCRIPTIONS), set(V2_ACTION_DESCRIPTIONS))
        for tag, description in V2_ACTION_DESCRIPTIONS.items():
            if tag == "POINT":
                self.assertNotEqual(V2_1_ACTION_DESCRIPTIONS[tag], description)
            else:
                self.assertEqual(V2_1_ACTION_DESCRIPTIONS[tag], description)


if __name__ == "__main__":
    unittest.main()
