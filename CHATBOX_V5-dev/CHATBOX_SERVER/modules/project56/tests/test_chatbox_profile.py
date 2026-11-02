import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVER_DIR))

from modules.project56.action_selector.duration_filter import DurationFilter  # noqa: E402
from modules.project56.robot_profile import PlaceholderRobotProfileProvider  # noqa: E402
from modules.project56.robot_profile.chatbox_profile import (  # noqa: E402
    CHATBOX_NATIVE_TAG,
    CHATBOX_PROFILE,
    EXPRESSION_SECONDS,
)

CLIENT = SERVER_DIR.parents[1] / "FINAL_DEMO" / "CHATBOX_CLIENT" / "robot.py"


def _client_emotion_map():
    for node in ast.parse(CLIENT.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EMOTION_MAP":
            return ast.literal_eval(node.value)
    raise AssertionError("EMOTION_MAP not found in robot.py")


class ChatBoxProfileTests(unittest.TestCase):
    def test_provider_returns_it_by_client_id(self):
        profile = PlaceholderRobotProfileProvider().get_profile("chatbox_jetson_001")
        self.assertIs(profile, CHATBOX_PROFILE)

    def test_every_action_has_a_native_tag_and_a_duration(self):
        self.assertEqual(set(CHATBOX_PROFILE.actions), set(CHATBOX_NATIVE_TAG))
        for action in CHATBOX_PROFILE.actions.values():
            self.assertTrue(action.profiled)
            self.assertEqual(action.duration_seconds, EXPRESSION_SECONDS)

    @unittest.skipUnless(CLIENT.exists(), "FINAL_DEMO client not in this checkout")
    def test_native_tags_are_known_to_the_jetson_client_and_distinct(self):
        emotion_map = _client_emotion_map()
        motions = []
        for native in CHATBOX_NATIVE_TAG.values():
            self.assertIn(native, emotion_map)
            motions.append(emotion_map[native])
        self.assertEqual(len(motions), len(set(motions)), "two actions share one motion")

    def test_gesture_must_fit_the_segment(self):
        tags = list(CHATBOX_PROFILE.actions)
        keep = lambda sec: DurationFilter().filter(  # noqa: E731
            SimpleNamespace(estimated_duration=sec), tags, CHATBOX_PROFILE.actions
        ).eligible_tags
        self.assertEqual(keep(1.0), ())
        self.assertEqual(set(keep(8.0)), set(tags))


if __name__ == "__main__":
    unittest.main()
