"""Action-tag generator (Method B, ChatBox): reply text -> one gesture tag per speech segment.

    "Hello! Nice to meet you."  ->  [{"text": "Hello!",             "action": "GREET",   "tag": "WAVE"},
                                     {"text": "Nice to meet you.",  "action": "DEFAULT", "tag": "DEFAULT"}]

Segment (project56 SpeechSegmenter) -> filter by gesture duration -> LLM picks one action per
segment (project56 ActionSelectorModule, "choice" mode) -> planner action -> the Jetson's native
EMOTION_MAP tag (CHATBOX_NATIVE_TAG). A segment with no fitting gesture gets tag None (plain speech).

The Jetson client fires only the FIRST [TAG] of a demo_step, so each segment goes out as its own
step ("[WAVE] Hello!"); RobotHub.say_steps paces them on the client's ack.

    python3 -m modules.action_tag_generator "Hello! Nice to meet you." --ai-server 10.42.0.118
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from modules.project56.action_selector import ActionContext, ActionSelectorModule
from modules.project56.co_speech import CoSpeechPlanOrchestrator, SpeechSegmenter
from modules.project56.llm.llm_provider import parse_response
from modules.project56.robot_profile.chatbox_profile import (
    CHATBOX_NATIVE_TAG, CHATBOX_PROFILE, EXPRESSION_SECONDS)

# CALIBRATION KNOB. A gesture lasts EXPRESSION_SECONDS (4.5 s) but most sentences are shorter, so
# with 0 every expressive action is rejected as "exceeds_segment_duration" (3.0 still drops a
# one-word "Hello!"). The next segment's tag replaces a running gesture, so overrun is harmless on
# the ChatBox: default = never filter on duration. Lower it once gesture/speech times are measured.
OVERRUN_SECONDS = EXPRESSION_SECONDS


class ChatLLM:
    """persona_chat backends (`respond(system, user, history, max_tokens)`) -> project56's `generate()`."""

    def __init__(self, backend, max_tokens: int = 120):
        self.backend, self.max_tokens, self.last_error = backend, max_tokens, ""

    def is_available(self) -> bool:
        return getattr(self.backend, "available", True) is not False

    def generate(self, system_prompt, user_message, temperature=None, response_format=None):
        raw = self.backend.respond(system_prompt, user_message, [], self.max_tokens)
        if raw.startswith("[LLM"):      # backend error text, not a model answer
            self.last_error = raw
        return parse_response(raw)


class ActionTagGenerator:
    def __init__(self, llm, overrun_seconds: float = OVERRUN_SECONDS):
        selector = ActionSelectorModule(llm, CHATBOX_PROFILE, language_scoring_mode="choice",
                                        duration_overrun_allowance_seconds=overrun_seconds)
        selector.initialize()
        self.segmenter = SpeechSegmenter(CHATBOX_PROFILE.speech.segmentation_timing)
        self.orchestrator = CoSpeechPlanOrchestrator(selector)

    def generate(self, reply: str, user_message: str = "") -> List[Dict[str, Optional[str]]]:
        segments = self.segmenter.segment(reply)
        plan = self.orchestrator.build_selected_plan(
            segments, allowed_tags=list(CHATBOX_PROFILE.actions),
            context=ActionContext(user_message=user_message, dialogue=reply,
                                  robot_id=CHATBOX_PROFILE.robot_id))
        return [{"text": p.speech_segment.text, "action": p.selected_tag,
                 "tag": CHATBOX_NATIVE_TAG[p.selected_tag] if p.selected_tag else None}
                for p in plan.segments]


def step_texts(items: List[Dict[str, Optional[str]]]) -> List[str]:
    """demo_step texts: '[TAG] words', or plain words when the segment has no gesture."""
    return [f"[{i['tag']}] {i['text']}" if i["tag"] else i["text"] for i in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("text", help="the robot's reply to plan gestures for")
    ap.add_argument("--user", default="", help="what the person said (context for the picker)")
    ap.add_argument("--ai-server", metavar="HOST[:PORT]")
    ap.add_argument("--ollama", nargs="?", const="qwen2.5:7b", metavar="MODEL")
    ap.add_argument("--overrun", type=float, default=OVERRUN_SECONDS)
    a = ap.parse_args()
    from modules.persona_chat import OllamaChat, ai_server_chat
    if not (a.ai_server or a.ollama):
        ap.error("pick an LLM: --ai-server HOST[:PORT] or --ollama [MODEL]")
    llm = ChatLLM(OllamaChat(a.ollama) if a.ollama else ai_server_chat(a.ai_server))
    items = ActionTagGenerator(llm, a.overrun).generate(a.text, a.user)
    for i in items:
        print(f"{str(i['action']):10} -> {str(i['tag']):10} {i['text']}")
    if llm.last_error:
        print("LLM:", llm.last_error)


if __name__ == "__main__":
    main()
