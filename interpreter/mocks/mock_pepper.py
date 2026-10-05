"""Mock Pepper: records the SpeechPlan and prints one line. No TTS, no NAOqi."""
from interpreter.contracts import SpeechPlan


class MockPepper:
    def __init__(self) -> None:
        self.received: list = []

    def speak(self, plan: SpeechPlan) -> None:
        self.received.append(plan)
        p, e = plan.prosody, plan.eye
        print(f"[pepper] {plan.frame_id} {plan.mode} {plan.utterance!r} "
              f"rate={p.rate} pitch={p.pitch} vol={p.volume} eye={e.rgb}")
