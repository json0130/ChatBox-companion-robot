"""Stub sign generator: records the SignPlan and prints one line. No avatar, no NZSL."""
from interpreter.contracts import SignPlan


class StubSignGenerator:
    def __init__(self) -> None:
        self.received: list = []

    def send(self, plan: SignPlan) -> None:
        self.received.append(plan)
        print(f"[signgen] {plan.frame_id} meaning={plan.meaning!r} "
              f"emotion={plan.emotion_target.label} register={plan.register}")
