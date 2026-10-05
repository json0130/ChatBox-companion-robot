"""Routes a frame to the Task 1 or Task 2 handler and delivers the plan to its sink.
Handlers and sinks are injected so later steps can swap them."""
from interpreter.contracts import InterpretationFrame, SignPlan, SpeechPlan


class Pipeline:
    def __init__(self, task1, task2, speech_sink, sign_sink) -> None:
        """task1: frame -> SpeechPlan, task2: frame -> SignPlan,
        speech_sink needs .speak(plan), sign_sink needs .send(plan)."""
        self.task1, self.task2 = task1, task2
        self.speech_sink, self.sign_sink = speech_sink, sign_sink

    def process(self, frame: InterpretationFrame):
        if frame.direction == "sign_to_speech":
            handler, plan_type, deliver = self.task1, SpeechPlan, self.speech_sink.speak
        else:
            handler, plan_type, deliver = self.task2, SignPlan, self.sign_sink.send
        plan = handler(frame)
        if not isinstance(plan, plan_type):
            raise TypeError(f"{frame.frame_id}: {frame.direction} handler returned "
                            f"{type(plan).__name__}, expected {plan_type.__name__}")
        if plan.frame_id != frame.frame_id:
            raise ValueError(f"plan.frame_id {plan.frame_id!r} != frame.frame_id {frame.frame_id!r}")
        deliver(plan)
        return plan
