"""No-context baseline, to be replaced by Task 1 / Task 2. Pass-through handlers only:
no KG, no culture layer, no PAD, no prosody or eye colour decisions."""
from interpreter.contracts import (EmotionTarget, InterpretationFrame, SignPlan, SignUnit, SpeechPlan)


def baseline_task1(frame: InterpretationFrame) -> SpeechPlan:
    """No-context baseline, to be replaced by Task 1: say the recogniser's text as-is, neutral delivery."""
    return SpeechPlan(frame_id=frame.frame_id, mode="interpret", utterance=frame.text_en,
                      emotion_label=frame.emotion.label)


def baseline_task2(frame: InterpretationFrame) -> SignPlan:
    """No-context baseline, to be replaced by Task 2: the whole transcript as one unit, no idiom
    resolution, no concept tags, no non-manual hints."""
    e = frame.emotion
    return SignPlan(frame_id=frame.frame_id, meaning=frame.text_en,
                    units=[SignUnit(concept="utterance", surface=frame.text_en)],
                    emotion_target=EmotionTarget(e.label, e.valence, e.arousal, intensity=0.5),
                    register="neutral", confidence=frame.input_confidence)
