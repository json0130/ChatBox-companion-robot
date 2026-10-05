"""Task 1 (sign_to_speech): rule-based speech and expression planner.

Interpret, never reply: the plan carries the SIGNER's words and the SIGNER's emotion. Pepper adds no
content; its own voice is used only for clarification. Rules only: no KG, PAD, LLM or TTS here.
"""
import re
from dataclasses import dataclass
from typing import Optional

from interpreter.context import ContextProvider
from interpreter.contracts import (EyeColour, InterpretationFrame, LexiconEntry, Pause, Prosody,
                                   SpeechPlan, Voice)
from interpreter.lexicon import LEXICON, lookup

# PLACEHOLDERS: colour-emotion associations vary between people and cultures; check them in the
# evaluation step before relying on them.
EYE_PALETTE = {
    "neutral": (200, 220, 255), "happy": (255, 200, 60), "excited": (255, 140, 40),
    "calm": (80, 180, 220), "sad": (50, 80, 200), "apologetic": (150, 120, 200),
    "angry": (220, 40, 40), "anxious": (220, 200, 60), "surprised": (180, 255, 255),
    "confused": (170, 130, 220),
}
CLARIFY_UTTERANCE = "Sorry, one moment, I didn't catch that."
CLARIFY_EYE = EyeColour(rgb=(255, 170, 0), intensity=0.9, pattern="pulse", transition_ms=200)
_PUNCT = ".,!?;:"


@dataclass(frozen=True)
class Task1Config:
    expressiveness: float = 0.8      # 0 = flat delivery, 1 = full emotional delivery
    clarify_threshold: float = 0.5   # input_confidence below this -> ask for a repeat


def _clamp(x: float, lo: float = 0.6, hi: float = 1.6) -> float:
    return max(lo, min(hi, x))


def _choose_words(frame: InterpretationFrame, ctx: ContextProvider) -> list:
    """(english, te_reo) pairs the speaker prefers in te reo."""
    return [(c.english, c.te_reo) for c in frame.concepts
            if c.te_reo and ctx.prefers_te_reo(frame.speaker_id, c.slug)]


def _swap(text: str, pairs: list) -> str:
    """Whole-word, case-insensitive replace, keeping a leading capital."""
    for english, te_reo in pairs:
        def repl(m: re.Match, te_reo: str = te_reo) -> str:
            return te_reo[0].upper() + te_reo[1:] if m.group(0)[0].isupper() else te_reo
        text = re.sub(rf"\b{re.escape(english)}\b", repl, text, flags=re.IGNORECASE)
    return text


def plan_speech(frame: InterpretationFrame, ctx: ContextProvider,
                cfg: Optional[Task1Config] = None) -> SpeechPlan:
    """Plan how Pepper speaks and shows the signer's emotion.

    Clarify: if input_confidence < cfg.clarify_threshold, return the robot-voice line plus an amber
    pulse. The pulse is a visual "please repeat" cue for the signer (a spoken request would not reach
    them); the spoken line is for the non-signer. The colour meaning must be validated with Deaf users.

    Word choice: concept.english is swapped for concept.te_reo when ctx says the speaker prefers it.
    Known limitation: no grammar rewriting, so "family is" becomes "whānau is".

    Prosody: raw multipliers from arousal/valence, scaled toward neutral by
    expressiveness x emotion.confidence. Eye colour comes from EYE_PALETTE.
    """
    cfg = cfg or Task1Config()
    if frame.direction != "sign_to_speech":
        raise ValueError(f"Task 1 handles sign_to_speech only, got {frame.direction!r}")

    if frame.input_confidence < cfg.clarify_threshold:
        return SpeechPlan(frame_id=frame.frame_id, mode="clarify", utterance=CLARIFY_UTTERANCE,
                          emotion_label="neutral", voice=Voice(persona="robot"), eye=CLARIFY_EYE)

    pairs = _choose_words(frame, ctx)
    utterance = _swap(frame.text_en, pairs)
    tokens = utterance.split()
    stripped = [t.strip(_PUNCT).casefold() for t in tokens]

    emo = frame.emotion
    e = cfg.expressiveness * emo.confidence

    def scale(m: float) -> float:
        return 1 + (_clamp(m) - 1) * e

    emphasis = []
    for n in frame.nuance:
        if n.kind != "emphasis":
            continue
        target = _swap(n.target, pairs).strip(_PUNCT)
        if target.casefold() in stripped:   # single-token targets only; the contract can't stress phrases
            word = tokens[stripped.index(target.casefold())].strip(_PUNCT)
            if word not in emphasis:
                emphasis.append(word)

    pauses = [Pause(i, 150 if emo.arousal >= 0 else 300) for i, t in enumerate(tokens) if t.endswith(",")]
    prosody = Prosody(rate=scale(1 + 0.25 * emo.arousal),
                      pitch=scale(1 + 0.15 * emo.arousal + 0.10 * emo.valence),
                      volume=scale(1 + 0.20 * emo.arousal),
                      emphasis=emphasis, pauses=pauses)

    lively = emo.arousal > 0.5
    eye = EyeColour(rgb=EYE_PALETTE[emo.label], intensity=0.5 + 0.4 * (emo.arousal + 1) / 2,
                    pattern="pulse" if lively else "solid", transition_ms=200 if lively else 400)

    lexicon = [LexiconEntry(w, lookup(w)) for w in LEXICON
               if re.search(rf"\b{re.escape(w)}\b", utterance, flags=re.IGNORECASE)]

    return SpeechPlan(frame_id=frame.frame_id, mode="interpret", utterance=utterance,
                      emotion_label=emo.label, voice=Voice(), prosody=prosody, eye=eye, lexicon=lexicon)
