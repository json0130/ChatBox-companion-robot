"""Interpreter contracts (Step 0): the three messages that connect the modules.

    InterpretationFrame  recogniser (direction 1) or speech front-end (direction 2) -> your decision layer
    SpeechPlan           Task 1 output -> Pepper (voice + eye colour)
    SignPlan             Task 2 output -> collaborator's sign generator   [PROPOSAL, to be agreed]

Rules
- Standard library only, no project imports (keeps the zero-import invariant).
- Every object validates itself on construction, so a bad message fails at the boundary.
- Language is NZ English. Te reo concepts travel as `Concept` tags, not as a second language.
- Prosody values are RELATIVE multipliers (1.0 = neutral), so the same plan can drive NAOqi
  or an external TTS. An adapter (Step 2) maps them to backend parameters.
- Emotion is the SPEAKER's emotion, never the robot's persona.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "0.1"

DIRECTIONS = ("sign_to_speech", "speech_to_sign")
EMOTIONS = ("neutral", "happy", "excited", "calm", "sad", "apologetic",
            "angry", "anxious", "surprised", "confused")
NUANCE_KINDS = ("statement", "question", "request", "command", "exclamation", "emphasis",
                "apology", "greeting", "farewell", "thanks", "idiom", "hedge", "humour")
REGISTERS = ("casual", "neutral", "respectful", "formal")
PERSONAS = ("interpreter", "robot")            # robot = Pepper speaking for itself (clarification only)
BACKENDS = ("auto", "naoqi", "external")
SPEECH_MODES = ("interpret", "clarify")
EYE_PATTERNS = ("solid", "pulse", "fade")
# Open for the sign-generation owner to extend; closed here so typos fail loudly.
NONMANUAL_HINTS = ("brows_up", "brows_down", "head_tilt", "head_nod", "head_shake",
                   "mouth_smile", "mouth_down", "eyes_wide", "slow_pace", "fast_pace")
SIGN_ROLES = ("topic", "comment")


# ---------------------------------------------------------------- validation helpers
def _in(name: str, value: Any, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {allowed}")


def _range(name: str, value: float, lo: float, hi: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (lo <= value <= hi):
        raise ValueError(f"{name}={value!r} must be a number in [{lo}, {hi}]")


def _text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


class _Base:
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


# ---------------------------------------------------------------- shared pieces
@dataclass
class Emotion(_Base):
    """Speaker's affect. valence/arousal/dominance in [-1, 1] (PAD convention), confidence in [0, 1]."""
    label: str
    valence: float
    arousal: float
    confidence: float = 1.0
    dominance: Optional[float] = None

    def __post_init__(self) -> None:
        _in("emotion.label", self.label, EMOTIONS)
        _range("emotion.valence", self.valence, -1, 1)
        _range("emotion.arousal", self.arousal, -1, 1)
        _range("emotion.confidence", self.confidence, 0, 1)
        if self.dominance is not None:
            _range("emotion.dominance", self.dominance, -1, 1)

    @classmethod
    def from_dict(cls, d: dict) -> "Emotion":
        return cls(**d)


@dataclass
class NuanceTag(_Base):
    """kind='emphasis' needs `target` (the emphasised word). kind='idiom' needs `target` (the idiom
    as spoken) and `meaning` (its plain-English meaning)."""
    kind: str
    target: Optional[str] = None
    meaning: Optional[str] = None

    def __post_init__(self) -> None:
        _in("nuance.kind", self.kind, NUANCE_KINDS)
        if self.kind in ("emphasis", "idiom"):
            _text(f"nuance[{self.kind}].target", self.target)
        if self.kind == "idiom":
            _text("nuance[idiom].meaning", self.meaning)

    @classmethod
    def from_dict(cls, d: dict) -> "NuanceTag":
        return cls(**d)


@dataclass
class Concept(_Base):
    """A culturally specific concept. `slug` is the ck: node id used in the KG culture layer."""
    slug: str
    english: str                      # the plain NZ English word ("family")
    te_reo: Optional[str] = None      # the kupu Māori form ("whānau"), if one is in common NZ English use

    def __post_init__(self) -> None:
        if not (isinstance(self.slug, str) and self.slug.startswith("ck:") and len(self.slug) > 3):
            raise ValueError(f"concept.slug={self.slug!r} must start with 'ck:'")
        _text("concept.english", self.english)
        if self.te_reo is not None:
            _text("concept.te_reo", self.te_reo)

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(**d)


# ---------------------------------------------------------------- 1. frame
@dataclass
class InterpretationFrame(_Base):
    frame_id: str
    direction: str
    speaker_id: str                   # KG PersonNode id, e.g. "person:aroha"
    addressee_id: str
    text_en: str                      # NZ English. Direction 1: recogniser's reading. Direction 2: STT transcript.
    emotion: Emotion
    input_confidence: float           # recogniser confidence (dir 1) or STT confidence (dir 2), [0, 1]
    nuance: list[NuanceTag] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    gloss: Optional[str] = None       # only for sign_to_speech
    setting: Optional[str] = None     # e.g. "lab_open_day", "marae"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text("frame_id", self.frame_id)
        _in("direction", self.direction, DIRECTIONS)
        _text("speaker_id", self.speaker_id)
        _text("addressee_id", self.addressee_id)
        _text("text_en", self.text_en)
        _range("input_confidence", self.input_confidence, 0, 1)
        if self.gloss is not None and self.direction != "sign_to_speech":
            raise ValueError("gloss is only valid when direction='sign_to_speech'")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "InterpretationFrame":
        d = dict(d)
        d["emotion"] = Emotion.from_dict(d["emotion"])
        d["nuance"] = [NuanceTag.from_dict(x) for x in d.get("nuance", [])]
        d["concepts"] = [Concept.from_dict(x) for x in d.get("concepts", [])]
        return cls(**d)


# ---------------------------------------------------------------- 2. speech plan (Task 1)
@dataclass
class Voice(_Base):
    persona: str = "interpreter"
    backend: str = "auto"             # "auto" lets the adapter choose; pin to "naoqi"/"external" for A/B tests
    voice_id: Optional[str] = None
    language: str = "en-NZ"

    def __post_init__(self) -> None:
        _in("voice.persona", self.persona, PERSONAS)
        _in("voice.backend", self.backend, BACKENDS)
        if self.language != "en-NZ":
            raise ValueError("language is fixed to 'en-NZ' for now")

    @classmethod
    def from_dict(cls, d: dict) -> "Voice":
        return cls(**d)


@dataclass
class Pause(_Base):
    after_word_index: int             # index into utterance.split()
    ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.after_word_index, int) or self.after_word_index < 0:
            raise ValueError("pause.after_word_index must be an int >= 0")
        _range("pause.ms", self.ms, 0, 2000)

    @classmethod
    def from_dict(cls, d: dict) -> "Pause":
        return cls(**d)


@dataclass
class Prosody(_Base):
    """Relative multipliers, 1.0 = neutral. Adapter clamps to what the backend supports."""
    rate: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emphasis: list[str] = field(default_factory=list)   # words in the utterance to stress
    pauses: list[Pause] = field(default_factory=list)

    def __post_init__(self) -> None:
        for n in ("rate", "pitch", "volume"):
            _range(f"prosody.{n}", getattr(self, n), 0.5, 2.0)

    @classmethod
    def from_dict(cls, d: dict) -> "Prosody":
        d = dict(d)
        d["pauses"] = [Pause.from_dict(x) for x in d.get("pauses", [])]
        return cls(**d)


@dataclass
class EyeColour(_Base):
    """Pepper eye LEDs. rgb is 0-255 per channel; intensity scales brightness."""
    rgb: tuple[int, int, int] = (255, 255, 255)
    intensity: float = 1.0
    pattern: str = "solid"
    transition_ms: int = 300

    def __post_init__(self) -> None:
        self.rgb = tuple(self.rgb)  # JSON round-trips lists
        if len(self.rgb) != 3:
            raise ValueError("eye.rgb must have 3 channels")
        for c in self.rgb:
            _range("eye.rgb channel", c, 0, 255)
        _range("eye.intensity", self.intensity, 0, 1)
        _in("eye.pattern", self.pattern, EYE_PATTERNS)
        _range("eye.transition_ms", self.transition_ms, 0, 5000)

    @classmethod
    def from_dict(cls, d: dict) -> "EyeColour":
        return cls(**d)


@dataclass
class LexiconEntry(_Base):
    """Pronunciation hint for a word the TTS may get wrong (mostly kupu Māori in NZ English)."""
    word: str
    respelling: str

    def __post_init__(self) -> None:
        _text("lexicon.word", self.word)
        _text("lexicon.respelling", self.respelling)

    @classmethod
    def from_dict(cls, d: dict) -> "LexiconEntry":
        return cls(**d)


@dataclass
class SpeechPlan(_Base):
    frame_id: str                     # the frame this plan answers
    mode: str                         # "interpret" | "clarify"
    utterance: str
    emotion_label: str                # emotion being shown (for logs and evaluation)
    voice: Voice = field(default_factory=Voice)
    prosody: Prosody = field(default_factory=Prosody)
    eye: EyeColour = field(default_factory=EyeColour)
    lexicon: list[LexiconEntry] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text("frame_id", self.frame_id)
        _in("mode", self.mode, SPEECH_MODES)
        _text("utterance", self.utterance)
        _in("emotion_label", self.emotion_label, EMOTIONS)
        if self.mode == "clarify" and self.voice.persona != "robot":
            raise ValueError("clarify mode must use persona='robot' (the interpreter never speaks for itself)")
        if self.mode == "interpret" and self.voice.persona != "interpreter":
            raise ValueError("interpret mode must use persona='interpreter'")
        words = self.utterance.split()
        for w in self.prosody.emphasis:
            if w not in words and w.strip(".,!?;:") not in [x.strip(".,!?;:") for x in words]:
                raise ValueError(f"prosody.emphasis word {w!r} not found in utterance")
        for p in self.prosody.pauses:
            if p.after_word_index >= len(words):
                raise ValueError(f"pause index {p.after_word_index} beyond utterance length {len(words)}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "SpeechPlan":
        d = dict(d)
        d["voice"] = Voice.from_dict(d.get("voice", {}))
        d["prosody"] = Prosody.from_dict(d.get("prosody", {}))
        d["eye"] = EyeColour.from_dict(d.get("eye", {}))
        d["lexicon"] = [LexiconEntry.from_dict(x) for x in d.get("lexicon", [])]
        return cls(**d)


# ---------------------------------------------------------------- 3. sign plan (Task 2) -- PROPOSAL
@dataclass
class SignUnit(_Base):
    """One meaning unit, in English order. The generator owns NZSL grammar and re-ordering."""
    concept: str                      # plain concept ("apology", "postpone") or a ck: slug
    surface: Optional[str] = None     # the English words it came from
    prefer_sign: Optional[str] = None # e.g. a Māori-concept lexicon entry id
    role: Optional[str] = None
    time: Optional[str] = None

    def __post_init__(self) -> None:
        _text("unit.concept", self.concept)
        if self.role is not None:
            _in("unit.role", self.role, SIGN_ROLES)

    @classmethod
    def from_dict(cls, d: dict) -> "SignUnit":
        return cls(**d)


@dataclass
class EmotionTarget(_Base):
    label: str
    valence: float
    arousal: float
    intensity: float = 0.5            # how strongly to show it, [0, 1]

    def __post_init__(self) -> None:
        _in("emotion_target.label", self.label, EMOTIONS)
        _range("emotion_target.valence", self.valence, -1, 1)
        _range("emotion_target.arousal", self.arousal, -1, 1)
        _range("emotion_target.intensity", self.intensity, 0, 1)

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionTarget":
        return cls(**d)


@dataclass
class SignPlan(_Base):
    frame_id: str
    meaning: str                      # plain NZ English, idioms resolved
    units: list[SignUnit]
    emotion_target: EmotionTarget
    register: str = "neutral"
    nonmanual_hints: list[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_clarification: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text("frame_id", self.frame_id)
        _text("meaning", self.meaning)
        if not self.units:
            raise ValueError("units must not be empty")
        _in("register", self.register, REGISTERS)
        for h in self.nonmanual_hints:
            _in("nonmanual_hint", h, NONMANUAL_HINTS)
        _range("confidence", self.confidence, 0, 1)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "SignPlan":
        d = dict(d)
        d["units"] = [SignUnit.from_dict(x) for x in d["units"]]
        d["emotion_target"] = EmotionTarget.from_dict(d["emotion_target"])
        return cls(**d)
