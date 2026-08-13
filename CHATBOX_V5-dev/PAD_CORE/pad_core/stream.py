from collections import deque


class AffectStream:
    """Converts raw valence/arousal signals into smoothed PAD offsets.

    Sits between a V/A emotion model and the affect pipeline, and does ONE job:
    average out frame-to-frame jitter over a short window. It deliberately does
    NOT attenuate the signal — `affect.feel` owns that, via the empathy fraction
    (0.60) that decides how far a face pulls the robot off its temperament.

    The gains therefore default to 1.0. They were 0.3, which multiplied with
    empathy to an effective 0.18 and made PAD look like it barely responded to
    anything; attenuating in two places is a bug, not a tuning choice.
    """

    def __init__(
        self,
        gain_valence: float = 1.0,
        gain_arousal: float = 1.0,
        smoothing_window: int = 5,
    ):
        self.gain_valence = gain_valence
        self.gain_arousal = gain_arousal
        self._p_buf: deque[float] = deque(maxlen=smoothing_window)
        self._a_buf: deque[float] = deque(maxlen=smoothing_window)

    def update(self, valence: float, arousal: float) -> tuple[float, float]:
        """Smooth over the window and return (valence, arousal)."""
        self._p_buf.append(valence * self.gain_valence)
        self._a_buf.append(arousal * self.gain_arousal)
        dP = sum(self._p_buf) / len(self._p_buf)
        dA = sum(self._a_buf) / len(self._a_buf)
        return dP, dA

    def mock_update(self, valence: float = 0.0, arousal: float = 0.0) -> tuple[float, float]:
        """For testing without a live camera feed."""
        return self.update(valence, arousal)
