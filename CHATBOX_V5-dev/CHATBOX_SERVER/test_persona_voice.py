from modules.persona_voice import LIMITS, voice_for

QUIET = {"O": 0, "C": 0.9, "E": -0.8, "A": 0.5, "N": 0}
LIVELY = {"O": 0, "C": 0, "E": 0.9, "A": 0, "N": 0.4}


def _n(v):
    return float(v.rstrip("%dBst"))


def test_extravert_is_faster_higher_brighter_than_introvert():
    q, l = voice_for("companion", QUIET), voice_for("companion", LIVELY)
    for key in ("rate", "pitch", "tone", "volume"):
        assert _n(l[key]) > _n(q[key]), key


def test_role_changes_the_voice_for_the_same_personality():
    assert voice_for("receptionist", LIVELY) != voice_for("storyteller", LIVELY)


def test_values_stay_within_the_limits_and_use_the_clients_units():
    v = voice_for("tutor", {k: 1.0 for k in "OCEAN"})
    w = voice_for("tutor", {k: -1.0 for k in "OCEAN"})
    for out in (v, w):
        assert out["rate"].endswith("%") and out["pitch"].endswith("st")
        assert out["tone"].endswith("dB") and out["volume"].endswith("dB")
        for key, lim in LIMITS.items():
            assert abs(_n(out[key])) <= lim + 0.05


def test_unknown_role_and_missing_traits_do_not_crash():
    assert voice_for("something else", {}) == {"rate": "+0%", "pitch": "+0.0st",
                                              "tone": "+0dB", "volume": "+0dB"}
