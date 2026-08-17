"""
PAD reaches the system prompt and the servos, without displacing memory.

The regression this exists to prevent used to be live: enabling PAD swapped in a
prompt built by pad_persona, so the identity block, the KG memory, the RAG hits
and the anti-hallucination rules all vanished. Affect and memory never ran
together, which is why PAD never appeared to change anything.

Also pins the descriptor compression that shapes how the prompt-grid experiment
must be read: CHATBOX shows 30% of its temperament, which collapses several
tiers onto identical wording, while ELLEBOT differentiates.

No LLM, no camera. Run directly or under pytest.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.affect_bridge import affect                              # noqa: E402
from modules.face_webcam.webcam_loop import WebcamKGLoop              # noqa: E402
from tools.pad_prompt_grid import _seed_person                        # noqa: E402

_TIERS = ("unknown", "visitor", "known", "close")
_BLOCKS = ("IDENTITY", "HOW TO REPLY", "WHO YOU'RE TALKING TO")


def _loop(robot="chatbox"):
    tmp = tempfile.mkdtemp(prefix="pad_prompt_")
    return WebcamKGLoop(
        robot_id=robot, show_window=False, seed=True, llm_client=None,
        kg_path=os.path.join(tmp, "kg.json"),
        faces_path=os.path.join(tmp, "faces.npz"),
        sessions_db=os.path.join(tmp, "s.db"),
        embed_fn=None, pad_enabled=True, emotion_enabled=True,
    )


def test_pad_does_not_displace_the_memory_system():
    """THE regression guard. With PAD on, the prompt must still carry every block
    it carries with PAD off — plus the manner line, and nothing removed."""
    loop = _loop()
    pid = "p_known"
    tier = _seed_person(loop.store, pid, "chatbox", "known")
    pad = loop._adapter().process_turn(0.0, 0.0, tier)

    without = loop._build_system_prompt(pid, rag_hits=[], pad=None)
    with_pad = loop._build_system_prompt(pid, rag_hits=[], pad=pad)

    for block in _BLOCKS:
        assert block in without, f"{block} missing even without PAD"
        assert block in with_pad, f"PAD dropped the {block} block"
    # the memory itself must survive
    assert "guitar" in with_pad, "PAD dropped the person's remembered topics"
    # PAD only ever ADDS
    assert len(with_pad) > len(without)
    print("1. PAD adds to the prompt; every block and the KG memory survive ✓")


def test_manner_line_present_once_and_last():
    """The manner line must sit after the 'don't offer emotional support' rule —
    the last instruction in a block is the one that sticks."""
    loop = _loop()
    pid = "p_close"
    tier = _seed_person(loop.store, pid, "chatbox", "close")
    pad = loop._adapter().process_turn(0.0, 0.0, tier)
    prompt = loop._build_system_prompt(pid, rag_hits=[], pad=pad)

    assert prompt.count("Your manner right now is") == 1
    assert prompt.index("Your manner right now is") > prompt.index("emotional support")
    assert "colours your WORDING only" in prompt
    print("2. the manner line appears once, after the emotional-support rule ✓")


def test_no_metrics_leak_into_the_prompt():
    """rapport/trust/interaction counts must never be shown to the model — it
    will narrate them back. The tier is communicated as one plain sentence."""
    loop = _loop()
    pid = "p_close"
    tier = _seed_person(loop.store, pid, "chatbox", "close")
    pad = loop._adapter().process_turn(0.0, 0.0, tier)
    prompt = loop._build_system_prompt(pid, rag_hits=[], pad=pad)
    for leak in ("rapport=", "trust=", "interactions=", "Relationship metrics"):
        assert leak not in prompt, f"{leak!r} leaked into the prompt"
    assert "You know this person well" in prompt
    print("3. no rapport/trust numbers in the prompt; tier is one sentence ✓")


def test_first_time_wording_follows_the_count_not_the_tier():
    """A remembered person can still derive as visitor/unknown, so 'first time'
    has to key off interaction_count or it contradicts the memory below it."""
    from modules.affect_bridge import prompt as pad_prompt
    tier_note = pad_prompt.tier_note
    assert "first time" in tier_note("unknown", 0)
    assert "first time" not in tier_note("unknown", 7)
    assert "first time" not in tier_note("visitor", 3)
    print("4. 'first time' follows interaction_count, not the tier ✓")


def test_tier_changes_the_prompt():
    """Different tiers must produce different prompts, or the experiment has
    nothing to measure."""
    loop = _loop("ellebot")
    prompts = {}
    for tier in _TIERS:
        pid = f"p_{tier}"
        derived = _seed_person(loop.store, pid, "ellebot", tier)
        pad = loop._adapter().process_turn(0.0, 0.0, derived)
        prompts[tier] = loop._build_system_prompt(pid, rag_hits=[], pad=pad)
    bodies = {t: p.split("WHO YOU'RE TALKING TO")[0] for t, p in prompts.items()}
    assert len(set(bodies.values())) > 1, "tier never changed the instruction block"
    print(f"4b. tier changes the prompt ({len(set(bodies.values()))} distinct "
          "instruction blocks across the ladder) ✓")


def test_every_tier_gets_its_own_word():
    """The third descriptor word is the relationship's only mark on the prompt,
    so a robot whose word never changes says nothing about its relationship.

    This used to fail badly: descriptors were read from the SHOWN coordinate and
    CHATBOX's show=0.30 collapsed twelve of sixteen emotion x tier cells onto
    'reserved'. Words now read FELT and the D bands resolve both robots' ranges.
    """
    for robot, key in (("chatbox", "CHATBOX"), ("ellebot", "ELLEBOT")):
        base = affect.to_pad(affect.ROBOTS[key]["ocean"])
        words = [affect.descriptors(
            affect.feel_with_relationship(base, 0.0, 0.0, t))[2]
            for t in affect.TIERS]
        assert len(set(words)) == len(affect.TIERS), f"{robot}: {words}"
    print("5. every tier yields a distinct dominance word on both robots ✓")


def test_grid_runs_headless():
    """The whole experiment must run with no LLM, so it is CI-safe."""
    import argparse
    from tools.pad_prompt_grid import run_grid
    args = argparse.Namespace(
        robot="chatbox", tiers="unknown,close", emotions="happy,sad",
        message="hi", model="none", temperature=0.0, repeats=1,
        control=True, no_llm=True, no_directive=False)
    rows = run_grid(args)
    assert len(rows) == 2 * 2 * 2, len(rows)          # tiers x emotions x pad on/off
    assert all(r["prompt"] for r in rows)
    on = [r for r in rows if r["pad_on"]]
    assert all(r["words"] for r in on)
    for r in rows:
        if r["tier"] in ("unknown", "close"):
            assert r["derived_tier"] == r["tier"], f"{r['tier']} -> {r['derived_tier']}"
    print(f"6. the grid runs headless ({len(rows)} cells, tiers derived correctly) ✓")


# ── the servo STYLE wire ─────────────────────────────────────────────────────

def _style_probe():
    """A loop pointed at a real local socket, plus the lines it receives."""
    import socket
    import threading
    got = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(8)

    def serve():
        while True:
            try:
                c, _ = srv.accept()
                with c:
                    got.append(c.recv(256).decode().strip())
            except OSError:
                break
    threading.Thread(target=serve, daemon=True).start()

    tmp = tempfile.mkdtemp(prefix="style_wire_")
    loop = WebcamKGLoop(
        robot_id="chatbox", show_window=False, seed=True, llm_client=None,
        kg_path=os.path.join(tmp, "kg.json"), faces_path=os.path.join(tmp, "f.npz"),
        sessions_db=os.path.join(tmp, "s.db"), embed_fn=None,
        pad_enabled=True, emotion_enabled=True,
        esp32_host="127.0.0.1", esp32_port=port)
    return loop, got, srv


def test_style_is_sent_only_when_the_mood_moves():
    """Style is STICKY in the firmware, so it only needs resending on change —
    and resending every tick would flood a link with a 0.5 s timeout."""
    loop, got, srv = _style_probe()
    try:
        ad = loop._adapter()
        style = lambda t: ad.process_turn(0.0, 0.0, t)["style"]

        assert loop._maybe_send_style(style("known")) is True
        loop._last_style_t = 0.0                       # bypass the rate limit
        assert loop._maybe_send_style(style("known")) is False, "deadband failed"
        loop._last_style_t = 0.0
        assert loop._maybe_send_style(style("close")) is True, "tier jump not sent"
        assert loop._maybe_send_style(style("unknown")) is False, "rate limit failed"
        assert loop._maybe_send_style(force=True) is True, "force must always send"

        time.sleep(0.2)
        assert all(l.startswith("STYLE ") for l in got), got
        assert len(got) == 3, got
    finally:
        srv.close()
    print("7. STYLE is sent on change, suppressed by deadband + rate limit, "
          "always sent when forced ✓")


def test_style_precedes_the_gesture_tag():
    """Ordering is load-bearing: the firmware styles a gesture with whatever it
    holds AT THE MOMENT THE TAG ARRIVES, so the STYLE line must land first."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "modules", "face_webcam", "webcam_loop.py")).read()
    i = src.index("self._maybe_send_style(force=True)")
    j = src.index("_send_esp32(expr,", i)
    assert i < j, "style must be forced before the tag is sent"
    print("8. the forced STYLE precedes the gesture tag in _apply_chat_result ✓")


def test_style_wire_off_without_pad():
    """Styling needs PAD; without it the robot must behave exactly as before."""
    tmp = tempfile.mkdtemp(prefix="style_off_")
    loop = WebcamKGLoop(
        robot_id="chatbox", show_window=False, seed=True, llm_client=None,
        kg_path=os.path.join(tmp, "kg.json"), faces_path=os.path.join(tmp, "f.npz"),
        sessions_db=os.path.join(tmp, "s.db"), embed_fn=None,
        pad_enabled=False, style_wire=True, esp32_host="127.0.0.1")
    assert loop._style_wire is False
    assert loop._maybe_send_style({"amplitude": 1.0}) is False
    print("9. no STYLE line is emitted when PAD is disabled ✓")


if __name__ == "__main__":
    test_pad_does_not_displace_the_memory_system()
    test_manner_line_present_once_and_last()
    test_no_metrics_leak_into_the_prompt()
    test_first_time_wording_follows_the_count_not_the_tier()
    test_tier_changes_the_prompt()
    test_every_tier_gets_its_own_word()
    test_grid_runs_headless()
    test_style_is_sent_only_when_the_mood_moves()
    test_style_precedes_the_gesture_tag()
    test_style_wire_off_without_pad()
    print("\nPAD reaches the prompt and the servos; memory is intact.")
