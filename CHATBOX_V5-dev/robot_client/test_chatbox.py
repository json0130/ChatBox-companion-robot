"""test_chatbox.py — the ChatBox adapter: tag -> ESP32 command, piper sentence played, barge-in stops it."""

import json
import subprocess
import threading

from run import load


def _robot(tmp_path, arduino=False):
    cfg = json.load(open("configs/chatbox.json"))
    cfg["modules"] = ["gpt"]                       # no mic in the test
    cfg["features"] = {"arduino_integration": arduino}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    return load("chatbox")(str(p))


def test_chatbox_builds_and_its_allowed_tags_all_map_to_a_command(tmp_path):
    from robots.chatbox import EMOTION_MAP
    robot = _robot(tmp_path)
    assert {"console_output", "speech"} <= set(robot.output_modules)
    for tag in robot.config["allowed_tags"]:
        assert tag.strip("[]") in EMOTION_MAP, tag


def test_a_tag_becomes_the_esp32_command_and_unknown_tags_go_default(tmp_path):
    robot = _robot(tmp_path)
    sent = []
    robot.arduino = type("A", (), {"is_connected": lambda s: True, "send_command": lambda s, c: sent.append(c)})()
    for tag in ("[HAPPY]", "hands_clap", "WAVE"):
        robot.on_emotion_detected(tag)
    assert sent == ["greeting", "hands_clap", "default"]


def test_say_plays_the_piper_wav_and_barge_in_terminates_the_player(monkeypatch):
    from robots.chatbox import Speech
    calls, procs = [], []

    class Player:
        def __init__(self, cmd, **kw):
            procs.append(self); self.cmd, self.killed = cmd, threading.Event()
        def poll(self): return 0 if self.killed.is_set() else None
        def terminate(self): self.killed.set()

    def run(cmd, **kw):
        calls.append(cmd[0])
        open(cmd[cmd.index("--output_file") + 1], "wb").write(b"RIFFxxxx")   # piper wrote something
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", Player)
    tts = Speech("speech", {"audio_cmd": ["aplay", "-D", "default"]})
    t = threading.Thread(target=tts.say, args=("Hello there.",)); t.start()
    for _ in range(100):
        if procs: break
        threading.Event().wait(0.02)
    assert calls == ["piper"] and procs[0].cmd[:3] == ["aplay", "-D", "default"]
    tts.interrupt()                                  # barge-in
    t.join(2)
    assert procs[0].killed.is_set() and not t.is_alive()
