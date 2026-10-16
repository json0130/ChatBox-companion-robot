"""Protocol + cleaning checks for ai_server_llm against an in-process fake server.
Run: python3 -m pytest modules/tests_ai_server_llm.py   (no network, no model)"""
import json
import socket
import threading

from modules.ai_server_llm import AIServerLLM, clean_reply, to_prompt


def _fake_server(handler):
    """One-shot server speaking the length/echo/JSON protocol; handler(request)->reply dict."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    seen = {}

    def run():
        conn, _ = srv.accept()
        with conn:
            header = conn.recv(32)
            conn.sendall(header)                       # echo, then read the body
            body = b""
            while len(body) < int(header):
                body += conn.recv(int(header) - len(body))
            seen["req"] = json.loads(body)
            out = json.dumps(handler(seen["req"])).encode()
            conn.sendall(str(len(out)).encode())
            assert conn.recv(32) == str(len(out)).encode()   # client must echo
            conn.sendall(out)
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1], seen


def _ok(text):
    return lambda r: {**r, "command": [], "content": text}


def _client(port):
    c = AIServerLLM("127.0.0.1", port, timeout=5)
    c.available = True
    return c


def test_round_trip_flattens_history_and_cleans_reply():
    port, seen = _fake_server(_ok("[HAPPY] Hi there!\n\nUser: and then you said"))
    out = _client(port).respond("Be kind.", "hello", history=[("hey", "[HAPPY] hey!")])
    assert out == "[HAPPY] Hi there!"
    req = seen["req"]
    assert req["command"] == ["LLM"] and req["parameters"] == {"max_tokens": 140}
    assert req["content"] == ("System: Be kind.\n\nUser: hey\n\nAssistant: [HAPPY] hey!"
                              "\n\nUser: hello\n\nAssistant:")


def test_server_error_becomes_error_string():
    port, _ = _fake_server(lambda r: {**r, "command": ["ERR"], "content": "boom"})
    assert _client(port).respond("s", "u").startswith("[LLM error: server error: boom")


def test_unreachable_server_is_unavailable():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    c = AIServerLLM("127.0.0.1", port, timeout=1)
    assert c.connect() is False and c.respond("s", "u").startswith("[LLM not connected")


def test_clean_reply_drops_reasoning_block():
    assert clean_reply("We need to think.\n</think>\n\nHello!") == "Hello!"
    assert clean_reply(None) == ""
    assert to_prompt([{"role": "user", "content": "x"}]).endswith("Assistant:")
