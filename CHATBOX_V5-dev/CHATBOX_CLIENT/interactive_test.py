#!/usr/bin/env python3
"""
interactive_test.py — Manually test ChatBox actions over serial or TCP.

Usage:
    python3 interactive_test.py              # auto-detect serial, fall back to TCP
    python3 interactive_test.py --serial     # force serial
    python3 interactive_test.py --tcp        # force TCP
    python3 interactive_test.py --port /dev/ttyUSB1   # specific serial port
"""
import glob
import socket
import sys
import time
from typing import Optional, Tuple

# ── All expressions the firmware accepts ─────────────────────────────────────
VALID_EXPRESSIONS = [
    "greeting",       "wave",           "point",          "confused",
    "shrug",          "angry",          "sad",            "sleep",
    "default",        "pose",           "idle",
    "hands_clap",     "hands_wave_both","hands_only_wave", "hands_only_tap",
    "head_nod",       "head_shake",
    "ears_wiggle",    "ears_perk",
    "dance_sway",     "dance_arms",     "dance_groove",
]

ESP32_HOST = 'chatbox.local'
ESP32_PORT = 8888

# ── Connection helpers ────────────────────────────────────────────────────────

def _find_serial_port(hint: str = "") -> Optional[str]:
    try:
        import serial as _ser
    except ImportError:
        return None
    candidates = []
    if hint:
        candidates.append(hint)
    candidates += sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    for p in candidates:
        try:
            s = _ser.Serial(p, timeout=0.5)
            s.close()
            return p
        except Exception:
            continue
    return None


def connect_serial(port_hint: str = "") -> Tuple:
    """Returns (conn, 'serial') or (None, None)."""
    try:
        import serial as _ser
    except ImportError:
        print("❌ pyserial not installed — run: pip install pyserial")
        return None, None

    port = _find_serial_port(port_hint)
    if not port:
        print("❌ No serial port found (tried /dev/ttyUSB*, /dev/ttyACM*)")
        return None, None

    try:
        baud = 115200
        conn = _ser.Serial(port, baudrate=baud, timeout=1.0)
        time.sleep(1.5)  # let ESP32 reset after DTR toggle
        conn.reset_input_buffer()
        print(f"✅ Connected via serial  {port} @ {baud} baud")
        return conn, "serial"
    except Exception as e:
        print(f"❌ Serial open failed: {e}")
        return None, None


def connect_tcp() -> Tuple:
    """Returns (conn, 'tcp') or (None, None)."""
    try:
        sock = socket.create_connection((ESP32_HOST, ESP32_PORT), timeout=5.0)
        sock.settimeout(1.0)
        print(f"✅ Connected via TCP  {ESP32_HOST}:{ESP32_PORT}")
        # drain startup banner
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                chunk = sock.recv(256)
                if chunk:
                    print(f"   ESP32: {chunk.decode('utf-8', errors='ignore').strip()}")
            except socket.timeout:
                break
        return sock, "tcp"
    except Exception as e:
        print(f"❌ TCP connection failed: {e}")
        return None, None


def close_conn(conn, mode: str):
    try:
        conn.close()
    except Exception:
        pass
    print(f"🔌 {mode.upper()} connection closed.")

# ── Send / receive ────────────────────────────────────────────────────────────

def send_command(conn, mode: str, command: str):
    try:
        payload = f"{command.strip()}\n".encode('utf-8')
        if mode == "serial":
            conn.write(payload)
            conn.flush()
        else:
            conn.sendall(payload)
        print(f"📤 Sent: {command}")
    except Exception as e:
        print(f"❌ Send error: {e}")
        return

    # Read response until DONE: / ERR: or timeout (longest animation ~5 s)
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            if mode == "serial":
                line = conn.readline().decode('utf-8', errors='ignore').strip()
            else:
                line = conn.recv(256).decode('utf-8', errors='ignore').strip()
            if line:
                print(f"📥 ESP32: {line}")
                if line.startswith("DONE:") or line.startswith("ERR:"):
                    break
        except (socket.timeout, Exception):
            pass

# ── Menus ─────────────────────────────────────────────────────────────────────

def _print_menu():
    print("\n┌─ Available actions " + "─" * 38 + "┐")
    cols = 4
    for i, expr in enumerate(VALID_EXPRESSIONS, 1):
        end = "\n" if i % cols == 0 else ""
        print(f"  {i:>2}. {expr:<18}", end=end)
    if len(VALID_EXPRESSIONS) % cols != 0:
        print()
    print("└" + "─" * 58 + "┘")
    print("  Type a number, an expression name, 'menu' to reshow, or 'exit'")


def interactive_mode(conn, mode: str):
    print(f"\n🎭  ChatBox Interactive Test  [{mode.upper()}]")
    _print_menu()
    while True:
        try:
            raw = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break

        if not raw:
            continue
        if raw.lower() in ('exit', 'quit', 'q'):
            print("👋 Exiting...")
            break
        if raw.lower() in ('menu', 'help', '?', 'h'):
            _print_menu()
            continue

        # Numeric selection
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(VALID_EXPRESSIONS):
                send_command(conn, mode, VALID_EXPRESSIONS[idx])
                _print_menu()
            else:
                print(f"❌ Number out of range (1–{len(VALID_EXPRESSIONS)})")
            continue

        # Name match (partial ok)
        cmd = raw.lower()
        matches = [e for e in VALID_EXPRESSIONS if e == cmd]
        if not matches:
            matches = [e for e in VALID_EXPRESSIONS if cmd in e]
        if len(matches) == 1:
            send_command(conn, mode, matches[0])
            _print_menu()
        elif len(matches) > 1:
            print(f"   Ambiguous — did you mean: {', '.join(matches)}?")
        else:
            print(f"❌ Unknown expression '{cmd}'. Type 'menu' to see all options.")


def automated_test(conn, mode: str):
    print(f"\n🤖  Automated Test Sequence  [{mode.upper()}]")
    seq = [
        "default", "greeting", "wave", "pose",
        "sad", "confused", "point",
        "hands_clap", "ears_wiggle", "ears_perk",
        "head_nod", "head_shake",
        "dance_sway", "default",
    ]
    for i, cmd in enumerate(seq, 1):
        print(f"\n--- {i}/{len(seq)}: {cmd} ---")
        send_command(conn, mode, cmd)
        time.sleep(0.3)
    print("\n✅ Automated test complete.")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    force_serial = '--serial' in args
    force_tcp    = '--tcp' in args
    port_hint    = ""
    if '--port' in args:
        idx = args.index('--port')
        if idx + 1 < len(args):
            port_hint = args[idx + 1]

    print("🤖  ChatBox Action Tester")
    print("=" * 40)

    conn, mode = None, None

    if force_tcp:
        conn, mode = connect_tcp()
    elif force_serial:
        conn, mode = connect_serial(port_hint)
    else:
        # Auto: try serial first, fall back to TCP
        conn, mode = connect_serial(port_hint)
        if not conn:
            print("   ↳ Falling back to TCP…")
            conn, mode = connect_tcp()

    if not conn:
        print("\n❌ Could not connect via serial or TCP. Exiting.")
        sys.exit(1)

    try:
        while True:
            print("\n  1. Interactive mode")
            print("  2. Automated test sequence")
            print("  3. Exit")
            choice = input("Choice: ").strip()
            if choice == '1':
                interactive_mode(conn, mode)
            elif choice == '2':
                automated_test(conn, mode)
            elif choice in ('3', 'exit', 'q'):
                print("👋 Goodbye!")
                break
            else:
                print("Enter 1, 2, or 3.")
    finally:
        close_conn(conn, mode)


if __name__ == "__main__":
    main()
