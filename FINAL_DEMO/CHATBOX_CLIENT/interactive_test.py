import glob
import sys
import time
from typing import Optional  # for X|None on Python <3.10

import serial  # pyserial

# ESP32 is wired to the Jetson by USB. Pass the port as an argument, or leave
# it out to use the first /dev/ttyUSB* or /dev/ttyACM* found.
BAUD = 115200

VALID_EXPRESSIONS = [
    "greeting", "wave", "point", "confused", "shrug",
    "angry", "sad", "sleep", "default", "pose"
]


def connect_to_esp32() -> Optional[serial.Serial]:
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        port = ports[0] if ports else None
    if not port:
        print("❌ No ESP32 serial port found (/dev/ttyUSB*, /dev/ttyACM*)")
        return None
    try:
        sock = serial.Serial(port, BAUD, timeout=1.0)
        time.sleep(2)   # ESP32 resets when the port opens
        print(f"✅ Connected to ESP32 on {port}")
        sock.reset_input_buffer()   # drop the boot banner
        sock.write(b"\n")           # flush reset junk out of the ESP32's line buffer
        return sock
    except Exception as e:
        print(f"❌ Error connecting to ESP32: {e}")
        return None


def send_command(sock: serial.Serial, command: str):
    try:
        sock.write(f"{command}\n".encode())
        print(f"📤 Sent: {command}")

        # Read until we get DONE: or ERR: (up to 8s — longest expression is ~4.5s)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            line = sock.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"📥 ESP32: {line}")
                if line.startswith("DONE:") or line.startswith("ERR:"):
                    break

    except Exception as e:
        print(f"❌ Error sending command: {e}")


def interactive_mode(sock: serial.Serial):
    print("\n🎭 ESP32 ChatBox Interactive Mode")
    print("Valid commands:", ", ".join(VALID_EXPRESSIONS))
    print("Type 'exit' to quit, 'help' for commands list\n")

    while True:
        try:
            command = input("Enter command: ").strip().lower()
            if command == 'exit':
                print("👋 Exiting...")
                break
            elif command == 'help':
                print("Valid expressions:", ", ".join(VALID_EXPRESSIONS))
            elif command:
                send_command(sock, command)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break


def automated_test(sock: serial.Serial):
    print("\n🤖 Running Automated Test Sequence...")

    test_sequence = [
        "default", "greeting", "wave",
        "happy",    # Invalid command test
        "sad", "angry", "confused", "shrug", "point", "pose", "sleep", "default",
    ]

    for i, command in enumerate(test_sequence, 1):
        print(f"\n--- Test {i}/{len(test_sequence)}: {command} ---")
        send_command(sock, command)

    print("\n✅ Automated test sequence completed!")


def main():
    print("🤖 ESP32 ChatBox Control System")
    print("=" * 40)

    sock = connect_to_esp32()
    if not sock:
        return

    try:
        while True:
            print("\nChoose test mode:")
            print("1. Interactive Mode")
            print("2. Automated Test")
            print("3. Exit")

            choice = input("Enter choice (1-3): ").strip()
            if choice == '1':
                interactive_mode(sock)
            elif choice == '2':
                automated_test(sock)
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    finally:
        sock.close()
        print("🔌 Serial connection closed.")


if __name__ == "__main__":
    main()
