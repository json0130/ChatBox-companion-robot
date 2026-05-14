import socket
import time

ESP32_HOST = 'chatbox.local'
ESP32_PORT = 8888

VALID_EXPRESSIONS = [
    "greeting", "wave", "point", "confused", "shrug",
    "angry", "sad", "sleep", "default", "pose"
]


def connect_to_esp32() -> socket.socket | None:
    try:
        sock = socket.create_connection((ESP32_HOST, ESP32_PORT), timeout=5.0)
        sock.settimeout(1.0)
        print(f"✅ Connected to ESP32 at {ESP32_HOST}:{ESP32_PORT}")

        # Drain any startup banner the ESP32 sends on first connect
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                chunk = sock.recv(256)
                if chunk:
                    print(f"ESP32: {chunk.decode('utf-8', errors='ignore').strip()}")
            except socket.timeout:
                break

        return sock
    except Exception as e:
        print(f"❌ Error connecting to ESP32: {e}")
        return None


def send_command(sock: socket.socket, command: str):
    try:
        sock.sendall(f"{command}\n".encode())
        print(f"📤 Sent: {command}")

        # Read until we get DONE: or ERR: (up to 8s — longest expression is ~4.5s)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                line = sock.recv(256).decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"📥 ESP32: {line}")
                    if line.startswith("DONE:") or line.startswith("ERR:"):
                        break
            except socket.timeout:
                pass

    except Exception as e:
        print(f"❌ Error sending command: {e}")


def interactive_mode(sock: socket.socket):
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


def automated_test(sock: socket.socket):
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
        print("🔌 TCP connection closed.")


if __name__ == "__main__":
    main()
