"""
robots/chatbox/arduino_output.py (copied from FINAL_DEMO/CHATBOX_CLIENT/OutputModules, the 11-11 queue/reopen version)

arduino_output.py — wired USB serial link to the ESP32.

The ESP32 is plugged straight into the Jetson by USB cable; there is no WiFi
between them. Gesture commands ("point", "default", ...) go out one per line,
and the ESP32 answers with OK:/ERR:/DONE: lines.
"""

import glob
import queue
import time
import threading
import logging
from typing import Optional, Dict, Any
from client import OutputModule

try:
    import serial  # pyserial
except ImportError:
    serial = None

logger = logging.getLogger(__name__)


class ArduinoOutputModule(OutputModule):
    def __init__(self, name: str = "arduino_output", config: Dict[str, Any] = None):
        super().__init__(name, config)

        self.serial_port: Optional[str] = self.config.get('serial_port')  # None -> auto-detect
        self.baud = self.config.get('baud', 115200)
        self._reconnect_delay = self.config.get('reconnect_delay', 1.0)
        self._max_delay = self.config.get('max_reconnect_delay', 30.0)

        self._ser = None
        self._port_in_use: Optional[str] = None
        self._write_lock = threading.Lock()
        self.connected = False
        self._running = False
        self.last_esp32_message = ""

        # The ESP32 runs one gesture at a time and keeps only ONE queued command, so a command sent while it
        # is mid-gesture is dropped or answered late. Commands go through a queue and a sender thread (the
        # caller, e.g. a demo step that is about to speak, never waits): each is held until the previous one
        # reports DONE:/ERR:, the board reboots, or `busy_timeout` passes. A board that does not even answer
        # OK: within `ack_timeout` is not receiving, so the queue does not wait on it.
        self._cmds: "queue.Queue[str]" = queue.Queue()
        self._idle = threading.Event()      # last command finished (DONE/ERR) or none in flight
        self._idle.set()
        self._silent = 0                    # commands in a row the ESP32 never answered
        self._acked = threading.Event()     # OK:/ERR: seen for the command in flight
        self._busy_timeout = self.config.get('busy_timeout', 30.0)
        self._ack_timeout = self.config.get('ack_timeout', 5.0)

        # Callbacks assigned by robot.py
        self.on_connected = None
        self.on_disconnected = None
        self.on_connection_error = None

    def initialize(self) -> bool:
        if serial is None:
            logger.error("[Arduino] pyserial missing — pip install pyserial")
            return False
        # Always register; the port is opened asynchronously in start()
        return True

    def _find_port(self) -> Optional[str]:
        if self.serial_port:
            return self.serial_port
        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        return ports[0] if ports else None

    def _connect(self) -> bool:
        port = self._find_port()
        if not port:
            logger.error("[Arduino] No ESP32 serial port found (/dev/ttyUSB*, /dev/ttyACM*)")
            if self.on_connection_error:
                self.on_connection_error("no serial port")
            return False
        try:
            s = serial.Serial(port, self.baud, timeout=1.0, write_timeout=1.0)
            time.sleep(2)   # ESP32 resets when the port opens
            s.reset_input_buffer()  # drop the boot banner
            # The reset leaves junk bytes in the ESP32's line buffer, which would
            # be glued onto the first real command. A bare newline flushes them;
            # the firmware ignores empty lines.
            s.write(b"\n")
            self._ser = s
            self._port_in_use = port
            self.connected = True
            logger.info(f"[Arduino] Connected on {port} @ {self.baud}")
            if self.on_connected:
                self.on_connected()
            return True
        except Exception as e:
            logger.error(f"[Arduino] Cannot open {port}: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    def _drop_connection(self, reason: str):
        if not self.connected:
            return
        logger.error(f"[Arduino] Lost connection: {reason}")
        self.connected = False
        self._idle.set()
        try:
            self._ser.close()
        except Exception:
            pass
        self._ser = None
        if self.on_disconnected:
            self.on_disconnected()

    def start(self) -> bool:
        self._running = True
        threading.Thread(target=self._maintain_connection, daemon=True).start()
        threading.Thread(target=self._send_worker, daemon=True, name="arduino-send").start()
        return True

    def stop(self):
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.connected = False
        logger.info("[Arduino] Serial connection closed.")
        if self.on_disconnected:
            self.on_disconnected()

    def process_output(self, data: Any) -> bool:
        return True

    def is_connected(self) -> bool:
        return self.connected

    def port_name(self) -> str:
        return self._port_in_use or self.serial_port or "auto-detect"

    def write_line(self, text: str) -> bool:
        """Write one line to the ESP32. Thread-safe; shared with face tracking."""
        if not self.connected or not self._ser:
            return False
        try:
            with self._write_lock:
                self._ser.write(f"{text.strip()}\n".encode('utf-8'))
            return True
        except Exception as e:
            self._drop_connection(f"write failed: {e}")
            return False

    def send_command(self, command: str) -> bool:
        """Queue one command for the ESP32; returns at once. Sent in order by _send_worker."""
        if not self.connected or not self._ser:
            logger.warning(f"[Arduino] Cannot send '{command}' — not connected")
            return False
        self._cmds.put(command)
        return True

    def _send_worker(self):
        while self._running:
            try:
                command = self._cmds.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self.connected:
                continue
            if not self._idle.wait(self._busy_timeout):
                logger.warning(f"[Arduino] No DONE from the ESP32 after {self._busy_timeout:.0f}s — sending '{command}' anyway")
            self._idle.clear()
            self._acked.clear()
            if not self.write_line(command):
                self._idle.set()
                logger.error(f"[Arduino] Send failed: {command.strip()}")
                continue
            logger.info(f"[Arduino] Sent: {command.strip()}")
            if self._acked.wait(self._ack_timeout):
                self._silent = 0
                continue
            self._silent += 1
            logger.warning(f"[Arduino] ESP32 did not answer '{command}' within {self._ack_timeout:.0f}s "
                           f"(last line from it: {self.last_esp32_message!r}) — it is not receiving "
                           "(check port / power / reset loop)")
            self._idle.set()
            if self._silent >= 2:   # stale fd after the CP2102 re-enumerated, or wrong port: reopen and rescan
                self._silent = 0
                self._drop_connection("no reply to 2 commands in a row")

    def _maintain_connection(self):
        delay = self._reconnect_delay
        first_attempt = True
        while self._running:
            if not self.connected:
                if first_attempt:
                    first_attempt = False
                else:
                    logger.info(f"[Arduino] Reconnecting in {delay:.0f}s...")
                    time.sleep(delay)
                if self._connect():
                    delay = self._reconnect_delay
                else:
                    delay = min(delay * 2, self._max_delay)
                continue

            try:
                line = self._ser.readline()   # returns b'' on the 1s timeout
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    self.last_esp32_message = line
                    # OK:/ERR:/DONE: are command replies; everything else is debug
                    if line.startswith(("OK:", "ERR:")):
                        self._acked.set()
                    if line.startswith(("DONE:", "ERR:")) or "ChatBox System Starting" in line:
                        self._acked.set()
                        self._idle.set()          # finished, rejected, or rebooted: free for the next command
                    if line.startswith(("OK:", "ERR:", "DONE:")):
                        logger.info(f"[Arduino] ESP32: {line}")
                    elif "ChatBox System Starting" in line:
                        logger.warning("[Arduino] ESP32 (re)booted — if this happens during a gesture it is a power problem")
                    else:
                        logger.debug(f"[Arduino] ESP32: {line}")
            except Exception as e:
                if self._running:
                    self._drop_connection(str(e))
