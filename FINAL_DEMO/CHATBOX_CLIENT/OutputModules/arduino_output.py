"""
arduino_output.py — wired USB serial link to the ESP32.

The ESP32 is plugged straight into the Jetson by USB cable; there is no WiFi
between them. Gesture commands ("wave", "default", ...) go out one per line,
and the ESP32 answers with OK:/ERR:/DONE: lines.
"""

import glob
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
        if not self.connected or not self._ser:
            logger.warning(f"[Arduino] Cannot send '{command}' — not connected")
            return False
        if self.write_line(command):
            logger.info(f"[Arduino] Sent: {command.strip()}")
            return True
        logger.error(f"[Arduino] Send failed: {command.strip()}")
        return False

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
                    if line.startswith(("OK:", "ERR:", "DONE:")):
                        logger.info(f"[Arduino] ESP32: {line}")
                    else:
                        logger.debug(f"[Arduino] ESP32: {line}")
            except Exception as e:
                if self._running:
                    self._drop_connection(str(e))
