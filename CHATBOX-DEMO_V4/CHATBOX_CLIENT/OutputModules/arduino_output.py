import glob
import socket
import struct
import subprocess
import time
import threading
import logging
from typing import Optional, Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)

try:
    import serial as _serial_mod
    _SERIAL_AVAILABLE = True
except ImportError:
    _serial_mod = None
    _SERIAL_AVAILABLE = False


class ArduinoOutputModule(OutputModule):
    def __init__(self, name: str = "arduino_output", config: Dict[str, Any] = None):
        super().__init__(name, config)

        self.host = self.config.get('host', 'chatbox.local')
        self.port = self.config.get('port', 8888)
        self._reconnect_delay = self.config.get('reconnect_delay', 1.0)
        self._max_delay = self.config.get('max_reconnect_delay', 30.0)
        self._serial_cfg: Dict[str, Any] = self.config.get('serial', {})

        self._sock: Optional[socket.socket] = None
        self._serial: Optional[Any] = None  # serial.Serial instance
        self._mode: str = "tcp"             # "tcp" or "serial", set in start()
        self.connected = False
        self._running = False
        self.last_esp32_message = ""
        self._resolved_ip: Optional[str] = None
        self._bind_address: Optional[str] = self.config.get('bind_address')

        self.on_connected = None
        self.on_disconnected = None
        self.on_connection_error = None

    def initialize(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Serial detection
    # ------------------------------------------------------------------

    def _detect_serial_port(self) -> Optional[str]:
        if not _SERIAL_AVAILABLE:
            return None
        candidates = []
        hint = self._serial_cfg.get("port", "")
        if hint:
            candidates.append(hint)
        candidates += sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        for port in candidates:
            try:
                s = _serial_mod.Serial(port, timeout=0.5)
                s.close()
                return port
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # TCP helpers (unchanged logic, plus TCP_NODELAY)
    # ------------------------------------------------------------------

    def _resolve_host(self) -> str:
        if self._resolved_ip:
            return self._resolved_ip

        if self.host.endswith('.local'):
            ip = self._mdns_resolve(self.host)
            if ip:
                self._resolved_ip = ip
                return ip

            try:
                result = subprocess.run(
                    ['avahi-resolve-host-name', self.host],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        logger.info(f"[Arduino] Resolved {self.host} → {ip} via avahi")
                        self._resolved_ip = ip
                        return ip
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"[Arduino] avahi-resolve failed: {e}")
        else:
            try:
                socket.getaddrinfo(self.host, self.port)
                return self.host
            except socket.gaierror:
                pass

        return self.host

    def _mdns_resolve(self, hostname: str, timeout: float = 3.0) -> Optional[str]:
        labels = hostname.rstrip('.').encode('ascii').split(b'.')
        qname = b''.join(bytes([len(l)]) + l for l in labels) + b'\x00'
        packet = struct.pack('!HHHHHH', 0, 0, 1, 0, 0, 0)
        packet += qname + struct.pack('!HH', 1, 1)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        sock.settimeout(1.0)
        if self._bind_address:
            try:
                sock.bind((self._bind_address, 0))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                socket.inet_aton(self._bind_address))
            except OSError as e:
                logger.warning(f"[Arduino] Cannot bind mDNS to {self._bind_address}: {e} — try --network host")
        try:
            sock.sendto(packet, ('224.0.0.251', 5353))
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if len(data) < 12:
                    continue
                flags = struct.unpack('!H', data[2:4])[0]
                if not (flags & 0x8000):
                    continue
                qdcount = struct.unpack('!H', data[4:6])[0]
                ancount = struct.unpack('!H', data[6:8])[0]
                if ancount == 0:
                    continue
                off = 12
                for _ in range(qdcount):
                    while off < len(data):
                        n = data[off]
                        if n == 0:      off += 1; break
                        if n >= 0xC0:   off += 2; break
                        off += 1 + n
                    off += 4
                for _ in range(ancount):
                    while off < len(data):
                        n = data[off]
                        if n == 0:      off += 1; break
                        if n >= 0xC0:   off += 2; break
                        off += 1 + n
                    if off + 10 > len(data):
                        break
                    rtype, _, _, rdlen = struct.unpack('!HHIH', data[off:off + 10])
                    off += 10
                    if rtype == 1 and rdlen == 4:
                        ip = '.'.join(str(b) for b in data[off:off + 4])
                        logger.info(f"[Arduino] mDNS: {hostname} → {ip}")
                        return ip
                    off += rdlen
        except Exception as e:
            logger.debug(f"[Arduino] raw mDNS query error: {e}")
        finally:
            sock.close()
        return None

    def _tcp_connect(self) -> bool:
        host = self._resolve_host()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self._bind_address:
                try:
                    s.bind((self._bind_address, 0))
                except OSError as e:
                    logger.warning(f"[Arduino] Cannot bind TCP to {self._bind_address}: {e} — try --network host")
            s.settimeout(5.0)
            s.connect((host, self.port))
            s.settimeout(1.0)
            self._sock = s
            self.connected = True
            logger.info(f"[Arduino] TCP mode — connected to {host}:{self.port}")
            if self.on_connected:
                self.on_connected()
            return True
        except socket.gaierror:
            self._resolved_ip = None
            logger.error(f"[Arduino] Cannot resolve '{self.host}'")
            if self.on_connection_error:
                self.on_connection_error(f"Cannot resolve {self.host}")
            return False
        except Exception as e:
            logger.error(f"[Arduino] Connection failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    def _serial_connect(self) -> bool:
        baudrate = self._serial_cfg.get("baudrate", 115200)
        timeout = self._serial_cfg.get("timeout", 1.0)
        try:
            self._serial = _serial_mod.Serial(
                self._serial_port, baudrate=baudrate,
                timeout=timeout, write_timeout=1.0
            )
            self.connected = True
            logger.info(f"[Arduino] Serial mode — connected to {self._serial_port} @ {baudrate}")
            if self.on_connected:
                self.on_connected()
            return True
        except Exception as e:
            logger.error(f"[Arduino] Serial connection failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        self._running = True
        self._serial_port = self._detect_serial_port()
        if self._serial_port:
            self._mode = "serial"
            logger.info(f"[Arduino] Wired USB detected — using serial mode ({self._serial_port})")
            threading.Thread(target=self._maintain_serial, daemon=True).start()
        else:
            self._mode = "tcp"
            logger.info(f"[Arduino] No serial device found — using TCP/WiFi mode ({self.host}:{self.port})")
            threading.Thread(target=self._maintain_connection, daemon=True).start()
        return True

    def stop(self):
        self._running = False
        if self._mode == "serial":
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            logger.info("[Arduino] Serial connection closed.")
        else:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            logger.info("[Arduino] TCP connection closed.")
        self.connected = False
        if self.on_disconnected:
            self.on_disconnected()

    def process_output(self, data: Any) -> bool:
        return True

    def is_connected(self) -> bool:
        return self.connected

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send_command(self, command: str) -> bool:
        if not self.connected:
            logger.warning(f"[Arduino] Cannot send '{command}' — not connected")
            return False
        try:
            payload = f"{command.strip()}\n".encode('utf-8')
            if self._mode == "serial":
                self._serial.write(payload)
                self._serial.flush()
            else:
                self._sock.sendall(payload)
            logger.info(f"[Arduino] Sent: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"[Arduino] Send failed: {e}")
            self.connected = False
            return False

    # ------------------------------------------------------------------
    # Connection maintenance loops
    # ------------------------------------------------------------------

    def _maintain_connection(self):
        """TCP reconnect loop with exponential backoff."""
        delay = self._reconnect_delay
        first_attempt = True
        while self._running:
            if not self.connected:
                if first_attempt:
                    first_attempt = False
                else:
                    logger.info(f"[Arduino] Reconnecting in {delay:.0f}s...")
                    time.sleep(delay)
                if self._tcp_connect():
                    delay = self._reconnect_delay
                else:
                    delay = min(delay * 2, self._max_delay)
                continue

            try:
                data = self._sock.recv(256)
                if not data:
                    raise ConnectionResetError("EOF")
                for line in data.decode('utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line:
                        self.last_esp32_message = line
                        logger.debug(f"[Arduino] ESP32: {line}")
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    logger.error(f"[Arduino] Lost connection: {e}")
                    self.connected = False
                    self._resolved_ip = None
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    if self.on_disconnected:
                        self.on_disconnected()

    def _maintain_serial(self):
        """Serial reconnect loop with exponential backoff."""
        delay = self._reconnect_delay
        first_attempt = True
        while self._running:
            if not self.connected:
                if first_attempt:
                    first_attempt = False
                else:
                    logger.info(f"[Arduino] Serial reconnecting in {delay:.0f}s...")
                    time.sleep(delay)
                if self._serial_connect():
                    delay = self._reconnect_delay
                else:
                    delay = min(delay * 2, self._max_delay)
                continue

            try:
                line = self._serial.readline()
                if line:
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        self.last_esp32_message = text
                        logger.debug(f"[Arduino] ESP32: {text}")
            except Exception as e:
                if self._running:
                    logger.error(f"[Arduino] Serial lost: {e}")
                    self.connected = False
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    self._serial = None
                    if self.on_disconnected:
                        self.on_disconnected()
