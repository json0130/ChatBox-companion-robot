import socket
import struct
import subprocess
import time
import threading
import logging
from typing import Optional, Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)


class ArduinoOutputModule(OutputModule):
    def __init__(self, name: str = "arduino_output", config: Dict[str, Any] = None):
        super().__init__(name, config)

        self.host = self.config.get('host', 'chatbox.local')
        self.port = self.config.get('port', 8888)
        self._reconnect_delay = self.config.get('reconnect_delay', 1.0)
        self._max_delay = self.config.get('max_reconnect_delay', 30.0)

        self._sock: Optional[socket.socket] = None
        self.connected = False
        self._running = False
        self.last_esp32_message = ""
        self._resolved_ip: Optional[str] = None  # cached from avahi-resolve
        self._bind_address: Optional[str] = self.config.get('bind_address')  # local IP of ESP32-side interface

        # Callbacks assigned by robot.py
        self.on_connected = None
        self.on_disconnected = None
        self.on_connection_error = None

    def initialize(self) -> bool:
        # Always register; TCP connection established asynchronously in start()
        return True

    def _resolve_host(self) -> str:
        """
        Return a connectable address for self.host.
        For .local mDNS names: try raw mDNS multicast first (fastest in Docker),
        then avahi-resolve, then fall back to system getaddrinfo.
        For regular hostnames: use getaddrinfo directly.
        Caches the resolved IP so resolution only runs once per connection cycle.
        """
        if self._resolved_ip:
            return self._resolved_ip

        if self.host.endswith('.local'):
            # Raw mDNS multicast — fastest path inside Docker (no avahi-daemon needed)
            ip = self._mdns_resolve(self.host)
            if ip:
                self._resolved_ip = ip
                return ip

            # Avahi fallback (needs avahi-daemon + D-Bus)
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
            # Regular hostname — system resolver is appropriate
            try:
                socket.getaddrinfo(self.host, self.port)
                return self.host
            except socket.gaierror:
                pass

        return self.host  # last resort

    def _mdns_resolve(self, hostname: str, timeout: float = 3.0) -> Optional[str]:
        """
        Send a raw mDNS A-record query to 224.0.0.251:5353 and return the first
        IPv4 address that responds.  Works inside Docker --network host without
        avahi-daemon (libnss-mdns >= 0.13 requires avahi; this does not).
        """
        # Build DNS query packet
        labels = hostname.rstrip('.').encode('ascii').split(b'.')
        qname = b''.join(bytes([len(l)]) + l for l in labels) + b'\x00'
        # Header: ID=0, standard query, 1 question, 0 answers/ns/ar
        packet = struct.pack('!HHHHHH', 0, 0, 1, 0, 0, 0)
        packet += qname + struct.pack('!HH', 1, 1)  # QTYPE=A, QCLASS=IN

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
                if not (flags & 0x8000):   # not a response
                    continue
                qdcount = struct.unpack('!H', data[4:6])[0]
                ancount = struct.unpack('!H', data[6:8])[0]
                if ancount == 0:
                    continue
                # Skip question section
                off = 12
                for _ in range(qdcount):
                    while off < len(data):
                        n = data[off]
                        if n == 0:      off += 1; break
                        if n >= 0xC0:   off += 2; break
                        off += 1 + n
                    off += 4  # QTYPE + QCLASS
                # Parse answer records looking for an A record
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
                    if rtype == 1 and rdlen == 4:  # A record
                        ip = '.'.join(str(b) for b in data[off:off + 4])
                        logger.info(f"[Arduino] mDNS: {hostname} → {ip}")
                        return ip
                    off += rdlen
        except Exception as e:
            logger.debug(f"[Arduino] raw mDNS query error: {e}")
        finally:
            sock.close()
        return None

    def _connect(self) -> bool:
        host = self._resolve_host()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
            logger.info(f"[Arduino] Connected to {host}:{self.port}")
            if self.on_connected:
                self.on_connected()
            return True
        except socket.gaierror:
            self._resolved_ip = None  # clear cache — may have changed
            logger.error(f"[Arduino] Cannot resolve '{self.host}'")
            if self.on_connection_error:
                self.on_connection_error(f"Cannot resolve {self.host}")
            return False
        except Exception as e:
            logger.error(f"[Arduino] Connection failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    def start(self) -> bool:
        self._running = True
        threading.Thread(target=self._maintain_connection, daemon=True).start()
        return True

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self.connected = False
        logger.info("[Arduino] TCP connection closed.")
        if self.on_disconnected:
            self.on_disconnected()

    def process_output(self, data: Any) -> bool:
        return True

    def is_connected(self) -> bool:
        return self.connected

    def send_command(self, command: str) -> bool:
        if not self.connected or not self._sock:
            logger.warning(f"[Arduino] Cannot send '{command}' — not connected")
            return False
        try:
            self._sock.sendall(f"{command.strip()}\n".encode('utf-8'))
            logger.info(f"[Arduino] Sent: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"[Arduino] Send failed: {e}")
            self.connected = False
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
                    self._resolved_ip = None  # re-resolve on next attempt
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    if self.on_disconnected:
                        self.on_disconnected()
