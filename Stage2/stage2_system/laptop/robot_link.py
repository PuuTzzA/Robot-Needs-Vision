"""Robot communication client and status ping receiver for Windows Mission Control."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Optional, Tuple

from shared.protocol import (
    DEFAULT_LAPTOP_PING_PORT,
    DEFAULT_VM_BEACON_PORT,
    DEFAULT_VM_CMD_PORT,
    DISCOVERY_BEACON_REQUEST,
    DISCOVERY_BEACON_RESPONSE_PREFIX,
    GotoRequest,
    RegisterRequest,
    RobotStatus,
    TeleopRequest,
)


class PingReceiverHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP POST /api/robot_status pings pushed by the VM."""

    link: RobotLink

    def do_POST(self) -> None:
        if self.path == "/api/robot_status":
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = self.rfile.read(length)
                data = json.loads(payload.decode("utf-8"))
                status = RobotStatus.from_dict(data)
                self.link._on_status_ping_received(status)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
                return
            except Exception as err:
                pass

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class RobotLink:
    """Manages connection, discovery, command dispatch, and pose tracking with VM."""

    def __init__(
        self,
        vm_ip: str = "auto",
        vm_port: int = DEFAULT_VM_CMD_PORT,
        ping_port: int = DEFAULT_LAPTOP_PING_PORT,
    ) -> None:
        self.vm_ip_arg = vm_ip
        self.vm_port = vm_port
        self.ping_port = ping_port

        self.active_vm_ip: Optional[str] = None
        self.latest_status: RobotStatus = RobotStatus()
        self._last_ping_time: float = 0.0
        self._status_callbacks: list[Callable[[RobotStatus], None]] = []
        self._lock = threading.Lock()

        # Ping Receiver Server
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._polling_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        """True if a valid ping or status response was received within the last 3 seconds."""
        with self._lock:
            return (time.time() - self._last_ping_time) < 3.0 and self.active_vm_ip is not None

    def add_status_callback(self, cb: Callable[[RobotStatus], None]) -> None:
        self._status_callbacks.append(cb)

    def start(self) -> None:
        self._running = True
        self._start_ping_receiver()

        # Connect / discover
        if self.vm_ip_arg.lower() in ("auto", "discover"):
            self.discover_and_connect()
        else:
            self.connect(self.vm_ip_arg, self.vm_port)

        # Start polling fallback thread in case UDP/HTTP push is blocked by firewall
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()

    def _start_ping_receiver(self) -> None:
        for port in [self.ping_port, 8072, 8073, 8074]:
            try:
                server = HTTPServer(("0.0.0.0", port), PingReceiverHandler)
                server.link = self  # type: ignore
                self._http_server = server
                self.ping_port = port
                self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                self._server_thread.start()
                print(f"[RobotLink] Listening for robot status pings on port {port}...")
                return
            except OSError:
                continue
        print(f"[RobotLink] [WARN] Could not bind ping receiver port {self.ping_port}. Polling fallback will be used.")

    def _on_status_ping_received(self, status: RobotStatus) -> None:
        with self._lock:
            self.latest_status = status
            self._last_ping_time = time.time()

        for cb in self._status_callbacks:
            try:
                cb(status)
            except Exception:
                pass

    def connect(self, ip: str, port: int) -> bool:
        """Test connection to VM and register laptop ping port."""
        self.active_vm_ip = ip
        self.vm_port = port

        try:
            # 1. Hello probe
            res = self._http_get("/api/hello", timeout=1.5)
            if res and res.get("status") == "ok":
                print(f"[RobotLink] Connected to Robot Controller at http://{ip}:{port} ({res.get('name')}).")

                # 2. Register laptop ping receiver
                self._http_post("/api/register", RegisterRequest(ping_port=self.ping_port).to_dict())

                # 3. Pull initial status
                stat_data = self._http_get("/api/status", timeout=1.5)
                if stat_data:
                    self._on_status_ping_received(RobotStatus.from_dict(stat_data))
                return True
        except Exception as e:
            print(f"[RobotLink] Connection to {ip}:{port} failed: {e}")

        return False

    def discover_and_connect(self) -> bool:
        """Run multi-layer auto discovery to find the VM."""
        print("[RobotLink] Searching for VM Robot Controller via auto-discovery...")

        # Strategy 1: VMware Guest IP inspect
        vm_ips = self._get_vmware_guest_ips()
        for ip in vm_ips:
            if self.connect(ip, self.vm_port):
                return True

        # Strategy 2: UDP Broadcast Beacon
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(0.3)
            sock.sendto(DISCOVERY_BEACON_REQUEST, ("255.255.255.255", DEFAULT_VM_BEACON_PORT))

            t_end = time.time() + 1.2
            while time.time() < t_end:
                try:
                    data, addr = sock.recvfrom(512)
                    msg = data.decode("utf-8", errors="ignore")
                    if msg.startswith(DISCOVERY_BEACON_RESPONSE_PREFIX):
                        offer_port = int(msg.split(":")[1])
                        sock.close()
                        if self.connect(addr[0], offer_port):
                            return True
                except (socket.timeout, BlockingIOError, OSError):
                    pass
            sock.close()
        except Exception:
            pass

        # Strategy 3: Common local / bridged IPs
        candidates = ["127.0.0.1", "192.168.31.253", "192.168.52.128"]
        for ip in candidates:
            if self.connect(ip, self.vm_port):
                return True

        print("[RobotLink] [WARN] Auto-discovery could not find VM Robot Controller. Will retry in background.")
        return False

    def _get_vmware_guest_ips(self) -> list[str]:
        ips = []
        candidates = [
            r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe",
            r"C:\Program Files\VMware\VMware Workstation\vmrun.exe",
            "vmrun",
        ]
        for exe in candidates:
            try:
                res = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=1.0)
                if res.returncode == 0:
                    lines = res.stdout.strip().splitlines()
                    for line in lines[1:]:
                        vmx = line.strip()
                        if vmx.endswith(".vmx"):
                            ip_res = subprocess.run([exe, "getGuestIPAddress", vmx], capture_output=True, text=True, timeout=1.0)
                            if ip_res.returncode == 0:
                                ip = ip_res.stdout.strip()
                                if ip and not ip.startswith("Error"):
                                    ips.append(ip)
                    if ips:
                        return ips
            except Exception:
                pass
        return ips

    def _polling_loop(self) -> None:
        """Fallback status polling at 2 Hz."""
        while self._running:
            time.sleep(0.5)
            if not self.active_vm_ip:
                continue

            # If we haven't received a pushed ping in 1.5 seconds, poll
            if (time.time() - self._last_ping_time) > 1.5:
                try:
                    data = self._http_get("/api/status", timeout=0.8)
                    if data:
                        self._on_status_ping_received(RobotStatus.from_dict(data))
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Command Dispatch Methods
    # -------------------------------------------------------------------------

    def goto_keypoint(self, keypoint_id: int) -> Tuple[bool, int, str]:
        """Dispatch a Nav2 navigation goal for the specified keypoint ID."""
        req = GotoRequest(keypoint_id=keypoint_id)
        res = self._http_post("/api/goto", req.to_dict())
        if res and res.get("accepted"):
            seq = int(res.get("goal_seq", 0))
            return True, seq, "Goal accepted by Nav2"
        msg = res.get("message", "Request failed or rejected") if res else "No response from robot"
        return False, 0, msg

    def teleop(self, key: str, duration: float = 0.6) -> bool:
        """Send a short-burst teleop command (w/a/s/d/space)."""
        req = TeleopRequest(key=key, duration=duration)
        res = self._http_post("/api/teleop", req.to_dict())
        return bool(res and res.get("ok"))

    def cancel_navigation(self) -> bool:
        """Cancel active navigation goal on robot."""
        res = self._http_post("/api/cancel", {})
        return bool(res and res.get("ok"))

    def fetch_keypoints(self) -> Optional[dict]:
        """Fetch current keypoints JSON and sha directly from VM."""
        return self._http_get("/api/keypoints")

    def fetch_map(self) -> Optional[dict]:
        """Fetch map yaml metadata and pgm base64 from VM."""
        return self._http_get("/api/map")

    def ping(self) -> Tuple[bool, float, str]:
        """Test RTT latency to robot controller."""
        if not self.active_vm_ip:
            return False, 0.0, "No active VM IP"
        t0 = time.perf_counter()
        res = self._http_get("/api/status", timeout=2.0)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if res:
            return True, rtt_ms, f"{self.active_vm_ip}:{self.vm_port}"
        return False, 0.0, "No response"

    def wait_for_goal(
        self,
        goal_seq: int,
        timeout: float = 60.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[bool, str]:
        """Block until target goal_seq finishes or times out."""
        t_end = time.time() + timeout

        while time.time() < t_end:
            if cancel_event and cancel_event.is_set():
                self.cancel_navigation()
                return False, "Canceled by user"

            with self._lock:
                st = self.latest_status
                if st.goal_seq == goal_seq:
                    if st.nav_state == "succeeded":
                        return True, "Reached destination"
                    elif st.nav_state in ("aborted", "canceled", "rejected"):
                        return False, f"Navigation {st.nav_state}"

            time.sleep(0.1)

        return False, f"Timed out after {timeout:.0f}s"

    # -------------------------------------------------------------------------
    # HTTP Transport Helpers
    # -------------------------------------------------------------------------

    def _http_get(self, path: str, timeout: float = 2.0) -> Optional[dict]:
        if not self.active_vm_ip:
            return None
        conn = None
        try:
            conn = http.client.HTTPConnection(self.active_vm_ip, self.vm_port, timeout=timeout)
            conn.request("GET", path)
            res = conn.getresponse()
            if res.status == 200:
                payload = res.read()
                return json.loads(payload.decode("utf-8"))
        except Exception:
            return None
        finally:
            if conn:
                conn.close()
        return None

    def _http_post(self, path: str, data: dict, timeout: float = 3.0) -> Optional[dict]:
        if not self.active_vm_ip:
            return None
        conn = None
        try:
            conn = http.client.HTTPConnection(self.active_vm_ip, self.vm_port, timeout=timeout)
            headers = {"Content-Type": "application/json"}
            body = json.dumps(data).encode("utf-8")
            conn.request("POST", path, body=body, headers=headers)
            res = conn.getresponse()
            if res.status == 200:
                payload = res.read()
                return json.loads(payload.decode("utf-8"))
        except Exception:
            return None
        finally:
            if conn:
                conn.close()
        return None

    def stop(self) -> None:
        self._running = False
        if self._http_server:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass

