#!/usr/bin/env python3
"""Stage 2: Laptop Camera Sender for Windows / Linux / macOS.

Captures frames from USB camera(s), compresses them to JPEG, and streams them
over the local network (HTTP POST to /api/push_frame) to the Ubuntu VM.

Key Features:
  • Single-camera mode (--single-cam 0) or stereo pair mode (--left-cam 0 --right-cam 2).
  • Multi-layered Auto-Discovery (--server auto):
      1. VMware Workstation / Player host integration (vmrun / DHCP lease inspect).
      2. Multi-interface UDP broadcast (binds to each physical adapter to traverse Bridged Wi-Fi/LAN).
      3. Active subnet / ARP fallback probe.
  • VPN Bypass: Automatically binds outbound TCP sockets to the matching local interface IP,
    preventing university/corporate VPNs (like eduVPN/OpenVPN) from hijacking local VM traffic.

Usage (Single Camera - Recommended when running from Windows to VM):
    python Stage2\laptop_camera_sender.py --server auto --single-cam 0

Usage (Stereo Pair):
    python Stage2\laptop_camera_sender.py --server auto --left-cam 0 --right-cam 2
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
from pathlib import Path
import platform
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["http", "tcp"],
        default="http",
        help="Streaming protocol: 'http' (recommended, port 8080/8081) or 'tcp' (default: http).",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="auto",
        help="Server IP address, 'auto' for multi-layer auto-discovery, or hostname (default: auto).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Target port (default: 8080; auto-discovers active port if set to auto).",
    )
    parser.add_argument(
        "--single-cam",
        type=int,
        default=None,
        help="Device index of single camera (e.g. 0). If set, runs in single-camera mode.",
    )
    parser.add_argument(
        "--left-cam",
        type=int,
        default=0,
        help="Device index of left camera (default: 0).",
    )
    parser.add_argument(
        "--right-cam",
        type=int,
        default=2,
        help="Device index of right camera (default: 2).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture frame width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture frame height (default: 720).",
    )
    parser.add_argument(
        "--stream-scale",
        type=float,
        default=1.0,
        help="Downscale factor before transmitting (default: 1.0).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help="JPEG compression quality (1-100, default: 70).",
    )
    parser.add_argument(
        "--http-conns",
        type=int,
        default=2,
        help="Number of concurrent persistent HTTP connections (default: 2).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Network request timeout in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=30.0,
        help="Maximum capture FPS limit (default: 30.0).",
    )
    return parser.parse_args()


def get_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera(
    idx: int,
    width: int,
    height: int,
    backend: int,
    retries: int = 5,
    retry_delay: float = 0.5,
) -> cv2.VideoCapture:
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY and platform.system() != "Windows":
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30.0)

            warmup_ok = False
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    warmup_ok = True
                    break
                time.sleep(0.04)

            if warmup_ok:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                print(f"[LaptopSender] Camera {idx} initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
                return cap
            else:
                cap.release()
        time.sleep(retry_delay)

    # Detect what camera indices are actually available on the system
    available = []
    for test_idx in range(4):
        test_cap = cv2.VideoCapture(test_idx, backend)
        if test_cap.isOpened():
            available.append(test_idx)
            test_cap.release()
    hint = f" Available camera device(s): {available}." if available else ""
    raise RuntimeError(f"Could not open camera device {idx} after {retries} attempts.{hint}")


class SingleCameraGrabber:
    """Threaded camera grabber for a single camera."""

    def __init__(self, cam: cv2.VideoCapture) -> None:
        self.cam = cam
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self.grab_count = 0
        self.grab_fps = 0.0
        self._fps_timer = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cam.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame_id += 1
                    self.grab_count += 1
                    self._latest_frame = frame
                    now = time.perf_counter()
                    dt = now - self._fps_timer
                    if dt >= 1.0:
                        self.grab_fps = self.grab_count / dt
                        self.grab_count = 0
                        self._fps_timer = now
            else:
                time.sleep(0.005)

    def get_latest_frame(self) -> tuple[np.ndarray, int] | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy(), self._frame_id

    def get_grab_fps(self) -> float:
        with self._lock:
            return self.grab_fps

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class StereoCameraGrabber:
    """Threaded camera grabber for stereo pair."""

    def __init__(self, cam_l: cv2.VideoCapture, cam_r: cv2.VideoCapture) -> None:
        self.cam_l = cam_l
        self.cam_r = cam_r
        self._latest_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self.grab_count = 0
        self.grab_fps = 0.0
        self._fps_timer = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            self.cam_l.grab()
            self.cam_r.grab()
            ret_l, frame_l = self.cam_l.retrieve()
            ret_r, frame_r = self.cam_r.retrieve()
            if ret_l and ret_r and frame_l is not None and frame_r is not None:
                with self._lock:
                    self._frame_id += 1
                    self.grab_count += 1
                    self._latest_pair = (frame_l, frame_r)
                    now = time.perf_counter()
                    dt = now - self._fps_timer
                    if dt >= 1.0:
                        self.grab_fps = self.grab_count / dt
                        self.grab_count = 0
                        self._fps_timer = now
            else:
                time.sleep(0.005)

    def get_latest_frame(self) -> tuple[np.ndarray, int] | None:
        with self._lock:
            if self._latest_pair is None:
                return None
            composite = np.hstack((self._latest_pair[0], self._latest_pair[1]))
            return composite, self._frame_id

    def get_grab_fps(self) -> float:
        with self._lock:
            return self.grab_fps

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


# ==============================================================================
# Network Helpers: Interface Discovery & VPN Route Bypass
# ==============================================================================

def get_local_interfaces() -> list[tuple[str, str]]:
    """Return a list of (interface_ip, broadcast_ip) for all active non-loopback IPv4 interfaces."""
    interfaces: list[tuple[str, str]] = []
    seen_ips = set()

    # 1. Query hostname addresses
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip and not ip.startswith("127.") and ip not in seen_ips:
                seen_ips.add(ip)
    except Exception:
        pass

    # 2. Query default outbound socket routing
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        out_ip = s.getsockname()[0]
        s.close()
        if out_ip and not out_ip.startswith("127.") and out_ip not in seen_ips:
            seen_ips.add(out_ip)
    except Exception:
        pass

    for ip in seen_ips:
        parts = ip.split(".")
        if len(parts) == 4:
            bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
            interfaces.append((ip, bcast))

    return interfaces


def get_matching_source_ip(target_ip: str) -> str | None:
    """Find the local IPv4 address sharing the subnet with target_ip.

    Binding outbound sockets to this IP prevents VPN tunnels (like OpenVPN / eduVPN)
    from capturing traffic intended for local Bridged or NAT subnets.
    """
    try:
        t_addr = ipaddress.IPv4Address(target_ip)
        if t_addr.is_loopback:
            return "127.0.0.1"

        interfaces = get_local_interfaces()
        # Prefer /24 match
        for lip, _ in interfaces:
            try:
                if ipaddress.IPv4Network(f"{lip}/24", strict=False) == ipaddress.IPv4Network(f"{target_ip}/24", strict=False):
                    return lip
            except Exception:
                pass
        # Fallback /16 match
        for lip, _ in interfaces:
            try:
                if ipaddress.IPv4Network(f"{lip}/16", strict=False) == ipaddress.IPv4Network(f"{target_ip}/16", strict=False):
                    return lip
            except Exception:
                pass
    except Exception:
        pass
    return None


def probe_http_receiver(ip: str, port: int, timeout: float = 0.35) -> bool:
    """Test if ip:port responds to HTTP and is the camera receiver."""
    source_ip = get_matching_source_ip(ip)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if source_ip and not ip.startswith("127."):
            s.bind((source_ip, 0))
        s.settimeout(timeout)
        if s.connect_ex((ip, port)) != 0:
            s.close()
            return False

        # Send minimal HTTP check
        req = f"GET /api/push_frame HTTP/1.0\r\nHost: {ip}\r\nConnection: close\r\n\r\n".encode("ascii")
        s.sendall(req)
        resp = s.recv(256)
        s.close()
        # Accept HTTP status (501 Unsupported method or 200 OK or 405)
        return b"HTTP/1." in resp
    except Exception:
        return False


def get_vmware_guest_ips() -> list[str]:
    """Query VMware Workstation/Player vmrun command or DHCP lease files for running VM IPs."""
    ips: list[str] = []

    # 1. Query vmrun CLI
    vmrun_candidates = [
        r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe",
        r"C:\Program Files\VMware\VMware Workstation\vmrun.exe",
        "vmrun",
    ]
    for exe in vmrun_candidates:
        try:
            res = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=1.5)
            if res.returncode == 0:
                lines = res.stdout.strip().splitlines()
                for line in lines[1:]:
                    vmx = line.strip()
                    if vmx.endswith(".vmx"):
                        res_ip = subprocess.run([exe, "getGuestIPAddress", vmx], capture_output=True, text=True, timeout=1.5)
                        if res_ip.returncode == 0:
                            candidate = res_ip.stdout.strip()
                            if candidate and candidate not in ips and not candidate.startswith("Error"):
                                ips.append(candidate)
                if ips:
                    return ips
        except Exception:
            pass

    # 2. Parse VMware DHCP lease file (useful for NAT mode fallback)
    lease_paths = [
        r"C:\ProgramData\VMware\vmnetdhcp.leases",
        r"C:\ProgramData\VMware\vmnetdhcp.leases~",
    ]
    for lp in lease_paths:
        try:
            p = Path(lp)
            if p.exists():
                text = p.read_text(errors="ignore")
                found = re.findall(r"lease\s+([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s*\{", text)
                for f in reversed(found):
                    if f not in ips:
                        ips.append(f)
        except Exception:
            pass

    return ips


def get_arp_ips() -> list[str]:
    """Read local ARP cache for reachable IPv4 addresses."""
    ips: list[str] = []
    try:
        res = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=1.0)
        if res.returncode == 0:
            found = re.findall(r"([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})", res.stdout)
            for f in found:
                if not f.endswith(".255") and not f.startswith("224.") and not f.startswith("239.") and f not in ips:
                    ips.append(f)
    except Exception:
        pass
    return ips


class HttpSendWorkerPool:
    """Worker pool for sending JPEG frames over HTTP POST with persistent connections."""

    def __init__(
        self,
        server: str,
        port: int,
        num_conns: int = 2,
        timeout: float = 5.0,
        source_ip: str | None = None,
    ) -> None:
        self.server = server
        self.port = port
        self.num_conns = max(1, num_conns)
        self.timeout = timeout
        self.source_ip = source_ip
        self.queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue(maxsize=self.num_conns)
        self.running = True
        self.sent_count = 0
        self.sent_bytes = 0
        self.current_fps = 0.0
        self.last_payload_kb = 0.0
        self._rtt_accum = 0.0
        self._rtt_samples = 0
        self._lock = threading.Lock()
        self.threads = [
            threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            for i in range(self.num_conns)
        ]
        for t in self.threads:
            t.start()

    def submit_frame(self, payload: bytes, timestamp: float, seq: int) -> None:
        try:
            self.queue.put_nowait((payload, timestamp, seq))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((payload, timestamp, seq))
            except queue.Full:
                pass

    def _worker_loop(self, worker_id: int) -> None:
        headers = {
            "Content-Type": "image/jpeg",
            "Connection": "keep-alive",
        }
        conn: http.client.HTTPConnection | None = None
        source_addr = (self.source_ip, 0) if self.source_ip else None

        while self.running:
            try:
                payload, timestamp, seq = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if conn is None:
                    conn = http.client.HTTPConnection(
                        self.server,
                        self.port,
                        timeout=self.timeout,
                        source_address=source_addr,
                    )
                    conn.connect()

                req_headers = dict(headers)
                req_headers["X-Capture-Timestamp"] = f"{timestamp:.4f}"
                with self._lock:
                    fps_val = self.current_fps
                req_headers["X-Sender-FPS"] = f"{fps_val:.1f}"
                req_headers["X-Frame-Sequence"] = str(seq)

                t0 = time.perf_counter()
                conn.request("POST", "/api/push_frame", body=payload, headers=req_headers)
                res = conn.getresponse()
                res.read()
                rtt = (time.perf_counter() - t0) * 1000.0

                with self._lock:
                    self.sent_count += 1
                    self.sent_bytes += len(payload)
                    self.last_payload_kb = len(payload) / 1024.0
                    self._rtt_accum += rtt
                    self._rtt_samples += 1
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(0.5)

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def get_and_reset_stats(self, dt: float) -> tuple[float, float, float, float]:
        with self._lock:
            self.current_fps = self.sent_count / dt if dt > 0 else 0.0
            rate_mb = (self.sent_bytes / (1024.0 * 1024.0)) / dt if dt > 0 else 0.0
            avg_rtt = (self._rtt_accum / self._rtt_samples) if self._rtt_samples > 0 else 0.0
            kb = self.last_payload_kb
            self.sent_count = 0
            self.sent_bytes = 0
            self._rtt_accum = 0.0
            self._rtt_samples = 0
            return self.current_fps, rate_mb, avg_rtt, kb

    def stop(self) -> None:
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)


def stream_http(args: argparse.Namespace, grabber) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    scale = max(0.1, min(1.0, float(args.stream_scale)))
    num_conns = max(1, int(args.http_conns))

    source_ip = get_matching_source_ip(args.server)
    if source_ip and not args.server.startswith("127."):
        print(f"[LaptopSender] Local interface bound to {source_ip} (VPN bypass active)")

    print(f"[LaptopSender] Streaming to http://{args.server}:{args.port}/api/push_frame...")
    worker_pool = HttpSendWorkerPool(args.server, args.port, num_conns=num_conns, timeout=args.timeout, source_ip=source_ip)
    last_seq = -1
    fps_timer = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            data = grabber.get_latest_frame()
            if data is None:
                time.sleep(0.003)
                continue

            frame, seq = data
            if seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq

            if scale < 0.999:
                ch, cw = frame.shape[:2]
                target_w = int(round(cw * scale))
                target_h = int(round(ch * scale))
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            success, encoded_jpg = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                continue

            payload = encoded_jpg.tobytes()
            worker_pool.submit_frame(payload, time.time(), seq)

            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                dt = now - fps_timer
                net_fps, rate_mb, avg_rtt, kb = worker_pool.get_and_reset_stats(dt)
                cam_fps = grabber.get_grab_fps()
                rtt_str = f"{avg_rtt:.0f} ms" if avg_rtt > 0 else "-- ms"
                sys.stdout.write(
                    f"\r[Stream] Camera: {cam_fps:.1f} FPS | Network: {net_fps:.1f} FPS | RTT: {rtt_str} | Frame: {kb:.1f} KB"
                )
                sys.stdout.flush()
                fps_timer = now

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\n[LaptopSender] Stopped by user.")
    finally:
        worker_pool.stop()


def discover_server(beacon_port: int = 8089, default_http_port: int = 8080, timeout: float = 1.5) -> tuple[str, int]:
    """Auto-discover the Ubuntu VM IP and port on Bridged or NAT subnets."""
    print("[LaptopSender] Searching for Stage 2 camera receiver via multi-layer discovery...")

    candidate_ports = [default_http_port, 8081, 8080, 8088, 5000]
    # Remove duplicates preserving order
    candidate_ports = list(dict.fromkeys(candidate_ports))

    # --------------------------------------------------------------------------
    # Strategy 1: VMware Guest Direct Query (Fastest & 100% reliable for VMs)
    # --------------------------------------------------------------------------
    vm_ips = get_vmware_guest_ips()
    for vm_ip in vm_ips:
        for p in candidate_ports:
            if probe_http_receiver(vm_ip, p, timeout=0.35):
                print(f"[LaptopSender] Auto-discovered VMware VM at http://{vm_ip}:{p} (via VMware host integration)!")
                return vm_ip, p

    # --------------------------------------------------------------------------
    # Strategy 2: Multi-Interface UDP Broadcast
    # --------------------------------------------------------------------------
    interfaces = get_local_interfaces()
    udp_sockets: list[socket.socket] = []
    try:
        for lip, bcast in interfaces:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind((lip, 0))
                sock.settimeout(0.2)
                for b_target in [bcast, "255.255.255.255"]:
                    try:
                        sock.sendto(b"TB4_CAMERA_DISCOVERY", (b_target, beacon_port))
                    except Exception:
                        pass
                udp_sockets.append(sock)
            except Exception:
                pass

        # Listen for response
        t_end = time.time() + timeout
        while time.time() < t_end:
            for s in udp_sockets:
                try:
                    data, addr = s.recvfrom(512)
                    msg = data.decode("utf-8", errors="ignore")
                    if msg.startswith("TB4_CAMERA_OFFER:"):
                        discovered_port = int(msg.split(":")[1])
                        discovered_ip = addr[0]
                        print(f"[LaptopSender] Auto-discovered VM camera receiver at http://{discovered_ip}:{discovered_port}!")
                        return discovered_ip, discovered_port
                except (socket.timeout, BlockingIOError, OSError):
                    continue
            time.sleep(0.02)
    finally:
        for s in udp_sockets:
            try:
                s.close()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Strategy 3: Local Subnet ARP & Candidate Sweep
    # --------------------------------------------------------------------------
    arp_ips = get_arp_ips()
    fallbacks = [
        "192.168.31.253",  # Known bridged VM address
        "192.168.52.128",  # VMware NAT default
        "127.0.0.1",
    ]
    all_candidates = list(dict.fromkeys(vm_ips + arp_ips + fallbacks))
    for ip in all_candidates:
        for p in candidate_ports:
            if probe_http_receiver(ip, p, timeout=0.25):
                print(f"[LaptopSender] Reached VM camera receiver at http://{ip}:{p} (via subnet probe)")
                return ip, p

    print("[LaptopSender] [WARN] Auto-discovery could not find an active HTTP receiver.")
    best_guess = vm_ips[0] if vm_ips else "192.168.31.253"
    print(f"[LaptopSender] Falling back to {best_guess}:{default_http_port}.")
    return best_guess, default_http_port


def main() -> None:
    args = parse_args()

    if args.server.lower() in ("auto", "discover"):
        server_ip, server_port = discover_server(default_http_port=args.port)
        args.server = server_ip
        args.port = server_port
    else:
        # If user explicitly provided an IP, verify if target port has the receiver
        # or if it relocated to 8081/8088
        if not probe_http_receiver(args.server, args.port, timeout=0.3):
            for alt_port in [8081, 8080, 8088, 5000]:
                if alt_port != args.port and probe_http_receiver(args.server, alt_port, timeout=0.3):
                    print(f"[LaptopSender] Auto-switched to active receiver port {alt_port} on {args.server}.")
                    args.port = alt_port
                    break

    backend = get_backend()
    print(f"[LaptopSender] OS: {platform.system()} | Camera backend: {backend}")

    if args.single_cam is not None:
        print(f"[LaptopSender] Opening Single Camera (ID {args.single_cam})...")
        cam = open_camera(args.single_cam, args.width, args.height, backend)
        grabber = SingleCameraGrabber(cam)
        try:
            stream_http(args, grabber)
        finally:
            grabber.stop()
            cam.release()
    else:
        print(f"[LaptopSender] Opening Left Camera (ID {args.left_cam}) and Right Camera (ID {args.right_cam})...")
        cam_l = open_camera(args.left_cam, args.width, args.height, backend)
        time.sleep(0.4)
        cam_r = open_camera(args.right_cam, args.width, args.height, backend)
        grabber = StereoCameraGrabber(cam_l, cam_r)
        try:
            stream_http(args, grabber)
        finally:
            grabber.stop()
            cam_l.release()
            cam_r.release()

    print("[LaptopSender] Complete.")


if __name__ == "__main__":
    main()
