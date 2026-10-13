"""Network frame receiver and sender utilities for remote camera streaming."""

from __future__ import annotations
import socket
import struct
import threading
import time
import cv2
import numpy as np


class NetworkStereoReceiver:
    """Listens on a TCP socket for paired stereo frames sent by laptop_camera_sender.py.

    Frames are received as length-prefixed JPEG buffers: [Left_Frame | Right_Frame].
    Thread-safe and automatically recovers when client disconnects or reconnects.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._latest_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = False
        self._frame_count = 0

    def start(self) -> None:
        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)
        print(f"[NetworkReceiver] Listening for laptop camera stream on {self.host}:{self.port}...")

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _recvall(self, conn: socket.socket, size: int) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            try:
                packet = conn.recv(size - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except (socket.timeout, BlockingIOError):
                if not self._running:
                    return None
                continue
            except Exception:
                return None
        return bytes(data)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            print(f"[NetworkReceiver] Connected to laptop camera client at {addr}")
            self._connected = True
            conn.settimeout(2.0)

            try:
                while self._running:
                    # Header: 8-byte big-endian unsigned int containing payload size
                    header = self._recvall(conn, 8)
                    if not header:
                        break
                    (payload_len,) = struct.unpack(">Q", header)
                    if payload_len == 0 or payload_len > 50_000_000:
                        break

                    payload = self._recvall(conn, payload_len)
                    if not payload:
                        break

                    # Decode composite JPEG [Left | Right]
                    np_arr = np.frombuffer(payload, dtype=np.uint8)
                    composite = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if composite is None:
                        continue

                    h, w = composite.shape[:2]
                    mid_x = w // 2
                    left_frame = composite[:, :mid_x]
                    right_frame = composite[:, mid_x:]

                    with self._lock:
                        self._latest_pair = (left_frame, right_frame)
                        self._frame_count += 1
            except Exception as e:
                print(f"[NetworkReceiver] Connection interrupted: {e}")
            finally:
                conn.close()
                self._connected = False
                print("[NetworkReceiver] Waiting for reconnection...")

    def get_latest_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if self._latest_pair is None:
                return None
            return self._latest_pair[0].copy(), self._latest_pair[1].copy()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)

