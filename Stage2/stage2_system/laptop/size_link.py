"""Client for SSH Size Estimation Server."""

from __future__ import annotations

import base64
import http.client
import json
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

from shared.protocol import DEFAULT_SSH_SIZE_PORT, DetectionItem, MeasurementResult


class SizeLink:
    """Sends stereo image pairs to the SSH GPU Size Server for height estimation."""

    def __init__(
        self,
        server_url: str = f"http://127.0.0.1:{DEFAULT_SSH_SIZE_PORT}",
        is_mock: bool = False,
    ) -> None:
        self.server_url = server_url
        self.is_mock = is_mock
        parsed = urlparse(server_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or DEFAULT_SSH_SIZE_PORT

    def ping(self) -> Tuple[bool, float, str]:
        """Test round-trip latency to the SSH size server."""
        if self.is_mock:
            return True, 1.2, "Mock Size Server (offline)"

        t0 = time.perf_counter()
        conn = None
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
            conn.request("GET", "/api/health")
            res = conn.getresponse()
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if res.status == 200:
                body = json.loads(res.read().decode("utf-8"))
                info = f"{body.get('estimator', 'crestereo')} on {body.get('device', 'GPU')}"
                return True, rtt_ms, info
        except Exception as err:
            return False, 0.0, str(err)
        finally:
            if conn:
                conn.close()

        return False, 0.0, "Health check failed"

    def measure(
        self,
        composite_png_bytes: bytes,
        point_name: Optional[str] = None,
        hint_class: Optional[str] = None,
        timeout_s: float = 15.0,
    ) -> MeasurementResult:
        """Send lossless stereo pair to size server and receive bounding box and height estimates."""
        if self.is_mock:
            return self._mock_measurement(point_name, hint_class)

        conn = None
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout_s)
            path = "/api/measure"
            params = []
            if point_name:
                params.append(f"point={point_name}")
            if hint_class:
                params.append(f"hint={hint_class}")
            if params:
                path += "?" + "&".join(params)

            headers = {
                "Content-Type": "image/png",
                "Content-Length": str(len(composite_png_bytes)),
            }
            conn.request("POST", path, body=composite_png_bytes, headers=headers)
            res = conn.getresponse()
            if res.status == 200:
                payload = res.read()
                data = json.loads(payload.decode("utf-8"))
                return MeasurementResult.from_dict(data)
            else:
                return MeasurementResult(
                    ok=False,
                    point=point_name,
                    error_message=f"Server returned HTTP {res.status}: {res.reason}",
                )
        except Exception as err:
            return MeasurementResult(
                ok=False,
                point=point_name,
                error_message=f"Size server communication failed: {err}",
            )
        finally:
            if conn:
                conn.close()

    def _mock_measurement(self, point_name: Optional[str], hint_class: Optional[str]) -> MeasurementResult:
        """Generate plausible canned measurement for testing."""
        pt = (point_name or "X").upper()
        mock_table = {
            "B": ("cup", 10.5, 8.2, 0.82),
            "C": ("apple", 8.0, 7.5, 0.75),
            "D": ("bottle", 31.5, 9.0, 0.90),
            "E": ("banana", 16.0, 4.5, 0.85),
            "F": ("vase", 27.0, 12.0, 0.95),
            "G": ("clock", 13.5, 12.0, 0.88),
            "H": ("bottle", 22.0, 7.0, 0.80),
            "I": ("teddy bear", 28.0, 18.0, 1.10),
        }
        c_name, h_cm, w_cm, z_m = mock_table.get(pt, (hint_class or "object", 15.0, 8.0, 0.85))

        det = DetectionItem(
            class_name=c_name,
            confidence=0.91,
            bbox_px=[350, 180, 520, 580],
            height_cm=h_cm,
            width_cm=w_cm,
            distance_m=z_m,
            valid_depth_pixels=14200,
        )
        return MeasurementResult(
            ok=True,
            point=point_name,
            primary=det,
            detections=[det],
            timing_ms={"rectify": 4.5, "stereo": 38.0, "yolo": 18.0, "total": 60.5},
        )

