"""Background manual image analysis; Tk widgets are only touched by the GUI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import threading
import uuid

import cv2
import numpy as np

from .llm.object_analysis import AnalysisError, analyze_object


class ManualCapture:
    def __init__(self, camera, runs_dir: Path, analyzer=analyze_object):
        self.camera, self.runs_dir, self.analyzer = camera, Path(runs_dir), analyzer
        self.busy = False
        self.result_text = ""
        self._events = queue.Queue()
        self._closed = threading.Event()

    def start(self, *, mock: bool = False) -> bool:
        # start() and poll() are both called on the Tk main thread.
        if self.busy or self._closed.is_set():
            return False
        self.busy = True
        self.result_text = "Capturing image..."
        threading.Thread(target=self._work, args=(mock,), daemon=True).start()
        return True

    def _work(self, mock):
        try:
            if self.camera.is_mock and not mock:
                raise AnalysisError("Camera is in mock mode. Connect a real camera before uploading.")
            # Atomically copy a recent frame; never open the USB device a second time.
            frame = self.camera.snapshot_left_frame(max_age_s=2.0)
            if frame is None:
                raise AnalysisError("No recent camera frame. Check the camera connection and try again.")
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise AnalysisError("Could not encode the captured image.")
            jpeg = encoded.tobytes()
            # Display the exact JPEG that will be sent, not a later live frame.
            captured = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if captured is None:
                raise AnalysisError("Could not decode the captured image.")
            folder = self.runs_dir / 'manual_captures'
            folder.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
            image_path = folder / (stem + '.jpg')
            image_path.write_bytes(jpeg)
            self._events.put(('captured', (captured, image_path.name)))
            if self._closed.is_set():
                return
            result = self.analyzer(jpeg, mock=mock)
            (folder / (stem + '.json')).write_text(result.model_dump_json(indent=2), encoding='utf-8')
            self._events.put(('result', ('[DEMO] ' if mock else '') + result.display()))
        except AnalysisError as error:
            self._events.put(('error', str(error)))
        except Exception:
            # Do not dump exception/request objects which may contain credentials.
            self._events.put(('error', 'Capture or analysis failed. Check the camera and output directory.'))

    def poll(self):
        if self._closed.is_set():
            return []
        events = []
        while True:
            try:
                kind, value = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == 'captured':
                self.result_text = 'Analyzing captured image...'
            elif kind == 'result':
                self.result_text, self.busy = value, False
            elif kind == 'error':
                self.result_text, self.busy = 'Error: ' + value, False
            events.append((kind, value))
        return events

    def close(self):
        self._closed.set()
