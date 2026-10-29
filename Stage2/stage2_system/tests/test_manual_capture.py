"""Offline tests: no robot, USB camera, API key or paid requests needed."""
from pathlib import Path
import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from pydantic import ValidationError
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laptop.camera import CameraManager
from laptop.manual_capture import ManualCapture
from laptop.llm.object_analysis import AnalysisError, ObjectAnalysis, analyze_object, PROMPT


class TestAnalysis(unittest.TestCase):
    def response(self, content, status=200, finish='stop'):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.json.return_value = {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
        return response

    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-key'})
    @patch('laptop.llm.object_analysis.requests.post')
    def test_image_prompt_and_result(self, post):
        post.return_value = self.response('{"object_name":"Cup","height_cm":10.5,"color":"blue"}')
        result = analyze_object(b'jpeg-test')
        self.assertEqual(result.display(), 'Cup, height: 10.5 cm, color: blue')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'deepseek-flash')
        content = payload['messages'][1]['content']
        self.assertEqual(content[0]['text'], PROMPT)
        self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]), b'jpeg-test')
        self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_unknown_and_invalid_height(self):
        self.assertIn('height: unknown', ObjectAnalysis(object_name='Cup', height_cm=None, color='blue').display())
        for height in (-1, 0, float('inf'), float('nan')):
            with self.assertRaises(ValidationError):
                ObjectAnalysis(object_name='Cup', height_cm=height, color='blue')

    @patch.dict(os.environ, {}, clear=True)
    @patch('laptop.llm.object_analysis.requests.post')
    def test_missing_key_does_not_call_api(self, post):
        with self.assertRaisesRegex(AnalysisError, 'DEEPSEEK_API_KEY'):
            analyze_object(b'jpeg')
        post.assert_not_called()

    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-key'})
    @patch('laptop.llm.object_analysis.requests.post')
    def test_invalid_and_truncated_responses(self, post):
        for response in (self.response('not json'), self.response('{}'), self.response('{}', finish='length')):
            post.return_value = response
            with self.assertRaises(AnalysisError):
                analyze_object(b'jpeg')

    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-key'})
    @patch('laptop.llm.object_analysis.requests.post')
    def test_http_and_timeout_errors_hide_payloads(self, post):
        post.return_value = self.response('sensitive error body', status=401)
        with self.assertRaisesRegex(AnalysisError, 'rejected the API key'):
            analyze_object(b'jpeg')
        post.side_effect = requests.Timeout('sensitive error body')
        with self.assertRaisesRegex(AnalysisError, '^DeepSeek request timed out'):
            analyze_object(b'jpeg')


class TestManualCapture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.camera = CameraManager(is_mock=False)
        self.camera._running = True
        self.camera._last_frame_at = time.monotonic()
        self.camera._latest_left = np.full((24, 32, 3), 120, dtype=np.uint8)
        self.analyzer = MagicMock(return_value=ObjectAnalysis(object_name='Cup', height_cm=10, color='blue'))
        self.capture = ManualCapture(self.camera, Path(self.directory.name), self.analyzer)

    def finish(self):
        events = []
        deadline = time.monotonic() + 3
        while self.capture.busy and time.monotonic() < deadline:
            events.extend(self.capture.poll())
            time.sleep(0.005)
        self.assertFalse(self.capture.busy)
        return events

    def test_exact_uploaded_image_is_displayed_and_saved(self):
        self.assertTrue(self.capture.start())
        events = self.finish()
        self.assertEqual([kind for kind, _ in events], ['captured', 'result'])
        displayed, name = events[0][1]
        jpeg = self.analyzer.call_args.args[0]
        self.assertEqual((Path(self.directory.name)/'manual_captures'/name).read_bytes(), jpeg)
        np.testing.assert_array_equal(displayed, cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        self.assertEqual(self.capture.result_text, 'Cup, height: 10 cm, color: blue')

    def test_repeat_click_and_close(self):
        gate = threading.Event()
        self.analyzer.side_effect = lambda *a, **k: (gate.wait(2), ObjectAnalysis(object_name='Cup', height_cm=None, color='blue'))[1]
        self.capture.start()
        self.assertFalse(self.capture.start())
        self.capture.close()
        gate.set()
        self.assertFalse(self.capture.start())
        self.assertEqual(self.capture.poll(), [])
        # Let worker finish before TemporaryDirectory cleanup.
        time.sleep(0.05)

    def test_stale_frame_does_not_upload(self):
        self.camera._last_frame_at = time.monotonic() - 10
        self.capture.start()
        self.finish()
        self.assertIn('No recent camera frame', self.capture.result_text)
        self.analyzer.assert_not_called()

    def test_synthetic_camera_does_not_upload(self):
        self.camera.is_mock = True
        self.capture.start()
        self.finish()
        self.assertIn('mock mode', self.capture.result_text)
        self.analyzer.assert_not_called()

    def test_mock_llm_result_is_labelled(self):
        self.camera.is_mock = True
        self.capture.start(mock=True)
        self.finish()
        self.assertTrue(self.capture.result_text.startswith('[DEMO]'))

    def test_snapshot_is_independent_and_stopped_feed_is_rejected(self):
        frame = self.camera.snapshot_left_frame()
        frame[:] = 0
        self.assertTrue(np.all(self.camera._latest_left == 120))
        self.camera._running = False
        self.assertIsNone(self.camera.snapshot_left_frame())


class TestPanelIntegration(unittest.TestCase):
    def test_main_thread_routes_capture_and_result_to_existing_panels(self):
        # Exercise GUI update logic without needing a display server/Tk installation.
        import importlib
        with patch.dict(sys.modules, {
            'tkinter': MagicMock(), 'tkinter.font': MagicMock(),
            'tkinter.ttk': MagicMock(), 'PIL.ImageTk': MagicMock(),
        }):
            gui_module = importlib.import_module('laptop.gui')
        gui = gui_module.MissionControlGUI.__new__(gui_module.MissionControlGUI)
        gui.m = MagicMock()
        gui.m.is_running.return_value = False
        gui.m.current_level = 0
        gui.m.level_results = {1: 'Original mission result'}
        frame = np.full((24, 32, 3), 120, dtype=np.uint8)
        gui.m.camera.get_latest_pair.return_value = (frame, frame, 30)
        gui.m.last_processed_frame = frame
        gui.m.last_processed_name = 'capture.jpg'
        gui.manual_capture = MagicMock()
        gui.manual_capture.busy = False
        gui.manual_capture.result_text = 'Cup, height: 10 cm, color: blue'
        gui.manual_capture.poll.return_value = [('captured', (frame, 'capture.jpg'))]
        for name in ('root', 'btn_capture', 'btn_level_results', 'lbl_status', 'lbl_level',
                     'lbl_extra_info', 'lbl_robot_conn', 'lbl_result', 'lbl_left_hdr',
                     'lbl_right_hdr', 'canvas_left_cam', 'canvas_right_cam', 'lbl_proc_hdr',
                     'canvas_proc', 'canvas_map', 'map_renderer', 'txt_console'):
            setattr(gui, name, MagicMock())
        gui._render_frame_to_label = MagicMock()
        with patch.object(gui_module.ImageTk, 'PhotoImage'):
            gui._update_loop()
        gui.m.set_processed_frame.assert_called_once_with(frame, 'capture.jpg')
        self.assertIn('Cup, height: 10 cm, color: blue', gui.lbl_result.config.call_args.kwargs['text'])
        gui.m.set_result.assert_not_called()
        gui._show_level_results()
        gui.manual_capture.poll.return_value = []
        with patch.object(gui_module.ImageTk, 'PhotoImage'):
            gui._update_loop()
        self.assertIn('Original mission result', gui.lbl_result.config.call_args.kwargs['text'])


if __name__ == '__main__':
    unittest.main()
