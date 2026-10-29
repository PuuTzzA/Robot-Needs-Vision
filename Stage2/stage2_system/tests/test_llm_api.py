"""Offline adapter and task integration tests; never contact DeepSeek."""
import base64
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laptop.llm import llm_api as api, tasks
from laptop.llm.schemas import Category, Level3TaskInfo

class TestDeepSeekAdapter(unittest.TestCase):
    def setUp(self):
        api.set_mock_mode(False)
        self.env = patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'})
        self.env.start()
        self.post = patch.object(api.requests, 'post').start()
        self.response = self.post.return_value.__enter__.return_value
        self.response.status_code = 200
        self.answer({'category': 'Cup', 'color': 'blue'})

    def tearDown(self):
        patch.stopall()
        api.set_mock_mode(False)

    def answer(self, value, finish='stop'):
        self.response.json.return_value = {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(value)}}]}

    def test_classification_request(self):
        result = tasks.classify_object(b'jpeg', 'B')
        self.assertEqual(result.category, 'Cup')
        kw = self.post.call_args.kwargs
        body = json.loads(kw['data'])
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertIn('Known object categories', body['messages'][1]['content'][0]['text'])
        self.assertIn('JSON Schema', body['messages'][0]['content'])
        self.assertEqual(body['messages'][1]['content'][1]['text'], 'Image label: photo_point_B')
        self.assertTrue(body['messages'][1]['content'][2]['image_url']['url'].endswith(base64.b64encode(b'jpeg').decode()))
        self.assertFalse(kw['allow_redirects'])

    def test_task_parsers(self):
        self.answer({'waypoints': [{'name': ' b '}, {'name': 'E'}]})
        self.assertEqual(tasks.parse_level1_task(b'task', ['B','E']), ['B','E'])
        self.answer({'categories': [{'category': 'Cup'}, {'category': 'Book'}]})
        self.assertEqual(tasks.parse_level2_task(b'task'), ['Cup','Book'])
        self.answer({'task_description': 'Count red objects', 'use_high_visibility': False})
        self.assertFalse(tasks.parse_level3_task(b'task').use_high_visibility)

    def test_level3_multiple_images(self):
        self.answer({'task_description': 'Count', 'inventory': [], 'answer': '2'})
        result = tasks.solve_level3(b'task', Level3TaskInfo(task_description='Count', use_high_visibility=False), {'D': b'd', 'B': b'b'}, {})
        self.assertEqual(result.answer, '2')
        content = json.loads(self.post.call_args.kwargs['data'])['messages'][1]['content']
        self.assertEqual([c['text'] for c in content[1::2]], ['Image label: photo_task_laptop','Image label: photo_point_B','Image label: photo_point_D'])

    def test_missing_key_and_empty_image(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''}):
            with self.assertRaises(api.LLMError):
                tasks.classify_object(b'jpeg', 'B')
        with self.assertRaises(api.LLMError):
            tasks.classify_object(b'', 'B')
        self.post.assert_not_called()

    def test_invalid_and_incomplete(self):
        for value, finish in [({},'stop'), ({'category':'Cup'},'length')]:
            with self.subTest(value=value, finish=finish):
                self.answer(value, finish)
                with self.assertRaises(api.LLMError):
                    tasks.classify_object(b'jpeg','B')
        self.response.json.return_value = {'choices': []}
        with self.assertRaises(api.LLMError):
            tasks.classify_object(b'jpeg','B')

    def test_http_errors_and_timeout_do_not_leak(self):
        for status in [401,402,429,500]:
            self.response.status_code = status
            with self.assertRaises(api.LLMError) as err:
                tasks.classify_object(b'jpeg','B')
            self.assertNotIn('test-only', str(err.exception))
        self.post.side_effect = requests.Timeout('test-only')
        with self.assertRaises(api.LLMError) as err:
            tasks.classify_object(b'jpeg','B')
        self.assertNotIn('test-only', str(err.exception))

    def test_mock_needs_no_key_or_network(self):
        api.set_mock_mode(True)
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(tasks.parse_level1_task(b'task', ['B','E']), ['B','E'])
            self.assertTrue(api.health_check()[0])
        self.post.assert_not_called()

    @patch.object(api.requests, 'get')
    def test_health_check(self, get):
        res = get.return_value.__enter__.return_value
        res.status_code = 200
        res.json.return_value = {'data':[{'id':'deepseek-flash'}]}
        self.assertTrue(api.health_check()[0])
        res.status_code = 401
        self.assertFalse(api.health_check()[0])
        self.post.assert_not_called()

if __name__ == '__main__':
    unittest.main()
