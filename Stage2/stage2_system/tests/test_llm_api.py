"""Offline adapter and task integration tests; never contact DeepSeek."""
import base64
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laptop.llm import llm_api as api, prompts, tasks
from laptop.llm.schemas import Category, TaskLegibility


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
        self.response.json.return_value = {
            'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(value)}}]
        }

    def test_level3_task_info_prompt_removed(self):
        self.assertFalse(hasattr(prompts, 'LEVEL3_TASK_INFO_PROMPT'))
        self.assertFalse(hasattr(tasks, 'parse_level3_task'))

    def test_classification_request(self):
        self.answer({'category': 'Cup', 'color': 'blue', 'height_cm': 11.0})
        result = tasks.classify_object(b'jpeg', 'B')
        self.assertEqual(result.category, 'Cup')
        self.assertAlmostEqual(result.height_cm, 11.0)
        kw = self.post.call_args.kwargs
        body = json.loads(kw['data'])
        self.assertEqual(body['model'], 'deepseek-flash')
        prompt_text = body['messages'][1]['content'][0]['text']
        self.assertIn('Known object categories', prompt_text)
        self.assertIn('None (no object', prompt_text)
        self.assertIn('Known possible objects (roughly from smallest to tallest', prompt_text)
        self.assertIn('Book laying down (~1.5 cm)', prompt_text)
        self.assertIn('Smaller Vase (~17.5 cm)', prompt_text)
        self.assertIn('Bottle 1.5L (~30-35 cm)', prompt_text)
        self.assertIn('JSON Schema', body['messages'][0]['content'])
        self.assertIn('height_cm', body['messages'][0]['content'])
        self.assertEqual(body['messages'][1]['content'][1]['text'], 'Image label: photo_point_B')
        self.assertTrue(
            body['messages'][1]['content'][2]['image_url']['url'].endswith(
                base64.b64encode(b'jpeg').decode()
            )
        )
        self.assertFalse(kw['allow_redirects'])

    def test_classification_empty_box_none_option(self):
        for raw_cat in ('None', 'none', 'No object', 'Empty box'):
            self.answer({'category': raw_cat, 'color': 'none', 'height_cm': 0.0})
            res = tasks.classify_object(b'jpeg', 'B')
            self.assertTrue(res.is_empty)
            self.assertEqual(res.category, 'None')
            self.assertIsNone(res.color)
            self.assertIsNone(res.height_cm)

    def test_task_parsers_and_legibility_check(self):
        self.answer({'waypoints': [{'name': ' b '}, {'name': 'E'}]})
        self.assertEqual(tasks.parse_level1_task(b'task', ['B', 'E']), ['B', 'E'])

        self.answer({'categories': [{'category': 'Cup'}, {'category': 'Book'}]})
        self.assertEqual(tasks.parse_level2_task(b'task'), ['Cup', 'Book'])

        self.answer({'legible': True, 'transcribed_text': 'How many objects are taller than B?', 'reason': None})
        leg_ok = tasks.check_level3_legibility(b'task')
        self.assertIsInstance(leg_ok, TaskLegibility)
        self.assertTrue(leg_ok.legible)

        self.answer({'legible': False, 'transcribed_text': None, 'reason': 'severe motion blur'})
        leg_bad = tasks.check_level3_legibility(b'task')
        self.assertFalse(leg_bad.legible)
        self.assertEqual(leg_bad.reason, 'severe motion blur')

    def test_level3_multiple_images_and_full_sentence_prompt(self):
        full_sentence = (
            '2. The Vase at point B and the Clock at point D are taller than the Cup at point C.'
        )
        self.answer({'task_description': 'Count taller objects', 'inventory': [], 'answer': full_sentence})
        result = tasks.solve_level3(b'task', {'D': b'd', 'B': b'b'}, {})
        self.assertEqual(result.answer, full_sentence)
        body = json.loads(self.post.call_args.kwargs['data'])
        self.assertIn('Complete full-sentence answer', body['messages'][0]['content'])
        content = body['messages'][1]['content']
        self.assertIn('complete, informative sentence', content[0]['text'])
        self.assertEqual(
            [c['text'] for c in content[1::2]],
            ['Image label: photo_task_laptop', 'Image label: photo_point_B', 'Image label: photo_point_D'],
        )

    def test_missing_key_and_empty_image(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''}):
            with self.assertRaises(api.LLMError):
                tasks.classify_object(b'jpeg', 'B')
        with self.assertRaises(api.LLMError):
            tasks.classify_object(b'', 'B')
        self.post.assert_not_called()

    def test_invalid_and_incomplete(self):
        for value, finish in [({}, 'stop'), ({'category': 'Cup'}, 'length')]:
            with self.subTest(value=value, finish=finish):
                self.answer(value, finish)
                with self.assertRaises(api.LLMError):
                    tasks.classify_object(b'jpeg', 'B')
        self.response.json.return_value = {'choices': []}
        with self.assertRaises(api.LLMError):
            tasks.classify_object(b'jpeg', 'B')

    def test_http_errors_and_timeout_do_not_leak(self):
        for status in [401, 402, 429, 500]:
            self.response.status_code = status
            with self.assertRaises(api.LLMError) as err:
                tasks.classify_object(b'jpeg', 'B')
            self.assertNotIn('test-only', str(err.exception))
        self.post.side_effect = requests.Timeout('test-only')
        with self.assertRaises(api.LLMError) as err:
            tasks.classify_object(b'jpeg', 'B')
        self.assertNotIn('test-only', str(err.exception))

    def test_mock_needs_no_key_or_network(self):
        api.set_mock_mode(True)
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(tasks.parse_level1_task(b'task', ['B', 'E']), ['B', 'E'])
            self.assertTrue(tasks.check_level3_legibility(b'task').legible)
            sol = tasks.solve_level3(b'task', {'B': b'b', 'C': b'c'}, {})
            self.assertIn('point', sol.answer.lower())
            self.assertGreater(len(sol.answer.split()), 5)
            self.assertTrue(api.health_check()[0])
        self.post.assert_not_called()

    @patch.object(api.requests, 'get')
    def test_health_check(self, get):
        res = get.return_value.__enter__.return_value
        res.status_code = 200
        res.json.return_value = {'data': [{'id': 'deepseek-flash'}]}
        self.assertTrue(api.health_check()[0])
        res.status_code = 401
        self.assertFalse(api.health_check()[0])
        self.post.assert_not_called()


class TestEnvLoader(unittest.TestCase):
    def test_loads_dotenv_with_bom_quotes_and_export(self):
        import os
        import tempfile
        from laptop.llm.env_loader import load_env_files

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / '.env'
            # Include UTF-8 BOM, comments, export prefix, and quotes
            env_file.write_bytes(
                '\ufeff# Comment line\n'
                'DEEPSEEK_API_KEY = "sk-from-dotenv"\n'
                'export EXTRA_TEST_VAR=\'hello\'\n'.encode('utf-8')
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_files([env_file])
                self.assertEqual(os.environ.get('DEEPSEEK_API_KEY'), 'sk-from-dotenv')
                self.assertEqual(os.environ.get('EXTRA_TEST_VAR'), 'hello')

    def test_existing_environ_takes_precedence_over_dotenv(self):
        import os
        import tempfile
        from laptop.llm.env_loader import load_env_files

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / '.env'
            env_file.write_text('DEEPSEEK_API_KEY=sk-from-dotenv\n', encoding='utf-8')
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'sk-from-shell'}, clear=True):
                load_env_files([env_file])
                self.assertEqual(os.environ.get('DEEPSEEK_API_KEY'), 'sk-from-shell')


if __name__ == '__main__':
    unittest.main()
