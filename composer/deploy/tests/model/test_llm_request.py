"""The replay verifier and facade share only explicit request normalization."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services/model'))
from llm_request import facade_chat_request


class FacadeRequestTests(unittest.TestCase):
    def test_guidance_is_portable_and_conflicts_fail(self):
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        for field in ({'guided_json': schema}, {'nvext': {'guided_json': schema}}):
            original = copy.deepcopy(field)
            normalized = facade_chat_request({'model': 'arbitrary/model', **field})
            self.assertEqual(field, original)
            self.assertEqual(normalized['response_format']['json_schema']['schema'], schema)
            self.assertNotIn('guided_json', normalized)
            self.assertNotIn('nvext', normalized)
        for fields in ({'nvext': 'bad'}, {'guided_json': 'bad'},
                       {'guided_json': schema, 'nvext': {'guided_json': schema}},
                       {'guided_json': schema, 'response_format': {'type': 'json_object'}}):
            with self.assertRaises(ValueError):
                facade_chat_request(fields)

    def test_sdk_defaults_and_explicit_reasoning(self):
        body = {'model': 'nvidia/nemotron-3.5-lightning-30b-a3b',
                'messages': [{'role': 'user', 'content': 'hello', 'cache_breakpoint': True}],
                'max_tokens': 1024, 'max_completion_tokens': 64, 'stop': []}
        before = copy.deepcopy(body)
        normalized = facade_chat_request(body)
        self.assertEqual(body, before)
        self.assertEqual(normalized['max_completion_tokens'], 64)
        self.assertNotIn('max_tokens', normalized)
        self.assertNotIn('stop', normalized)
        self.assertEqual(normalized['messages'], body['messages'])
        self.assertEqual(normalized['chat_template_kwargs'], {'enable_thinking': False})
        explicit = {**body, 'chat_template_kwargs': {'enable_thinking': True}, 'stop': ['end']}
        self.assertEqual(facade_chat_request(explicit)['chat_template_kwargs'], {'enable_thinking': True})
        self.assertEqual(facade_chat_request(explicit)['stop'], ['end'])
        self.assertNotIn('chat_template_kwargs', facade_chat_request({**body, 'model': 'another-model'}))
