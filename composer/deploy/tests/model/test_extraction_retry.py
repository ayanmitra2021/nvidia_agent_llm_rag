"""Structured extraction must not retry the same malformed cached answer."""
import ast
import copy
import json
from pathlib import Path
import unittest

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableAssign
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'


class Summary(BaseModel):
    answer: str


class ExtractionRetry(unittest.TestCase):
    def test_every_extractor_repairs_once_with_a_distinct_request(self):
        graph = json.loads((ROOT / 'composer/translation_web.json').read_text())
        found = 0
        for name, entry in graph['notebooks'].items():
            for cell in entry['cells']:
                source = cell.get('content')
                if not isinstance(source, str) or 'def RExtract(' not in source:
                    continue
                found += 1
                tree = ast.parse(source)
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'RExtract')
                scope = {'PydanticOutputParser': PydanticOutputParser,
                         'RunnableAssign': RunnableAssign, 'OutputParserException': OutputParserException}
                exec(compile(ast.Module(body=[function], type_ignores=[]), name, 'exec'), scope)
                for always_invalid in (False, True):
                    calls = []
                    def respond(prompt):
                        calls.append(prompt.to_string())
                        return '{"answer":"supported"}\n}' if len(calls) == 1 or always_invalid else '{"answer":"supported"}'
                    prompt = ChatPromptTemplate.from_messages([('user', '{input}\n{format_instructions}')])
                    chain = scope['RExtract'](Summary, RunnableLambda(respond), prompt)
                    data = {'input': 'Keep this original document.'}
                    before = copy.deepcopy(data)
                    with self.subTest(notebook=name, invalid=always_invalid):
                        if always_invalid:
                            with self.assertRaises(OutputParserException):
                                chain.invoke(data)
                        else:
                            self.assertEqual(chain.invoke(data).answer, 'supported')
                        self.assertEqual(data, before)
                        self.assertEqual(len(calls), 2)
                        self.assertNotEqual(calls[0], calls[1])
                        self.assertTrue(all(data['input'] in call for call in calls))
        self.assertGreater(found, 0)
