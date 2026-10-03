"""Exercise the actual grading function with external model/retrieval fixtures."""
import ast
import builtins
import json
from pathlib import Path
import re
import secrets
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_documents import DOCS, ROOT


class AssessmentContract(unittest.TestCase):
    def run_assessment(self, *, bad_document=False, bad_retriever=False, bad_generator=False, stream_error=False):
        # Compile the real gate without starting Gradio or requiring UI dependencies.
        source = ast.parse((ROOT / 'frontend/frontend_block.py').read_text())
        functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'rag_eval', 'add_message', 'assert_docs'}]
        document = DOCS.Document
        shared = ' '.join(f'overlap{i}' for i in range(30))
        documents = [document(page_content=f'Paper header section {i} {shared} unique conclusion {i}')
                     for i in range(12)]
        if bad_document:
            documents[3].page_content += '\x03'

        class Retrieval:
            def invoke(self, query):
                if bad_retriever:
                    return []
                # Full chunks identify their source. A shortened shared overlap
                # legitimately retrieves an adjacent chunk instead.
                return [next((d for d in documents if d.page_content == query), documents[0])]

        class Generation:
            def invoke(self, inputs):
                return 'ignores context' if bad_generator else inputs['context']

        class Stream:
            def __init__(self, text):
                self.text = text

            def stream(self, inputs):
                if stream_error:
                    raise RuntimeError('External provider unavailable')
                yield self.text

        class Prompt:
            def __init__(self, text):
                self.text = text

            def __or__(self, other):
                return Stream(self.text)

        with tempfile.TemporaryDirectory() as directory:
            passed = Path(directory) / 'PASSED'
            passed.touch()  # A failed rerun must remove stale credit.
            environment = {
                'os': SimpleNamespace(remove=lambda path: passed.unlink()),
                'open': lambda path, mode: builtins.open(passed, mode),
                're': re, 'secrets': secrets, 'Document': document,
                'Globals': SimpleNamespace(doc_chunks=[], doc_names=set()),
                'FAISS': SimpleNamespace(load_local=lambda *a, **k: SimpleNamespace(
                    docstore=SimpleNamespace(_dict=dict(enumerate(documents))))),
                'validate_documents': DOCS.validate_documents,
                'chains_dict': {'retriever': Retrieval(), 'generator': Generation()},
                'basic_chain': object(), 'rag_chain': Stream('A grounded answer.'),
                'get_synth_prompt': lambda *a: Prompt('Question: What does this document explain?\n\nAnswer: Its conclusion.'),
                'get_eval_prompt': lambda: Prompt('The baseline is preferable. Final score: [1]'),
                'logger': Mock(), 'gr': SimpleNamespace(Warning=lambda *a: None),
                'get_traceback': str,
            }
            exec(compile(ast.Module(body=functions, type_ignores=[]), 'frontend-grading', 'exec'), environment)
            history = []
            for history in environment['rag_eval'](history, 'RAG'):
                pass
            return passed.exists(), history

    def test_complete_key_contract_passes_with_overlapping_chunks(self):
        passed, history = self.run_assessment()
        self.assertTrue(passed)
        self.assertEqual(sum('Preference probe:' in r['content'] for r in history), 8)

    def test_failures_revoke_stale_credit(self):
        for case, message in [('bad_document', 'Unreadable document text'),
                              ('bad_retriever', 'Retriever check failed'),
                              ('bad_generator', 'Generator check failed'),
                              ('stream_error', 'External provider unavailable')]:
            with self.subTest(case=case):
                passed, history = self.run_assessment(**{case: True})
                self.assertFalse(passed)
                self.assertIn(message, str(history))

    def test_separate_key_does_not_enter_solution_projection(self):
        graph = json.loads((ROOT / 'composer/translation_web.json').read_text())
        for path in ROOT.rglob('*.ipynb'):
            notebook = json.loads(path.read_text())
            if notebook.get('metadata', {}).get('course_role') == 'assessment_solution':
                self.assertNotIn(path.relative_to(ROOT).as_posix(), graph['notebooks'])
                self.assertTrue(any('%%writefile' in ''.join(c['source']) for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
