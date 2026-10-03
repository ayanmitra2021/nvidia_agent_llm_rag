"""Source text must be readable before it enters a RAG index or grader."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SPEC = importlib.util.spec_from_file_location('course_documents', ROOT / 'composer/documents.py')
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


class DocumentQuality(unittest.TestCase):
    def test_arxiv_uses_checked_pdf_loading_and_preserves_paper_metadata(self):
        paper = SimpleNamespace(
            get_short_id=lambda: '9901.12345v2', title='A selected paper',
            summary='Its abstract', authors=['Author One'],
            published=datetime(2026, 1, 2), entry_id='https://arxiv.org/abs/9901.12345v2',
            download_pdf=Mock(),
        )
        client = SimpleNamespace(results=lambda search: iter([paper]))
        pages = [DOCS.Document(page_content='First page'), DOCS.Document(page_content='Second page')]
        with tempfile.TemporaryDirectory() as temporary:
            bundled = Path(temporary) / '9901.12345v2.pdf'
            bundled.touch()
            with patch('arxiv.Client', return_value=client), patch.object(DOCS, 'load_pdf', return_value=pages) as loader:
                document = DOCS.load_arxiv(['9901.12345'], bundled_dir=temporary)[0]
                loader.assert_called_once_with(bundled)
                paper.download_pdf.assert_not_called()
                self.assertEqual(document.page_content, 'First page\n\nSecond page')
                self.assertEqual(document.metadata['Title'], paper.title)
                self.assertEqual(document.metadata['Summary'], paper.summary)
                self.assertEqual(document.metadata['entry_id'], paper.entry_id)
            bundled.unlink()
            paper.download_pdf.side_effect = lambda dirpath, filename: str(Path(dirpath) / filename)
            with patch('arxiv.Client', return_value=client), patch.object(DOCS, 'load_pdf', return_value=pages) as loader:
                DOCS.load_arxiv(['9901.12345'], bundled_dir=temporary)
                self.assertEqual(loader.call_args.args[0].name, 'paper.pdf')
                self.assertFalse(loader.call_args.args[0].parent.exists())
            with patch('arxiv.Client', return_value=client), patch.object(DOCS, 'load_pdf', side_effect=ValueError('Unreadable PDF')):
                with self.assertRaisesRegex(ValueError, 'Unreadable PDF'):
                    DOCS.load_arxiv(['9901.12345'], bundled_dir=temporary)

    def test_arxiv_missing_papers_and_empty_selection_fail_visibly(self):
        client = SimpleNamespace(results=lambda search: iter([]))
        with patch('arxiv.Client', return_value=client):
            with self.assertRaisesRegex(ValueError, 'No arXiv paper found'):
                DOCS.load_arxiv(['9901.99999'])
            with self.assertRaisesRegex(ValueError, 'No document text'):
                DOCS.load_arxiv([])

    def test_every_bundled_pdf_has_matching_readable_text(self):
        papers = list((ROOT / 'cached_papers').rglob('*.pdf'))
        self.assertTrue(papers)
        for pdf in papers:
            self.assertTrue(pdf.with_suffix('.text.json').is_file(), str(pdf))
            documents = DOCS.load_pdf(pdf)
            self.assertTrue(documents)
            self.assertTrue(all(d.metadata['pdf_sha256'] for d in documents))
            self.assertTrue(all(d.metadata.get('Title') and d.metadata.get('Summary') for d in documents))
            snapshot = json.loads(pdf.with_suffix('.text.json').read_text())
            review = json.loads(pdf.with_suffix('.review.json').read_text())
            self.assertEqual(review['pdf_sha256'], snapshot['pdf_sha256'])
            self.assertEqual(review['text_sha256'], snapshot['text_sha256'])
            repaired = {p['page'] for p in snapshot['pages'] if p['extraction'] != 'text'}
            self.assertEqual(repaired, {p['page'] for p in review['pages']})
            for page in review['pages']:
                for anchor in page['anchors']:
                    self.assertIn(anchor, snapshot['pages'][page['page']]['text'])

    def test_corruption_fails_before_indexing_or_grading(self):
        for text in ('7KRXJKW\x03\x14', '\ufffdtext', '\ue123text', '\x85text', ''):
            with self.subTest(text=repr(text)), self.assertRaisesRegex(ValueError, 'Unreadable'):
                DOCS.validate_documents([DOCS.Document(page_content=text)])
        with self.assertRaises(ValueError):
            DOCS.validate_documents([])
        DOCS.validate_documents([DOCS.Document(page_content='中文、日本語, naïve, ∑ x²\n\t[1]')])

    def test_renamed_changed_deleted_and_malformed_snapshots(self):
        original = next((ROOT / 'cached_papers').rglob('*.pdf'))
        with tempfile.TemporaryDirectory() as temporary:
            renamed = Path(temporary) / 'novel-paper.pdf'
            shutil.copyfile(original, renamed)
            snapshot = renamed.with_suffix('.text.json')
            good = json.loads(original.with_suffix('.text.json').read_text())
            snapshot.write_text(json.dumps(good))
            self.assertTrue(DOCS.load_pdf(renamed))
            for field in ('pdf_sha256', 'text_sha256', 'schema'):
                bad = copy.deepcopy(good)
                bad[field] = 'wrong'
                snapshot.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'snapshot does not match'):
                    DOCS.load_pdf(renamed)
            bad = copy.deepcopy(good)
            bad['pages'][0]['text'] += '\x03'
            bad['text_sha256'] = DOCS.text_digest(bad['pages'])
            snapshot.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Unreadable'):
                DOCS.load_pdf(renamed)
            for mutation in ('delete', 'duplicate', 'reorder'):
                bad = copy.deepcopy(good)
                if mutation == 'delete':
                    del bad['pages'][1]
                elif mutation == 'duplicate':
                    bad['pages'][1] = bad['pages'][0]
                else:
                    bad['pages'].reverse()
                bad['text_sha256'] = DOCS.text_digest(bad['pages'])
                snapshot.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'snapshot does not match'):
                    DOCS.load_pdf(renamed)
            snapshot.write_text('{broken')
            with self.assertRaises(ValueError):
                DOCS.load_pdf(renamed)
            snapshot.write_text(json.dumps(good))
            renamed.write_bytes(renamed.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'snapshot does not match'):
                DOCS.load_pdf(renamed)

    def test_quality_guards_cover_ingestion_save_reload_and_assessment(self):
        graph = json.loads((ROOT / 'composer/translation_web.json').read_text())
        for name, entry in graph['notebooks'].items():
            for cell in entry['cells']:
                if cell['cell_type'] != 'code':
                    continue
                for key in ('content', 'solution'):
                    source = cell.get(key, '')
                    if 'docstore.save_local(' in source or 'FAISS.load_local(' in source:
                        if entry.get('role') == 'assessment':
                            continue  # The unfinished assessment remains the learner's work.
                        self.assertIn('validate_documents(', source, (name, cell['id'], key))
        frontend = (ROOT / 'frontend/frontend_block.py').read_text()
        self.assertIn('validate_documents(list(docstore.docstore._dict.values()))', frontend)


if __name__ == '__main__':
    unittest.main()
