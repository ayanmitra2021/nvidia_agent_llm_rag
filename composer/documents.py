"""Readable PDF documents and checked, source-bound text snapshots."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unicodedata

from langchain_core.documents import Document


def text_digest(pages):
    return hashlib.sha256(json.dumps(pages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def unreadable_characters(text):
    return {c for c in text if c == '\ufffd' or (unicodedata.category(c) in {'Cc', 'Cs', 'Co'} and c not in '\t\n\r')}


def validate_documents(documents):
    """Reject empty text and invalid encoding; semantic/OCR review remains separate."""
    if not documents:
        raise ValueError('No document text was loaded.')
    for index, document in enumerate(documents):
        text = document.page_content
        bad = unreadable_characters(text)
        if not text.strip() or bad:
            source = document.metadata.get('source', document.metadata.get('Title', 'document'))
            page = document.metadata.get('page', index)
            raise ValueError(
                f'Unreadable document text in {source}, page/chunk {page}. '
                'Re-extract the PDF with OCR, then rebuild the index in Notebook 7. '
                'The existing index must not be used for evaluation.'
            )
    return documents


def load_arxiv(paper_ids, *, bundled_dir=None):
    """Load selected papers through the same checked PDF path as local documents."""
    import arxiv

    bundled_dir = Path(bundled_dir) if bundled_dir is not None else Path(__file__).resolve().parents[1] / 'cached_papers'
    client = arxiv.Client()
    documents = []
    for paper_id in paper_ids:
        paper = next(client.results(arxiv.Search(id_list=[paper_id])), None)
        if paper is None:
            raise ValueError(f'No arXiv paper found for {paper_id}.')
        bundled_pdf = bundled_dir / (paper.get_short_id().replace('/', '_') + '.pdf')
        with tempfile.TemporaryDirectory(prefix='course-paper-') as temporary:
            pdf = bundled_pdf if bundled_pdf.is_file() else Path(paper.download_pdf(dirpath=temporary, filename='paper.pdf'))
            pages = load_pdf(pdf)
        documents.append(Document(
            page_content='\n\n'.join(page.page_content for page in pages),
            metadata={
                'Title': paper.title, 'Summary': paper.summary,
                'Authors': ', '.join(str(author) for author in paper.authors),
                'Published': str(paper.published.date()), 'entry_id': paper.entry_id,
            },
        ))
    return validate_documents(documents)


def load_pdf(path, *, prefer_snapshot=True):
    """Use verified text when bundled; otherwise extract, using OCR only where needed."""
    path = Path(path)
    pdf_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot = path.with_suffix('.text.json')
    metadata = {}
    import pymupdf
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
    if prefer_snapshot and snapshot.is_file():
        saved = json.loads(snapshot.read_text(encoding='utf-8'))
        if (saved.get('schema') != 'dli-pdf-text/1' or saved.get('pdf_sha256') != pdf_sha
                or saved.get('text_sha256') != text_digest(saved.get('pages'))
                or not isinstance(saved.get('pages'), list)
                or [p.get('page') for p in saved['pages']] != list(range(page_count))):
            raise ValueError(f'PDF text snapshot does not match {path}. Rebuild the snapshot before use.')
        pages, metadata = saved['pages'], saved.get('metadata', {})
    else:
        pages = []
        with pymupdf.open(path) as pdf:
            for page in pdf:
                # Unknown glyph IDs must not masquerade as Unicode characters.
                text = page.get_text(flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE)
                method = 'text'
                if not text.strip() or unreadable_characters(text):
                    try:
                        pixels = page.get_pixmap(dpi=300, alpha=False)
                        result = subprocess.run(
                            ['tesseract', 'stdin', 'stdout', '-l', 'eng'],
                            input=pixels.tobytes('png'), capture_output=True, check=True,
                            timeout=30, env={**os.environ, 'OMP_THREAD_LIMIT': '1'},
                        )
                        text = result.stdout.decode('utf-8')
                    except (OSError, subprocess.SubprocessError) as error:
                        raise ValueError(f'OCR failed for {path}, page {page.number + 1}. Check tesseract-ocr and the 30-second page timeout before retrying.') from error
                    method = 'ocr'
                # Keep every page: an empty OCR result must fail validation visibly.
                pages.append({'page': page.number, 'text': text, 'extraction': method})
    documents = [Document(page_content=page['text'], metadata={
        **metadata, 'source': str(path), 'page': page['page'],
        'extraction': page['extraction'], 'pdf_sha256': pdf_sha,
    }) for page in pages]
    return validate_documents(documents)
