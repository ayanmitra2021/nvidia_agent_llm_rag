"""Regenerate text snapshots from PDFs; OCR is required for unmapped fonts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'notebooks'
sys.path.insert(0, str(ROOT))
from composer.documents import load_pdf, text_digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pdf', type=Path)
parser.add_argument('--output', type=Path, help='Defaults to a candidate; never overwrite reviewed text implicitly.')
args = parser.parse_args()
output = args.output or args.pdf.with_suffix('.candidate.text.json')
reviewed = args.pdf.with_suffix('.text.json')
previous = json.loads(reviewed.read_text()) if reviewed.exists() else {}
docs = load_pdf(args.pdf, prefer_snapshot=False)
pages = [{'page': d.metadata['page'], 'text': d.page_content, 'extraction': d.metadata['extraction']} for d in docs]
corrections = args.pdf.with_suffix('.corrections.json')
if corrections.is_file():
    corrected = json.loads(corrections.read_text())
    if corrected['pdf_sha256'] != hashlib.sha256(args.pdf.read_bytes()).hexdigest():
        raise ValueError('Text corrections belong to a different PDF version.')
    for correction in corrected['pages']:
        index = correction['page']
        pages[index] = {'page': index, 'text': correction['text'],
                        'extraction': previous['pages'][index]['extraction']}
result = {'schema': 'dli-pdf-text/1', 'pdf_sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
          'text_sha256': text_digest(pages), 'metadata': previous.get('metadata', {}), 'pages': pages,
          'provenance': previous.get('provenance', {})}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'pages': len(pages), 'ocr_pages': [p['page'] + 1 for p in pages if p['extraction'] == 'ocr']}))
