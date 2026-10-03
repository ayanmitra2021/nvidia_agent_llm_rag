"""Fail on unreviewed runtime drift, including newly added or renamed modules."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'SOURCE.json').read_text())
    expected = {row['path']: row['sha256'] for row in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate runtime provenance path')
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
              if p.is_file() and p.name != 'SOURCE.json'}
    if actual != expected:
        raise ValueError('Runtime files changed; review and refresh SOURCE.json')
    return len(actual)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'services/model'
    print(f'Runtime provenance verified: {verify(root)} files')
