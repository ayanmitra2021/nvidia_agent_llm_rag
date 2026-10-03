"""Import reviewed RAG wire records without changing prompts or response content."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1] / 'services/model'
sys.path.insert(0, str(RUNTIME))
from llm_course_cache import CourseCache, package_records, response_model
from llm_request import facade_chat_request


def digest(data):
    return hashlib.sha256(data).hexdigest()


def import_records(source, seed, output):
    raw = source.read_bytes()
    manifest = json.loads(Path(str(source) + '.manifest.json').read_bytes())
    if digest(raw) != manifest['sha256']:
        raise ValueError('Source bundle checksum mismatch')
    document = json.loads(gzip.decompress(raw))
    if document['format'] != 'dli-model-cache/1' or document['provenance'] != manifest['provenance']:
        raise ValueError('Source bundle provenance mismatch')
    wire = {entry['key']: entry for entry in document['entries']}
    if len(wire) != len(document['entries']) or len(wire) != manifest['entries']:
        raise ValueError('Source bundle identities are not unique')
    seed_cache = CourseCache(seed)
    seed_cache._load()
    if seed_cache.unavailable:
        raise ValueError('Seed recordings unavailable')
    seed_document = json.loads(gzip.decompress(seed.read_bytes()))
    records = list(seed_document['entries'])
    imports, exclusions = [], []
    for record in document['provenance']['records']:
        sources = [item for item in record['sources'] if item['course'] == 'c-fx-15-v1']
        if not sources:
            continue
        if record['review']['disposition'] not in {'accept', 'intentional_failure'} or (
                record['review']['disposition'] == 'intentional_failure' and not record['review'].get('pedagogic_evidence')):
            raise ValueError('Unreviewed RAG recording')
        if record['request']['model'] != 'nvidia/nemotron-3.5-lightning-30b-a3b':
            exclusions.append({'source_key': record['key'], 'reason': 'Model is not used by this course revision',
                               'model': record['request']['model']})
            continue
        transport = wire[record['key']]
        payload = base64.b64decode(transport['body'], validate=True)
        if digest(payload) != transport['sha256']:
            raise ValueError('Source response checksum mismatch')
        if record['transport_admission']['actual_capture_endpoint'] != 'https://integrate.api.nvidia.com/v1/chat/completions':
            raise ValueError('Unexpected capture endpoint')
        if transport['content_type'] == 'text/event-stream':
            frames = [line[5:].strip() for line in payload.splitlines() if line.startswith(b'data:')]
            if frames.count(b'[DONE]') != 1 or frames[-1] != b'[DONE]':
                raise ValueError('Incomplete recorded stream')
            response = [json.loads(frame) for frame in frames[:-1]]
        elif transport['content_type'] == 'application/json':
            response = json.loads(payload)
        else:
            raise ValueError('Unknown response transport')
        # These are delivery annotations, not provider-generated model content.
        for chunk in response if isinstance(response, list) else [response]:
            chunk.pop('dli_cache', None)
            chunk.pop('dli_routing', None)
        provenance = {'source_key': record['key'], 'wire_response_sha256': transport['sha256'],
                      'sources': [{k: value[k] for k in ('course', 'notebook', 'cell_id', 'language', 'mode')}
                                  for value in sources], 'review': record['review']['disposition'],
                      'reason': record['review'].get('reason'),
                      'pedagogic_evidence': record['review'].get('pedagogic_evidence')}
        records.append({'request': facade_chat_request(record['request'], 'nvidia/nemotron-3.5-lightning-30b-a3b'), 'response': response,
                        'origin_schema': 'dli-course-origin-v2', 'origin_provider': 'nvidia',
                        'origin_model': response_model(response), 'provider_request': record['request'],
                        'recorded_at': 0, 'recorded_at_unknown': True, 'import': provenance})
        imports.append(provenance)
    if not imports:
        raise ValueError('No reviewed RAG recordings found')
    provenance = dict(seed_document['provenance'])
    provenance['rag_import'] = {'source_sha256': digest(raw), 'seed_sha256': digest(seed.read_bytes()),
                               'records': imports, 'excluded': exclusions,
                               'coverage': 'Exact recorded requests only; misses use the configured live provider and fallback.'}
    return package_records(records, output, provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = import_records(args.source, args.seed, args.output)
    print(json.dumps({k: result[k] for k in ('entries', 'distinct_requests', 'sha256', 'compressed_bytes')}))
