"""Test-only exact-record replay through the real ASGI completion boundary."""
import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile


def sample_index_for_record(cache, record):
    first = cache.get(record['request'], sample=True, sample_index=0)
    if first is None:
        raise ValueError('Recorded request is unreachable: ' + record['id'])
    for index in range(first[1]['sample_count']):
        hit = first if index == 0 else cache.get(record['request'], sample=True, sample_index=index)
        if hit is not None and hit[1]['entry_id'] == record['id']:
            if hit[0] != record['response']:
                raise ValueError('Recorded response changed: ' + record['id'])
            return index
    raise ValueError('Recorded entry is unreachable in its sampling pool: ' + record['id'])


async def replay_records(client, records, cache):
    failures = []
    for record in records:
        try:
            index = sample_index_for_record(cache, record)
            body = {**record['request'], 'cache_sample': True, 'cache_sample_index': index, 'cache_pacing': 'off'}
            response = await client.post('/v1/chat/completions', json=body)
            if response.status_code != 200 or response.headers.get('x-dli-cache') != 'BUNDLE_HIT':
                failures.append({'key': record['key'], 'entry_id': record['id'], 'status': response.status_code})
                continue
            if body.get('stream'):
                actual = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith('data: ') and line != 'data: [DONE]']
                metadata = [chunk.pop('dli_cache', None) for chunk in actual]
                if not response.text.rstrip().endswith('data: [DONE]'):
                    failures.append({'key': record['key'], 'entry_id': record['id'], 'error': 'missing DONE'})
            else:
                actual = response.json()
                metadata = [actual.pop('dli_cache', None)]
            if actual != record['response'] or not metadata or any(
                    not isinstance(item, dict) or item.get('entry_id') != record['id']
                    or item.get('upstream_request') is not False for item in metadata):
                failures.append({'key': record['key'], 'entry_id': record['id'], 'error': 'record or provenance mismatch'})
        except (ValueError, KeyError, TypeError) as error:
            failures.append({'key': record['key'], 'entry_id': record['id'], 'error': str(error)})
    return failures


async def verify_bundle(bundle, runtime):
    sys.path.insert(0, str(runtime))
    import httpx
    from fastapi.responses import JSONResponse
    from llm_course_cache import CourseCache
    with tempfile.TemporaryDirectory() as directory:
        os.environ.update(DLI_LLM_PROVIDERS=json.dumps([{'base_url': 'http://127.0.0.1:19101/v1',
            'api_key_env': 'COURSE_REPLAY_TEST_KEY'}]), COURSE_REPLAY_TEST_KEY='synthetic-test-only',
            LLM_STORE_PATH=directory + '/empty.sqlite', LLM_COURSE_CACHE_PATH=str(bundle),
            LLM_COURSE_CACHE_ENABLED='true', LLM_CACHE_PACING='off')
        import litellm_service as service
        upstream = []
        async def tripwire(*args, **kwargs):
            upstream.append(True)
            return JSONResponse({'test': 'unexpected live fallback'}, status_code=599)
        service.perform_completion = tripwire
        records = json.loads(gzip.decompress(bundle.read_bytes()))['entries']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url='http://test',
                headers={'x-dli-base-url': 'http://127.0.0.1:19101/v1'}) as client:
            failures = await replay_records(client, records, CourseCache(bundle))
        return {'entries': len(records), 'covered_upstream_calls': len(upstream), 'failures': failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(verify_bundle(args.bundle, args.runtime))
    print(json.dumps(report, indent=2))
    raise SystemExit(int(bool(report['failures'] or report['covered_upstream_calls'])))
