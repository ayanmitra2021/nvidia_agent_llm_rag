"""Exercise built CPU images through public APIs, with isolated external fixtures."""
import gzip
import json
from pathlib import Path
import time

import httpx

BASE = 'http://127.0.0.1:9000'
with httpx.Client(timeout=45, trust_env=False) as client:
    for _ in range(120):
        try:
            if client.get(BASE + '/health').status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    else:
        raise AssertionError('Built facade did not become ready')
    models = client.get(BASE + '/v1/models?scope=all').json()['data']
    assert {'course/embedding', 'course/reranker'}.issubset({m['id'] for m in models})
    assert client.get(BASE + '/v1/models/course/embedding').json()['model_type'] == 'embedding'
    for input_type in ('query', 'passage'):
        response = client.post(BASE + '/v1/embeddings', json={'model': 'course/embedding',
            'input': ['retrieval augmented generation', 'unrelated weather report'], 'input_type': input_type})
        response.raise_for_status()
        vectors = response.json()['data']
        assert len(vectors) == 2 and all(len(item['embedding']) == 384 for item in vectors)
        assert vectors[0]['embedding'] != vectors[1]['embedding']
        assert response.headers['x-dli-route'] == 'local'
    response = client.post(BASE + '/v1/ranking', json={'model': 'course/reranker',
        'query': {'text': 'retrieval augmented generation'},
        'passages': [{'text': 'RAG retrieves context before generation.'}, {'text': 'A recipe for apple pie.'}]})
    response.raise_for_status()
    assert [item['index'] for item in response.json()['rankings']] == [0, 1]
    assert response.headers['x-dli-inference-executed'] == 'true'
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
    from langchain_core.documents import Document
    embedder = NVIDIAEmbeddings(model='course/embedding', base_url=BASE + '/v1', api_key='synthetic-course')
    assert len(embedder.embed_query('retrieval augmented generation')) == 384
    assert len(embedder.embed_documents(['RAG retrieves context.', 'Apple pie recipe.'])) == 2
    reranker = NVIDIARerank(model='course/reranker', base_url=BASE + '/v1', api_key='synthetic-course', top_n=1)
    ranked = reranker.compress_documents([Document(page_content='RAG retrieves context before generation.'),
        Document(page_content='A recipe for apple pie.')], 'retrieval augmented generation')
    assert len(ranked) == 1 and ranked[0].page_content.startswith('RAG')
    document = json.loads(gzip.decompress(Path('/recordings/responses.json.gz').read_bytes()))
    records = [entry for entry in document['entries'] if 'import' in entry]
    assert records
    for entry in records:
        response = client.post(BASE + '/v1/chat/completions', json={**entry['request'], 'cache_pacing': 'off'})
        assert response.status_code == 200, (entry['id'], response.status_code)
        assert response.headers['x-dli-cache'] == 'BUNDLE_HIT', entry['id']
        assert response.headers['x-dli-inference-executed'] == 'false'
        if entry['request'].get('stream'):
            assert response.text.rstrip().endswith('data: [DONE]')
            assert '"error"' not in response.text
    assert client.get('http://127.0.0.1:9101/counts').json() == {'primary': 0, 'fallback': 0}
    body = {'model': records[0]['request']['model'], 'messages': [{'role': 'user', 'content': 'uncached native fixture'}],
            'use_cache': False}
    response = client.post(BASE + '/v1/chat/completions', json=body)
    response.raise_for_status()
    assert response.headers['x-dli-cache'] == 'BYPASS'
    assert response.headers['x-dli-fallback'] == 'true'
    assert response.headers['x-dli-inference-executed'] == 'true'
    assert response.json()['choices'][0]['message']['content'] == 'Synthetic fallback response'
    counts = client.get('http://127.0.0.1:9101/counts').json()
    assert counts['primary'] > 0 and counts['fallback'] == 1, counts
    print(json.dumps({'recordings_replayed': len(records), 'cached_provider_calls': 0,
        'retrieval': 'CPU embedding and reranking passed', 'bypass_fallback': 'passed', 'provider_fixture_calls': counts}))
