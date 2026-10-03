"""RAG-specific boundaries complement the shared provider regression suite."""
import asyncio
import copy
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
from test_llm_course_cache import CourseCache
from llm_request import facade_chat_request
import local_retrieval

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / 'task1/services/model'


class RetrievalInputTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, body, operation):
        class Request:
            async def json(self):
                return body
        return await getattr(local_retrieval, operation)(Request())

    async def test_cold_models_report_unavailable(self):
        with patch.object(local_retrieval, 'READY', False):
            for operation in ('embeddings', 'ranking'):
                with self.assertRaises(Exception) as caught:
                    await self.request({}, operation)
                self.assertEqual(caught.exception.status_code, 503)

    async def test_malformed_inputs_fail_before_model_execution(self):
        bodies = {'embeddings': [None, [], {}, {'input': ''}, {'input': [None]},
                  {'input': 'hello', 'dimensions': 128}, {'input': 'hello', 'encoding_format': 'base64'}],
                  'ranking': [None, [], {}, {'query': None}, {'query': 'string'},
                  {'query': {'text': 'hello'}, 'passages': [None]},
                  {'query': {'text': 'hello'}, 'passages': [{'text': 'ok'}] * 65}]}
        with patch.object(local_retrieval, 'READY', True):
            for operation, cases in bodies.items():
                for body in cases:
                    with self.subTest(operation=operation, body=body), self.assertRaises(Exception) as caught:
                        await self.request(body, operation)
                    self.assertEqual(caught.exception.status_code, 400)


class RAGDeploymentTests(unittest.TestCase):
    def test_declared_production_references_activate_managed_fallback(self):
        references = dict(line.split('=', 1) for line in (ROOT / 'secrets.env').read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith('#') and '=' in line)
        required = {'NVIDIA_API_KEY', 'NEBIUS_TOKEN_FACTORY_KEY', 'NEBIUS_BASE_URL'}
        self.assertTrue(required <= references.keys(), 'Missing production provider references')
        for name in required:
            self.assertTrue(bool(re.fullmatch(r'[A-Za-z0-9_./-]+:[A-Za-z0-9_/-]+', references[name])),
                            'Production providers must use managed secret references')
        deployment = yaml.safe_load((ROOT / 'task1/docker-compose.deployment.yml').read_text())
        self.assertEqual(deployment['services']['litellm_wrapper']['env_file'], ['/dli/secrets/.env'])
        program = '''
import runpy
runpy.run_path('litellm_wrapper.py', run_name='configuration_test')
import litellm_service as service
from provider_policy import model_allowed
assert service.FALLBACK_URL == 'https://fallback.invalid/v1'
provider = service.PROVIDERS[service.FALLBACK_URL]
assert provider[1]['nvidia/nemotron-3.5-lightning-30b-a3b'] == 'nvidia/Nemotron-3_5-Lightning'
assert model_allowed(provider, 'nvidia/nemotron-3.5-lightning-30b-a3b')
assert not model_allowed(provider, 'unregistered/vision-model')
'''
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                env = {name: 'synthetic-provider-key' for name in required}
                env.update(PATH=os.defpath, NEBIUS_BASE_URL='https://fallback.invalid/v1',
                           LLM_STORE_PATH=str(Path(directory) / 'requests.sqlite'))
                if legacy:
                    env['NEBUIS_TOKEN_FACTORY_KEY'] = env.pop('NEBIUS_TOKEN_FACTORY_KEY')
                result = subprocess.run([sys.executable, '-c', program], cwd=RUNTIME, env=env,
                                        capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, 'Managed fallback configuration failed')
                self.assertNotIn(b'synthetic-provider-key', result.stdout + result.stderr)

    def test_release_requires_runtime_checks_before_build_and_publication(self):
        config = yaml.safe_load((ROOT / '.gitlab-ci.yml').read_text())
        checks = {name for name in config if name.startswith('verify ')} | {'verify tasks'}
        for job in ('build tasks', '.publish-course'):
            self.assertTrue(checks.issubset(set(config[job]['needs'])))
        self.assertEqual(config['build tasks']['rules'][0], {'if': '$CI_COMMIT_TAG', 'when': 'on_success'})
        self.assertEqual(config['stage to staging']['rules'], [{'if': '$CI_COMMIT_TAG', 'when': 'on_success'}])
        self.assertTrue({'stage to staging', 'sync staging to china'}.issubset(config['publish to production']['needs']))
        self.assertNotIn('rules', config['publish to production'])

    def test_every_discovered_build_context_has_its_dockerfile(self):
        root = (ROOT / 'task1').resolve()
        config = yaml.safe_load((root / 'docker-compose.override.yml').read_text())
        for name, service in config['services'].items():
            build = service.get('build')
            if build is None:
                continue
            context = (root / (build['context'] if isinstance(build, dict) else build)).resolve()
            self.assertTrue(context.is_relative_to(root), name)
            dockerfile = context / (build.get('dockerfile', 'Dockerfile') if isinstance(build, dict) else 'Dockerfile')
            self.assertTrue(dockerfile.is_file(), str(dockerfile))

    def test_credentials_only_reach_private_wrapper(self):
        for filename in ('docker-compose.override.yml', 'docker-compose.deployment.yml'):
            services = yaml.safe_load((ROOT / 'task1' / filename).read_text())['services']
            self.assertIn('env_file', services['litellm_wrapper'])
            self.assertNotIn('env_file', services['llm_client'])
            self.assertNotIn('ports', services['litellm_wrapper'])
        services = yaml.safe_load((ROOT / 'task1/docker-compose.yml').read_text())['services']
        self.assertNotIn('ports', services['litellm_wrapper'])
        self.assertEqual(services['llm_client']['depends_on']['litellm_wrapper']['condition'], 'service_healthy')
        self.assertIn('LLM_LOCAL_RETRIEVAL=true', services['llm_client']['environment'])
        self.assertFalse(any('API_KEY' in line for line in services['llm_client']['environment']))

    def test_reviewed_rag_records_are_present_without_relabeling_ultra(self):
        cache = CourseCache()
        cache._load()
        document = json.loads(gzip.decompress(cache.path.read_bytes()))
        imported = [entry for entry in document['entries'] if 'import' in entry]
        self.assertTrue(imported)
        self.assertTrue(any(entry['import']['review'] == 'intentional_failure' for entry in imported))
        for entry in imported:
            self.assertEqual(entry['request'], facade_chat_request(entry['provider_request'], entry['request']['model']))
            self.assertEqual(entry['request']['model'], 'nvidia/nemotron-3.5-lightning-30b-a3b')
            self.assertTrue(cache.get(entry['request']))
            if entry['import']['review'] == 'intentional_failure':
                self.assertTrue(entry['import']['pedagogic_evidence'])
        self.assertEqual(len(document['provenance']['rag_import']['records']), len(imported))

    def test_discovery_outage_keeps_recordings_available_only_in_course_mode(self):
        program = '''
import asyncio, json
from unittest.mock import patch
import httpx
import litellm_service as service
async def run():
    async def unavailable(*args, **kwargs):
        raise httpx.ConnectError('synthetic unavailable')
    state = service.runtime()
    with patch.object(httpx.AsyncClient, 'get', unavailable):
        catalog = await service.fetch_catalog(state, state.base_url)
        assert catalog and all(x['dli_catalog_source'] == 'recorded' for x in catalog.values())
        for mode, enabled in [('custom', True), ('course', False)]:
            state.mode, state.COURSE_CACHE_ENABLED = mode, enabled
            try:
                await service.fetch_catalog(state, state.base_url)
            except httpx.ConnectError:
                pass
            else:
                raise AssertionError('Recorded catalog leaked outside course replay')
asyncio.run(run())
'''
        with tempfile.TemporaryDirectory() as directory:
            env = {'PATH': os.environ['PATH'], 'HOME': directory, 'PYTHONDONTWRITEBYTECODE': '1',
                   'LLM_STORE_PATH': directory + '/cache.sqlite', 'TEST_KEY': 'synthetic-test-only',
                   'DLI_LLM_PROVIDERS': json.dumps([{'base_url': 'http://127.0.0.1:19101/v1', 'api_key_env': 'TEST_KEY'}])}
            result = subprocess.run([sys.executable, '-c', program], cwd=RUNTIME, env=env,
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ProvenanceTests(unittest.TestCase):
    def test_discovered_runtime_and_unknown_file_mutations(self):
        sys.path.insert(0, str(ROOT / 'task1/scripts'))
        from verify_model_runtime import verify
        self.assertGreater(verify(RUNTIME), 0)
        for mutation in ('new', 'deleted', 'renamed', 'malformed', 'duplicate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'novel.py').write_text('x = 1\n')
                manifest = {'files': [{'path': 'novel.py', 'sha256': hashlib.sha256(b'x = 1\n').hexdigest()}]}
                if mutation == 'new':
                    (root / 'unexpected.py').write_text('x = 2\n')
                elif mutation == 'deleted':
                    (root / 'novel.py').unlink()
                elif mutation == 'renamed':
                    (root / 'novel.py').rename(root / 'renamed.py')
                elif mutation == 'malformed':
                    manifest['files'][0]['sha256'] = 'near-match'
                else:
                    manifest['files'].append(manifest['files'][0])
                (root / 'SOURCE.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    verify(root)

if __name__ == '__main__':
    unittest.main()
