"""Entry contracts: locale state, scoped projection, backups, and delivery sources."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
import zipfile

TASK = Path(__file__).resolve().parents[1]
COURSE = TASK / ('dli' if (TASK / 'dli').is_dir() else 'notebooks')
spec = importlib.util.spec_from_file_location('course_language_server', COURSE / 'composer/runtime/course_language_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = json.loads((COURSE / 'composer/translation_web.json').read_text())
        shutil.copytree(COURSE / 'composer', self.root / 'composer', ignore=shutil.ignore_patterns('microservices', 'runtime', '__pycache__', 'deploy'))
        for name in self.graph['notebooks']:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(COURSE / name, target)
        self.controller = server.CourseLanguage(self.root)

    def test_every_language_and_existing_mode(self):
        extra = self.root / 'my-new-notebook.ipynb'
        extra.write_text('{"cells": [], "metadata": {"learner": true}}')
        for language in self.graph['languages']:
            for mode in self.graph['modes']:
                state = self.controller.project('', language, mode)
                self.assertEqual(state['language'], language)
                self.assertEqual(state['mode'], mode)
                with zipfile.ZipFile(self.root / state['backup']) as archive:
                    self.assertEqual(set(archive.namelist()), set(self.graph['notebooks']))
                for name, definition in self.graph['notebooks'].items():
                    notebook = json.loads((self.root / name).read_text())
                    self.assertEqual(state['projected_metadata'][name], {key: value for key, value in notebook.get('metadata', {}).items() if key in ('course_language', 'course_mode')})
                    if definition.get('role') == 'assessment':
                        self.assertNotEqual(notebook.get('metadata', {}).get('course_mode'), 'solutions')
        self.assertIn('"learner": true', extra.read_text())

    def test_rollback_retains_exact_learner_bytes(self):
        name = self.graph['entrypoint']
        original = (self.root / name).read_bytes()
        def fail(*args, **kwargs):
            (self.root / name).write_text('partial projection')
            raise RuntimeError('failure')
        with patch.object(server.subprocess, 'run', side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.controller.project('', self.graph['languages'][0], 'exercises')
        self.assertEqual((self.root / name).read_bytes(), original)

    def test_novel_registered_notebook_and_missing_file(self):
        name = 'new-module/new-example.ipynb'
        self.graph['notebooks'][name] = {'cells': []}
        (self.root / name).parent.mkdir()
        (self.root / name).write_text('{"cells": [], "metadata": {}}')
        (self.root / 'composer/translation_web.json').write_text(json.dumps(self.graph))
        self.assertIn(name, self.controller.inspect('')[2]['notebooks'])
        self.controller.project('', self.graph['languages'][0], 'exercises')
        (self.root / name).unlink()
        with self.assertRaises(Exception):
            self.controller.project('', self.graph['languages'][0], 'exercises')

    def test_traversal_and_symlink_escape(self):
        for value in ('../escape', '/absolute', 'bad\\path'):
            with self.assertRaises(ValueError):
                self.controller.inspect(value)
        target = self.root / self.graph['entrypoint']
        target.unlink()
        target.symlink_to('/etc/hosts')
        with self.assertRaises(ValueError):
            self.controller.inspect('')

    def test_production_startup_enables_native_language_widget(self):
        entrypoint = TASK / 'entrypoint.sh' if (TASK / 'entrypoint.sh').is_file() else COURSE / 'composer/entrypoint.sh'
        source = entrypoint.read_text()
        # Server config fragments do not configure the LabApp browser global.
        self.assertIn('--LabApp.expose_app_in_browser=True', source)

    def test_dockerfile_visible_and_source_identical(self):
        actual = COURSE / 'composer/Dockerfile'
        self.assertTrue(actual.is_file())
        if (TASK / 'Dockerfile').is_file():
            self.assertEqual(actual.read_bytes(), (TASK / 'Dockerfile').read_bytes())
        ignore = (TASK / '.dockerignore').read_text()
        self.assertNotIn('Dockerfile', [line.strip() for line in ignore.splitlines() if not line.startswith('#')])


class CollaborationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_documents_refresh_before_completion(self):
        notified = []
        class Loaders(dict):
            file_id_manager = SimpleNamespace(get_id=lambda path: {'nested/renamed.ipynb': 'known', 'unopened.ipynb': 'unopened'}.get(path))
        async def notify():
            notified.append('refreshed')
        loaders = Loaders(known=SimpleNamespace(maybe_notify=notify))
        app = SimpleNamespace(disable_rtc=False, file_loaders=loaders)
        serverapp = SimpleNamespace(extension_manager=SimpleNamespace(extension_apps={'jupyter_server_ydoc': [app]}))
        controller = server.CourseLanguage('.', serverapp)
        await controller.refresh_collaboration(['nested/renamed.ipynb', 'unopened.ipynb', 'missing.ipynb'])
        self.assertEqual(notified, ['refreshed'])
        self.assertEqual(set(loaders), {'known'})
        loaders['known'].maybe_notify = AsyncMock(side_effect=RuntimeError('cannot synchronize'))
        with self.assertRaisesRegex(RuntimeError, 'cannot synchronize'):
            await controller.refresh_collaboration(['nested/renamed.ipynb'])

    async def test_refresh_failure_does_not_claim_projection_rollback(self):
        result = {'notebooks': ['lesson.ipynb'], 'language': 'en', 'backup': 'course-backups/saved.zip'}
        controller = SimpleNamespace(lock=asyncio.Lock(), finished=0,
            inspect=lambda course: (None, None, {'notebooks': []}),
            project=Mock(return_value=result),
            refresh_collaboration=AsyncMock(side_effect=RuntimeError('refresh failed')))
        handler = SimpleNamespace(controller=controller,
            get_json_body=lambda: {'course': '', 'language': 'en', 'mode': 'exercises'},
            session_manager=SimpleNamespace(list_sessions=AsyncMock(return_value=[])),
            set_status=Mock(), finish=Mock())
        await server.LanguageHandler.post.__wrapped__(handler)
        handler.set_status.assert_called_once_with(500)
        response = handler.finish.call_args.args[0]
        self.assertTrue(response['applied'])
        self.assertEqual(response['backup'], result['backup'])
        self.assertNotIn('restored', response['message'])

    async def test_absent_or_disabled_collaboration_is_supported(self):
        await server.CourseLanguage('.').refresh_collaboration(['new.ipynb'])
        for apps in ({}, {'jupyter_server_ydoc': [SimpleNamespace(disable_rtc=True)]}):
            serverapp = SimpleNamespace(extension_manager=SimpleNamespace(extension_apps=apps))
            await server.CourseLanguage('.', serverapp).refresh_collaboration(['new.ipynb'])


if __name__ == '__main__':
    unittest.main()
