"""Authenticated, course-scoped notebook projection for the native language UI."""
import asyncio
import fcntl
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import uuid
import zipfile

from jupyter_server.base.handlers import APIHandler, JupyterHandler
from jupyter_server.utils import url_path_join
from tornado import web
from tornado.httpclient import AsyncHTTPClient

INTERFACE = 'jupyter-course-language-v1'


class CourseLanguage:
    def __init__(self, root, serverapp=None):
        self.serverapp = serverapp
        self.root = Path(root).resolve()
        self.lock = asyncio.Lock()
        self.finished = 0.0

    async def refresh_collaboration(self, paths):
        if self.serverapp is None:
            return
        apps = self.serverapp.extension_manager.extension_apps.get('jupyter_server_ydoc', ())
        for app in apps:
            if app.disable_rtc:
                continue
            for path in paths:
                file_id = app.file_loaders.file_id_manager.get_id(path)
                if file_id is not None and file_id in app.file_loaders:
                    await app.file_loaders[file_id].maybe_notify()

    def inspect(self, relative):
        if not isinstance(relative, str) or '\\' in relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
            raise ValueError('Invalid course path')
        root = (self.root / relative).resolve()
        if not root.is_relative_to(self.root):
            raise ValueError('Course path escapes the notebook workspace')
        for support in ('composer/translation_web.json', 'composer/switch_language.py'):
            if not (root / support).resolve().is_relative_to(root):
                raise ValueError('Course support path escapes the course')
        graph = json.loads((root / 'composer/translation_web.json').read_text())
        if graph.get('selection_interface') != INTERFACE:
            raise ValueError('This course has no registered native selector')
        names = list(graph['notebooks'])
        for name in names:
            path = PurePosixPath(name)
            if '\\' in name or path.is_absolute() or '..' in path.parts or not name.endswith('.ipynb') or not (root / name).resolve().is_relative_to(root):
                raise ValueError('Invalid course notebook path')
        if graph['entrypoint'] not in names:
            raise ValueError('Course entrypoint is not registered')
        selectors = graph.get('selector_entrypoints', [])
        actual = {name for name, entry in graph['notebooks'].items()
                  if any(cell.get('id') == 'course-language-switch' for cell in entry['cells'])}
        if not selectors or len(selectors) != len(set(selectors)) or set(selectors) != actual or not actual.issubset(names):
            raise ValueError('Course selector entrypoints do not match authored controls')
        current = json.loads((root / graph['entrypoint']).read_text()).get('metadata', {})
        return root, graph, {'languages': [{'value': language, 'label': graph.get('lang_names', {}).get(language, language)}
            for language in graph['languages']], 'language': current.get('course_language', graph['languages'][0]),
            'mode': current.get('course_mode', 'exercises'), 'modes': graph['modes'],
            'entrypoint': graph['entrypoint'], 'titles': graph['course_titles'],
            'locale_tags': graph['locale_tags'], 'notebooks': [str(PurePosixPath(relative) / name) for name in names]}

    def project(self, relative, language, mode):
        root, graph, _ = self.inspect(relative)
        if language not in graph['languages'] or mode not in graph['modes']:
            raise ValueError('Unknown language or practice mode')
        # Share the CLI lock before taking snapshots, including failure rollback.
        lock_path = root / '.course-mode.lock'
        if not lock_path.resolve().is_relative_to(root):
            raise ValueError('Course lock escapes the workspace')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._project_locked(root, graph, relative, language, mode)

    def _project_locked(self, root, graph, relative, language, mode):
        originals = {name: (root / name).read_bytes() for name in graph['notebooks']}
        backup = root / 'course-backups' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.zip')
        if not backup.parent.resolve().is_relative_to(root):
            raise ValueError('Backup directory escapes the course')
        backup.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in originals.items():
                archive.writestr(name, data)
        try:
            command = ('import sys; from composer.switch_language import _switch_language; '
                       'raise SystemExit(0 if _switch_language(sys.argv[1], mode=sys.argv[2], '
                       'preserve_extra_notebooks=True) else 1)')
            subprocess.run([sys.executable, '-c', command, language, mode],
                           cwd=root, check=True, timeout=60, capture_output=True)
        except Exception:
            for name, data in originals.items():
                (root / name).write_bytes(data)
            raise
        projected_metadata = {}
        for name in graph['notebooks']:
            metadata = json.loads((root / name).read_text()).get('metadata', {})
            projected_metadata[str(PurePosixPath(relative) / name)] = {
                key: metadata[key] for key in ('course_language', 'course_mode') if key in metadata}
        return {**self.inspect(relative)[2], 'backup': backup.relative_to(self.root).as_posix(),
                'projected_metadata': projected_metadata}


class LanguageHandler(APIHandler):
    def initialize(self, controller):
        self.controller = controller

    @web.authenticated
    def get(self):
        try:
            root, _, value = self.controller.inspect(self.get_argument('course', ''))
            config = json.loads((root / 'composer/course_home.json').read_text())
            value['messages'] = json.loads((root / 'composer/course_language_text.json').read_text())
            value['services'] = [{key: item[key] for key in ('id', 'href')} for item in config['services']]
            self.set_header('Cache-Control', 'no-store')
            self.finish(value)
        except (ValueError, OSError, KeyError) as error:
            raise web.HTTPError(400, 'Invalid course selection or notebook graph') from error

    @web.authenticated
    async def post(self):
        if self.controller.lock.locked() or time.monotonic() - self.controller.finished < 1:
            raise web.HTTPError(409, 'A language change is already running or cooling down')
        body = self.get_json_body()
        if not isinstance(body, dict):
            raise web.HTTPError(400, 'Expected a language selection')
        async with self.controller.lock:
            try:
                _, _, state = self.controller.inspect(body.get('course', ''))
                sessions = await self.session_manager.list_sessions()
                affected = [item for item in sessions if item.get('path') in state['notebooks']]
                if any(item['kernel']['execution_state'] == 'busy' for item in affected):
                    raise web.HTTPError(409, 'Wait for the course notebook to finish running before changing language.')
                if body.get('source') == 'foyer' and affected:
                    raise web.HTTPError(409, 'Use the language widget in your open course notebook to preserve unsaved edits')
                managed = body.get('sessions', [])
                if not isinstance(managed, list) or any(not isinstance(value, str) for value in managed):
                    raise web.HTTPError(400, 'Invalid notebook session list')
                if body.get('source') != 'foyer' and any(item['id'] not in managed or item['kernel'].get('connections', 0) > 1 for item in affected):
                    raise web.HTTPError(409, 'Save and close other course browser tabs, and shut down unused course kernels before switching language')
                result = await asyncio.to_thread(self.controller.project, body.get('course', ''), body.get('language'), body.get('mode'))
                try:
                    await self.controller.refresh_collaboration(result['notebooks'])
                except Exception:
                    self.set_status(500)
                    self.finish({**result, 'applied': True, 'message':
                        'Language changed and a backup was saved, but open notebooks could not refresh. Reload JupyterLab before editing.'})
                    return
                self.finish(result)
            except web.HTTPError:
                raise
            except (ValueError, KeyError) as error:
                raise web.HTTPError(400, str(error)) from error
            except BlockingIOError as error:
                raise web.HTTPError(409, 'Another course switch is running; no notebooks were changed') from error
            except Exception as error:
                raise web.HTTPError(500, 'Language change failed; previous notebooks were restored') from error
            finally:
                self.controller.finished = time.monotonic()


def _jupyter_server_extension_points():
    return [{'module': 'course_language_server'}]


def _load_jupyter_server_extension(serverapp):
    controller = CourseLanguage(serverapp.root_dir, serverapp)
    serverapp.web_app.add_handlers('.*', [(url_path_join(serverapp.base_url, 'api/course/language'), LanguageHandler,
                                         {'controller': controller}),
        (url_path_join(serverapp.base_url, 'course-home'), CourseHomeHandler, {'controller': controller}),
        (url_path_join(serverapp.base_url, 'api/course/status'), CourseStatusHandler, {'controller': controller})])

class CourseHomeHandler(JupyterHandler):
    def initialize(self, controller):
        self.controller = controller

    @web.authenticated
    def get(self):
        # Establish XSRF and the normal Jupyter login cookie before showing links.
        self.xsrf_token
        self.set_header('Content-Type', 'text/html; charset=utf-8')
        self.set_header('Cache-Control', 'no-store')
        self.set_header('Referrer-Policy', 'same-origin')
        self.set_header('X-Content-Type-Options', 'nosniff')
        self.finish((self.controller.root / 'composer/runtime/course_home.html').read_text())


class CourseStatusHandler(APIHandler):
    def initialize(self, controller):
        self.controller = controller

    @web.authenticated
    async def get(self):
        config = json.loads((self.controller.root / 'composer/course_home.json').read_text())
        client = AsyncHTTPClient()
        async def probe(service):
            result = {'id': service['id'], 'state': 'unavailable'}
            try:
                response = await client.fetch(service['probe'], request_timeout=3, connect_timeout=2,
                                              follow_redirects=False, raise_error=False)
                if response.code == 200:
                    if service.get('json'):
                        value = json.loads(response.body)
                        if not isinstance(value, dict):
                            return result
                    result['state'] = 'available'
            except Exception:
                pass  # Never return upstream bodies, exception URLs, or credentials.
            return result
        values = await asyncio.gather(*(probe(item) for item in config['services']))
        self.set_header('Cache-Control', 'no-store')
        self.finish({'services': [{'id': 'jupyter', 'state': 'available'}, *values]})
