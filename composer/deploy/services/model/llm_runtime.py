"""Private, atomic profile revisions pinned for the complete request lifetime."""
import asyncio
import contextvars
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from llm_tracing import profile_attribute

CONTEXT = contextvars.ContextVar("course_runtime", default=None)


class Runtime(SimpleNamespace):
    def __init__(self, **values):
        super().__init__(**values)
        self.users = 0
        self.created = time.monotonic()
        self.retired = None
        self.catalog = {}
        self.semaphore = asyncio.Semaphore(self.settings["requests"]["max_concurrency"])

    def clean(self, value):
        encoded = json.dumps(value, ensure_ascii=True)
        for key, *_ in self.PROVIDERS.values():
            encoded = encoded.replace(json.dumps(key, ensure_ascii=True)[1:-1], "[REDACTED]")
        return json.loads(encoded)


class RuntimeManager:
    def __init__(self, initial, path):
        self.initial = self.current = initial
        self.states = {initial.id: initial}
        self.path = Path(path)
        self.lock = asyncio.Lock()
        self.workers = set()

    def get(self):
        return CONTEXT.get() or self.current

    async def run_in_thread(self, function, *args, **kwargs):
        """Keep the selected runtime alive until work actually exits its thread.

        Cancelling an HTTP request cannot stop an executor thread. The shield
        lets that request exit promptly while completion releases this lease.
        """
        state = self.get()
        state.users += 1
        token = CONTEXT.set(state)
        try:
            context = contextvars.copy_context()
            worker = asyncio.get_running_loop().run_in_executor(
                None, context.run, partial(function, *args, **kwargs))
        except BaseException:
            state.users -= 1
            self.prune()
            raise
        finally:
            CONTEXT.reset(token)
        self.workers.add(worker)

        def finished(done):
            self.workers.discard(done)
            # Retrieve failures even when the original caller was cancelled.
            if not done.cancelled():
                done.exception()
            state.users -= 1
            self.prune()

        worker.add_done_callback(finished)
        return await asyncio.shield(worker)

    def prune(self):
        now = time.monotonic()
        for key, value in list(self.states.items()):
            if (value is not self.initial and value is not self.current and not value.users
                    and value.retired is not None and now - value.retired > 120):
                value.STORE.db.close()
                del self.states[key]

    def persist(self, document):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("Private profile storage cannot be a symbolic link")
        os.chmod(self.path.parent, 0o700)
        fd, filename = tempfile.mkstemp(prefix=".profile-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(document, output)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(filename, 0o600)
            os.replace(filename, self.path)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    async def activate(self, profile, factory):
        async with self.lock:
            self.prune()
            if len(self.states) >= 8:
                raise ValueError("Several profile revisions are still in use; try again shortly")
            candidate = await factory(profile)
            try:
                self.persist({"profile": profile, "catalog": candidate.catalog, "revision": candidate.id})
            except BaseException:
                candidate.STORE.db.close()
                raise
            self.states[candidate.id] = candidate
            self.current.retired = time.monotonic()
            self.current = candidate
            return candidate

    async def restore(self, factory):
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_mode & 0o077:
                raise ValueError("Private profile storage permissions are invalid")
            document = json.loads(self.path.read_text())
            candidate = await factory(document["profile"], document["catalog"], document["revision"])
            self.states[candidate.id] = candidate
            self.current = candidate


class RuntimeRequests:
    def __init__(self, app, manager):
        self.app, self.manager = app, manager

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/v1/profile":
            return await self.app(scope, receive, send)
        requested = dict(scope.get("headers", [])).get(b"x-dli-profile")
        state = self.manager.states.get(requested.decode("ascii", errors="ignore")) if requested else self.manager.current
        if state is None:
            return await JSONResponse({"error": {"code": "profile_expired", "message": "Refresh the active profile"}},
                status_code=409)(scope, receive, send)
        state.users += 1
        token = CONTEXT.set(state)
        profile_attribute(state.id)
        try:
            if scope.get("method") == "POST":
                async with state.semaphore:
                    await self.app(scope, receive, send)
            else:
                await self.app(scope, receive, send)
        finally:
            CONTEXT.reset(token)
            state.users -= 1
            self.manager.prune()
