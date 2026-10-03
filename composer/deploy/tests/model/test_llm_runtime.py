"""Profile retirement and cancelled worker lifetimes, without provider access."""
import asyncio
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services/model'))
from llm_runtime import CONTEXT, Runtime, RuntimeManager, RuntimeRequests


class RuntimeLifetimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = 10.0
        clock = patch('llm_runtime.time', SimpleNamespace(monotonic=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.initial = self.state('deployment')
        self.manager = RuntimeManager(self.initial, Path(self.directory.name) / 'profile.json')

    def state(self, revision):
        database = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(database.close)
        return Runtime(id=revision, settings={'requests': {'max_concurrency': 2}},
            PROVIDERS={}, STORE=SimpleNamespace(db=database))

    async def activate(self, revision):
        async def factory(profile):
            return self.state(revision)
        return await self.manager.activate({}, factory)

    async def drain_workers(self):
        await asyncio.wait_for(asyncio.gather(*self.manager.workers, return_exceptions=True), 2)
        await asyncio.sleep(0)

    async def test_old_current_gets_full_retirement_grace_and_live_requests_survive(self):
        previous = await self.activate('previous')
        self.now = 1000
        await self.activate('current')
        self.assertEqual(previous.retired, 1000)
        self.manager.prune()
        self.assertIn('previous', self.manager.states)
        previous.users = 1
        self.now = 1121
        self.manager.prune()
        self.assertEqual(previous.STORE.db.execute('SELECT 1').fetchone(), (1,))
        previous.users = 0
        self.manager.prune()
        self.assertNotIn('previous', self.manager.states)
        self.assertIn('deployment', self.manager.states)
        with self.assertRaises(sqlite3.ProgrammingError):
            previous.STORE.db.execute('SELECT 1')

    async def test_failed_persistence_does_not_retire_current(self):
        previous = await self.activate('previous')
        with patch.object(self.manager, 'persist', side_effect=OSError('synthetic failure')):
            with self.assertRaises(OSError):
                await self.activate('failed')
        self.assertIs(self.manager.current, previous)
        self.assertIsNone(previous.retired)
        self.assertNotIn('failed', self.manager.states)

    async def test_cancelled_asgi_request_keeps_worker_database_and_context_alive(self):
        previous = await self.activate('previous')
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        observations = []

        def work():
            loop.call_soon_threadsafe(started.set)
            if not release.wait(2):
                raise TimeoutError('test failed to release worker')
            observations.append((self.manager.get().id, previous.STORE.db.execute('SELECT 1').fetchone()))

        async def application(scope, receive, send):
            await self.manager.run_in_thread(work)

        request = asyncio.create_task(RuntimeRequests(application, self.manager)(
            {'type': 'http', 'method': 'GET', 'path': '/v1/activity',
             'headers': [(b'x-dli-profile', b'previous')]}, None, None))
        try:
            await asyncio.wait_for(started.wait(), 2)
            self.assertEqual(previous.users, 2)
            await self.activate('current')
            self.now += 121
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(request, .5)
            self.assertEqual(previous.users, 1)
            self.manager.prune()
            self.assertIn('previous', self.manager.states)
            # A cancelled request must not block the event loop on thread exit.
            await asyncio.wait_for(asyncio.sleep(0), .5)
        finally:
            release.set()
            await self.drain_workers()
        self.assertEqual(observations, [('previous', (1,))])
        self.assertEqual(previous.users, 0)
        self.assertNotIn('previous', self.manager.states)
        self.assertIsNone(CONTEXT.get())

    async def test_worker_result_error_and_cancelled_error_release_the_lease(self):
        state = await self.activate('current')
        self.assertEqual(await self.manager.run_in_thread(lambda *, value: value, value=7), 7)
        self.assertEqual(state.users, 0)

        def fail():
            raise ValueError('synthetic worker failure')

        with self.assertRaisesRegex(ValueError, 'synthetic worker failure'):
            await self.manager.run_in_thread(fail)
        self.assertEqual(state.users, 0)
        release = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        errors = []
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))

        def delayed_failure():
            loop.call_soon_threadsafe(started.set)
            release.wait(2)
            fail()

        request = asyncio.create_task(self.manager.run_in_thread(delayed_failure))
        try:
            await asyncio.wait_for(started.wait(), 2)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertEqual(state.users, 1)
        finally:
            release.set()
            await self.drain_workers()
            loop.set_exception_handler(old_handler)
        self.assertEqual(state.users, 0)
        self.assertFalse(self.manager.workers)
        self.assertEqual(errors, [])

    async def test_capacity_never_evicts_recent_or_referenced_revisions(self):
        for index in range(7):
            await self.activate(str(index))
        with self.assertRaises(ValueError):
            await self.activate('over-capacity')
        self.assertEqual(len(self.manager.states), 8)
        pinned = self.manager.states['0']
        pinned.users = 1
        self.now += 121
        await self.activate('after-grace')
        self.assertIn('0', self.manager.states)
        self.assertEqual(pinned.STORE.db.execute('SELECT 1').fetchone(), (1,))


if __name__ == '__main__':
    unittest.main()
