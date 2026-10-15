import asyncio
import concurrent.futures
import inspect
import threading
import time

import pytest

import src.plugin_api.hooks as plugin_hooks
from backend import tables
from backend.watcher import Watcher
from hooks import lifecycle_hooks
from src.plugin_api import HookContext


def _context(hook, subject="model-browser-c"):
    return HookContext(hook_name=hook, plugin_id="model-browser-c", data={"plugin_id": subject})


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


class _FakeWatcher:
    def __init__(self, loop=None):
        self.loop = loop
        self.stop_requests = 0

    def request_stop(self):
        self.stop_requests += 1


def _install(monkeypatch, watcher, future):
    monkeypatch.setattr(lifecycle_hooks, "_watcher", watcher)
    monkeypatch.setattr(lifecycle_hooks, "_watcher_future", future)


@pytest.fixture(autouse=True)
def _no_watcher(monkeypatch):
    _install(monkeypatch, None, None)

    async def run_pass(self, *, reconcile=False):
        return None

    monkeypatch.setattr(Watcher, "run_pass", run_pass)


@pytest.fixture
def process_start(monkeypatch):
    scheduled = []

    def get_app_loop():
        raise RuntimeError("AppContainer not initialized yet")

    def run_on_app_loop(coro):
        scheduled.append(coro)
        coro.close()
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(plugin_hooks, "get_app_loop", get_app_loop)
    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    return scheduled


async def _cancel_remaining_tasks():
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def app_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="fake-app-loop", daemon=True)
    thread.start()
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5)
    monkeypatch.setattr(plugin_hooks, "get_app_loop", lambda: loop)
    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", lambda coro: asyncio.run_coroutine_threadsafe(coro, loop))
    yield loop
    asyncio.run_coroutine_threadsafe(_cancel_remaining_tasks(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_on_boot_creates_the_tables_and_returns_its_context(monkeypatch, process_start):
    calls = []
    monkeypatch.setattr(tables, "create_tables", lambda: calls.append("create_tables"))
    context = _context("plugin.lifecycle.boot")

    assert lifecycle_hooks.on_boot(context) is context
    assert calls == ["create_tables"]
    assert process_start == []
    assert lifecycle_hooks._watcher_future is None


def test_on_boot_logs_and_swallows_a_table_error(monkeypatch, process_start, caplog):
    def create_tables():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(tables, "create_tables", create_tables)
    context = _context("plugin.lifecycle.boot")

    assert lifecycle_hooks.on_boot(context) is context
    assert "Could not create the plugin's tables" in caplog.text


def test_on_ready_schedules_the_watcher_on_the_app_loop(monkeypatch):
    scheduled = []
    future = concurrent.futures.Future()

    def run_on_app_loop(coro):
        scheduled.append(coro)
        coro.close()
        return future

    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    context = _context("plugin.lifecycle.ready")

    assert lifecycle_hooks.on_ready(context) is context
    assert len(scheduled) == 1
    assert inspect.iscoroutine(scheduled[0])
    assert scheduled[0].__qualname__ == "Watcher.run"
    assert isinstance(lifecycle_hooks._watcher, Watcher)
    assert lifecycle_hooks._watcher_future is future

    lifecycle_hooks.on_ready(_context("plugin.lifecycle.ready"))

    assert len(scheduled) == 1


def test_on_ready_logs_and_swallows_an_unavailable_loop(process_start, caplog):
    context = _context("plugin.lifecycle.ready")

    assert lifecycle_hooks.on_ready(context) is context
    assert lifecycle_hooks._watcher_future is None
    assert "Could not start the download watcher" in caplog.text


def test_on_disable_of_another_plugin_leaves_the_watcher_alone(monkeypatch):
    watcher, future = _FakeWatcher(), concurrent.futures.Future()
    _install(monkeypatch, watcher, future)
    context = _context("plugin.lifecycle.disable", subject="civitai-provider")

    assert lifecycle_hooks.on_disable(context) is context
    assert watcher.stop_requests == 0
    assert not future.cancelled()
    assert lifecycle_hooks._watcher is watcher


def test_on_disable_stops_the_watcher_and_drops_a_task_not_started_yet(monkeypatch):
    watcher, future = _FakeWatcher(loop=None), concurrent.futures.Future()
    _install(monkeypatch, watcher, future)
    context = _context("plugin.lifecycle.disable")

    assert lifecycle_hooks.on_disable(context) is context
    assert watcher.stop_requests == 1
    assert future.cancelled()
    assert lifecycle_hooks._watcher is None
    assert lifecycle_hooks._watcher_future is None


def test_disable_stops_a_running_watcher_from_another_thread(app_loop):
    lifecycle_hooks.on_ready(_context("plugin.lifecycle.ready"))
    watcher, future = lifecycle_hooks._watcher, lifecycle_hooks._watcher_future
    _wait_until(lambda: watcher.loop is not None)

    lifecycle_hooks.on_disable(_context("plugin.lifecycle.disable"))

    assert future.result(timeout=2) is None
    assert watcher.loop is None


def test_disable_cancels_a_watcher_that_does_not_return(app_loop, monkeypatch):
    monkeypatch.setattr(Watcher, "request_stop", lambda self: None)
    monkeypatch.setattr(lifecycle_hooks, "STOP_GRACE_SECONDS", 0.05)
    lifecycle_hooks.on_ready(_context("plugin.lifecycle.ready"))
    watcher, future = lifecycle_hooks._watcher, lifecycle_hooks._watcher_future
    _wait_until(lambda: watcher.loop is not None)

    lifecycle_hooks.on_disable(_context("plugin.lifecycle.disable"))

    concurrent.futures.wait([future], timeout=2)
    assert future.cancelled()
    _wait_until(lambda: watcher.loop is None)


def test_on_boot_starts_the_watcher_when_the_plugin_is_enabled_at_runtime(app_loop, monkeypatch):
    monkeypatch.setattr(tables, "create_tables", lambda: None)

    lifecycle_hooks.on_boot(_context("plugin.lifecycle.boot"))

    watcher, future = lifecycle_hooks._watcher, lifecycle_hooks._watcher_future
    assert future is not None
    _wait_until(lambda: watcher.loop is not None)

    lifecycle_hooks.on_ready(_context("plugin.lifecycle.ready"))

    assert lifecycle_hooks._watcher_future is future
    lifecycle_hooks.on_disable(_context("plugin.lifecycle.disable"))
    assert future.result(timeout=2) is None
