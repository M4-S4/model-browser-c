import logging
import threading

from src.plugin_api import HookContext

if __name__.count(".") >= 2:
    from ..backend import PLUGIN_ID, settings, tables
    from ..backend.watcher import Watcher
else:
    from backend import PLUGIN_ID, settings, tables
    from backend.watcher import Watcher

logger = logging.getLogger("plugins.model-browser")

STOP_GRACE_SECONDS = 10.0

_lock = threading.Lock()
_watcher = None
_watcher_future = None


def on_boot(context: HookContext) -> HookContext:
    try:
        tables.create_tables()
        logger.info("[MODEL-BROWSER] Tables ready: %s", ", ".join(tables.TABLE_NAMES))
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not create the plugin's tables")
    try:
        rewritten = settings.normalise_stored_booleans()
        if rewritten:
            logger.info("[MODEL-BROWSER] Settings rewritten as true/false: %s", ", ".join(rewritten))
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not normalise the plugin's stored settings")
    try:
        if _app_loop_running():
            _start_watcher()
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not start the download watcher")
    return context


def on_ready(context: HookContext) -> HookContext:
    try:
        _start_watcher()
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not start the download watcher")
    return context


def on_disable(context: HookContext) -> HookContext:
    if context.get("plugin_id") != PLUGIN_ID:
        return context
    try:
        _stop_watcher()
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not stop the download watcher")
    try:
        _shutdown_http()
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not close the sources' HTTP session")
    return context


def _shutdown_http() -> None:
    if __name__.count(".") >= 2:
        from ..backend import http
    else:
        from backend import http
    http.shutdown()


def _app_loop_running() -> bool:
    try:
        from src.plugin_api.hooks import get_app_loop

        return get_app_loop().is_running()
    except (ImportError, RuntimeError):
        return False


def _start_watcher() -> None:
    global _watcher, _watcher_future
    from src.plugin_api.hooks import run_on_app_loop

    with _lock:
        if _watcher_future is not None and not _watcher_future.done():
            return
        watcher = Watcher()
        _watcher_future = run_on_app_loop(watcher.run())
        _watcher = watcher


def _stop_watcher() -> None:
    global _watcher, _watcher_future
    with _lock:
        watcher, future = _watcher, _watcher_future
        _watcher = _watcher_future = None
    if watcher is None or future is None or future.done():
        logger.info("[MODEL-BROWSER] Disabled; no download watcher was running")
        return

    watcher.request_stop()
    loop = watcher.loop
    if loop is None:
        future.cancel()
    elif not loop.is_closed():
        loop.call_soon_threadsafe(loop.call_later, STOP_GRACE_SECONDS, future.cancel)
    logger.info("[MODEL-BROWSER] Disabled; stopping the download watcher")
