import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from . import settings

logger = logging.getLogger("plugins.model-browser")

TABLE = "model_browser_cache"
MAX_ROWS = 5000
PURGE_EVERY_WRITES = 50


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(source: str, request: Any) -> str:
    return hashlib.md5((source + canonical_json(request)).encode("utf-8"), usedforsecurity=False).hexdigest()


def format_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


class Cache:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        max_rows: int = MAX_ROWS,
        purge_every: int = PURGE_EVERY_WRITES,
    ) -> None:
        self.clock = clock
        self.max_rows = max_rows
        self.purge_every = purge_every
        self._lock = threading.Lock()
        self._writes = 0
        self._long_lived: Dict[str, float] = {}

    @staticmethod
    def default_ttl_seconds() -> float:
        return float(settings.get_setting("cache_ttl_minutes")) * 60.0

    def get(self, source: str, request: Any, *, ttl_seconds: Optional[float] = None) -> Optional[Any]:
        from src.plugin_api import db

        key = cache_key(source, request)
        ttl = self.default_ttl_seconds() if ttl_seconds is None else float(ttl_seconds)
        cutoff = format_timestamp(self.clock() - ttl)
        with db.get_cursor() as cursor:
            cursor.execute(f"SELECT payload FROM {TABLE} WHERE key = ? AND fetched_at >= ?", (key, cutoff))
            row = cursor.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (ValueError, TypeError):
            logger.warning("[MODEL-BROWSER] Ignoring an unreadable cache entry for %s", source)
            return None

    def set(self, source: str, request: Any, payload: Any, *, ttl_seconds: Optional[float] = None) -> None:
        from src.plugin_api import db

        key = cache_key(source, request)
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with db.get_cursor() as cursor:
            cursor.execute(
                f"INSERT OR REPLACE INTO {TABLE} (key, source, payload, fetched_at) VALUES (?, ?, ?, ?)",
                (key, source, text, format_timestamp(self.clock())),
            )
        with self._lock:
            if ttl_seconds is not None:
                self._long_lived[key] = float(ttl_seconds)
            self._writes += 1
            due = self.purge_every > 0 and (self._writes - 1) % self.purge_every == 0
        if due:
            try:
                self.purge()
            except Exception:
                logger.warning("[MODEL-BROWSER] Could not purge the cache", exc_info=True)

    def purge(self) -> int:
        from src.plugin_api import db

        now = self.clock()
        default_cutoff = format_timestamp(now - self.default_ttl_seconds())
        with self._lock:
            long_lived = dict(self._long_lived)
        deleted = 0
        with db.get_cursor() as cursor:
            if long_lived:
                marks = ",".join("?" * len(long_lived))
                cursor.execute(
                    f"DELETE FROM {TABLE} WHERE fetched_at < ? AND key NOT IN ({marks})",
                    (default_cutoff, *long_lived),
                )
            else:
                cursor.execute(f"DELETE FROM {TABLE} WHERE fetched_at < ?", (default_cutoff,))
            deleted += max(cursor.rowcount, 0)
            for key, ttl in long_lived.items():
                cursor.execute(
                    f"DELETE FROM {TABLE} WHERE key = ? AND fetched_at < ?", (key, format_timestamp(now - ttl))
                )
                deleted += max(cursor.rowcount, 0)
            cursor.execute(
                f"DELETE FROM {TABLE} WHERE key IN "
                f"(SELECT key FROM {TABLE} ORDER BY fetched_at DESC, key LIMIT -1 OFFSET ?)",
                (max(int(self.max_rows), 0),),
            )
            deleted += max(cursor.rowcount, 0)
        if deleted:
            logger.debug("[MODEL-BROWSER] Cache purge removed %d rows", deleted)
        return deleted


cache = Cache()
