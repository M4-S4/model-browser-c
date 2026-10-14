import asyncio
import concurrent.futures
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Coroutine, Dict, Iterable, List, Mapping, Optional, Tuple

from src.plugin_api import User

from . import PLUGIN_ID, core_bridge, limits, notifications, pending, permissions, requests
from .core_bridge import CoreBridgeError
from .pending import PendingRow
from .requests import Request

logger = logging.getLogger("plugins.model-browser")

POLL_INTERVAL_SECONDS = 60
PATH_FALLBACK_AFTER = 2
HASH_FALLBACK_AFTER = 6
MAX_INDEX_LOOKUPS = 30
UNQUEUED_GRACE_PASSES = 5
MAX_ROW_ERRORS = 10
MAX_ERROR_LENGTH = 2000

WAITING_STATUSES = frozenset({"pending", "downloading", "paused"})
FINISHING_STATUSES = ("indexed", "assigned")

PENDING_PROVIDER_ID = "model-browser"
NOTIFY_DOWNLOAD_DONE = notifications.DOWNLOAD_DONE
NOTIFY_DOWNLOAD_FAILED = notifications.DOWNLOAD_FAILED
NOT_ASSIGNED_TO_REQUESTER = "The model was downloaded but could not be made visible to you"

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def model_attributes(info: Mapping[str, Any]) -> Dict[str, str]:
    model = info.get("model") if isinstance(info.get("model"), Mapping) else {}
    versions = [version for version in model.get("versions") or [] if isinstance(version, Mapping)]
    version = next((version for version in versions if str(version.get("id")) == str(info.get("version_id"))), None)
    if version is None:
        version = versions[0] if versions else {}
    license_ = model.get("license") if isinstance(model.get("license"), Mapping) else {}
    values = {"mb_base_model": version.get("base_model"), "mb_license": license_.get("summary")}
    return {key: value.strip() for key, value in values.items() if isinstance(value, str) and value.strip()}


def _timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def indexed_since(model: Mapping[str, Any], row: PendingRow) -> bool:
    created, written = _timestamp(model.get("created_at")), _timestamp(row.created_at)
    return created is not None and written is not None and created >= written


def sha256_of_file(path: str) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _label(row: PendingRow) -> str:
    model = row.info.get("model") if isinstance(row.info.get("model"), Mapping) else {}
    name = model.get("name")
    if isinstance(name, str) and name.strip() and name.strip() != row.file_name:
        return f"{name.strip()} ({row.file_name})"
    return row.file_name


def _where(assign_to: str) -> str:
    if assign_to.startswith("group:"):
        return "it is now available to your group"
    if assign_to == "none":
        return "it is now installed"
    return "it is now in your models"


def _metadata(row: PendingRow) -> Dict[str, Any]:
    return {
        "source": row.source,
        "source_model_id": row.source_model_id,
        "source_version_id": row.source_version_id,
        "sha256": None if row.nohash else row.sha256,
        "model_id": row.model_id,
        "download_id": row.download_id,
    }


def schedule_step(step: Coroutine[Any, Any, Any], key: str) -> bool:
    try:
        from src.plugin_api.hooks import run_on_app_loop

        future = run_on_app_loop(step)
    except Exception as exc:
        step.close()
        logger.warning(
            "[MODEL-BROWSER] Could not hand %s over to the application loop (%s); the sweep will take it up", key, exc
        )
        return False
    future.add_done_callback(lambda done: _log_step_error(done, key))
    return True


def _log_step_error(future: concurrent.futures.Future, key: str) -> None:
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.error("[MODEL-BROWSER] Could not move %s on; the sweep will retry", key, exc_info=error)


async def carry_on(key: str) -> Optional[PendingRow]:
    row = await asyncio.to_thread(pending.get, key)
    if row is None or row.status not in FINISHING_STATUSES:
        return row
    return await finish(row)


async def finish(row: PendingRow) -> Optional[PendingRow]:
    assigned: Dict[str, str] = {}
    if row.status == "indexed":
        row, assigned = await assign(row)
    if row is not None and row.status == "assigned":
        row = await enrich(row, assigned)
    return row


def record_indexed(row: PendingRow, model: Mapping[str, Any]) -> Optional[PendingRow]:
    key = row.sha256
    model_sha = str(model.get("sha256") or "").strip().lower()
    if model_sha and model_sha != key:
        if pending.get(model_sha) is None:
            moved = pending.rekey(key, model_sha)
            requests.relink(key, model_sha)
            if moved is None or moved.download_id != row.download_id:
                logger.warning("[MODEL-BROWSER] %s merged into the row of %s meanwhile", key, model_sha)
                return moved
            key = model_sha
        else:
            logger.info("[MODEL-BROWSER] %s is the model of another row (%s); keeping its key", key, model_sha)
    updated = pending.update(key, only_from=("queued",), model_id=model.get("id"), status="indexed", error=None)
    if updated is None:
        logger.info("[MODEL-BROWSER] %s was no longer queued: model %s not recorded again", row.file_name, model.get("id"))
        return None
    logger.info("[MODEL-BROWSER] %s indexed as model %s", row.file_name, model.get("id"))
    return updated


async def assign(row: PendingRow) -> Tuple[Optional[PendingRow], Dict[str, str]]:
    if not row.model_id:
        return await fail(row, "No model to assign"), {}
    kind, _, target = row.assign_to.partition(":")
    assign_to, note = row.assign_to, None
    try:
        if row.assign_to == "none":
            pass
        elif kind == "group" and target:
            admin = await asyncio.to_thread(_acting_admin, row, target)
            if admin is None:
                note = f"No admin could act for the group '{target}': the model was assigned to its requester instead"
            else:
                try:
                    await asyncio.to_thread(core_bridge.assign_to_group, target, [row.model_id], admin)
                except CoreBridgeError as exc:
                    if exc.code != "not_found":
                        raise
                    note = f"The group '{target}' no longer exists: the model was assigned to its requester instead"
            if note is not None:
                logger.warning("[MODEL-BROWSER] %s", note)
                await asyncio.to_thread(core_bridge.assign_to_user, row.model_id, row.requested_by)
                assign_to = f"user:{row.requested_by}"
        else:
            user_id = target if kind == "user" and target else row.requested_by
            if kind != "user":
                logger.warning("[MODEL-BROWSER] Unknown assign_to %r on %s: assigning to the requester", row.assign_to, row.sha256)
            await asyncio.to_thread(core_bridge.assign_to_user, row.model_id, user_id)
            assign_to = f"user:{user_id}"
    except CoreBridgeError as exc:
        return await fail(row, f"The model was downloaded but could not be made visible: {exc.message}"), {}
    fields: Dict[str, Any] = {"status": "assigned", "error": note}
    if assign_to != row.assign_to:
        fields["assign_to"] = assign_to
    updated = await asyncio.to_thread(pending.update, row.sha256, only_from=("indexed",), **fields)
    if updated is None:
        logger.info("[MODEL-BROWSER] %s was no longer indexed: left to the step that moved it on", row.file_name)
        return None, {}
    return updated, await _assign_requests(updated)


def _acting_admin(row: PendingRow, group_id: str) -> Optional[User]:
    request = requests.get(row.request_id) if row.request_id else None
    return permissions.acting_admin(row.requested_by, request.decided_by if request is not None else None, group_id)


def _linked_requests(row: PendingRow) -> List[Request]:
    linked = {request.id: request for request in requests.list_approved_for_sha256(row.sha256)}
    if row.request_id and row.request_id not in linked:
        own = requests.get(row.request_id)
        if own is not None and own.status == requests.APPROVED:
            linked[own.id] = own
    return list(linked.values())


async def _assign_requests(row: PendingRow) -> Dict[str, str]:
    assigned: Dict[str, str] = {}
    try:
        linked = await asyncio.to_thread(_linked_requests, row)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not read the requests waiting for %s", row.sha256, exc_info=True)
        return assigned
    for request in linked:
        if request.id == row.request_id:
            continue
        target = await _assign_request(row, request)
        if target is not None:
            assigned[request.id] = target
    return assigned


async def _assign_request(row: PendingRow, request: Request) -> Optional[str]:
    target = request.assign_to or f"user:{request.user_id}"
    kind, _, target_id = target.partition(":")
    try:
        if target == "none":
            return target
        if kind == "group" and target_id:
            admin = await asyncio.to_thread(permissions.acting_admin, request.user_id, request.decided_by, target_id)
            if admin is None:
                logger.warning(
                    "[MODEL-BROWSER] No admin could act for the group '%s' of request %s: assigning its requester",
                    target_id,
                    request.id,
                )
            else:
                try:
                    await asyncio.to_thread(core_bridge.assign_to_group, target_id, [row.model_id], admin)
                    return target
                except CoreBridgeError as exc:
                    if exc.code != "not_found":
                        raise
                    logger.warning(
                        "[MODEL-BROWSER] The group '%s' of request %s no longer exists: assigning its requester",
                        target_id,
                        request.id,
                    )
            await asyncio.to_thread(core_bridge.assign_to_user, row.model_id, request.user_id)
            return f"user:{request.user_id}"
        user_id = target_id if kind == "user" and target_id else request.user_id
        await asyncio.to_thread(core_bridge.assign_to_user, row.model_id, user_id)
        return f"user:{user_id}"
    except Exception as exc:
        message = exc.message if isinstance(exc, CoreBridgeError) else str(exc)
        logger.warning("[MODEL-BROWSER] Could not assign model %s for request %s: %s", row.model_id, request.id, message)
        return None


async def settle_requests(row: PendingRow, *, done: bool, assigned: Optional[Mapping[str, str]] = None) -> None:
    try:
        linked = await asyncio.to_thread(_linked_requests, row)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not read the requests waiting for %s", row.sha256, exc_info=True)
        return
    assigned = assigned or {}
    for request in linked:
        target: Optional[str] = None
        if done:
            if request.id == row.request_id:
                target = row.assign_to
            elif request.id in assigned:
                target = assigned[request.id]
            else:
                target = await _assign_request(row, request)
        status = requests.DONE if target is not None else requests.FAILED
        try:
            closed = await asyncio.to_thread(
                lambda: requests.set_status(request.id, status, assign_to=target, only_from=(requests.APPROVED,))
            )
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not close request %s", request.id, exc_info=True)
            continue
        if closed is None:
            continue
        logger.info("[MODEL-BROWSER] Request %s %s", closed.id, status)
        if closed.user_id != row.requested_by:
            await _notify_requester(row, closed, served=target is not None)


async def _notify_requester(row: PendingRow, request: Request, *, served: bool) -> None:
    metadata = {**_metadata(row), "request_id": request.id}
    if served:
        where = _where(request.assign_to or f"user:{request.user_id}")
        await asyncio.to_thread(notifications.download_done, request.user_id, _label(row), where, metadata)
    else:
        reason = NOT_ASSIGNED_TO_REQUESTER if row.status == "enriched" else row.error
        await asyncio.to_thread(notifications.download_failed, request.user_id, _label(row), reason, metadata)


async def enrich(row: PendingRow, assigned: Optional[Mapping[str, str]] = None) -> Optional[PendingRow]:
    notes: List[str] = [row.error] if row.error else []
    notes += await _fetch(PENDING_PROVIDER_ID, row.model_id)
    attributes = model_attributes(row.info)
    if attributes:
        try:
            await asyncio.to_thread(core_bridge.write_attributes, row.model_id, attributes)
        except Exception as exc:
            message = exc.message if isinstance(exc, CoreBridgeError) else str(exc)
            logger.warning("[MODEL-BROWSER] Could not write the attributes of model %s: %s", row.model_id, message)
            notes.append(f"attributes: {message}")
    error = "; ".join(notes)[:MAX_ERROR_LENGTH] or None
    updated = await asyncio.to_thread(pending.update, row.sha256, only_from=("assigned",), status="enriched", error=error)
    if updated is None:
        logger.info("[MODEL-BROWSER] %s was no longer assigned: left to the step that ended it", row.file_name)
        return None
    logger.info("[MODEL-BROWSER] %s ready (model %s)%s", row.file_name, row.model_id, f": {error}" if error else "")
    await _notify(updated, done=True)
    await settle_requests(updated, done=True, assigned=assigned)
    return updated


async def _fetch(provider: str, model_id: Optional[str]) -> List[str]:
    try:
        summary = await core_bridge.run_provider_fetch(provider, [model_id] if model_id else [])
    except Exception as exc:
        logger.warning("[MODEL-BROWSER] The %s fetch for model %s failed: %s", provider, model_id, exc)
        return [f"{provider}: {exc}"]
    if summary.get("successful"):
        return []
    reasons = [
        str(error.get("reason"))
        for error in summary.get("errors") or []
        if isinstance(error, Mapping) and error.get("reason")
    ]
    if reasons:
        return [f"{provider}: {reason}" for reason in reasons]
    return [f"{provider}: no information found"] if summary.get("failed") else []


async def fail(row: PendingRow, message: str) -> Optional[PendingRow]:
    updated = await asyncio.to_thread(
        pending.update, row.sha256, only_from=pending.UNFINISHED, status="failed", error=message[:MAX_ERROR_LENGTH]
    )
    if updated is None:
        logger.info("[MODEL-BROWSER] %s had ended already, not failing it: %s", row.file_name, message)
        return None
    logger.warning("[MODEL-BROWSER] %s failed: %s", row.file_name, message)
    if updated.limit_ref_id:
        refunded = await asyncio.to_thread(limits.refund, updated.limit_ref_id)
        if refunded:
            logger.info("[MODEL-BROWSER] Refunded %d quota event(s) of %s", refunded, row.file_name)
    await _notify(updated, done=False)
    await settle_requests(updated, done=False)
    return updated


async def _notify(row: PendingRow, *, done: bool) -> None:
    if done:
        await asyncio.to_thread(
            notifications.download_done, row.requested_by, _label(row), _where(row.assign_to), _metadata(row)
        )
    else:
        await asyncio.to_thread(notifications.download_failed, row.requested_by, _label(row), row.error, _metadata(row))


class Watcher:
    def __init__(self) -> None:
        self._stop_requested = False
        self._stop_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lookups: Dict[str, int] = {}
        self._unqueued: Dict[str, int] = {}
        self._digests: Dict[str, Optional[str]] = {}
        self._errors: Dict[str, int] = {}
        self._pass_failing = False

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    @property
    def running(self) -> bool:
        return self._loop is not None

    async def run(self) -> None:
        self._stop_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        logger.info("[MODEL-BROWSER] Download watcher started (a sweep every %ss)", POLL_INTERVAL_SECONDS)
        try:
            reconcile = True
            while not self._stop_requested:
                await self.run_pass(reconcile=reconcile)
                reconcile = False
                if self._stop_requested:
                    break
                await self._sleep(POLL_INTERVAL_SECONDS)
        finally:
            self._loop = None
            await self._release_resources()
            logger.info("[MODEL-BROWSER] Download watcher stopped")

    async def _release_resources(self) -> None:
        try:
            from .http import http as source_http

            await source_http.close()
        except Exception:
            logger.debug("[MODEL-BROWSER] Could not close the sources' HTTP session", exc_info=True)

    def request_stop(self) -> None:
        self._stop_requested = True
        loop, event = self._loop, self._stop_event
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            pass

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def run_pass(self, *, reconcile: bool = False) -> None:
        try:
            rows = await asyncio.to_thread(pending.list_unfinished)
        except Exception:
            if not self._pass_failing:
                logger.exception("[MODEL-BROWSER] Could not read the pending downloads; retrying every pass")
            self._pass_failing = True
            return
        self._pass_failing = False
        self._prune(rows)
        if reconcile and rows:
            logger.info("[MODEL-BROWSER] Resuming %d unfinished download(s)", len(rows))
        for row in rows:
            if self._stop_requested:
                return
            await self._advance_safely(row)
        if reconcile and not self._stop_requested:
            await self._settle_orphans()

    def _prune(self, rows: Iterable[PendingRow]) -> None:
        rows = list(rows)
        queued = {row.sha256 for row in rows if row.status == "queued"}
        unfinished = {row.sha256 for row in rows}
        for counters, kept in (
            (self._lookups, queued),
            (self._unqueued, queued),
            (self._digests, queued),
            (self._errors, unfinished),
        ):
            for key in [key for key in counters if key not in kept]:
                del counters[key]

    async def _settle_orphans(self) -> None:
        try:
            approved = await asyncio.to_thread(lambda: requests.list_rows(statuses=[requests.APPROVED], limit=1000))
            ended: Dict[str, PendingRow] = {}
            for request in approved:
                if request.sha256 and request.sha256 not in ended:
                    row = await asyncio.to_thread(pending.get, request.sha256)
                    if row is not None and row.finished:
                        ended[row.sha256] = row
            for row in ended.values():
                await settle_requests(row, done=row.status == "enriched")
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not close the requests of finished downloads", exc_info=True)

    async def _advance_safely(self, row: PendingRow) -> None:
        try:
            await self.advance(row)
        except Exception as exc:
            count = self._errors.get(row.sha256, 0) + 1
            self._errors[row.sha256] = count
            if count == 1:
                logger.exception("[MODEL-BROWSER] Could not move %s on (%s); retrying on the next pass", row.sha256, row.status)
            else:
                logger.debug("[MODEL-BROWSER] %s still failing (%d): %s", row.sha256, count, exc)
            if count >= MAX_ROW_ERRORS:
                self._errors.pop(row.sha256, None)
                try:
                    await self._fail(row, f"Stopped following this download after repeated errors: {exc}")
                except Exception:
                    logger.exception("[MODEL-BROWSER] Could not mark %s failed", row.sha256)
        else:
            self._errors.pop(row.sha256, None)

    async def advance(self, row: PendingRow) -> Optional[PendingRow]:
        if row.status == "queued":
            row = await self._follow_download(row)
        if row is not None and row.status in FINISHING_STATUSES:
            row = await finish(row)
        return row

    async def _follow_download(self, row: PendingRow) -> Optional[PendingRow]:
        key = row.sha256
        if not row.download_id:
            waits = self._unqueued.get(key, 0) + 1
            self._unqueued[key] = waits
            if waits > UNQUEUED_GRACE_PASSES:
                return await self._fail(row, "The download was never put on the queue (the server stopped while queuing it)")
            return row
        self._unqueued.pop(key, None)

        download = await asyncio.to_thread(core_bridge.get_download, row.download_id)
        if download is None:
            if not row.nohash:
                model = await asyncio.to_thread(core_bridge.find_model_by_sha256, key)
                if model is not None:
                    return await self._indexed(row, model)
            return await self._fail(row, "The download is no longer on the download queue")
        status = download.get("status")
        if status in WAITING_STATUSES:
            return row
        if status in ("failed", "cancelled"):
            reason = download.get("error_message") or (
                "The download was cancelled" if status == "cancelled" else "The download failed"
            )
            return await self._fail(row, str(reason))
        if status != "completed":
            logger.warning("[MODEL-BROWSER] Download %s has an unknown status %r", row.download_id, status)
            return row
        return await self._locate(row, download)

    async def _locate(self, row: PendingRow, download: Mapping[str, Any]) -> Optional[PendingRow]:
        key = row.sha256
        attempts = self._lookups.get(key, 0) + 1
        self._lookups[key] = attempts
        path = str(download.get("destination_path") or "")
        model = None
        if not row.nohash:
            model = await asyncio.to_thread(core_bridge.find_model_by_sha256, key)
        if model is None and path and (row.nohash or attempts >= PATH_FALLBACK_AFTER):
            candidate = await asyncio.to_thread(core_bridge.find_model_by_path, path)
            if candidate is not None and indexed_since(candidate, row):
                model = candidate
        if model is None and row.nohash and attempts >= HASH_FALLBACK_AFTER and path:
            if not download.get("destination_backend_id"):
                digest = await self._digest(key, path)
                if digest:
                    model = await asyncio.to_thread(core_bridge.find_model_by_sha256, digest)
        if model is None:
            if attempts >= MAX_INDEX_LOOKUPS:
                return await self._fail(
                    row,
                    "The file was downloaded but never showed up in the model index; "
                    "an admin can index it from the Models page",
                )
            return row
        return await self._indexed(row, model)

    async def _digest(self, key: str, path: str) -> Optional[str]:
        if key not in self._digests:
            logger.info("[MODEL-BROWSER] Hashing %s to find its model", path)
            self._digests[key] = await asyncio.to_thread(sha256_of_file, path)
        return self._digests[key]

    async def _indexed(self, row: PendingRow, model: Mapping[str, Any]) -> Optional[PendingRow]:
        self._forget(row.sha256)
        return await asyncio.to_thread(record_indexed, row, model)

    async def _fail(self, row: PendingRow, message: str) -> Optional[PendingRow]:
        self._forget(row.sha256)
        return await fail(row, message)

    def _forget(self, key: str) -> None:
        for counters in (self._lookups, self._unqueued, self._digests):
            counters.pop(key, None)
