import logging
from typing import Any, Dict, Optional

from src.plugin_api import HookContext

if __name__.count(".") >= 2:
    from ..backend import core_bridge, pending, watcher
    from ..backend.pending import PendingRow
else:
    from backend import core_bridge, pending, watcher
    from backend.pending import PendingRow

logger = logging.getLogger("plugins.model-browser")

NOT_INDEXED = "The file was downloaded but not indexed"


def on_download_complete(context: HookContext) -> HookContext:
    try:
        _complete(context)
    except Exception:
        logger.exception(
            "[MODEL-BROWSER] Could not take up the completed download %s; the sweep will", context.get("download_id")
        )
    return context


def _complete(context: HookContext) -> None:
    download_id = context.get("download_id")
    row = _row_of(download_id, context.get("checksum_sha256"), context.get("url"))
    if row is None:
        return
    if row.status != "queued":
        logger.info(
            "[MODEL-BROWSER] Download %s completed, but %s is %s already: left as it is", download_id, row.file_name, row.status
        )
        return
    model_id = context.get("model_id") or context.get("duplicate_of")
    if not model_id:
        if context.get("destination_backend_id"):
            logger.info(
                "[MODEL-BROWSER] %s landed on the remote backend %s; the sweep will find its model",
                row.file_name,
                context.get("destination_backend_id"),
            )
            return
        warning = context.get("warning")
        message = f"{NOT_INDEXED}: {warning}" if warning else f"{NOT_INDEXED}; an admin can index it from the Models page"
        watcher.schedule_step(watcher.fail(row, message), row.sha256)
        return
    if not context.get("model_id"):
        logger.info("[MODEL-BROWSER] %s is a copy of model %s", row.file_name, model_id)
    model = _model(row, model_id, copy=not context.get("model_id"), path=context.get("destination_path"))
    updated = watcher.record_indexed(row, model)
    if updated is not None and updated.status in watcher.FINISHING_STATUSES:
        watcher.schedule_step(watcher.carry_on(updated.sha256), updated.sha256)


def _row_of(download_id: Optional[str], checksum: Optional[str], url: Optional[str]) -> Optional[PendingRow]:
    row = pending.get_by_download_id(download_id)
    if row is not None or not url:
        return row
    row = pending.get(pending.key_for(checksum, url))
    if row is not None and row.download_id in (None, download_id):
        return row
    return None


def _model(row: PendingRow, model_id: str, *, copy: bool, path: Optional[str]) -> Dict[str, Any]:
    if not row.nohash:
        return {"id": model_id}
    found = core_bridge.find_model_by_path(path) if path and not copy else None
    if found is None or found.get("id") != model_id:
        found = core_bridge.find_model_by_id(model_id)
    sha256 = found.get("sha256") if found is not None and found.get("id") == model_id else None
    return {"id": model_id, "sha256": sha256}
