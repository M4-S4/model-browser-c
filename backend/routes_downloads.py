import logging
import re
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import core_bridge, downloads, nsfw, pending
from .downloads import DownloadError, DownloadRequest
from .http import SourceError
from .installed import installed_map
from .permissions import Principal, require_any, require_browse, require_download
from .routes_sources import source_http_error

logger = logging.getLogger("plugins.model-browser")

router = APIRouter()

require_downloads_access = require_any("download", "request")

MAX_INSTALLED_HASHES = 200
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _error(status: int, code: str, message: str, extra: Optional[Mapping[str, Any]] = None) -> HTTPException:
    return HTTPException(
        status_code=status, detail={"success": False, "error": code, "message": message, **dict(extra or {})}
    )


def download_http_error(exc: DownloadError) -> HTTPException:
    return _error(exc.status, exc.code, exc.message, exc.extra)


@router.post("/downloads")
async def post_download(
    body: Any = Body(...),
    principal: Principal = Depends(require_download),
) -> Dict[str, Any]:
    try:
        request = DownloadRequest.model_validate(body)
    except ValidationError as error:
        raise _error(400, "invalid_request", downloads.describe_validation_error(error)) from None
    try:
        data = await downloads.start_download(principal, request)
    except DownloadError as exc:
        raise download_http_error(exc) from None
    except SourceError as exc:
        raise source_http_error(exc) from None
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not start the download of %s %s", request.source, request.model_id)
        raise _error(500, "internal_error", "The download could not be started; the server log has the details") from None
    return {"success": True, "data": data}


@router.get("/downloads")
def list_downloads(
    all_users: Optional[bool] = Query(None, alias="all"),
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    principal: Principal = Depends(require_downloads_access),
) -> Dict[str, Any]:
    user = principal.user
    everyone = (principal.is_admin or principal.grant.has("approve")) and all_users is not False
    rows = pending.list_rows(requested_by=None if everyone else user.id, limit=limit)
    ctx = nsfw.resolve_nsfw(user)
    return {
        "success": True,
        "data": [
            {
                "pending": downloads.row_for_viewer(row, ctx),
                "download": downloads.public_download(_download(row), admin=principal.is_admin),
            }
            for row in rows
        ],
    }


def _download(row: pending.PendingRow) -> Optional[Dict[str, Any]]:
    if not row.download_id:
        return None
    try:
        return core_bridge.get_download(row.download_id)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not read download %s", row.download_id, exc_info=True)
        return None


class InstalledRequest(BaseModel):
    sha256s: List[str] = Field(max_length=MAX_INSTALLED_HASHES)

    @field_validator("sha256s")
    @classmethod
    def _hashes(cls, values: List[str]) -> List[str]:
        hashes: Dict[str, None] = {}
        for value in values:
            sha256 = value.strip().lower()
            if not _SHA256.fullmatch(sha256):
                raise ValueError(f"'{value[:80]}' is not a SHA256 (64 hexadecimal characters)")
            hashes[sha256] = None
        return list(hashes)


@router.post("/installed")
async def installed(
    body: Any = Body(...),
    principal: Principal = Depends(require_browse),
) -> Dict[str, Any]:
    try:
        request = InstalledRequest.model_validate(body)
    except ValidationError as error:
        raise _error(400, "invalid_request", downloads.describe_validation_error(error, "hash list")) from None
    return {"success": True, "data": {"installed": await installed_map(request.sha256s, principal)}}
