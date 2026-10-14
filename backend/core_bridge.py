import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, get_args

from src.plugin_api import (
    DownloadQueueException,
    InvalidModelMetadataException,
    ModelAssignmentException,
    ModelNotFoundException,
    UnsafeFilenameException,
    User,
)
from src.plugin_api import downloads as api_downloads
from src.plugin_api import groups as api_groups
from src.plugin_api import limits as api_limits
from src.plugin_api import models as api_models
from src.plugin_api import notifications as api_notifications

logger = logging.getLogger("plugins.model-browser")

ContentPolicy = Literal["allowed", "blur", "blocked"]
CONTENT_POLICIES = get_args(ContentPolicy)

NOTIFICATION_SOURCE = "model-browser"

MODEL_KEYS = ("id", "sha256", "is_available", "created_at", "filename", "model_type")

_FILE_UNAVAILABLE = "ModelFileUnavailable"

_INVALID_QUEUE_MESSAGES = (
    "must be a relative path",
    "escapes the configured directory",
    "No backend configured with id",
    "is not a native.remote backend",
    "is not fully configured",
    "Remote destinations are not available",
)


class CoreBridgeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _queue_refusal_code(message: str) -> str:
    return "invalid" if any(part in message for part in _INVALID_QUEUE_MESSAGES) else "blocked"


async def queue_model_download(
    url: str,
    *,
    filename: Optional[str] = None,
    model_type: Optional[str] = None,
    subdir: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    checksum_sha256: Optional[str] = None,
    provider_id: Optional[str] = None,
    created_by: Optional[str] = None,
    destination_backend_id: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        return await api_downloads.queue_model_download(
            url,
            filename=filename,
            model_type=model_type,
            subdir=subdir,
            tags=list(tags) if tags is not None else None,
            checksum_sha256=checksum_sha256,
            provider_id=provider_id,
            created_by=created_by,
            destination_backend_id=destination_backend_id,
        )
    except UnsafeFilenameException as exc:
        raise CoreBridgeError("invalid", str(exc)) from exc
    except DownloadQueueException as exc:
        raise CoreBridgeError(_queue_refusal_code(str(exc)), str(exc)) from exc


def get_download(download_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not download_id:
        return None
    return api_downloads.get_download(download_id)


def _trimmed(model: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    return {key: model.get(key) for key in MODEL_KEYS}


def find_model_by_sha256(sha256: str) -> Optional[Dict[str, Any]]:
    if not sha256:
        return None
    return _trimmed(api_models.model_for_sha256(sha256, admin=True))


def find_model_by_path(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return _trimmed(api_models.model_for_path(path, admin=True))


def find_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    if not model_id:
        return None
    try:
        path = api_models.resolve_model_file(model_id)
    except Exception as exc:
        if type(exc).__name__ != _FILE_UNAVAILABLE:
            raise
        return None
    return _trimmed(api_models.model_for_path(str(path), admin=True))


def visible_model_ids(user_id: str) -> List[str]:
    return api_models.visible_model_ids(user_id)


def assign_to_user(model_id: str, user_id: str) -> bool:
    try:
        return api_models.assign_model_to_user(model_id, user_id)
    except ModelAssignmentException as exc:
        raise CoreBridgeError("refused", str(exc)) from exc


def assign_to_group(group_id: str, model_ids: Sequence[str], acting_user: User) -> List[str]:
    if group_id not in {group["id"] for group in api_groups.list_groups()}:
        raise CoreBridgeError("not_found", f"There is no user group with the id '{group_id}'")
    if not isinstance(acting_user, User):
        raise CoreBridgeError("refused", f"Assigning models to the group '{group_id}' needs an acting admin")
    model_ids = [model_id for model_id in model_ids if model_id]
    if not model_ids:
        return []
    try:
        return api_groups.assign_models_to_group(group_id, model_ids, acting_user)
    except ValueError as exc:
        raise CoreBridgeError("refused", str(exc)) from exc


async def run_provider_fetch(provider: str, model_ids: Sequence[str]) -> Dict[str, Any]:
    model_ids = [model_id for model_id in model_ids if model_id]
    if not model_ids:
        return {"successful": 0, "failed": 0}
    return await api_models.run_provider_fetch(provider, model_ids)


def write_attributes(model_id: str, values: Mapping[str, Any]) -> None:
    try:
        api_models.update_model_metadata(model_id, dict(values))
    except InvalidModelMetadataException as exc:
        logger.warning("[MODEL-BROWSER] The core refused the attributes of model %s: %s", model_id, exc)
        raise CoreBridgeError("invalid", str(exc)) from exc
    except ModelNotFoundException as exc:
        raise CoreBridgeError("not_found", str(exc)) from exc


def effective_policy(user_id: str) -> ContentPolicy:
    mode = api_groups.effective_content_policy(user_id)
    return mode if mode in CONTENT_POLICIES else "blocked"


def user_groups(user_id: str) -> List[Dict[str, Any]]:
    return api_groups.get_user_groups(user_id)


def list_groups() -> List[Dict[str, Any]]:
    return api_groups.list_groups()


def group_member_ids(group_id: str) -> List[str]:
    return api_groups.group_member_ids(group_id)


def notify(
    *,
    user_id: str,
    notification_type: str,
    title: str,
    message: str = "",
    level: str = "info",
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if not user_id:
        logger.warning("[MODEL-BROWSER] Not sending %r: no user to notify", notification_type)
        return
    try:
        api_notifications.notify(
            source=NOTIFICATION_SOURCE,
            title=title,
            message=message,
            user_id=user_id,
            type=notification_type,
            level=level,
            category="plugin",
            metadata=dict(metadata) if metadata else None,
        )
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not send the %r notification to %s", notification_type, user_id, exc_info=True)


def register_notification_types(registry: Any, types: Sequence[Any]) -> List[str]:
    specs = [
        api_notifications.NotificationTypeSpec(
            key=item.key, label=item.label, description=item.description, category=item.category
        )
        for item in types
    ]
    return api_notifications.register_notification_types(registry, specs)


def _unlimited(kind: str) -> Dict[str, Any]:
    return {"kind": kind, "limit": None, "used": None, "exempt": False, "resets_at": None, "contact_line": ""}


def _utc_iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.astimezone(timezone.utc).isoformat() if moment is not None else None


def limit_status(user_id: str, kind: str) -> Dict[str, Any]:
    try:
        status = api_limits.limit_status(user_id, kind)
        return {
            "kind": status.kind,
            "limit": status.limit,
            "used": status.used,
            "exempt": status.exempt,
            "resets_at": _utc_iso(status.resets_at),
            "contact_line": status.contact_line,
        }
    except Exception:
        logger.warning(
            "[MODEL-BROWSER] Could not read the %s limit of user %s; treating it as unlimited", kind, user_id, exc_info=True
        )
        return _unlimited(kind)
