import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.plugin_api import LimitEvents, generate_ulid

from . import PLUGIN_ID, core_bridge

logger = logging.getLogger("plugins.model-browser")

DOWNLOADS = "model-browser.downloads"
DOWNLOAD_BYTES = "model-browser.download_bytes"
KINDS = (DOWNLOADS, DOWNLOAD_BYTES)
LABELS = {DOWNLOADS: "Model downloads per month", DOWNLOAD_BYTES: "Downloaded model bytes per month"}

_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB")


class QuotaExceeded(Exception):
    def __init__(
        self,
        kind: str,
        used: float,
        limit: float,
        incoming: Optional[float],
        resets_at: Optional[str],
        message: str,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.used = used
        self.limit = limit
        self.incoming = incoming
        self.resets_at = resets_at
        self.message = message

    def payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "label": LABELS.get(self.kind, self.kind),
            "used": _number(self.used),
            "limit": _number(self.limit),
            "incoming": None if self.incoming is None else _number(self.incoming),
            "resets_at": self.resets_at,
        }


def _number(value: float) -> float:
    value = float(value)
    return int(value) if value.is_integer() else value


def _trim(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def format_amount(kind: str, value: float) -> str:
    if kind != DOWNLOAD_BYTES:
        return _trim(float(value))
    value = float(value)
    index = 0
    while index < len(_BYTE_UNITS) - 1 and value >= 1024 ** (index + 1):
        index += 1
    return f"{_trim(value / 1024 ** index)} {_BYTE_UNITS[index]}"


def exceeded(limit: float, used: float, incoming: Optional[float]) -> bool:
    if incoming is None:
        return used >= limit
    return used + incoming > limit


def _resets(resets_at: Optional[str]) -> str:
    if not resets_at:
        return ""
    try:
        moment = datetime.fromisoformat(resets_at)
    except ValueError:
        return ""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return f" It resets on {moment.strftime('%b')} {moment.day} at {moment.strftime('%H:%M')} UTC."


def refusal_message(
    kind: str,
    used: float,
    limit: float,
    incoming: Optional[float],
    resets_at: Optional[str] = None,
    contact_line: str = "",
    requester_name: Optional[str] = None,
) -> str:
    label = LABELS.get(kind, kind).lower()
    usage = f"{format_amount(kind, used)} of {format_amount(kind, limit)}"
    file_goes_over = kind == DOWNLOAD_BYTES and incoming is not None and used < limit
    size = format_amount(kind, incoming) if incoming is not None else ""
    if requester_name is None:
        if file_goes_over:
            text = f"This file ({size}) would go over your {label} limit: you have used {usage}."
        else:
            text = f"You have reached your {label} limit: {usage} used."
    elif file_goes_over:
        text = f"This file ({size}) would go over the {label} limit of {requester_name}: they have used {usage}."
    else:
        text = f"{requester_name} has reached their {label} limit: {usage} used."
    text += _resets(resets_at)
    if requester_name is None and contact_line.strip():
        text += f" {contact_line.strip()}"
    return text


def check_quota(user_id: str, size_bytes: Optional[int], *, requester_name: Optional[str] = None) -> None:
    for kind in KINDS:
        try:
            status = core_bridge.limit_status(user_id, kind)
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not read the %s limit of %s; not enforcing it", kind, user_id, exc_info=True)
            continue
        limit, used = status.get("limit"), status.get("used")
        if limit is None or used is None or status.get("exempt"):
            continue
        if kind == DOWNLOADS:
            incoming: Optional[float] = 1
        else:
            incoming = size_bytes if isinstance(size_bytes, int) and size_bytes > 0 else None
        if exceeded(float(limit), float(used), incoming):
            message = refusal_message(
                kind,
                float(used),
                float(limit),
                incoming,
                status.get("resets_at"),
                str(status.get("contact_line") or ""),
                requester_name,
            )
            raise QuotaExceeded(kind, float(used), float(limit), incoming, status.get("resets_at"), message)


def new_ref_id() -> str:
    return f"{PLUGIN_ID}:{generate_ulid()}"


def record(user_id: str, size_bytes: Optional[int], ref_id: str) -> None:
    events = [(DOWNLOADS, 1)]
    if isinstance(size_bytes, int) and size_bytes > 0:
        events.append((DOWNLOAD_BYTES, size_bytes))
    for kind, units in events:
        try:
            LimitEvents.record(user_id, kind, units, ref_id)
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not record a %s event for %s", kind, user_id, exc_info=True)


def refund(ref_id: Optional[str]) -> int:
    if not ref_id:
        return 0
    try:
        return int(LimitEvents.refund(ref_id) or 0)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not refund the limit events of %s", ref_id, exc_info=True)
        return 0
