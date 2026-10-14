import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional

TABLE = "model_browser_pending"
STATUSES = ("queued", "indexed", "assigned", "enriched", "failed")
UNFINISHED = ("queued", "indexed", "assigned")
NOHASH_PREFIX = "nohash:"

_COLUMNS = (
    "sha256", "source", "source_model_id", "source_version_id", "file_name", "info",
    "requested_by", "assign_to", "request_id", "download_id", "limit_ref_id", "model_id",
    "status", "error", "created_at", "updated_at",
)
_UPDATABLE = frozenset({"download_id", "limit_ref_id", "model_id", "status", "error", "assign_to", "request_id"})


def key_for(sha256: Optional[str], download_url: str) -> str:
    if sha256:
        return sha256.strip().lower()
    return NOHASH_PREFIX + download_url


def is_nohash(key: str) -> bool:
    return key.startswith(NOHASH_PREFIX)


@dataclass(frozen=True)
class PendingRow:
    sha256: str
    source: str
    source_model_id: str
    file_name: str
    requested_by: str
    assign_to: str
    info: Dict[str, Any] = field(default_factory=dict)
    source_version_id: Optional[str] = None
    request_id: Optional[str] = None
    download_id: Optional[str] = None
    limit_ref_id: Optional[str] = None
    model_id: Optional[str] = None
    status: str = "queued"
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def nohash(self) -> bool:
        return is_nohash(self.sha256)

    @property
    def finished(self) -> bool:
        return self.status not in UNFINISHED

    def to_json(self) -> Dict[str, Any]:
        return {column: getattr(self, column) for column in _COLUMNS}


def _row_from_record(record: Mapping[str, Any]) -> PendingRow:
    try:
        info = json.loads(record["info"] or "{}")
    except (TypeError, ValueError):
        info = {}
    values = {column: record[column] for column in _COLUMNS if column != "info"}
    for stamp in ("created_at", "updated_at"):
        if values.get(stamp) is not None:
            values[stamp] = str(values[stamp])
    return PendingRow(info=info if isinstance(info, dict) else {}, **values)


def get(key: str) -> Optional[PendingRow]:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT {', '.join(_COLUMNS)} FROM {TABLE} WHERE sha256 = ?", (key,))
        record = cursor.fetchone()
    return _row_from_record(record) if record else None


def get_by_download_id(download_id: Optional[str]) -> Optional[PendingRow]:
    from src.plugin_api import db

    if not download_id:
        return None
    with db.get_cursor() as cursor:
        cursor.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {TABLE} WHERE download_id = ? ORDER BY created_at DESC LIMIT 1",
            (download_id,),
        )
        record = cursor.fetchone()
    return _row_from_record(record) if record else None


def insert(row: PendingRow) -> PendingRow:
    from src.plugin_api import db

    if row.status not in STATUSES:
        raise ValueError(f"unknown pending status {row.status!r}")
    with db.get_cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {TABLE} (sha256, source, source_model_id, source_version_id, file_name, info, "
            f"requested_by, assign_to, request_id, download_id, limit_ref_id, model_id, status, error) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.sha256, row.source, row.source_model_id, row.source_version_id, row.file_name,
                json.dumps(row.info), row.requested_by, row.assign_to, row.request_id, row.download_id,
                row.limit_ref_id, row.model_id, row.status, row.error,
            ),
        )
    return get(row.sha256) or row


def update(key: str, *, only_from: Optional[Iterable[str]] = None, **fields: Any) -> Optional[PendingRow]:
    from src.plugin_api import db

    unknown = set(fields) - _UPDATABLE
    if unknown:
        raise ValueError(f"cannot update {sorted(unknown)} on a pending row")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"unknown pending status {fields['status']!r}")
    if not fields:
        return get(key)
    assignments = ", ".join(f"{name} = ?" for name in fields)
    query = f"UPDATE {TABLE} SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE sha256 = ?"
    params: List[Any] = [*fields.values(), key]
    if only_from is not None:
        allowed = [status for status in only_from if status in STATUSES]
        if not allowed:
            return None
        query += f" AND status IN ({', '.join('?' * len(allowed))})"
        params.extend(allowed)
    with db.get_cursor() as cursor:
        cursor.execute(query, params)
        changed = cursor.rowcount > 0
    return get(key) if changed or only_from is None else None


def rekey(old_key: str, sha256: str) -> Optional[PendingRow]:
    from src.plugin_api import db

    new_key = sha256.strip().lower()
    existing = get(new_key)
    with db.get_cursor() as cursor:
        if existing is not None:
            cursor.execute(f"DELETE FROM {TABLE} WHERE sha256 = ?", (old_key,))
        else:
            cursor.execute(
                f"UPDATE {TABLE} SET sha256 = ?, updated_at = CURRENT_TIMESTAMP WHERE sha256 = ?",
                (new_key, old_key),
            )
    return get(new_key)


def delete(key: str) -> bool:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        cursor.execute(f"DELETE FROM {TABLE} WHERE sha256 = ?", (key,))
        return cursor.rowcount > 0


def list_rows(
    *,
    requested_by: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
    limit: int = 200,
) -> List[PendingRow]:
    from src.plugin_api import db

    clauses, params = [], []
    if requested_by is not None:
        clauses.append("requested_by = ?")
        params.append(requested_by)
    if statuses is not None:
        wanted = [status for status in statuses if status in STATUSES]
        if not wanted:
            return []
        clauses.append(f"status IN ({', '.join('?' * len(wanted))})")
        params.extend(wanted)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.get_cursor() as cursor:
        cursor.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {TABLE}{where} ORDER BY created_at DESC, sha256 LIMIT ?",
            (*params, max(1, min(int(limit), 1000))),
        )
        records = cursor.fetchall()
    return [_row_from_record(record) for record in records]


def list_unfinished() -> List[PendingRow]:
    rows = list_rows(statuses=UNFINISHED, limit=1000)
    return list(reversed(rows))


def with_status(row: PendingRow, status: str, error: Optional[str] = None) -> PendingRow:
    return replace(row, status=status, error=error)
