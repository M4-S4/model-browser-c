import json
import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

TABLE = "model_browser_group_permissions"

CAPABILITIES = ("browse", "request", "download", "approve")

SOURCE_IDS = ("civitai", "huggingface", "openmodeldb", "civarchive")

ASSIGN_TO = ("user", "group")

DEFAULT_CAPABILITIES = {
    "all_users": frozenset({"browse", "request"}),
    "all_admins": frozenset(CAPABILITIES),
}

DEFAULT_SOURCES = frozenset({"civitai", "huggingface", "openmodeldb"})


@dataclass(frozen=True)
class Row:
    group_id: str
    capabilities: FrozenSet[str] = frozenset()
    sources: FrozenSet[str] = DEFAULT_SOURCES
    model_types: FrozenSet[str] = frozenset()
    max_file_gb: Optional[float] = None
    assign_to: str = "user"
    allow_remote: bool = False
    updated_by: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("capabilities", "sources", "model_types"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))

    def params_json(self) -> Dict[str, Any]:
        return {
            "sources": sorted(self.sources),
            "model_types": sorted(self.model_types),
            "max_file_gb": self.max_file_gb,
            "assign_to": self.assign_to,
            "allow_remote": self.allow_remote,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "capabilities": sorted(self.capabilities),
            "params": self.params_json(),
            "updated_by": self.updated_by,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Row":
        params = data.get("params")
        if not isinstance(params, Mapping):
            params = {}
        assign_to = params.get("assign_to")
        updated_by = data.get("updated_by")
        return cls(
            group_id=str(data["group_id"]),
            capabilities=_strings(data.get("capabilities"), frozenset()) & frozenset(CAPABILITIES),
            sources=_strings(params.get("sources"), DEFAULT_SOURCES) & frozenset(SOURCE_IDS),
            model_types=_strings(params.get("model_types"), frozenset()),
            max_file_gb=_size_gb(params.get("max_file_gb")),
            assign_to=assign_to if assign_to in ASSIGN_TO else "user",
            allow_remote=params.get("allow_remote") is True,
            updated_by=updated_by if isinstance(updated_by, str) and updated_by else None,
        )


def default_row(group_id: str) -> Row:
    return Row(group_id=group_id, capabilities=DEFAULT_CAPABILITIES.get(group_id, frozenset()))


def row_or_default(rows: Mapping[str, Row], group_id: str) -> Row:
    row = rows.get(group_id)
    return row if row is not None else default_row(group_id)


def load_rows() -> Dict[str, Row]:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT group_id, capabilities, params, updated_by FROM {TABLE}")
        records = cursor.fetchall()
    return {
        record["group_id"]: Row.from_json(
            {
                "group_id": record["group_id"],
                "capabilities": _loads(record["capabilities"]),
                "params": _loads(record["params"]),
                "updated_by": record["updated_by"],
            }
        )
        for record in records
    }


def save_rows(rows: Iterable[Row], updated_by: Optional[str] = None) -> None:
    from src.plugin_api import db

    by_group = {row.group_id: row for row in rows}
    with db.get_cursor() as cursor:
        cursor.execute(f"DELETE FROM {TABLE}")
        cursor.executemany(
            f"INSERT INTO {TABLE} (group_id, capabilities, params, updated_by, updated_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [
                (row.group_id, json.dumps(sorted(row.capabilities)), json.dumps(row.params_json()), updated_by)
                for row in by_group.values()
            ],
        )


def delete_rows(group_ids: Iterable[str]) -> int:
    from src.plugin_api import db

    ids = sorted(set(group_ids))
    if not ids:
        return 0
    with db.get_cursor() as cursor:
        cursor.execute(f"DELETE FROM {TABLE} WHERE group_id IN ({', '.join('?' * len(ids))})", ids)
        return cursor.rowcount


def _loads(text: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _strings(value: Any, default: FrozenSet[str]) -> FrozenSet[str]:
    if not isinstance(value, list):
        return default
    return frozenset(str(item) for item in value)


def _size_gb(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None
