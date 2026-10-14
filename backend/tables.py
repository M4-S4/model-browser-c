_COLUMNS = {
    "model_browser_group_permissions": """
        group_id TEXT PRIMARY KEY,
        capabilities TEXT NOT NULL,
        params TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by TEXT
    """,
    "model_browser_requests": """
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        source TEXT NOT NULL,
        source_model_id TEXT NOT NULL,
        source_version_id TEXT,
        file_id TEXT NOT NULL,
        snapshot TEXT NOT NULL,
        note TEXT,
        status TEXT NOT NULL,
        decided_by TEXT,
        decision_note TEXT,
        sha256 TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        assign_to TEXT
    """,
    "model_browser_cache": """
        key TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        payload TEXT NOT NULL,
        fetched_at TIMESTAMP NOT NULL
    """,
    "model_browser_pending": """
        sha256 TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_model_id TEXT NOT NULL,
        source_version_id TEXT,
        file_name TEXT NOT NULL,
        info TEXT NOT NULL,
        requested_by TEXT NOT NULL,
        assign_to TEXT NOT NULL,
        request_id TEXT,
        download_id TEXT,
        limit_ref_id TEXT,
        model_id TEXT,
        status TEXT NOT NULL,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """,
}

TABLE_NAMES = tuple(_COLUMNS)

_ADDED_COLUMNS = (
    ("model_browser_requests", "assign_to", "TEXT"),
    ("model_browser_group_permissions", "updated_by", "TEXT"),
)


def create_tables() -> None:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        for name, columns in _COLUMNS.items():
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {name} ({columns})")
        for table, column, column_type in _ADDED_COLUMNS:
            cursor.execute(f"PRAGMA table_info({table})")
            if column not in {row["name"] for row in cursor.fetchall()}:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
