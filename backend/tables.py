_COLUMNS = {
    "model_browser_group_permissions": """
        group_id TEXT PRIMARY KEY,      -- user_groups.id
        capabilities TEXT NOT NULL,     -- JSON list: "browse", "request", "download", "approve"
        params TEXT NOT NULL,           -- JSON: {sources, model_types, max_file_gb, assign_to, allow_remote}
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by TEXT                 -- id of the admin who last saved the rows (phase 9)
    """,
    "model_browser_requests": """
        id TEXT PRIMARY KEY,            -- generate_ulid()
        user_id TEXT NOT NULL,
        source TEXT NOT NULL,
        source_model_id TEXT NOT NULL,
        source_version_id TEXT,
        file_id TEXT NOT NULL,
        snapshot TEXT NOT NULL,         -- JSON: name, type, size, nsfw, license, preview when requested
        note TEXT,
        status TEXT NOT NULL,           -- open | approved | rejected | cancelled | done | failed
        decided_by TEXT,
        decision_note TEXT,
        sha256 TEXT,                    -- model_browser_pending.sha256 once the download is queued
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        assign_to TEXT                  -- "user:<id>" | "group:<id>" | "none", decided at approval (phase 5)
    """,
    "model_browser_cache": """
        key TEXT PRIMARY KEY,           -- md5(source + normalized query); raw, unfiltered response
        source TEXT NOT NULL,
        payload TEXT NOT NULL,
        fetched_at TIMESTAMP NOT NULL   -- purged every few writes (TTL and row cap, backend/cache.py)
    """,
    "model_browser_pending": """
        sha256 TEXT PRIMARY KEY,        -- lowercase
        source TEXT NOT NULL,
        source_model_id TEXT NOT NULL,
        source_version_id TEXT,
        file_name TEXT NOT NULL,
        info TEXT NOT NULL,             -- JSON: ModelDetail cut down to the chosen version and file
        requested_by TEXT NOT NULL,     -- user the model is downloaded for
        assign_to TEXT NOT NULL,        -- "user:<id>" | "group:<id>" | "none"
        request_id TEXT,                -- model_browser_requests.id when it comes from a request
        download_id TEXT,
        limit_ref_id TEXT,              -- LimitEvents.record(...) ref, refunded on failure
        model_id TEXT,
        status TEXT NOT NULL,           -- queued | indexed | enriched | assigned | failed
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
