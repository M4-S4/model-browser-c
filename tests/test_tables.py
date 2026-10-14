from backend.tables import TABLE_NAMES, create_tables

_EXPECTED_COLUMNS = {
    "model_browser_group_permissions": [
        "group_id", "capabilities", "params", "updated_at",
        "updated_by",
    ],
    "model_browser_requests": [
        "id", "user_id", "source", "source_model_id", "source_version_id", "file_id", "snapshot",
        "note", "status", "decided_by", "decision_note", "sha256", "created_at", "updated_at",
        "assign_to",
    ],
    "model_browser_cache": ["key", "source", "payload", "fetched_at"],
    "model_browser_pending": [
        "sha256", "source", "source_model_id", "source_version_id", "file_name", "info",
        "requested_by", "assign_to", "request_id", "download_id", "limit_ref_id", "model_id",
        "status", "error", "created_at", "updated_at",
    ],
}


def _table_names(db):
    with db.get_cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        return {row["name"] for row in cursor.fetchall()}


def _columns(db, table):
    with db.get_cursor() as cursor:
        cursor.execute(f"PRAGMA table_info({table})")
        return [row["name"] for row in cursor.fetchall()]


def test_table_names_carry_the_plugin_prefix():
    assert len(TABLE_NAMES) == 4
    assert all(name.startswith("model_browser_") for name in TABLE_NAMES)


def test_create_tables_creates_exactly_the_plugin_tables(plugin_db):
    create_tables()

    assert _table_names(plugin_db) == set(TABLE_NAMES)


def test_tables_have_the_planned_columns(plugin_db):
    create_tables()

    assert {table: _columns(plugin_db, table) for table in TABLE_NAMES} == _EXPECTED_COLUMNS


def test_a_requests_table_from_phase_3_gets_the_new_column_and_keeps_its_rows(plugin_db):
    with plugin_db.get_cursor() as cursor:
        cursor.execute(
            "CREATE TABLE model_browser_requests (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source TEXT NOT NULL, "
            "source_model_id TEXT NOT NULL, source_version_id TEXT, file_id TEXT NOT NULL, snapshot TEXT NOT NULL, "
            "note TEXT, status TEXT NOT NULL, decided_by TEXT, decision_note TEXT, sha256 TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cursor.execute(
            "INSERT INTO model_browser_requests (id, user_id, source, source_model_id, file_id, snapshot, status) "
            "VALUES ('r-1', 'user-1', 'civitai', '1', '3', '{}', 'open')"
        )

    create_tables()
    create_tables()

    assert _columns(plugin_db, "model_browser_requests") == _EXPECTED_COLUMNS["model_browser_requests"]
    with plugin_db.get_cursor() as cursor:
        cursor.execute("SELECT id, status, assign_to FROM model_browser_requests")
        assert [tuple(row) for row in cursor.fetchall()] == [("r-1", "open", None)]


def test_create_tables_twice_keeps_tables_and_rows(plugin_db):
    create_tables()
    with plugin_db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO model_browser_group_permissions (group_id, capabilities, params) VALUES (?, ?, ?)",
            ("all_users", '["browse", "request"]', "{}"),
        )

    create_tables()

    assert _table_names(plugin_db) == set(TABLE_NAMES)
    with plugin_db.get_cursor() as cursor:
        cursor.execute("SELECT group_id, capabilities FROM model_browser_group_permissions")
        assert [tuple(row) for row in cursor.fetchall()] == [("all_users", '["browse", "request"]')]
