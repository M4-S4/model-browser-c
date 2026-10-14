import json
from dataclasses import replace
from typing import get_args

import pytest

from backend.permissions_store import (
    CAPABILITIES,
    DEFAULT_SOURCES,
    SOURCE_IDS,
    TABLE,
    Row,
    default_row,
    delete_rows,
    load_rows,
    row_or_default,
    save_rows,
)
from backend.schema import SourceId
from backend.tables import create_tables

_ARTISTS = Row(
    group_id="artists",
    capabilities=frozenset({"browse", "download"}),
    sources=frozenset({"civitai", "civarchive"}),
    model_types=frozenset({"lora", "embedding"}),
    max_file_gb=2.5,
    assign_to="group",
    allow_remote=True,
)
_VIEWERS = Row(group_id="viewers", capabilities=frozenset({"browse"}))


@pytest.fixture
def table(plugin_db):
    create_tables()
    return plugin_db


def _insert(db, group_id, capabilities, params):
    with db.get_cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {TABLE} (group_id, capabilities, params) VALUES (?, ?, ?)",
            (group_id, capabilities, params),
        )


def _stored(db):
    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT group_id, capabilities, params FROM {TABLE} ORDER BY group_id")
        return [tuple(record) for record in cursor.fetchall()]


def test_the_vocabulary_matches_the_schema_and_the_plan():
    assert CAPABILITIES == ("browse", "request", "download", "approve")
    assert set(SOURCE_IDS) == set(get_args(SourceId))
    assert DEFAULT_SOURCES == {"civitai", "huggingface", "openmodeldb"}


def test_default_rows():
    assert default_row("all_users") == Row(
        group_id="all_users",
        capabilities=frozenset({"browse", "request"}),
        sources=DEFAULT_SOURCES,
        model_types=frozenset(),
        max_file_gb=None,
        assign_to="user",
        allow_remote=False,
    )
    assert default_row("all_admins").capabilities == set(CAPABILITIES)
    assert default_row("all_admins").params_json() == default_row("all_users").params_json()
    assert default_row("restricted_content").capabilities == frozenset()
    assert default_row("artists").capabilities == frozenset()


def test_row_or_default_prefers_the_stored_row():
    rows = {"all_users": Row(group_id="all_users")}

    assert row_or_default(rows, "all_users") is rows["all_users"]
    assert row_or_default(rows, "all_admins") == default_row("all_admins")


def test_a_row_holds_frozensets_whatever_it_is_given():
    row = Row(group_id="g", capabilities={"browse"}, sources=["civitai"], model_types=("lora",))

    assert row.capabilities == frozenset({"browse"}) and isinstance(row.capabilities, frozenset)
    assert isinstance(row.sources, frozenset) and isinstance(row.model_types, frozenset)
    hash(row)


def test_to_json_is_the_shape_the_routes_send_with_sorted_lists():
    assert _ARTISTS.to_json() == {
        "group_id": "artists",
        "capabilities": ["browse", "download"],
        "params": {
            "sources": ["civarchive", "civitai"],
            "model_types": ["embedding", "lora"],
            "max_file_gb": 2.5,
            "assign_to": "group",
            "allow_remote": True,
        },
        "updated_by": None,
    }


def test_the_saving_admin_is_carried_beside_the_params_not_in_them():
    row = replace(_VIEWERS, updated_by="admin-1")

    assert row.to_json()["updated_by"] == "admin-1"
    assert "updated_by" not in row.params_json()


@pytest.mark.parametrize(
    "row",
    [_ARTISTS, _VIEWERS, default_row("all_users"), replace(_ARTISTS, updated_by="admin-1")],
    ids=["full", "browse", "default", "saved-by-an-admin"],
)
def test_from_json_reads_back_what_to_json_wrote(row):
    assert Row.from_json(row.to_json()) == row
    assert Row.from_json(json.loads(json.dumps(row.to_json()))) == row


def test_save_then_load_round_trips_every_field(table):
    save_rows([_ARTISTS, _VIEWERS])

    assert load_rows() == {"artists": _ARTISTS, "viewers": _VIEWERS}


def test_save_rows_records_the_saving_admin_on_every_row_it_stores(table):
    save_rows([_ARTISTS, replace(_VIEWERS, updated_by="someone-else")], "admin-1")

    assert load_rows() == {
        "artists": replace(_ARTISTS, updated_by="admin-1"),
        "viewers": replace(_VIEWERS, updated_by="admin-1"),
    }
    with table.get_cursor() as cursor:
        cursor.execute(f"SELECT group_id, updated_by FROM {TABLE} ORDER BY group_id")
        assert [tuple(record) for record in cursor.fetchall()] == [("artists", "admin-1"), ("viewers", "admin-1")]

    save_rows([_VIEWERS], "admin-2")

    assert load_rows() == {"viewers": replace(_VIEWERS, updated_by="admin-2")}


@pytest.mark.parametrize("updated_by", [7, "", ["admin-1"]], ids=["number", "empty", "list"])
def test_a_malformed_saving_admin_reads_as_none(updated_by):
    assert Row.from_json({"group_id": "g", "updated_by": updated_by}).updated_by is None


def test_a_table_created_before_the_saving_admin_was_recorded_gets_the_column_and_keeps_its_rows(plugin_db):
    with plugin_db.get_cursor() as cursor:
        cursor.execute(
            f"CREATE TABLE {TABLE} (group_id TEXT PRIMARY KEY, capabilities TEXT NOT NULL, params TEXT NOT NULL, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
    _insert(plugin_db, "viewers", '["browse"]', "{}")

    create_tables()
    create_tables()

    with plugin_db.get_cursor() as cursor:
        cursor.execute(f"PRAGMA table_info({TABLE})")
        assert [record["name"] for record in cursor.fetchall()][-1] == "updated_by"
    assert load_rows() == {"viewers": _VIEWERS}
    save_rows([_VIEWERS], "admin-1")
    assert load_rows()["viewers"].updated_by == "admin-1"


def test_rows_are_stored_as_json_text(table):
    save_rows([_ARTISTS])

    [(group_id, capabilities, params)] = _stored(table)
    assert group_id == "artists"
    assert json.loads(capabilities) == ["browse", "download"]
    assert json.loads(params) == _ARTISTS.params_json()


def test_save_rows_replaces_the_whole_table(table):
    save_rows([_ARTISTS, _VIEWERS])
    changed = Row(group_id="viewers", capabilities=frozenset({"browse", "request"}))

    save_rows([changed])

    assert load_rows() == {"viewers": changed}


def test_save_rows_with_nothing_empties_the_table(table):
    save_rows([_ARTISTS, _VIEWERS])

    save_rows([])

    assert load_rows() == {}


def test_save_rows_keeps_the_last_row_of_a_group_listed_twice(table):
    second = Row(group_id="artists", capabilities=frozenset({"request"}))

    save_rows([_ARTISTS, second])

    assert load_rows() == {"artists": second}


def test_delete_rows_deletes_only_those_groups_and_counts_them(table):
    save_rows([_ARTISTS, _VIEWERS])

    assert delete_rows(["artists", "deleted-long-ago"]) == 1
    assert delete_rows([]) == 0
    assert load_rows() == {"viewers": _VIEWERS}


def test_a_row_with_empty_params_takes_every_default(table):
    _insert(table, "artists", '["download"]', "{}")

    assert load_rows()["artists"] == Row(group_id="artists", capabilities=frozenset({"download"}))


@pytest.mark.parametrize(
    "capabilities, params",
    [
        ("not json", "not json either"),
        ("null", "null"),
        ('{"browse": true}', '["civitai"]'),
        ("", ""),
    ],
    ids=["garbage", "null", "wrong-containers", "empty-strings"],
)
def test_an_unreadable_row_grants_nothing_and_takes_the_default_params(table, capabilities, params):
    _insert(table, "artists", capabilities, params)

    assert load_rows()["artists"] == Row(group_id="artists")


def test_malformed_params_each_fall_back_to_their_default(table):
    _insert(
        table,
        "artists",
        '["download", "manage", 7]',
        json.dumps(
            {
                "sources": "civitai",
                "model_types": {"lora": True},
                "max_file_gb": "5",
                "assign_to": "everyone",
                "allow_remote": "yes",
                "unknown_param": 1,
            }
        ),
    )

    assert load_rows()["artists"] == Row(group_id="artists", capabilities=frozenset({"download"}))


@pytest.mark.parametrize("size", [0, -1, True, None, "Infinity"])
def test_a_size_that_is_not_a_positive_number_means_no_limit(size):
    params = {"max_file_gb": float(size) if size == "Infinity" else size}

    assert Row.from_json({"group_id": "g", "params": params}).max_file_gb is None


def test_unknown_capabilities_and_sources_are_dropped_but_model_types_are_kept():
    row = Row.from_json(
        {
            "group_id": "g",
            "capabilities": ["browse", "manage"],
            "params": {"sources": ["civitai", "pixiv"], "model_types": ["no_such_type"]},
        }
    )

    assert row.capabilities == {"browse"}
    assert row.sources == {"civitai"}
    assert row.model_types == {"no_such_type"}


def test_from_json_needs_a_group_id():
    with pytest.raises(KeyError):
        Row.from_json({"capabilities": ["browse"]})
