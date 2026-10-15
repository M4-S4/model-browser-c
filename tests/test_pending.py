import sqlite3

import pytest

from backend import pending
from backend.pending import NOHASH_PREFIX, PendingRow
from backend.tables import create_tables

SHA = "a" * 64
OTHER_SHA = "b" * 64
URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
INFO = {
    "model": {"source": "civitai", "id": "122359", "name": "Detail Tweaker XL", "versions": [{"id": "135867", "files": [{"id": "1"}]}]},
    "version_id": "135867",
    "file_id": "1",
    "mb_type": "lora",
}


@pytest.fixture
def table(plugin_db):
    create_tables()
    return plugin_db


def _row(key=SHA, **fields):
    values = dict(
        sha256=key,
        source="civitai",
        source_model_id="122359",
        source_version_id="135867",
        file_name="add-detail-xl.safetensors",
        requested_by="user-1",
        assign_to="user:user-1",
        info=INFO,
    )
    values.update(fields)
    return PendingRow(**values)


def _set_created_at(db, key, stamp):
    with db.get_cursor() as cursor:
        cursor.execute(f"UPDATE {pending.TABLE} SET created_at = ? WHERE sha256 = ?", (stamp, key))


def test_the_key_is_the_lower_case_hash_or_the_url_behind_the_nohash_prefix():
    assert pending.key_for("  " + "AB" * 32 + " ", URL) == "ab" * 32
    assert pending.key_for(None, URL) == NOHASH_PREFIX + URL
    assert pending.key_for("", URL) == NOHASH_PREFIX + URL
    assert pending.is_nohash(NOHASH_PREFIX + URL)
    assert not pending.is_nohash(SHA)


def test_a_row_knows_whether_it_is_finished_and_whether_it_has_a_hash():
    assert not _row().nohash
    assert _row(key=NOHASH_PREFIX + URL).nohash
    assert [_row(status=status).finished for status in pending.STATUSES] == [False, False, False, True, True]


def test_insert_then_get_round_trips_every_column(table):
    stored = pending.insert(_row(request_id="req-1", limit_ref_id="ref-1"))

    assert stored == pending.get(SHA)
    assert stored.info == INFO
    assert (stored.status, stored.error, stored.download_id, stored.model_id) == ("queued", None, None, None)
    assert (stored.request_id, stored.limit_ref_id) == ("req-1", "ref-1")
    assert stored.created_at and stored.updated_at
    assert set(stored.to_json()) == set(pending._COLUMNS)
    assert stored.to_json()["info"] == INFO


def test_get_of_an_unknown_key_is_none(table):
    assert pending.get(SHA) is None


def test_get_by_download_id_finds_the_row_of_a_core_download(table):
    pending.insert(_row(download_id="dl-1"))
    pending.insert(_row(key=OTHER_SHA, download_id="dl-2"))
    pending.insert(_row(key=NOHASH_PREFIX + URL))

    assert pending.get_by_download_id("dl-2") == pending.get(OTHER_SHA)
    assert pending.get_by_download_id("dl-1").sha256 == SHA
    assert pending.get_by_download_id("dl-9") is None
    assert pending.get_by_download_id(None) is None
    assert pending.get_by_download_id("") is None


def test_a_second_row_for_the_same_key_is_refused(table):
    pending.insert(_row())

    with pytest.raises(sqlite3.IntegrityError):
        pending.insert(_row(requested_by="user-2"))


def test_insert_refuses_an_unknown_status(table):
    with pytest.raises(ValueError, match="unknown pending status"):
        pending.insert(_row(status="downloading"))
    assert pending.get(SHA) is None


def test_an_unreadable_info_column_reads_as_an_empty_dict(table):
    pending.insert(_row())
    with table.get_cursor() as cursor:
        cursor.execute(f"UPDATE {pending.TABLE} SET info = ? WHERE sha256 = ?", ("{not json", SHA))
    assert pending.get(SHA).info == {}

    with table.get_cursor() as cursor:
        cursor.execute(f"UPDATE {pending.TABLE} SET info = ? WHERE sha256 = ?", ("[1, 2]", SHA))
    assert pending.get(SHA).info == {}


def test_update_sets_the_given_fields_only(table):
    pending.insert(_row())

    updated = pending.update(SHA, download_id="dl-1", status="indexed", model_id="model-1")

    assert (updated.download_id, updated.status, updated.model_id) == ("dl-1", "indexed", "model-1")
    assert updated.assign_to == "user:user-1" and updated.info == INFO
    assert pending.update(SHA, error="boom", assign_to="group:artists", request_id="req-2").error == "boom"
    assert pending.get(SHA).assign_to == "group:artists"


def test_update_refuses_columns_that_are_not_updatable_and_unknown_statuses(table):
    pending.insert(_row())

    with pytest.raises(ValueError, match="cannot update"):
        pending.update(SHA, info={}, status="indexed")
    with pytest.raises(ValueError, match="unknown pending status"):
        pending.update(SHA, status="done")
    assert pending.get(SHA).status == "queued"


def test_update_without_fields_returns_the_row_and_an_unknown_key_none(table):
    pending.insert(_row())

    assert pending.update(SHA) == pending.get(SHA)
    assert pending.update(OTHER_SHA, status="failed") is None


def test_update_only_from_some_statuses_changes_nothing_on_a_row_in_another(table):
    pending.insert(_row(status="indexed", model_id="model-1"))

    assert pending.update(SHA, only_from=("queued",), status="indexed", model_id="model-2") is None
    assert (pending.get(SHA).status, pending.get(SHA).model_id) == ("indexed", "model-1")

    moved = pending.update(SHA, only_from=("indexed",), status="assigned", error="a note")
    assert moved == pending.get(SHA)
    assert (moved.status, moved.error) == ("assigned", "a note")
    assert pending.update(SHA, only_from=pending.UNFINISHED, status="failed") is not None
    assert pending.update(SHA, only_from=pending.UNFINISHED, status="failed", error="again") is None
    assert pending.get(SHA).error == "a note"
    assert pending.update(SHA, only_from=("done",), status="enriched") is None
    assert pending.update(OTHER_SHA, only_from=("queued",), status="failed") is None


def test_rekey_moves_a_nohash_row_to_its_real_hash(table):
    old_key = NOHASH_PREFIX + URL
    pending.insert(_row(key=old_key, download_id="dl-1"))

    moved = pending.rekey(old_key, "  " + SHA.upper())

    assert moved.sha256 == SHA
    assert moved.download_id == "dl-1" and moved.info == INFO
    assert pending.get(old_key) is None


def test_rekey_onto_an_existing_row_drops_the_old_row_and_keeps_the_existing_one(table):
    old_key = NOHASH_PREFIX + URL
    pending.insert(_row(key=old_key, requested_by="user-2", download_id="dl-2"))
    existing = pending.insert(_row(download_id="dl-1"))

    result = pending.rekey(old_key, SHA)

    assert result == existing
    assert pending.get(old_key) is None
    assert [row.sha256 for row in pending.list_rows()] == [SHA]


def test_delete_says_whether_a_row_was_removed(table):
    pending.insert(_row())

    assert pending.delete(SHA) is True
    assert pending.delete(SHA) is False
    assert pending.get(SHA) is None


def _three_rows(db):
    pending.insert(_row(key="1" * 64, requested_by="user-1", status="queued"))
    pending.insert(_row(key="2" * 64, requested_by="user-2", status="enriched"))
    pending.insert(_row(key="3" * 64, requested_by="user-1", status="failed"))
    _set_created_at(db, "1" * 64, "2026-10-11 10:00:00")
    _set_created_at(db, "2" * 64, "2026-10-11 11:00:00")
    _set_created_at(db, "3" * 64, "2026-10-11 12:00:00")


def test_list_rows_is_newest_first_and_filters_by_requester_and_status(table):
    _three_rows(table)

    assert [row.sha256[0] for row in pending.list_rows()] == ["3", "2", "1"]
    assert [row.sha256[0] for row in pending.list_rows(requested_by="user-1")] == ["3", "1"]
    assert [row.sha256[0] for row in pending.list_rows(statuses=["queued", "failed"])] == ["3", "1"]
    assert [row.sha256[0] for row in pending.list_rows(requested_by="user-1", statuses=["failed"])] == ["3"]
    assert [row.sha256[0] for row in pending.list_rows(limit=2)] == ["3", "2"]


def test_list_rows_with_only_unknown_statuses_lists_nothing(table):
    _three_rows(table)

    assert pending.list_rows(statuses=["downloading"]) == []
    assert pending.list_rows(statuses=[]) == []


def test_list_rows_clamps_the_limit(table):
    _three_rows(table)

    assert len(pending.list_rows(limit=0)) == 1
    assert len(pending.list_rows(limit=10_000)) == 3


def test_list_unfinished_is_every_unfinished_row_oldest_first(table):
    pending.insert(_row(key="1" * 64, status="assigned"))
    pending.insert(_row(key="2" * 64, status="enriched"))
    pending.insert(_row(key="3" * 64, status="queued"))
    pending.insert(_row(key="4" * 64, status="indexed"))
    pending.insert(_row(key="5" * 64, status="failed"))
    for index, key in enumerate(("3", "4", "1", "2", "5")):
        _set_created_at(table, key * 64, f"2026-10-11 1{index}:00:00")

    assert [(row.sha256[0], row.status) for row in pending.list_unfinished()] == [
        ("3", "queued"),
        ("4", "indexed"),
        ("1", "assigned"),
    ]
