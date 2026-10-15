import pytest

from backend import permissions, requests
from backend.requests import Usernames, build_snapshot, for_viewer, snapshot_for_viewer
from backend.schema import Image
from backend.tables import create_tables

from .civitai_fakes import context
from .download_fakes import NSFW_IMAGE, SAFE_IMAGE, SHA, admin, civitai_detail, principal, user


@pytest.fixture
def table(plugin_db, monkeypatch):
    create_tables()
    users = {"user-1": user("user-1"), "user-2": user("user-2"), "admin-1": user("admin-1", admin=True)}
    looked_up = []

    def get_user(user_id):
        looked_up.append(user_id)
        return users.get(user_id)

    monkeypatch.setattr(permissions, "get_user", get_user)
    return looked_up


def _create(user_id="user-1", file_id="1001", version_id="135867", **fields):
    values = dict(
        user_id=user_id,
        source="civitai",
        source_model_id="122359",
        source_version_id=version_id,
        file_id=file_id,
        snapshot={"name": "Detail Tweaker XL", "file_name": "add-detail-xl.safetensors"},
    )
    values.update(fields)
    return requests.create(**values)


def _age(db, request_id, stamp):
    with db.get_cursor() as cursor:
        cursor.execute(f"UPDATE {requests.TABLE} SET created_at = ? WHERE id = ?", (stamp, request_id))


def test_a_new_request_is_open_and_reads_back_whole(table):
    created = _create(note="For the SDXL workflow")

    assert created.status == "open" and created.is_open
    assert len(created.id) == 26
    again = requests.get(created.id)
    assert again == created
    assert (again.user_id, again.source, again.source_model_id, again.source_version_id, again.file_id) == (
        "user-1", "civitai", "122359", "135867", "1001",
    )
    assert again.snapshot == {"name": "Detail Tweaker XL", "file_name": "add-detail-xl.safetensors"}
    assert again.note == "For the SDXL workflow"
    assert (again.decided_by, again.decision_note, again.sha256, again.assign_to) == (None, None, None, None)
    assert again.created_at and again.updated_at
    assert requests.get("missing") is None


def test_find_open_finds_the_same_users_open_request_for_the_same_file_only(table):
    first = _create()
    _create(user_id="user-2")
    _create(file_id="1002")

    assert requests.find_open("user-1", "civitai", "122359", "135867", "1001") == first
    assert requests.find_open("user-1", "civitai", "122359", "999", "1001") is None
    assert requests.find_open("user-3", "civitai", "122359", "135867", "1001") is None

    requests.set_status(first.id, "cancelled")
    assert requests.find_open("user-1", "civitai", "122359", "135867", "1001") is None


def test_find_open_matches_a_request_without_a_version(table):
    created = _create(version_id=None)

    assert requests.find_open("user-1", "civitai", "122359", None, "1001") == created
    assert requests.find_open("user-1", "civitai", "122359", "135867", "1001") is None


def test_list_rows_is_newest_first_and_filters_by_author_and_status(table, plugin_db):
    older = _create()
    newer = _create(user_id="user-2")
    done = _create(file_id="1002")
    _age(plugin_db, older.id, "2026-10-11 09:00:00")
    _age(plugin_db, newer.id, "2026-10-11 10:00:00")
    _age(plugin_db, done.id, "2026-10-11 11:00:00")
    requests.set_status(done.id, "rejected", decided_by="admin-1")

    assert [row.id for row in requests.list_rows()] == [done.id, newer.id, older.id]
    assert [row.id for row in requests.list_rows(user_id="user-1")] == [done.id, older.id]
    assert [row.id for row in requests.list_rows(statuses=["open"])] == [newer.id, older.id]
    assert [row.id for row in requests.list_rows(user_id="user-1", statuses=["open", "rejected"])] == [done.id, older.id]
    assert requests.list_rows(statuses=["nonsense"]) == []
    assert [row.id for row in requests.list_rows(limit=1)] == [done.id]


def test_set_status_records_the_decision_and_leaves_the_rest(table):
    created = _create()

    approved = requests.set_status(
        created.id, "approved", decided_by="admin-1", decision_note="Enjoy", assign_to="user:user-1"
    )

    assert (approved.status, approved.decided_by, approved.decision_note) == ("approved", "admin-1", "Enjoy")
    assert approved.assign_to == "user:user-1"
    done = requests.set_status(created.id, "done")
    assert (done.status, done.decided_by, done.assign_to) == ("done", "admin-1", "user:user-1")


def test_a_conditional_change_only_happens_from_the_statuses_given(table):
    created = _create()

    assert requests.set_status(created.id, "approved", decided_by="admin-1", only_from=("open",)).status == "approved"
    assert requests.set_status(created.id, "rejected", decided_by="admin-2", only_from=("open",)) is None
    assert requests.get(created.id).decided_by == "admin-1"
    assert requests.set_status("missing", "cancelled", only_from=("open",)) is None
    assert requests.set_status(created.id, "open", only_from=()) is None


def test_update_clears_a_value_set_to_none(table):
    created = _create()
    requests.set_status(created.id, "approved", decided_by="admin-1", decision_note="ok")

    reopened = requests.update(created.id, only_from=("approved",), status="open", decided_by=None, decision_note=None)

    assert (reopened.status, reopened.decided_by, reopened.decision_note) == ("open", None, None)


@pytest.mark.parametrize("fields", [{"status": "lost"}, {"user_id": "user-2"}, {"snapshot": "{}"}])
def test_update_refuses_an_unknown_status_or_column(table, fields):
    created = _create()

    with pytest.raises(ValueError):
        requests.update(created.id, **fields)


def test_approved_requests_are_found_by_their_pending_row_and_follow_a_rekey(table):
    first = _create()
    second = _create(user_id="user-2")
    _create(file_id="1002")
    other = _create(user_id="user-2", file_id="1003")
    for request in (first, second):
        requests.update(request.id, status="approved", sha256="nohash:https://x/y", decided_by="admin-1")
    requests.update(other.id, status="rejected", sha256="nohash:https://x/y", decided_by="admin-1")

    assert [row.id for row in requests.list_approved_for_sha256("nohash:https://x/y")] == [first.id, second.id]
    assert requests.relink("nohash:https://x/y", SHA) == 3
    assert [row.id for row in requests.list_approved_for_sha256(SHA)] == [first.id, second.id]
    assert requests.list_approved_for_sha256("nohash:https://x/y") == []
    assert requests.list_approved_for_sha256("") == []
    assert requests.relink(SHA, SHA) == 0


def test_open_count_counts_the_open_requests_each_approver_may_decide_on(table):
    first = _create()
    _create(user_id="user-2")
    _create(user_id="admin-1", file_id="1003")
    requests.set_status(_create(file_id="1002").id, "cancelled")

    assert requests.open_count() == 3
    assert requests.open_count(exclude_user_id="user-1") == 2
    assert requests.open_count_for(admin()) == 3
    assert requests.open_count_for(principal("approver-9", capabilities=frozenset({"browse", "approve"}))) == 3
    assert requests.open_count_for(principal("user-1", capabilities=frozenset({"browse", "approve"}))) == 2
    assert requests.open_count_for(principal(capabilities=frozenset({"browse", "request"}))) == 0
    requests.set_status(first.id, "rejected", decided_by="admin-1")
    assert requests.open_count() == 2


def test_a_snapshot_that_is_not_json_reads_as_empty(table, plugin_db):
    created = _create()
    with plugin_db.get_cursor() as cursor:
        cursor.execute(f"UPDATE {requests.TABLE} SET snapshot = 'not json' WHERE id = ?", (created.id,))

    assert requests.get(created.id).snapshot == {}


def test_to_json_has_the_columns_and_the_usernames(table):
    created = _create(note="please")
    approved = requests.set_status(created.id, "approved", decided_by="admin-1", assign_to="user:user-1")

    data = approved.to_json()

    assert data["username"] == "user-1" and data["decided_by_username"] == "admin-1"
    assert set(data) == {
        "id", "user_id", "username", "source", "source_model_id", "source_version_id", "file_id", "snapshot",
        "note", "status", "decided_by", "decided_by_username", "decision_note", "sha256", "assign_to",
        "created_at", "updated_at",
    }
    assert (data["status"], data["assign_to"], data["note"]) == ("approved", "user:user-1", "please")


def test_a_user_who_no_longer_exists_has_no_username(table):
    created = _create(user_id="gone")

    data = requests.set_status(created.id, "rejected", decided_by="also-gone").to_json()

    assert (data["user_id"], data["username"], data["decided_by"], data["decided_by_username"]) == (
        "gone", None, "also-gone", None,
    )


def test_usernames_are_looked_up_once_each(table):
    names = Usernames()
    rows = [_create(), _create(file_id="1002"), _create(user_id="user-2")]

    for row in rows:
        row.to_json(names)

    assert sorted(table) == ["user-1", "user-2"]
    assert names(None) is None


def test_the_snapshot_describes_the_file_with_a_safe_thumbnail():
    detail = civitai_detail()
    version = detail.versions[0]

    snapshot = build_snapshot(detail, version, version.files[0], "lora", already_installed=True)

    assert snapshot == {
        "name": "Detail Tweaker XL",
        "type": "LORA",
        "mb_type": "lora",
        "version_name": "v1.0",
        "base_model": "SDXL 1.0",
        "file_name": "add-detail-xl.safetensors",
        "size_bytes": 228_000_000,
        "nsfw": False,
        "poi": False,
        "license": "CreativeML Open RAIL++-M; no selling",
        "thumbnail_url": SAFE_IMAGE,
        "source_url": "https://civitai.com/models/122359",
        "sha256": SHA,
        "already_installed": True,
    }


@pytest.mark.parametrize(
    "images",
    [
        [Image(url=NSFW_IMAGE, nsfw=True, nsfw_level=8)],
        [Image(url="https://image.civitai.com/person.jpeg", poi=True)],
        [Image(url="https://image.civitai.com/clip.mp4", kind="video")],
        [Image(url="http://image.civitai.com/plain.jpeg")],
        [Image(url="https://192.168.1.10/x.jpeg")],
        [Image(url="https://localhost/x.jpeg")],
        [Image(url="javascript:alert(1)")],
        [Image(url=None, blurhash="LKO2?U%2Tw=w", nsfw=True)],
        [],
    ],
    ids=["nsfw", "real-person", "video", "http", "ip", "localhost", "script", "no-url", "none"],
)
def test_the_thumbnail_is_never_an_unsafe_image(images):
    detail = civitai_detail()
    version = detail.versions[0].model_copy(update={"images": images})

    assert build_snapshot(detail, version, version.files[0], "lora", already_installed=False)["thumbnail_url"] is None


def test_the_first_safe_image_is_the_thumbnail_even_after_unsafe_ones():
    detail = civitai_detail()
    images = [Image(url=NSFW_IMAGE, nsfw=True, nsfw_level=8), Image(url=SAFE_IMAGE, nsfw_level=1)]
    version = detail.versions[0].model_copy(update={"images": images})

    assert build_snapshot(detail, version, version.files[0], "lora", already_installed=False)["thumbnail_url"] == SAFE_IMAGE


def test_an_nsfw_version_or_model_is_an_nsfw_snapshot_without_a_hash_when_the_file_has_none():
    detail = civitai_detail(version_nsfw=True)
    version = detail.versions[0]
    file = version.files[0].model_copy(update={"sha256": None})

    snapshot = build_snapshot(detail, version, file, "lora", already_installed=False)

    assert snapshot["nsfw"] is True and snapshot["sha256"] is None


@pytest.mark.parametrize(
    "ctx, nsfw, poi, hidden",
    [
        (context("allowed", "show"), True, False, False),
        (context("blur", "blur"), True, False, False),
        (context("allowed", "hide"), True, False, True),
        (context("blocked", "hide"), True, False, True),
        (context("blocked", "hide"), False, False, False),
        (context("allowed", "show", hide_poi=True), False, True, True),
        (context("allowed", "show"), False, True, False),
    ],
)
def test_a_viewer_who_does_not_see_the_model_gets_no_thumbnail(ctx, nsfw, poi, hidden):
    snapshot = {"name": "X", "nsfw": nsfw, "poi": poi, "thumbnail_url": SAFE_IMAGE}

    shown = snapshot_for_viewer(snapshot, ctx)

    assert shown["name"] == "X"
    assert (shown["thumbnail_url"] is None) is hidden
    assert shown.get("hidden", False) is hidden
    assert snapshot["thumbnail_url"] == SAFE_IMAGE


def test_for_viewer_applies_the_viewers_context_to_the_snapshot(table):
    created = _create(snapshot={"name": "X", "nsfw": True, "thumbnail_url": SAFE_IMAGE})

    assert for_viewer(created, context("blocked", "hide"))["snapshot"]["hidden"] is True
    assert for_viewer(created, context("allowed", "show"))["snapshot"]["thumbnail_url"] == SAFE_IMAGE
