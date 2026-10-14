import asyncio
import hashlib
import inspect
import logging

import pytest

import src.plugin_api.hooks as plugin_hooks
from backend import core_bridge, downloads, pending, requests
from backend import watcher as watcher_module
from backend.core_bridge import CoreBridgeError
from backend.pending import NOHASH_PREFIX, PendingRow
from backend.permissions_store import Row, save_rows
from backend.tables import create_tables
from backend.watcher import Watcher, indexed_since, model_attributes, sha256_of_file

from .download_fakes import HF_URL, OTHER_SHA, SHA, FakeCore, civitai_detail, model

ATTRIBUTES = {"mb_base_model": "SDXL 1.0", "mb_license": "CreativeML Open RAIL++-M; no selling"}
HF_PATH = "/app/models/diffusion_models/transformer/flux1-dev.safetensors"


async def _until(predicate, timeout=1.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


def _info():
    detail = civitai_detail()
    version = detail.versions[0]
    return downloads.pending_info(detail, version, version.files[0], "lora")


def _insert(key=SHA, **fields):
    values = dict(
        sha256=key,
        source="civitai",
        source_model_id="122359",
        source_version_id="135867",
        file_name="add-detail-xl.safetensors",
        requested_by="user-1",
        assign_to="user:user-1",
        info=_info(),
        download_id="dl-1",
    )
    values.update(fields)
    return pending.insert(PendingRow(**values))


def _row(key=SHA):
    return pending.get(key)


async def _passes(watcher, count=1):
    for _ in range(count):
        await watcher.run_pass()


def _group_saved_by(user_id, group_id="artists"):
    save_rows([Row(group_id=group_id, capabilities=frozenset({"browse"}))], user_id)


@pytest.fixture
def no_pass(monkeypatch):
    passes = []

    async def run_pass(self, *, reconcile=False):
        passes.append(reconcile)

    monkeypatch.setattr(Watcher, "run_pass", run_pass)
    return passes


async def test_run_returns_promptly_after_request_stop(monkeypatch, caplog, no_pass):
    caplog.set_level(logging.INFO, logger="plugins.model-browser")
    monkeypatch.setattr(watcher_module, "POLL_INTERVAL_SECONDS", 0.01)
    watcher = Watcher()
    task = asyncio.create_task(watcher.run())
    await _until(lambda: watcher.running)
    await asyncio.sleep(0.05)
    assert not task.done()

    watcher.request_stop()
    await asyncio.wait_for(task, timeout=1)

    assert not watcher.running
    assert no_pass[0] is True and len(no_pass) > 1 and not any(no_pass[1:])
    messages = [record.getMessage() for record in caplog.records]
    assert any("Download watcher started" in message for message in messages)
    assert any("Download watcher stopped" in message for message in messages)


async def test_a_stop_from_another_thread_cuts_the_poll_interval_short(monkeypatch, no_pass):
    monkeypatch.setattr(watcher_module, "POLL_INTERVAL_SECONDS", 60)
    watcher = Watcher()
    task = asyncio.create_task(watcher.run())
    await _until(lambda: watcher.running and no_pass)

    await asyncio.to_thread(watcher.request_stop)

    await asyncio.wait_for(task, timeout=1)
    assert no_pass == [True]


async def test_a_stop_requested_before_run_returns_at_once(monkeypatch, no_pass):
    monkeypatch.setattr(watcher_module, "POLL_INTERVAL_SECONDS", 60)
    watcher = Watcher()

    watcher.request_stop()

    await asyncio.wait_for(watcher.run(), timeout=1)
    assert not watcher.running
    assert no_pass == []


async def test_the_first_pass_takes_up_the_rows_a_restart_left_unfinished(core, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="plugins.model-browser")
    monkeypatch.setattr(watcher_module, "POLL_INTERVAL_SECONDS", 60)
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    _insert(SHA)
    _insert("c" * 64, status="indexed", model_id="model-2", download_id=None)
    _insert("d" * 64, status="assigned", model_id="model-3", download_id=None)
    watcher = Watcher()
    task = asyncio.create_task(watcher.run())

    await _until(lambda: all(_row(key).status == "enriched" for key in (SHA, "c" * 64, "d" * 64)), timeout=5)
    watcher.request_stop()
    await asyncio.wait_for(task, timeout=1)

    assert "Resuming 3 unfinished download(s)" in caplog.text
    assert sorted(core.calls_to("assign_to_user")) == [("model-1", "user-1"), ("model-2", "user-1")]
    assert sorted(call[1] for call in core.calls_to("run_provider_fetch")) == [["model-1"], ["model-2"], ["model-3"]]


async def test_a_pass_that_cannot_read_the_table_logs_once_and_carries_on(monkeypatch, caplog):
    def list_unfinished():
        raise RuntimeError("no such table: model_browser_pending")

    monkeypatch.setattr(pending, "list_unfinished", list_unfinished)
    watcher = Watcher()

    await _passes(watcher, 3)

    assert caplog.text.count("Could not read the pending downloads") == 1


async def test_a_completed_and_indexed_download_goes_all_the_way_in_one_pass(core):
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    _insert()

    await Watcher().run_pass()

    row = _row()
    assert (row.status, row.model_id, row.error) == ("enriched", "model-1", None)
    assert core.calls == [
        ("get_download", "dl-1"),
        ("find_model_by_sha256", SHA),
        ("assign_to_user", "model-1", "user-1"),
        ("run_provider_fetch", "model-browser", ["model-1"]),
        ("write_attributes", "model-1", ATTRIBUTES),
        ("notify", "model-browser.download_done"),
    ]
    [notification] = core.notifications
    assert (notification["user_id"], notification["level"], notification["title"]) == ("user-1", "success", "Model downloaded")
    assert notification["message"] == "Detail Tweaker XL (add-detail-xl.safetensors): it is now in your models."
    assert notification["metadata"] == {
        "source": "civitai",
        "source_model_id": "122359",
        "source_version_id": "135867",
        "sha256": SHA,
        "model_id": "model-1",
        "download_id": "dl-1",
    }


async def test_only_the_model_browser_provider_enriches_even_a_civitai_download(core):
    _insert(status="assigned", model_id="model-1")

    await Watcher().run_pass()

    assert core.calls_to("run_provider_fetch") == [("model-browser", ["model-1"])]
    assert _row().status == "enriched"


@pytest.mark.parametrize("status", ["pending", "downloading", "paused"])
async def test_a_download_still_on_its_way_is_left_as_it_is(core, status):
    core.add_download("dl-1", status)
    _insert()

    await Watcher().run_pass()

    assert _row().status == "queued"
    assert core.calls == [("get_download", "dl-1")]


async def test_the_model_is_found_on_a_later_pass_when_indexing_comes_after_completion(core):
    core.add_download("dl-1", "completed")
    _insert()
    watcher = Watcher()

    await watcher.run_pass()
    assert _row().status == "queued"

    core.models_by_sha[SHA] = model("model-1")
    await watcher.run_pass()

    assert (_row().status, _row().model_id) == ("enriched", "model-1")
    assert core.calls_to("find_model_by_sha256") == [(SHA,), (SHA,)]
    assert core.calls_to("find_model_by_path") == []


async def test_a_failed_download_fails_the_row_and_tells_the_requester(core):
    core.add_download("dl-1", "failed", error_message="Access denied (HTTP 401) fetching 'add-detail-xl.safetensors'")
    _insert()

    await Watcher().run_pass()

    row = _row()
    assert (row.status, row.error) == ("failed", "Access denied (HTTP 401) fetching 'add-detail-xl.safetensors'")
    assert "assign_to_user" not in core.names() and "run_provider_fetch" not in core.names()
    [notification] = core.notifications
    assert (notification["notification_type"], notification["level"]) == ("model-browser.download_failed", "error")
    assert notification["title"] == "Model download failed"
    assert notification["message"].endswith("Access denied (HTTP 401) fetching 'add-detail-xl.safetensors'")


async def test_a_cancelled_download_fails_the_row(core):
    core.add_download("dl-1", "cancelled")
    _insert()

    await Watcher().run_pass()

    assert (_row().status, _row().error) == ("failed", "The download was cancelled")


async def test_a_download_gone_from_the_queue_fails_the_row_unless_its_file_is_indexed(core):
    _insert(SHA)
    _insert(OTHER_SHA, download_id="dl-2")
    core.models_by_sha[OTHER_SHA] = model("model-2", OTHER_SHA)

    await Watcher().run_pass()

    assert (_row(SHA).status, _row(SHA).error) == ("failed", "The download is no longer on the download queue")
    assert (_row(OTHER_SHA).status, _row(OTHER_SHA).model_id) == ("enriched", "model-2")


async def test_a_row_that_never_got_a_download_id_fails_after_the_grace_passes(core, monkeypatch):
    monkeypatch.setattr(watcher_module, "UNQUEUED_GRACE_PASSES", 1)
    _insert(download_id=None)
    watcher = Watcher()

    await watcher.run_pass()
    assert _row().status == "queued"
    await watcher.run_pass()

    assert _row().status == "failed"
    assert "never put on the queue" in _row().error
    assert "get_download" not in core.names()


async def test_a_completed_download_that_never_shows_up_in_the_index_fails(core, monkeypatch):
    monkeypatch.setattr(watcher_module, "MAX_INDEX_LOOKUPS", 2)
    core.add_download("dl-1", "completed")
    _insert()
    watcher = Watcher()

    await _passes(watcher, 2)

    assert _row().status == "failed"
    assert "never showed up in the model index" in _row().error
    assert core.notifications[-1]["notification_type"] == "model-browser.download_failed"


async def test_a_nohash_row_moves_to_the_hash_of_the_model_found_at_its_path(core):
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=HF_PATH)
    core.models_by_path[HF_PATH] = model("model-2", OTHER_SHA)
    _insert(key, source="huggingface")

    await Watcher().run_pass()

    assert _row(key) is None
    row = _row(OTHER_SHA)
    assert (row.status, row.model_id, row.download_id) == ("enriched", "model-2", "dl-1")
    assert core.names()[:2] == ["get_download", "find_model_by_path"]
    assert "find_model_by_sha256" not in core.names()
    assert core.calls_to("run_provider_fetch") == [("model-browser", ["model-2"])]


async def test_a_nohash_row_keeps_its_key_when_another_row_already_has_the_models_hash(core):
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=HF_PATH)
    core.models_by_path[HF_PATH] = model("model-2", OTHER_SHA)
    _insert(key, source="huggingface")
    _insert(OTHER_SHA, status="enriched", requested_by="user-2", model_id="model-2", download_id="dl-9")

    await Watcher().run_pass()

    row = _row(key)
    assert (row.status, row.model_id) == ("enriched", "model-2")
    assert _row(OTHER_SHA).requested_by == "user-2"
    assert core.calls_to("assign_to_user") == [("model-2", "user-1")]


async def test_a_model_found_by_path_but_indexed_before_the_download_is_not_taken(core):
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=HF_PATH)
    core.models_by_path[HF_PATH] = model("someone-elses-model", OTHER_SHA, created="long_ago")
    _insert(key, source="huggingface")

    await Watcher().run_pass()

    assert _row(key).status == "queued"
    assert "assign_to_user" not in core.names()


async def test_a_nohash_file_identical_to_an_installed_model_is_found_by_hashing_it(core, monkeypatch, tmp_path):
    monkeypatch.setattr(watcher_module, "HASH_FALLBACK_AFTER", 2)
    weights = tmp_path / "flux1-dev.safetensors"
    weights.write_bytes(b"the same weights as an installed model")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    hashed = []
    monkeypatch.setattr(watcher_module, "sha256_of_file", lambda path: hashed.append(path) or sha256_of_file(path))
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=str(weights))
    core.models_by_sha[digest] = model("model-4", digest, created="long_ago")
    _insert(key, source="huggingface")
    watcher = Watcher()

    await watcher.run_pass()
    assert _row(key).status == "queued" and hashed == []
    await watcher.run_pass()

    assert hashed == [str(weights)]
    assert _row(key) is None
    assert (_row(digest).status, _row(digest).model_id) == ("enriched", "model-4")


async def test_a_file_on_a_remote_backend_is_never_hashed(core, monkeypatch):
    monkeypatch.setattr(watcher_module, "HASH_FALLBACK_AFTER", 1)
    monkeypatch.setattr(watcher_module, "sha256_of_file", lambda path: pytest.fail("hashed a remote file"))
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=HF_PATH, destination_backend_id="worker-1")
    _insert(key, source="huggingface")

    await _passes(Watcher(), 2)

    assert _row(key).status == "queued"


async def test_a_hash_row_tries_the_path_after_some_lookups(core, monkeypatch):
    monkeypatch.setattr(watcher_module, "PATH_FALLBACK_AFTER", 2)
    path = "/app/models/loras/add-detail-xl.safetensors"
    core.add_download("dl-1", "completed", destination_path=path)
    core.models_by_path[path] = model("model-5", OTHER_SHA)
    _insert()
    watcher = Watcher()

    await watcher.run_pass()
    assert _row().status == "queued" and core.calls_to("find_model_by_path") == []
    await watcher.run_pass()

    assert core.calls_to("find_model_by_path") == [(path,)]
    assert _row(SHA) is None
    assert (_row(OTHER_SHA).status, _row(OTHER_SHA).model_id) == ("enriched", "model-5")


async def test_a_group_download_is_assigned_to_the_group(core):
    core.add_user("admin-1", admin=True)
    _group_saved_by("admin-1")
    _insert(status="indexed", model_id="model-1", assign_to="group:artists")

    await Watcher().run_pass()

    assert core.calls_to("assign_to_group") == [("artists", ["model-1"])]
    assert core.group_admins == ["admin-1"]
    assert "assign_to_user" not in core.names()
    assert _row().status == "enriched"
    assert core.notifications[0]["message"].endswith("it is now available to your group.")


async def test_a_group_that_no_longer_exists_falls_back_to_the_requester(core):
    core.add_user("admin-1", admin=True)
    _group_saved_by("admin-1")
    core.assign_group_error = CoreBridgeError("not_found", "There is no user group with the id 'artists'")
    _insert(status="indexed", model_id="model-1", assign_to="group:artists")

    await Watcher().run_pass()

    row = _row()
    assert core.calls_to("assign_to_group") == [("artists", ["model-1"])]
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert (row.status, row.assign_to) == ("enriched", "user:user-1")
    assert "no longer exists" in row.error
    assert core.notifications[0]["notification_type"] == "model-browser.download_done"


async def test_a_group_assignment_the_core_refuses_fails_the_row(core):
    core.add_user("admin-1", admin=True)
    _group_saved_by("admin-1")
    core.assign_group_error = CoreBridgeError("refused", "Only admins can assign models to user groups")
    _insert(status="indexed", model_id="model-1", assign_to="group:artists", limit_ref_id="model-browser:ref-1")

    await Watcher().run_pass()

    row = _row()
    assert (row.status, row.assign_to) == ("failed", "group:artists")
    assert row.error == "The model was downloaded but could not be made visible: Only admins can assign models to user groups"
    assert "assign_to_user" not in core.names() and "run_provider_fetch" not in core.names()
    assert core.refunds == ["model-browser:ref-1"]


async def test_a_refused_assignment_fails_the_row(core):
    core.assign_user_error = CoreBridgeError("refused", "Could not assign model 'model-1' to user 'user-1' - no such user")
    _insert(status="indexed", model_id="model-1")

    await Watcher().run_pass()

    row = _row()
    assert row.status == "failed"
    assert row.error == "The model was downloaded but could not be made visible: Could not assign model 'model-1' to user 'user-1' - no such user"
    assert "run_provider_fetch" not in core.names()
    assert core.notifications[0]["notification_type"] == "model-browser.download_failed"


async def test_an_admins_download_for_the_instance_is_assigned_to_nobody(core):
    _insert(status="indexed", model_id="model-1", assign_to="none", requested_by="admin-1")

    await Watcher().run_pass()

    assert "assign_to_user" not in core.names() and "assign_to_group" not in core.names()
    assert _row().status == "enriched"
    [notification] = core.notifications
    assert notification["user_id"] == "admin-1"
    assert notification["message"].endswith("it is now installed.")


async def test_enrichment_failures_are_noted_and_the_row_still_ends_enriched(core):
    core.fetch_results["model-browser"] = {
        "successful": 0,
        "failed": 1,
        "errors": [{"model_id": "model-1", "reason": "Not found on Model-Browser"}],
    }
    core.attributes_error = CoreBridgeError("invalid", "'mb_license' is not a declared attribute for model type 'lora'")
    _insert(status="assigned", model_id="model-1")

    await Watcher().run_pass()

    row = _row()
    assert row.status == "enriched"
    assert row.error == (
        "model-browser: Not found on Model-Browser; "
        "attributes: 'mb_license' is not a declared attribute for model type 'lora'"
    )
    assert core.notifications[0]["notification_type"] == "model-browser.download_done"


async def test_a_provider_fetch_that_raises_is_noted_too(core):
    core.fetch_errors["model-browser"] = RuntimeError("model_index_manager.provider_info.run_provider_fetch moved")
    _insert(status="assigned", model_id="model-1")

    await Watcher().run_pass()

    assert _row().status == "enriched"
    assert _row().error == "model-browser: model_index_manager.provider_info.run_provider_fetch moved"
    assert core.calls_to("write_attributes") == [("model-1", ATTRIBUTES)]


async def test_a_note_from_the_assignment_is_kept_with_the_enrichment_notes(core):
    core.add_user("admin-1", admin=True)
    _group_saved_by("admin-1")
    core.assign_group_error = CoreBridgeError("not_found", "gone")
    core.fetch_results["model-browser"] = {"successful": 0, "failed": 1}
    _insert(status="indexed", model_id="model-1", assign_to="group:artists")

    await Watcher().run_pass()

    assert _row().error.endswith("; model-browser: no information found")
    assert "no longer exists" in _row().error


async def test_nothing_is_written_when_the_info_has_no_attribute_value(core):
    info = _info()
    info["model"]["license"] = None
    info["model"]["versions"][0]["base_model"] = "  "
    _insert(status="assigned", model_id="model-1", info=info)

    await Watcher().run_pass()

    assert core.calls_to("write_attributes") == []
    assert _row().status == "enriched"


async def test_a_notification_that_cannot_be_sent_does_not_change_the_row(core, monkeypatch):
    def notify(**kwargs):
        raise RuntimeError("notifications are down")

    monkeypatch.setattr(core_bridge, "notify", notify)
    _insert(status="assigned", model_id="model-1")

    await Watcher().run_pass()

    assert _row().status == "enriched"


async def test_one_rows_error_does_not_stop_the_others_and_a_row_that_keeps_failing_is_given_up(
    core, plugin_db, monkeypatch, caplog
):
    monkeypatch.setattr(watcher_module, "MAX_ROW_ERRORS", 2)
    real_get_download = core.get_download

    def get_download(download_id):
        if download_id == "dl-bad":
            raise RuntimeError("database is locked")
        return real_get_download(download_id)

    monkeypatch.setattr(core_bridge, "get_download", get_download)
    _insert("c" * 64, download_id="dl-bad")
    _insert(SHA, download_id="dl-1")
    with plugin_db.get_cursor() as cursor:
        cursor.execute(f"UPDATE {pending.TABLE} SET created_at = '2026-10-11 09:00:00' WHERE sha256 = ?", ("c" * 64,))
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    watcher = Watcher()

    await watcher.run_pass()

    assert _row("c" * 64).status == "queued"
    assert _row(SHA).status == "enriched"
    assert caplog.text.count("Could not move") == 1

    await watcher.run_pass()

    assert _row("c" * 64).status == "failed"
    assert _row("c" * 64).error == "Stopped following this download after repeated errors: database is locked"
    assert caplog.text.count("Could not move") == 1


LABEL = "Detail Tweaker XL (add-detail-xl.safetensors)"


def _approved_request(user_id, *, sha256=SHA, assign_to="user", decided_by="admin-1"):
    created = requests.create(
        user_id=user_id,
        source="civitai",
        source_model_id="122359",
        source_version_id="135867",
        file_id="1001",
        snapshot={"name": "Detail Tweaker XL", "file_name": "add-detail-xl.safetensors"},
    )
    return requests.set_status(
        created.id,
        "approved",
        decided_by=decided_by,
        sha256=sha256,
        assign_to=f"user:{user_id}" if assign_to == "user" else assign_to,
    )


def _status(request):
    return requests.get(request.id).status


def _sent(core, kind):
    return [(n["user_id"], n["message"]) for n in core.notifications if n["notification_type"] == kind]


async def test_other_approved_requesters_are_assigned_with_the_row_and_closed_when_the_model_is_ready(core):
    core.add_user("admin-1", admin=True)
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    _insert(requested_by="user-2", assign_to="user:user-2")
    first = _approved_request("user-1")
    grouped = _approved_request("user-3", assign_to="group:artists")

    await Watcher().run_pass()

    assert _row().status == "enriched"
    assert core.calls_to("assign_to_user") == [("model-1", "user-2"), ("model-1", "user-1")]
    assert core.calls_to("assign_to_group") == [("artists", ["model-1"])]
    assert core.group_admins == ["admin-1"]
    names = core.names()
    assert names.index("assign_to_group") < names.index("run_provider_fetch")
    assert (_status(first), _status(grouped)) == ("done", "done")
    assert _sent(core, "model-browser.download_done") == [
        ("user-2", f"{LABEL}: it is now in your models."),
        ("user-1", f"{LABEL}: it is now in your models."),
        ("user-3", f"{LABEL}: it is now available to your group."),
    ]
    assert core.notifications[1]["metadata"]["request_id"] == first.id
    assert core.notifications[1]["metadata"]["model_id"] == "model-1"


async def test_the_rows_own_request_is_done_and_its_requester_notified_once(core):
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    own = _approved_request("user-1")
    _insert(request_id=own.id)

    await Watcher().run_pass()

    assert _status(own) == "done"
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert [n["user_id"] for n in core.notifications] == ["user-1"]


async def test_another_request_of_the_rows_own_requester_is_closed_without_a_second_notification(core):
    _insert(status="assigned", model_id="model-1")
    duplicate = _approved_request("user-1")

    await Watcher().run_pass()

    assert _status(duplicate) == "done"
    assert [n["user_id"] for n in core.notifications] == ["user-1"]


@pytest.mark.parametrize(
    "requested_by, decided_by, admin_id",
    [("admin-2", "admin-1", "admin-2"), ("user-1", "admin-1", "admin-1"), ("user-1", "approver-1", "admin-3")],
    ids=["requester-is-an-admin", "approver-is-an-admin", "group-saved-by-an-admin"],
)
async def test_a_group_is_assigned_in_the_name_of_the_acting_admin(core, requested_by, decided_by, admin_id):
    for user_id in ("admin-1", "admin-2", "admin-3"):
        core.add_user(user_id, admin=True)
    core.add_user("approver-1")
    _group_saved_by("admin-3")
    own = _approved_request(requested_by, assign_to="group:artists", decided_by=decided_by)
    _insert(status="indexed", model_id="model-1", assign_to="group:artists", request_id=own.id, requested_by=requested_by)

    await Watcher().run_pass()

    assert core.calls_to("assign_to_group") == [("artists", ["model-1"])]
    assert core.group_admins == [admin_id]
    assert "assign_to_user" not in core.names()
    assert (_row().status, _row().assign_to, _row().error) == ("enriched", "group:artists", None)
    assert (_status(own), requests.get(own.id).assign_to) == ("done", "group:artists")


@pytest.mark.parametrize(
    "decided_by, saved_by",
    [("approver-1", "approver-1"), ("gone", None), (None, None)],
    ids=["non-admin-approver-and-editor", "approver-gone-default-row", "direct-download"],
)
async def test_without_an_acting_admin_a_group_download_goes_to_its_requester_with_a_note(core, decided_by, saved_by):
    core.add_user("admin-1", admin=True)
    core.add_user("approver-1")
    if saved_by:
        _group_saved_by(saved_by)
    own = _approved_request("user-1", assign_to="group:artists", decided_by=decided_by) if decided_by else None
    _insert(status="indexed", model_id="model-1", assign_to="group:artists", request_id=own.id if own else None)

    await Watcher().run_pass()

    row = _row()
    assert "assign_to_group" not in core.names()
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert (row.status, row.assign_to) == ("enriched", "user:user-1")
    assert row.error == "No admin could act for the group 'artists': the model was assigned to its requester instead"
    assert core.notifications[0]["message"].endswith("it is now in your models.")
    if own is not None:
        assert (_status(own), requests.get(own.id).assign_to) == ("done", "user:user-1")


@pytest.mark.parametrize(
    "saved_by, admin_ids, assigned_to",
    [("admin-3", ["admin-3"], "group:artists"), ("approver-1", [], "user:user-3")],
    ids=["group-saved-by-an-admin", "no-admin"],
)
async def test_another_requests_group_follows_the_acting_admin_rule_too(core, saved_by, admin_ids, assigned_to):
    core.add_user("admin-3", admin=True)
    core.add_user("approver-1")
    _group_saved_by(saved_by)
    _insert(status="indexed", model_id="model-1", requested_by="user-2", assign_to="user:user-2")
    grouped = _approved_request("user-3", assign_to="group:artists", decided_by="approver-1")

    await Watcher().run_pass()

    assert core.group_admins == admin_ids
    closed = requests.get(grouped.id)
    assert (closed.status, closed.assign_to) == ("done", assigned_to)
    requester_calls = [("model-1", "user-3")] if assigned_to.startswith("user:") else []
    assert core.calls_to("assign_to_user") == [("model-1", "user-2"), *requester_calls]


async def test_a_request_approved_after_the_assignment_step_is_assigned_when_the_model_is_ready(core):
    _insert(status="assigned", model_id="model-1", requested_by="user-2", assign_to="user:user-2")
    created = requests.create(
        user_id="user-1", source="civitai", source_model_id="122359", source_version_id="135867", file_id="1001",
        snapshot={},
    )
    late = requests.set_status(created.id, "approved", decided_by="admin-1", sha256=SHA)

    await Watcher().run_pass()

    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert _status(late) == "done"


async def test_a_requester_who_cannot_be_assigned_gets_a_failed_request(core, monkeypatch):
    real_assign = core.assign_to_user

    def assign_to_user(model_id, user_id):
        if user_id == "user-1":
            raise CoreBridgeError("refused", "Could not assign model 'model-1' to user 'user-1' - no such user")
        return real_assign(model_id, user_id)

    monkeypatch.setattr(core_bridge, "assign_to_user", assign_to_user)
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    _insert(requested_by="user-2", assign_to="user:user-2")
    request = _approved_request("user-1")

    await Watcher().run_pass()

    assert _row().status == "enriched"
    assert _status(request) == "failed"
    assert _sent(core, "model-browser.download_failed") == [
        ("user-1", f"{LABEL}: The model was downloaded but could not be made visible to you"),
    ]


async def test_a_group_gone_since_the_approval_falls_back_to_the_requester(core):
    core.add_user("admin-1", admin=True)
    core.assign_group_error = CoreBridgeError("not_found", "There is no user group with the id 'artists'")
    _insert(status="assigned", model_id="model-1", requested_by="user-2", assign_to="user:user-2")
    request = _approved_request("user-1", assign_to="group:artists")

    await Watcher().run_pass()

    assert core.calls_to("assign_to_group") == [("artists", ["model-1"])]
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    closed = requests.get(request.id)
    assert (closed.status, closed.assign_to) == ("done", "user:user-1")
    assert _sent(core, "model-browser.download_done")[-1] == ("user-1", f"{LABEL}: it is now in your models.")


async def test_the_rows_own_request_records_where_the_model_went(core):
    core.add_user("admin-1", admin=True)
    core.assign_group_error = CoreBridgeError("not_found", "There is no user group with the id 'artists'")
    own = _approved_request("user-1", assign_to="group:artists")
    _insert(status="indexed", model_id="model-1", assign_to="group:artists", request_id=own.id)

    await Watcher().run_pass()

    assert core.group_admins == ["admin-1"]
    assert (_row().assign_to, requests.get(own.id).assign_to) == ("user:user-1", "user:user-1")


async def test_a_failed_download_fails_its_requests_refunds_the_quota_and_tells_every_requester(core):
    core.add_download("dl-1", "failed", error_message="Access denied (HTTP 401)")
    own = _approved_request("user-1")
    other = _approved_request("user-2")
    _insert(request_id=own.id, limit_ref_id="model-browser:ref-1")

    await Watcher().run_pass()

    assert _row().status == "failed"
    assert core.refunds == ["model-browser:ref-1"]
    assert (_status(own), _status(other)) == ("failed", "failed")
    assert _sent(core, "model-browser.download_failed") == [
        ("user-1", f"{LABEL}: Access denied (HTTP 401)"),
        ("user-2", f"{LABEL}: Access denied (HTTP 401)"),
    ]
    assert "assign_to_user" not in core.names()


@pytest.mark.parametrize("ending", ["cancelled", "gone-from-the-queue", "never-queued", "never-indexed"])
async def test_a_download_that_ends_without_a_model_refunds_its_quota(core, monkeypatch, ending):
    monkeypatch.setattr(watcher_module, "UNQUEUED_GRACE_PASSES", 0)
    monkeypatch.setattr(watcher_module, "MAX_INDEX_LOOKUPS", 1)
    if ending == "cancelled":
        core.add_download("dl-1", "cancelled")
    elif ending == "never-indexed":
        core.add_download("dl-1", "completed")
    _insert(limit_ref_id="model-browser:ref-1", download_id=None if ending == "never-queued" else "dl-1")

    await Watcher().run_pass()

    assert _row().status == "failed"
    assert core.refunds == ["model-browser:ref-1"]


async def test_a_download_that_succeeds_or_fails_without_a_ref_refunds_nothing(core):
    core.add_download("dl-1", "completed")
    core.add_download("dl-2", "failed", error_message="HTTP 500")
    core.models_by_sha[SHA] = model("model-1")
    _insert(SHA, limit_ref_id="model-browser:ref-1")
    _insert(OTHER_SHA, download_id="dl-2")

    await Watcher().run_pass()

    assert (_row(SHA).status, _row(OTHER_SHA).status) == ("enriched", "failed")
    assert core.refunds == []


async def test_a_nohash_row_moved_to_its_models_hash_takes_its_requests_along(core):
    key = NOHASH_PREFIX + HF_URL
    core.add_download("dl-1", "completed", destination_path=HF_PATH)
    core.models_by_path[HF_PATH] = model("model-2", OTHER_SHA)
    own = _approved_request("user-1", sha256=key)
    other = _approved_request("user-2", sha256=key)
    _insert(key, source="huggingface", request_id=own.id)

    await Watcher().run_pass()

    assert _row(OTHER_SHA).status == "enriched"
    assert requests.get(own.id).sha256 == OTHER_SHA
    assert (_status(own), _status(other)) == ("done", "done")
    assert core.calls_to("assign_to_user") == [("model-2", "user-1"), ("model-2", "user-2")]


async def test_the_reconciliation_closes_the_requests_of_rows_that_ended_meanwhile(core):
    _insert(SHA, status="enriched", model_id="model-1", requested_by="user-2", download_id="dl-1")
    _insert(OTHER_SHA, status="failed", error="HTTP 500", requested_by="user-2", download_id="dl-2")
    done = _approved_request("user-1")
    failed = _approved_request("user-3", sha256=OTHER_SHA)
    unlinked = _approved_request("user-4", sha256=None)
    watcher = Watcher()

    await watcher.run_pass()
    assert (_status(done), _status(failed)) == ("approved", "approved")

    await watcher.run_pass(reconcile=True)

    assert (_status(done), _status(failed), _status(unlinked)) == ("done", "failed", "approved")
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert sorted((n["user_id"], n["notification_type"]) for n in core.notifications) == [
        ("user-1", "model-browser.download_done"),
        ("user-3", "model-browser.download_failed"),
    ]


async def test_requests_that_cannot_be_read_never_stop_a_row(core, monkeypatch, caplog):
    def broken(sha256):
        raise RuntimeError("no such table: model_browser_requests")

    monkeypatch.setattr(requests, "list_approved_for_sha256", broken)
    _insert(status="indexed", model_id="model-1")

    await Watcher().run_pass()

    assert _row().status == "enriched"
    assert "Could not read the requests waiting for" in caplog.text


async def test_carry_on_finishes_an_indexed_row_without_a_watcher(core):
    own = _approved_request("user-1")
    other = _approved_request("user-2")
    _insert(status="indexed", model_id="model-1", request_id=own.id)

    row = await watcher_module.carry_on(SHA)

    assert row == _row()
    assert (row.status, row.model_id, row.error) == ("enriched", "model-1", None)
    assert [name for name in core.names() if name in ("assign_to_user", "run_provider_fetch")] == [
        "assign_to_user",
        "assign_to_user",
        "run_provider_fetch",
    ]
    assert core.calls_to("assign_to_user") == [("model-1", "user-1"), ("model-1", "user-2")]
    assert (_status(own), _status(other)) == ("done", "done")
    assert _sent(core, "model-browser.download_done") == [
        ("user-1", f"{LABEL}: it is now in your models."),
        ("user-2", f"{LABEL}: it is now in your models."),
    ]


@pytest.mark.parametrize("status", ["queued", "enriched", "failed"])
async def test_carry_on_leaves_a_row_that_is_neither_indexed_nor_assigned_alone(core, status):
    _insert(status=status, model_id="model-1")

    row = await watcher_module.carry_on(SHA)

    assert row == _row() and row.status == status
    assert core.calls == []


async def test_carry_on_of_a_row_that_is_gone_does_nothing(core):
    assert await watcher_module.carry_on(SHA) is None
    assert core.calls == []


async def test_a_step_handed_over_runs_on_the_application_loop_and_its_error_is_logged(core, monkeypatch, caplog):
    loop = asyncio.get_running_loop()
    futures = []

    def run_on_app_loop(coro):
        futures.append(asyncio.run_coroutine_threadsafe(coro, loop))
        return futures[-1]

    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    _insert(status="indexed", model_id="model-1")

    async def broken():
        raise RuntimeError("database is locked")

    assert watcher_module.schedule_step(watcher_module.carry_on(SHA), SHA) is True
    assert watcher_module.schedule_step(broken(), OTHER_SHA) is True
    assert (_row().status, core.calls) == ("indexed", [])

    await asyncio.wrap_future(futures[0])
    with pytest.raises(RuntimeError):
        await asyncio.wrap_future(futures[1])

    assert _row().status == "enriched"
    assert f"Could not move {OTHER_SHA} on; the sweep will retry" in caplog.text
    assert f"Could not move {SHA} on" not in caplog.text


async def test_without_an_application_loop_a_step_is_left_to_the_sweep(core, monkeypatch, caplog):
    def run_on_app_loop(coro):
        coro.close()
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    _insert(status="indexed", model_id="model-1")
    step = watcher_module.carry_on(SHA)

    assert watcher_module.schedule_step(step, SHA) is False

    assert inspect.getcoroutinestate(step) == inspect.CORO_CLOSED
    assert f"Could not hand {SHA} over to the application loop (AppContainer not initialized yet)" in caplog.text
    assert _row().status == "indexed"
    await Watcher().run_pass()
    assert _row().status == "enriched"


async def test_a_sweep_on_an_older_copy_of_a_row_stops_where_another_step_moved_it_on(core):
    _insert(status="indexed", model_id="model-1")
    older = _row()
    await watcher_module.carry_on(SHA)
    core.calls.clear()

    assert await Watcher().advance(older) is None

    assert _row().status == "enriched"
    assert core.names() == ["assign_to_user"]
    assert [n["notification_type"] for n in core.notifications] == ["model-browser.download_done"]


async def test_an_older_copy_of_an_assigned_row_is_not_made_ready_twice(core):
    _insert(status="assigned", model_id="model-1")
    older = _row()
    await watcher_module.carry_on(SHA)

    assert await watcher_module.finish(older) is None

    assert _row().status == "enriched"
    assert len(core.notifications) == 1


async def test_a_row_that_has_ended_is_not_failed_again(core):
    _insert(status="enriched", model_id="model-1", limit_ref_id="model-browser:ref-1")

    assert await watcher_module.fail(_row(), "Stopped following this download after repeated errors") is None

    assert (_row().status, _row().error) == ("enriched", None)
    assert (core.refunds, core.notifications) == ([], [])


def test_a_model_is_recorded_only_for_a_row_that_is_still_queued(core):
    _insert(status="assigned", model_id="model-1")
    older = pending.with_status(_row(), "queued")

    assert watcher_module.record_indexed(older, model("model-2")) is None

    assert (_row().status, _row().model_id) == ("assigned", "model-1")


async def test_the_sweeps_counters_of_a_row_another_step_ended_start_over(core, monkeypatch):
    monkeypatch.setattr(watcher_module, "MAX_INDEX_LOOKUPS", 2)
    core.add_download("dl-1", "completed")
    _insert()
    watcher = Watcher()
    await watcher.run_pass()
    await watcher_module.fail(_row(), "The file was downloaded but not indexed")
    await watcher.run_pass()
    pending.delete(SHA)
    core.add_download("dl-2", "completed")
    _insert(download_id="dl-2")

    await watcher.run_pass()

    assert _row().status == "queued"


def test_model_attributes_come_from_the_chosen_version_and_the_license():
    info = _info()
    assert model_attributes(info) == ATTRIBUTES

    info["model"]["versions"].insert(0, dict(info["model"]["versions"][0], id="other", base_model="SD 1.5"))
    assert model_attributes(info)["mb_base_model"] == "SDXL 1.0"
    info["version_id"] = "missing"
    assert model_attributes(info)["mb_base_model"] == "SD 1.5"
    assert model_attributes({}) == {}
    assert model_attributes({"model": {"versions": ["junk"], "license": "MIT"}}) == {}


def test_indexed_since_compares_the_models_first_index_with_the_rows_creation():
    row = PendingRow(
        sha256=SHA, source="civitai", source_model_id="1", file_name="x", requested_by="u", assign_to="none",
        created_at="2026-10-11 10:00:00",
    )

    assert indexed_since({"created_at": "2026-10-11T10:00:00+00:00"}, row)
    assert indexed_since({"created_at": "2026-10-11T12:30:00+02:00"}, row)
    assert not indexed_since({"created_at": "2026-10-11T09:59:59+00:00"}, row)
    assert not indexed_since({"created_at": None}, row)
    assert not indexed_since({"created_at": "yesterday"}, row)


def test_sha256_of_file_is_none_for_a_file_that_cannot_be_read(tmp_path):
    assert sha256_of_file(str(tmp_path / "missing.safetensors")) is None
    present = tmp_path / "x.bin"
    present.write_bytes(b"abc")
    assert sha256_of_file(str(present)) == hashlib.sha256(b"abc").hexdigest()
