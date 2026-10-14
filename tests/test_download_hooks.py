import asyncio
import logging

import pytest

import src.plugin_api.hooks as plugin_hooks
from backend import core_bridge, downloads, pending, requests
from backend.pending import NOHASH_PREFIX, PendingRow
from backend.tables import create_tables
from backend.watcher import Watcher
from hooks.download_hooks import on_download_complete
from src.plugin_api import HookContext

from .download_fakes import HF_URL, OTHER_SHA, SHA, URL, FakeCore, civitai_detail, model

PATH = "/app/models/loras/add-detail-xl.safetensors"
HF_PATH = "/app/models/diffusion_models/transformer/flux1-dev.safetensors"
NOHASH_KEY = NOHASH_PREFIX + HF_URL


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


def _insert(key=SHA, **fields):
    detail = civitai_detail()
    version = detail.versions[0]
    values = dict(
        sha256=key,
        source="civitai",
        source_model_id="122359",
        source_version_id="135867",
        file_name="add-detail-xl.safetensors",
        requested_by="user-1",
        assign_to="user:user-1",
        info=downloads.pending_info(detail, version, version.files[0], "lora"),
        download_id="dl-1",
    )
    values.update(fields)
    return pending.insert(PendingRow(**values))


def _row(key=SHA):
    return pending.get(key)


def _completed(**fields):
    data = {
        "download_id": "dl-1",
        "url": URL,
        "type": "model",
        "filename": "add-detail-xl.safetensors",
        "destination_path": PATH,
        "checksum_sha256": SHA,
        "created_by": "user-1",
        "destination_backend_id": None,
        "model_id": "model-1",
        "duplicate_of": None,
        "warning": None,
    }
    data.update(fields)
    return HookContext(hook_name="download.after_complete", plugin_id="model-browser", data=data)


def _hf_completed(**fields):
    return _completed(url=HF_URL, checksum_sha256=None, destination_path=HF_PATH, filename="flux1-dev.safetensors", **fields)


async def _handle(context):
    return await asyncio.to_thread(on_download_complete, context)


async def test_a_download_of_the_plugin_is_indexed_and_the_rest_handed_over(core, scheduled):
    _insert()
    context = _completed()
    payload = dict(context.data)

    assert await _handle(context) is context

    assert context.data == payload
    row = _row()
    assert (row.status, row.model_id, row.download_id) == ("indexed", "model-1", "dl-1")
    assert core.calls == []
    assert len(scheduled) == 1
    await scheduled[0]
    assert (_row().status, _row().error) == ("enriched", None)
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert [n["notification_type"] for n in core.notifications] == ["model-browser.download_done"]


async def test_a_row_whose_download_id_is_not_stored_yet_is_found_by_its_files_key(core, scheduled):
    _insert(download_id=None)

    await _handle(_completed())

    assert (_row().status, _row().model_id) == ("indexed", "model-1")
    assert len(scheduled) == 1


async def test_a_copy_of_a_model_already_indexed_is_recorded_as_that_model(core, scheduled):
    _insert()

    await _handle(_completed(model_id=None, duplicate_of="model-7"))

    assert (_row().status, _row().model_id) == ("indexed", "model-7")
    assert core.calls == []
    await scheduled[0]
    assert core.calls_to("assign_to_user") == [("model-7", "user-1")]
    assert _row().status == "enriched"


@pytest.mark.parametrize(
    "warning, error",
    [
        (
            "Downloaded, but another model named 'add-detail-xl.safetensors' already exists as a checkpoint, "
            "so this file was not indexed separately.",
            "The file was downloaded but not indexed: Downloaded, but another model named "
            "'add-detail-xl.safetensors' already exists as a checkpoint, so this file was not indexed separately.",
        ),
        (None, "The file was downloaded but not indexed; an admin can index it from the Models page"),
    ],
    ids=["with-the-indexers-warning", "without-a-warning"],
)
async def test_a_file_the_catalog_did_not_index_fails_its_row(core, scheduled, warning, error):
    _insert(limit_ref_id="model-browser:ref-1")

    await _handle(_completed(model_id=None, duplicate_of=None, warning=warning))

    assert _row().status == "queued"
    assert len(scheduled) == 1
    await scheduled[0]
    row = _row()
    assert (row.status, row.error) == ("failed", error)
    assert core.refunds == ["model-browser:ref-1"]
    [notification] = core.notifications
    assert (notification["user_id"], notification["notification_type"]) == ("user-1", "model-browser.download_failed")
    assert notification["message"].endswith(error)
    assert "assign_to_user" not in core.names()


def _approved_request(user_id, sha256):
    created = requests.create(
        user_id=user_id, source="huggingface", source_model_id="black-forest-labs/FLUX.1-dev",
        source_version_id="main", file_id="transformer/flux1-dev.safetensors", snapshot={},
    )
    return requests.set_status(created.id, "approved", decided_by="admin-1", sha256=sha256, assign_to=f"user:{user_id}")


async def test_a_nohash_row_moves_to_the_hash_of_the_model_indexed_at_its_path(core, scheduled):
    core.models_by_path[HF_PATH] = model("model-2", OTHER_SHA)
    request = _approved_request("user-2", NOHASH_KEY)
    _insert(NOHASH_KEY, source="huggingface", download_id=None)

    await _handle(_hf_completed(model_id="model-2"))

    assert _row(NOHASH_KEY) is None
    row = _row(OTHER_SHA)
    assert (row.status, row.model_id) == ("indexed", "model-2")
    assert requests.get(request.id).sha256 == OTHER_SHA
    assert core.calls == [("find_model_by_path", HF_PATH)]
    await scheduled[0]
    assert _row(OTHER_SHA).status == "enriched"
    assert core.calls_to("assign_to_user") == [("model-2", "user-1"), ("model-2", "user-2")]


async def test_a_nohash_copy_of_an_indexed_model_moves_to_that_models_hash(core, scheduled):
    core.models_by_sha[OTHER_SHA] = model("model-4", OTHER_SHA, created="long_ago")
    _insert(NOHASH_KEY, source="huggingface")

    await _handle(_hf_completed(model_id=None, duplicate_of="model-4"))

    assert _row(NOHASH_KEY) is None
    assert (_row(OTHER_SHA).status, _row(OTHER_SHA).model_id) == ("indexed", "model-4")
    assert core.calls == [("find_model_by_id", "model-4")]


async def test_a_nohash_row_keeps_its_key_when_another_row_has_the_models_hash(core, scheduled):
    core.models_by_path[HF_PATH] = model("model-2", OTHER_SHA)
    _insert(NOHASH_KEY, source="huggingface")
    other = _insert(OTHER_SHA, status="enriched", requested_by="user-2", model_id="model-2", download_id="dl-9")

    await _handle(_hf_completed(model_id="model-2"))

    assert (_row(NOHASH_KEY).status, _row(NOHASH_KEY).model_id) == ("indexed", "model-2")
    assert _row(OTHER_SHA) == other
    await scheduled[0]
    assert _row(NOHASH_KEY).status == "enriched"
    assert core.calls_to("assign_to_user") == [("model-2", "user-1")]


async def test_a_nohash_row_keeps_its_key_when_the_model_at_its_path_is_another(core, scheduled):
    core.models_by_path[HF_PATH] = model("someone-elses-model", OTHER_SHA, created="long_ago")
    _insert(NOHASH_KEY, source="huggingface")

    await _handle(_hf_completed(model_id="model-2"))

    assert (_row(NOHASH_KEY).status, _row(NOHASH_KEY).model_id) == ("indexed", "model-2")
    assert _row(OTHER_SHA) is None
    assert core.calls == [("find_model_by_path", HF_PATH), ("find_model_by_id", "model-2")]


async def test_without_a_row_of_its_own_the_handler_writes_nothing(core, scheduled):
    context = _completed()
    payload = dict(context.data)

    assert await _handle(context) is context

    assert context.data == payload
    assert pending.list_rows() == []
    assert (scheduled, core.calls) == ([], [])


@pytest.mark.parametrize(
    "fields",
    [
        {"download_id": "dl-9", "checksum_sha256": OTHER_SHA, "url": "https://example.org/other.safetensors"},
        {"download_id": "dl-9"},
        {"download_id": "dl-9", "type": "media", "url": "https://example.org/x.png", "checksum_sha256": None, "model_id": None},
    ],
    ids=["another-file", "the-same-file-another-download", "a-media-download"],
)
async def test_another_download_of_the_instance_is_ignored(core, scheduled, fields):
    stored = _insert()

    await _handle(_completed(**fields))

    assert pending.list_rows() == [stored]
    assert (scheduled, core.calls) == ([], [])


@pytest.mark.parametrize("status", ["indexed", "assigned", "enriched", "failed"])
async def test_a_row_that_is_no_longer_queued_is_left_as_it_is(core, scheduled, caplog, status):
    caplog.set_level(logging.INFO, logger="plugins.model-browser")
    stored = _insert(status=status, model_id="model-0", error="HTTP 500" if status == "failed" else None)

    await _handle(_completed())

    assert _row() == stored
    assert (scheduled, core.calls) == ([], [])
    assert f"Download dl-1 completed, but add-detail-xl.safetensors is {status} already: left as it is" in caplog.text


async def test_a_late_completion_of_a_download_the_plugin_gave_up_on_revives_nothing(core, scheduled):
    core.add_download("dl-1", "failed", error_message="HTTP 500")
    _insert(limit_ref_id="model-browser:ref-1")
    await Watcher().run_pass()
    failed = _row()

    await _handle(_completed())

    assert _row() == failed and failed.status == "failed"
    assert core.refunds == ["model-browser:ref-1"] and core.limit_events == []
    assert [n["notification_type"] for n in core.notifications] == ["model-browser.download_failed"]
    assert scheduled == []


async def test_a_download_sent_to_a_remote_backend_is_left_to_the_sweep(core, scheduled):
    _insert(limit_ref_id="model-browser:ref-1")

    await _handle(_completed(model_id=None, destination_backend_id="worker-1"))

    assert (_row().status, scheduled, core.refunds) == ("queued", [], [])
    core.add_download("dl-1", "completed", destination_backend_id="worker-1")
    core.models_by_sha[SHA] = model("model-1")
    await Watcher().run_pass()
    assert _row().status == "enriched"


async def test_without_an_application_loop_the_row_is_indexed_and_the_sweep_finishes_it(core, monkeypatch, caplog):
    def run_on_app_loop(coro):
        coro.close()
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    _insert()
    context = _completed()

    assert await _handle(context) is context

    assert _row().status == "indexed"
    assert "the sweep will take it up" in caplog.text
    await Watcher().run_pass()
    assert _row().status == "enriched"


async def test_a_sweep_that_saw_the_row_queued_does_not_record_it_again(core, scheduled):
    core.add_download("dl-1", "completed")
    core.models_by_sha[SHA] = model("model-1")
    _insert()
    older = _row()
    await _handle(_completed())
    await scheduled[0]

    assert await Watcher().advance(older) is None

    assert _row().status == "enriched"
    assert len(core.notifications) == 1


async def test_the_handler_never_raises_when_the_database_fails(core, scheduled, monkeypatch, caplog):
    def broken(download_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(pending, "get_by_download_id", broken)
    context = _completed()

    assert await _handle(context) is context

    assert "Could not take up the completed download dl-1; the sweep will" in caplog.text
    assert scheduled == []


async def test_the_handler_never_raises_when_the_core_fails(core, scheduled, monkeypatch, caplog):
    def broken(path):
        raise RuntimeError("no such table: models")

    monkeypatch.setattr(core_bridge, "find_model_by_path", broken)
    stored = _insert(NOHASH_KEY, source="huggingface")
    context = _hf_completed(model_id="model-2")

    assert await _handle(context) is context

    assert _row(NOHASH_KEY) == stored
    assert scheduled == []
    assert "Could not take up the completed download dl-1; the sweep will" in caplog.text
