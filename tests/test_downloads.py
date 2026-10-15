import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.plugin_api.hooks as plugin_hooks
from backend import downloads, nsfw, pending
from backend.core_bridge import CoreBridgeError
from backend.downloads import DownloadError, DownloadRequest, destination, info_for_viewer, public_download
from backend.pending import PendingRow
from backend.schema import File
from backend.tables import create_tables

from .civitai_fakes import context
from .download_fakes import (
    FP16_URL,
    HF_URL,
    NSFW_IMAGE,
    OTHER_SHA,
    SAFE_IMAGE,
    SHA,
    URL,
    FakeCore,
    FakeSource,
    admin,
    body,
    civitai_detail,
    hf_detail,
    install_sources,
    limited,
    model,
    principal,
    requesting,
)

DOWNLOADS_KIND = "model-browser.downloads"
BYTES_KIND = "model-browser.download_bytes"


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


@pytest.fixture
def viewer(monkeypatch):
    state = SimpleNamespace(ctx=context("allowed", "show"))
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda user: state.ctx)
    return state


@pytest.fixture
def civitai(monkeypatch, core, viewer):
    fakes = install_sources(monkeypatch, FakeSource("civitai", civitai_detail()), FakeSource("huggingface", hf_detail()))
    return fakes["civitai"]


@pytest.fixture
def huggingface(civitai, monkeypatch):
    import backend.sources as sources_module

    return sources_module.get_source("huggingface")


async def _start(who, **overrides):
    return await downloads.start_download(who, DownloadRequest.model_validate(body(**overrides)))


async def _refused(who, **overrides):
    with pytest.raises(DownloadError) as raised:
        await _start(who, **overrides)
    return raised.value


async def test_a_download_is_queued_with_exactly_these_arguments(core, civitai):
    answer = await _start(principal())

    assert core.queued == [
        {
            "url": URL,
            "filename": "add-detail-xl.safetensors",
            "model_type": "lora",
            "subdir": None,
            "tags": ["model-browser", "model-browser:civitai"],
            "checksum_sha256": SHA,
            "provider_id": "civitai",
            "created_by": "user-1",
            "destination_backend_id": None,
        }
    ]
    assert civitai.calls == [("122359", 2)]
    assert (answer["already_installed"], answer["already_queued"]) == (False, False)
    assert answer["download"]["id"] == "dl-1" and answer["download"]["status"] == "pending"


async def test_the_pending_row_holds_the_chosen_version_and_file_for_the_provider(core, civitai):
    await _start(principal())

    row = pending.get(SHA)
    assert (row.status, row.download_id, row.model_id) == ("queued", "dl-1", None)
    assert (row.source, row.source_model_id, row.source_version_id) == ("civitai", "122359", "135867")
    assert (row.file_name, row.requested_by, row.assign_to) == ("add-detail-xl.safetensors", "user-1", "user:user-1")
    info = row.info
    assert (info["version_id"], info["file_id"], info["mb_type"]) == ("135867", "1001", "lora")
    assert info["model"]["name"] == "Detail Tweaker XL"
    assert info["model"]["license"]["summary"] == "CreativeML Open RAIL++-M; no selling"
    [version] = info["model"]["versions"]
    assert version["id"] == "135867" and version["base_model"] == "SDXL 1.0"
    assert [file["id"] for file in version["files"]] == ["1001"]
    assert [(image["url"], image["nsfw"], image["nsfw_level"]) for image in version["images"]] == [
        (SAFE_IMAGE, False, 1),
        (NSFW_IMAGE, True, 8),
    ]


async def test_the_info_stored_is_what_the_requester_may_see(core, civitai, viewer):
    viewer.ctx = context("blur", "blur")

    await _start(principal())

    images = pending.get(SHA).info["model"]["versions"][0]["images"]
    assert [(image["url"], image["nsfw"]) for image in images] == [(SAFE_IMAGE, False), (None, True)]


async def test_a_non_admin_gets_the_public_fields_of_the_download_only(core, civitai):
    answer = await _start(principal())

    assert set(answer["download"]) == set(downloads.PUBLIC_DOWNLOAD_FIELDS)
    assert "destination_path" not in answer["download"] and "url" not in answer["download"]
    assert answer["pending"]["sha256"] == SHA and answer["pending"]["download_id"] == "dl-1"


async def test_an_admin_downloads_for_the_instance_and_sees_the_whole_download(core, civitai):
    answer = await _start(admin())

    assert pending.get(SHA).assign_to == "none"
    assert answer["download"]["destination_path"] == "/app/models/lora/add-detail-xl.safetensors"
    assert core.queued[0]["created_by"] == "admin-1"


async def test_another_file_type_and_folders_are_passed_on(core, civitai):
    await _start(principal(), file_id="1002", model_type="checkpoint", subdir="sdxl/detail/")

    [queued] = core.queued
    assert (queued["url"], queued["model_type"], queued["subdir"]) == (FP16_URL, "checkpoint", "sdxl/detail")
    assert queued["checksum_sha256"] == OTHER_SHA
    assert pending.get(OTHER_SHA).info["mb_type"] == "checkpoint"


async def test_a_request_id_is_kept_on_the_row(core, civitai):
    prepared = await downloads.prepare_download(principal(), DownloadRequest.model_validate(body()))
    await downloads.queue_prepared(prepared, request_id="req-1")

    assert pending.get(SHA).request_id == "req-1"


async def test_a_source_the_grant_does_not_allow_is_refused_before_reading_it(core, civitai):
    error = await _refused(principal(sources=frozenset({"huggingface"})))

    assert (error.status, error.code) == (403, "forbidden")
    assert "civitai" in error.message
    assert civitai.calls == [] and core.queued == [] and pending.list_rows() == []


async def test_an_unknown_source_is_a_404(core, civitai):
    error = await _refused(admin(), source="civarchive")

    assert (error.status, error.code) == (404, "unknown_source")


async def test_a_type_the_grant_does_not_allow_is_refused(core, civitai):
    error = await _refused(principal(model_types=frozenset({"embedding"})))

    assert (error.status, error.code) == (403, "forbidden")
    assert "'lora'" in error.message
    assert core.queued == []


async def test_a_type_the_grant_allows_can_be_chosen(core, civitai):
    await _start(principal(model_types=frozenset({"embedding"})), model_type="embedding")

    assert core.queued[0]["model_type"] == "embedding"


async def test_a_file_above_the_size_limit_is_refused(core, civitai):
    error = await _refused(principal(max_file_gb=0.1))

    assert (error.status, error.code) == (403, "forbidden")
    assert "0.21 GB" in error.message and "0.1 GB" in error.message
    assert core.queued == []


async def test_a_file_within_the_size_limit_or_of_unknown_size_is_accepted(core, civitai):
    await _start(principal(max_file_gb=0.5))
    civitai.detail_value.versions[0].files[1] = civitai.detail_value.versions[0].files[1].model_copy(
        update={"size_bytes": None}
    )
    await _start(principal(max_file_gb=0.5), file_id="1002")

    assert len(core.queued) == 2


async def test_a_remote_destination_needs_allow_remote(core, civitai):
    error = await _refused(principal(), destination_backend_id="worker-1")
    assert (error.status, error.code) == (403, "forbidden")

    await _start(principal(allow_remote=True), destination_backend_id="worker-1")
    assert core.queued[0]["destination_backend_id"] == "worker-1"


async def test_a_download_url_off_the_sources_hosts_is_refused(core, civitai):
    files = civitai.detail_value.versions[0].files
    files[0] = files[0].model_copy(update={"download_url": "https://civitai.evil.example/api/download/models/1"})

    error = await _refused(admin())

    assert (error.status, error.code) == (403, "forbidden")
    assert core.queued == []


@pytest.mark.parametrize("ceiling, mode", [("blocked", "hide"), ("allowed", "hide"), ("blur", "hide")])
async def test_an_nsfw_model_is_not_found_for_a_user_who_hides_nsfw(core, civitai, viewer, ceiling, mode):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewer.ctx = context(ceiling, mode)

    error = await _refused(principal())

    assert (error.status, error.code) == (404, "not_found")
    assert core.queued == [] and pending.list_rows() == []


async def test_an_nsfw_version_is_not_found_for_a_blocked_user(core, civitai, viewer):
    civitai.detail_value = civitai_detail(version_nsfw=True)
    viewer.ctx = context("blocked", "hide")

    error = await _refused(principal())

    assert (error.status, error.code, error.message) == (404, "not_found", "Version not found")


async def test_an_nsfw_model_can_be_downloaded_by_a_user_who_blurs_it(core, civitai, viewer):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewer.ctx = context("blur", "blur")

    await _start(principal())

    assert len(core.queued) == 1


async def test_a_real_person_model_is_not_found_when_hide_poi_is_on(core, civitai, viewer):
    civitai.detail_value = civitai_detail(poi=True)
    viewer.ctx = context("allowed", "show", hide_poi=True)

    assert (await _refused(principal())).status == 404


@pytest.mark.parametrize(
    "overrides, message",
    [({"version_id": "999"}, "Version not found"), ({"file_id": "999"}, "File not found")],
)
async def test_an_unknown_version_or_file_is_not_found(core, civitai, overrides, message):
    error = await _refused(principal(), **overrides)

    assert (error.status, error.code, error.message) == (404, "not_found", message)


async def test_a_model_without_a_proposed_type_needs_one_from_the_request(core, civitai):
    civitai.detail_value = civitai.detail_value.model_copy(update={"mb_type": None})

    error = await _refused(principal())
    assert (error.status, error.code) == (400, "invalid_request")

    await _start(principal(), model_type="lora")
    assert core.queued[0]["model_type"] == "lora"


async def test_the_source_error_is_raised_for_the_route(core, civitai):
    from backend.http import SourceNotFound

    civitai.error = SourceNotFound("Model not found")

    with pytest.raises(SourceNotFound):
        await _start(principal())


async def test_a_user_picks_one_of_their_groups_when_their_grant_assigns_to_a_group(core, civitai):
    core.memberships["user-1"] = ["all_users", "artists"]

    await _start(principal(assign_to="group"), assign_group_id="artists")

    assert pending.get(SHA).assign_to == "group:artists"


async def test_a_group_that_is_not_the_users_is_refused(core, civitai):
    core.memberships["user-1"] = ["all_users", "artists"]

    error = await _refused(principal(assign_to="group"), assign_group_id="testers")

    assert (error.status, error.code) == (403, "forbidden")
    assert core.queued == []


async def test_a_group_is_refused_when_the_grant_assigns_to_the_user(core, civitai):
    core.memberships["user-1"] = ["artists"]

    error = await _refused(principal(assign_to="user"), assign_group_id="artists")

    assert (error.status, error.code) == (403, "forbidden")


async def test_without_a_group_the_model_goes_to_the_user_even_when_groups_are_allowed(core, civitai):
    await _start(principal(assign_to="group"))

    assert pending.get(SHA).assign_to == "user:user-1"


async def test_an_admin_may_pick_any_existing_group(core, civitai):
    await _start(admin(), assign_group_id="testers")
    assert pending.get(SHA).assign_to == "group:testers"

    error = await _refused(admin(), file_id="1002", assign_group_id="ghosts")
    assert (error.status, error.code) == (400, "invalid_request")


def _request(**overrides):
    return DownloadRequest.model_validate(body(**overrides))


def _file(name):
    return File(id=name, name=name, download_url=HF_URL)


def test_a_nested_source_path_puts_its_folders_under_the_subdir():
    assert destination(_request(), _file("transformer/flux1-dev.safetensors")) == ("transformer", "flux1-dev.safetensors")
    assert destination(_request(subdir="flux"), _file("a/b/x.safetensors")) == ("flux/a/b", "x.safetensors")
    assert destination(_request(), _file("x.safetensors")) == (None, "x.safetensors")


def test_a_file_name_from_the_request_replaces_the_source_path():
    assert destination(_request(filename="flux-dev.safetensors", subdir="flux"), _file("transformer/x.safetensors")) == (
        "flux",
        "flux-dev.safetensors",
    )
    assert destination(_request(filename="FLUX.SafeTensors"), _file("x.safetensors")) == (None, "FLUX.SafeTensors")


def test_a_file_name_from_the_request_keeps_the_files_extension():
    with pytest.raises(DownloadError) as raised:
        destination(_request(filename="flux.ckpt"), _file("x.safetensors"))

    assert (raised.value.status, raised.value.code) == (400, "invalid_request")
    assert "'.safetensors'" in raised.value.message


def test_a_source_file_name_that_could_leave_its_folder_needs_a_name_from_the_request():
    with pytest.raises(DownloadError) as raised:
        destination(_request(), _file("../x.safetensors"))

    assert raised.value.code == "invalid_request"
    assert destination(_request(filename="x.safetensors"), _file("../x.safetensors")) == (None, "x.safetensors")


async def test_a_nested_hugging_face_file_is_queued_without_a_hash(core, huggingface):
    await _start(
        principal(),
        source="huggingface",
        model_id="black-forest-labs/FLUX.1-dev",
        version_id="main",
        file_id="transformer/flux1-dev.safetensors",
        subdir="flux",
    )

    [queued] = core.queued
    assert (queued["filename"], queued["subdir"], queued["model_type"]) == ("flux1-dev.safetensors", "flux/transformer", "diffusion_model")
    assert (queued["checksum_sha256"], queued["provider_id"]) == (None, "huggingface")
    assert queued["tags"] == ["model-browser", "model-browser:huggingface"]
    row = pending.get(pending.NOHASH_PREFIX + HF_URL)
    assert row.nohash and row.file_name == "flux1-dev.safetensors"
    assert "find_model_by_sha256" not in core.names()


@pytest.mark.parametrize(
    "overrides",
    [
        {"subdir": "../models"},
        {"subdir": "a/../b"},
        {"subdir": "/etc"},
        {"subdir": "a\\b"},
        {"subdir": "C:/models"},
        {"subdir": "a//b"},
        {"filename": "sub/x.safetensors"},
        {"filename": ".."},
        {"filename": "x\n.safetensors"},
        {"model_type": "lycoris"},
        {"model_id": "../122359"},
        {"model_id": ""},
        {"file_id": "a\\b"},
        {"source": ""},
    ],
)
def test_the_body_refuses_paths_and_values_that_could_escape(overrides):
    with pytest.raises(ValidationError):
        _request(**overrides)


def test_the_body_accepts_numbers_for_ids_and_reads_blank_options_as_absent():
    request = _request(model_id=122359, version_id=135867, file_id=1001, subdir=" ", filename="", model_type="", assign_group_id="")

    assert (request.model_id, request.version_id, request.file_id) == ("122359", "135867", "1001")
    assert (request.subdir, request.filename, request.model_type, request.assign_group_id) == (None, None, None, None)


async def test_an_installed_file_is_assigned_rather_than_downloaded_again(core, civitai):
    core.models_by_sha[SHA] = model("model-9")

    answer = await _start(principal())

    assert core.queued == []
    row = pending.get(SHA)
    assert (row.status, row.model_id, row.download_id, row.assign_to) == ("indexed", "model-9", None, "user:user-1")
    assert (answer["already_installed"], answer["already_queued"], answer["download"]) == (True, False, None)
    assert answer["pending"]["status"] == "indexed"


async def test_an_installed_file_is_handed_over_to_the_application_loop_at_once(core, civitai, scheduled):
    core.models_by_sha[SHA] = model("model-9")

    await _start(principal())

    assert len(scheduled) == 1
    assert (pending.get(SHA).status, core.calls_to("assign_to_user")) == ("indexed", [])
    await scheduled[0]
    row = pending.get(SHA)
    assert (row.status, row.model_id) == ("enriched", "model-9")
    assert core.calls_to("assign_to_user") == [("model-9", "user-1")]
    assert core.notifications[0]["notification_type"] == "model-browser.download_done"


async def test_a_file_queued_or_already_on_its_way_hands_nothing_over(core, civitai, scheduled):
    await _start(principal())
    await _start(principal("user-2"))

    assert scheduled == []


async def test_an_installed_file_is_answered_even_without_an_application_loop(core, civitai, monkeypatch, caplog):
    def run_on_app_loop(coro):
        coro.close()
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(plugin_hooks, "run_on_app_loop", run_on_app_loop)
    core.models_by_sha[SHA] = model("model-9")

    answer = await _start(principal())

    assert answer["already_installed"] is True
    assert pending.get(SHA).status == "indexed"
    assert "the sweep will take it up" in caplog.text


async def test_an_installed_model_whose_file_is_gone_is_downloaded_again(core, civitai):
    core.models_by_sha[SHA] = model("model-9", is_available=False)

    answer = await _start(principal())

    assert len(core.queued) == 1
    assert pending.get(SHA).status == "queued" and answer["already_installed"] is False


async def test_a_file_already_on_its_way_is_not_queued_twice(core, civitai):
    first = await _start(principal())
    core.downloads["dl-1"]["status"] = "downloading"

    again = await _start(principal("user-2"))

    assert len(core.queued) == 1
    assert (again["already_queued"], again["already_installed"]) == (True, False)
    assert again["pending"]["requested_by"] == "user-1"
    assert again["download"]["status"] == "downloading"
    assert first["pending"]["sha256"] == again["pending"]["sha256"]


async def test_a_finished_row_is_replaced_by_a_new_download(core, civitai):
    pending.insert(
        PendingRow(
            sha256=SHA, source="civitai", source_model_id="122359", file_name="old.safetensors",
            requested_by="user-2", assign_to="user:user-2", status="failed", error="HTTP 500",
        )
    )

    answer = await _start(principal())

    row = pending.get(SHA)
    assert (row.status, row.requested_by, row.error, row.download_id) == ("queued", "user-1", None, "dl-1")
    assert answer["already_queued"] is False and len(core.queued) == 1


async def test_a_row_written_meanwhile_by_another_request_is_answered_as_already_queued(core, civitai, monkeypatch):
    real_get = pending.get
    pending.insert(
        PendingRow(
            sha256=SHA, source="civitai", source_model_id="122359", file_name="add-detail-xl.safetensors",
            requested_by="user-2", assign_to="user:user-2", download_id="dl-7",
        )
    )
    looks = []

    def get(key):
        looks.append(key)
        return None if len(looks) == 1 else real_get(key)

    monkeypatch.setattr(pending, "get", get)

    answer = await _start(principal())

    assert answer["already_queued"] is True and answer["pending"]["requested_by"] == "user-2"
    assert core.queued == []


async def test_an_insert_that_fails_for_another_reason_is_raised(core, civitai, monkeypatch):
    def insert(row):
        raise sqlite3.IntegrityError("NOT NULL constraint failed")

    monkeypatch.setattr(pending, "insert", insert)

    with pytest.raises(sqlite3.IntegrityError):
        await _start(principal())


@pytest.mark.parametrize("code, status", [("blocked", 409), ("invalid", 400)])
async def test_a_refusal_from_the_core_deletes_the_row(core, civitai, code, status):
    core.queue_error = CoreBridgeError(code, "Quota plugin: monthly limit reached")

    error = await _refused(principal())

    assert (error.status, error.code, error.message) == (status, code, "Quota plugin: monthly limit reached")
    assert pending.get(SHA) is None


async def test_an_unexpected_queue_failure_deletes_the_row_and_is_raised(core, civitai):
    core.queue_error = RuntimeError("model-browser: PotionUI's AppContainer no longer provides download_queue.queue_model_download")

    with pytest.raises(RuntimeError):
        await _start(principal())

    assert pending.get(SHA) is None


def _stored_info(detail=None):
    detail = detail or civitai_detail()
    version = detail.versions[0]
    return downloads.pending_info(detail, version, version.files[0], "lora")


def test_info_for_a_viewer_who_shows_nsfw_is_unchanged():
    info = _stored_info()

    assert info_for_viewer(info, context("allowed", "show")) == info


def test_info_for_a_blurring_viewer_has_no_nsfw_url_and_a_hiding_viewer_no_nsfw_image():
    info = _stored_info()

    blurred = info_for_viewer(info, context("blur", "blur"))["model"]["versions"][0]["images"]
    assert [(image["url"], image["blurhash"]) for image in blurred] == [(SAFE_IMAGE, None), (None, "LKO2?U%2Tw=w")]
    hidden = info_for_viewer(info, context("blocked", "hide"))["model"]["versions"][0]["images"]
    assert [image["url"] for image in hidden] == [SAFE_IMAGE]
    assert info["model"]["versions"][0]["images"][1]["url"] == NSFW_IMAGE


def test_an_nsfw_model_is_hidden_from_a_viewer_who_hides_nsfw_but_keeps_its_name_and_file():
    info = _stored_info(civitai_detail(nsfw=True))

    shown = info_for_viewer(info, context("blocked", "hide"))

    assert shown["hidden"] is True
    assert shown["model"]["name"] == "Detail Tweaker XL"
    assert shown["model"]["description_html"] is None
    assert shown["model"]["versions"][0]["images"] == []
    assert [file["id"] for file in shown["model"]["versions"][0]["files"]] == ["1001"]


def test_info_without_a_readable_model_is_left_alone_or_emptied():
    assert info_for_viewer({"version_id": "1"}, context("blocked", "hide")) == {"version_id": "1"}
    assert info_for_viewer({"model": {"id": 1}}, context("blocked", "hide"))["model"] is None


def test_public_download_keeps_the_public_fields_for_a_non_admin():
    download = {"id": "dl-1", "status": "completed", "destination_path": "/app/models/x", "url": "https://x", "progress": 1.0}

    assert public_download(download, admin=True) == download
    shown = public_download(download, admin=False)
    assert shown["id"] == "dl-1" and shown["status"] == "completed" and shown["progress"] == 1.0
    assert "destination_path" not in shown and "url" not in shown
    assert public_download(None, admin=False) is None


def test_row_for_viewer_filters_the_info_of_the_row():
    row = PendingRow(
        sha256=SHA, source="civitai", source_model_id="122359", file_name="x.safetensors",
        requested_by="user-1", assign_to="user:user-1", info=_stored_info(),
    )

    data = downloads.row_for_viewer(row, context("blur", "blur"))

    assert data["sha256"] == SHA
    assert data["info"]["model"]["versions"][0]["images"][1]["url"] is None
    assert data["info"]["model"]["description_html"] is None
    assert data["info"]["model"]["name"] == "Detail Tweaker XL"
    assert row.info["model"]["description_html"] == "<p>Adds detail.</p>"
    assert row.info["model"]["versions"][0]["images"][1]["url"] == NSFW_IMAGE


async def test_a_queued_file_counts_one_download_and_its_size_for_the_requester(core, civitai):
    await _start(principal())

    row = pending.get(SHA)
    assert row.limit_ref_id.startswith("model-browser-c:")
    assert core.limit_events == [
        ("user-1", DOWNLOADS_KIND, 1, row.limit_ref_id),
        ("user-1", BYTES_KIND, 228_000_000, row.limit_ref_id),
    ]
    assert core.calls_to("limit_status") == [("user-1", DOWNLOADS_KIND), ("user-1", BYTES_KIND)]


async def test_each_queued_file_gets_its_own_ref_id(core, civitai):
    await _start(principal())
    await _start(principal(), file_id="1002")

    assert pending.get(SHA).limit_ref_id != pending.get(OTHER_SHA).limit_ref_id


async def test_a_download_over_the_count_is_refused_before_anything_is_written(core, civitai):
    core.limits[("user-1", DOWNLOADS_KIND)] = limited(DOWNLOADS_KIND, 1, 1)

    error = await _refused(principal())

    assert (error.status, error.code) == (403, "limit_exceeded")
    assert error.message.startswith("You have reached your model downloads per month limit: 1 of 1 used.")
    assert error.extra == {
        "kind": DOWNLOADS_KIND,
        "label": "Model downloads per month",
        "used": 1,
        "limit": 1,
        "incoming": 1,
        "resets_at": "2026-11-01T00:00:00+00:00",
    }
    assert core.queued == [] and pending.list_rows() == [] and core.limit_events == []


async def test_a_file_over_the_volume_is_refused(core, civitai):
    core.limits[("user-1", BYTES_KIND)] = limited(BYTES_KIND, 3 * 1024 ** 3, 2.5 * 1024 ** 3)

    await _start(principal())
    error = await _refused(principal(), file_id="1002")

    assert (error.code, error.extra["kind"], error.extra["incoming"]) == ("limit_exceeded", BYTES_KIND, 3 * 1024 ** 3)
    assert len(core.queued) == 1


async def test_an_installed_file_and_a_file_already_on_its_way_use_no_quota(core, civitai):
    core.limits[("user-1", DOWNLOADS_KIND)] = limited(DOWNLOADS_KIND, 1, 1)
    core.limits[("user-2", DOWNLOADS_KIND)] = limited(DOWNLOADS_KIND, 1, 1)
    core.models_by_sha[SHA] = model("model-9")

    installed = await _start(principal())
    await _start(admin(), file_id="1002")
    queued = await _start(principal("user-2"), file_id="1002")

    assert installed["already_installed"] and queued["already_queued"]
    assert pending.get(SHA).limit_ref_id is None
    assert [event[0] for event in core.limit_events] == ["admin-1", "admin-1"]
    assert [call[0] for call in core.calls_to("limit_status")] == ["admin-1", "admin-1"]


async def test_a_download_the_core_refuses_records_nothing(core, civitai):
    core.queue_error = CoreBridgeError("blocked", "Disk full")

    await _refused(principal())

    assert core.limit_events == []


async def test_a_request_is_checked_against_the_request_view_only(core, civitai):
    may_only_request = principal(sources=frozenset(), request_params=requesting())

    prepared = await downloads.prepare_download(may_only_request, _request(), purpose="request")

    assert (prepared.on_behalf, prepared.assign_to) == (False, "user:user-1")
    with pytest.raises(DownloadError) as raised:
        await downloads.prepare_download(may_only_request, _request())
    assert raised.value.message == "Your permissions do not allow downloads from 'civitai'"

    may_only_download = principal(request_params=requesting(sources=frozenset()))
    assert (await downloads.prepare_download(may_only_download, _request())).assign_to == "user:user-1"
    with pytest.raises(DownloadError):
        await downloads.prepare_download(may_only_download, _request(), purpose="request")


@pytest.mark.parametrize(
    "view, overrides, message",
    [
        ({"sources": frozenset({"huggingface"})}, {}, "Your permissions do not allow requesting models from 'civitai'"),
        ({"model_types": frozenset({"embedding"})}, {}, "Your permissions do not allow requesting 'lora' models"),
        ({"max_file_gb": 0.1}, {}, "This file is 0.21 GB: your permissions allow requesting files of at most 0.1 GB"),
        (
            {},
            {"assign_group_id": "artists"},
            "Your requests are assigned to you: your permissions do not let you pick a group",
        ),
        (
            {"assign_to": "group"},
            {"assign_group_id": "testers"},
            "You can only assign a request to one of your own groups",
        ),
    ],
    ids=["source", "type", "size", "group-not-allowed", "group-not-theirs"],
)
async def test_request_refusals_speak_of_requesting(core, civitai, view, overrides, message):
    core.memberships["user-1"] = ["all_users", "artists"]

    with pytest.raises(DownloadError) as raised:
        await downloads.prepare_download(principal(request_params=requesting(**view)), _request(**overrides), purpose="request")

    assert (raised.value.status, raised.value.code, raised.value.message) == (403, "forbidden", message)


async def test_a_request_never_names_a_remote_destination(core, civitai):
    with pytest.raises(DownloadError) as raised:
        await downloads.prepare_download(admin(), _request(destination_backend_id="worker-1"), purpose="request")

    assert raised.value.message == "A request cannot choose a remote destination: the file is downloaded to this server"
    prepared = await downloads.prepare_download(admin(), _request(destination_backend_id="worker-1"))
    assert prepared.request.destination_backend_id == "worker-1"


async def test_an_unknown_purpose_is_a_programming_error(core, civitai):
    with pytest.raises(ValueError):
        await downloads.prepare_download(principal(), _request(), purpose="upload")


async def _prepare_for(who, **overrides):
    return await downloads.prepare_download(who, _request(**overrides), purpose="request", on_behalf=True)


async def _refused_for(who, **overrides):
    with pytest.raises(DownloadError) as raised:
        await _prepare_for(who, **overrides)
    return raised.value


async def test_a_file_queued_for_a_requester_is_theirs_created_by_the_approver_and_shown_to_the_approver(core, civitai):
    prepared = await _prepare_for(principal("user-1"))

    answer = await downloads.queue_prepared(
        prepared, request_id="req-1", created_by="admin-1", viewer_ctx=context("blocked", "hide"), viewer_admin=True
    )

    assert core.queued[0]["created_by"] == "admin-1"
    row = pending.get(SHA)
    assert (row.requested_by, row.assign_to, row.request_id) == ("user-1", "user:user-1", "req-1")
    assert answer["download"]["destination_path"]
    images = answer["pending"]["info"]["model"]["versions"][0]["images"]
    assert [image["url"] for image in images] == [SAFE_IMAGE]
    assert {event[0] for event in core.limit_events} == {"user-1"}


async def test_without_a_viewer_the_answer_is_for_the_requester(core, civitai):
    answer = await downloads.queue_prepared(await _prepare_for(principal()), created_by="admin-1")

    assert set(answer["download"]) == set(downloads.PUBLIC_DOWNLOAD_FIELDS)
    assert len(answer["pending"]["info"]["model"]["versions"][0]["images"]) == 2


@pytest.mark.parametrize(
    "view, overrides, message",
    [
        (
            {"sources": frozenset({"huggingface"})},
            {},
            "The requester's permissions do not allow requesting models from 'civitai'",
        ),
        (
            {"model_types": frozenset({"embedding"})},
            {},
            "The requester's permissions do not allow requesting 'lora' models",
        ),
        (
            {"max_file_gb": 0.1},
            {},
            "This file is 0.21 GB: the requester's permissions allow requesting files of at most 0.1 GB",
        ),
        (
            {},
            {"assign_group_id": "artists"},
            "The requester's requests are assigned to them: their permissions do not let you pick a group",
        ),
        (
            {"assign_to": "group"},
            {"assign_group_id": "testers"},
            "A request can only be assigned to one of the requester's groups",
        ),
    ],
    ids=["source", "type", "size", "group-not-allowed", "group-not-theirs"],
)
async def test_on_behalf_refusals_speak_of_the_requester(core, civitai, view, overrides, message):
    core.memberships["user-1"] = ["all_users", "artists"]

    error = await _refused_for(principal(request_params=requesting(**view)), **overrides)

    assert (error.status, error.code, error.message) == (403, "forbidden", message)


async def test_on_behalf_the_requesters_download_view_does_not_count(core, civitai):
    prepared = await _prepare_for(principal(sources=frozenset(), max_file_gb=0.01))

    assert prepared.on_behalf


async def test_an_approver_may_pick_one_of_the_requesters_groups(core, civitai):
    core.memberships["user-1"] = ["all_users", "artists"]

    prepared = await _prepare_for(principal(request_params=requesting(assign_to="group")), assign_group_id="artists")

    assert prepared.assign_to == "group:artists" and prepared.on_behalf


async def test_a_model_the_requester_may_not_see_is_not_visible_to_requester(core, civitai, viewer):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewer.ctx = context("blocked", "hide")

    error = await _refused_for(principal())

    assert (error.status, error.code) == (403, "not_visible_to_requester")
    assert core.queued == []


async def test_a_hidden_version_is_not_visible_but_a_missing_one_is_not_found(core, civitai, viewer):
    civitai.detail_value = civitai_detail(version_nsfw=True)
    viewer.ctx = context("blocked", "hide")

    assert (await _refused_for(principal())).code == "not_visible_to_requester"
    missing = await _refused_for(principal(), version_id="999")
    assert (missing.status, missing.code, missing.message) == (404, "not_found", "Version not found")


async def test_the_quota_refusal_on_behalf_names_the_requester(core, civitai):
    core.limits[("user-1", DOWNLOADS_KIND)] = limited(DOWNLOADS_KIND, 1, 1)
    prepared = await _prepare_for(principal())

    with pytest.raises(DownloadError) as raised:
        await downloads.queue_prepared(prepared, created_by="admin-1")

    assert raised.value.code == "limit_exceeded"
    assert raised.value.message == (
        "user-1 has reached their model downloads per month limit: 1 of 1 used. It resets on Nov 1 at 00:00 UTC."
    )
