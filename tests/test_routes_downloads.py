from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import core_bridge, downloads, nsfw, pending, permissions, routes_downloads
from backend.api import router
from backend.core_bridge import CoreBridgeError
from backend.http import SourceNotFound, SourceRateLimited, SourceUnavailable
from backend.pending import PendingRow
from backend.permissions import require_browse, require_download
from backend.tables import create_tables
from src.plugin_api import get_current_active_user

from .civitai_fakes import context
from .download_fakes import (
    NSFW_IMAGE,
    OTHER_SHA,
    SAFE_IMAGE,
    SHA,
    USER_GRANT,
    FakeCore,
    FakeSource,
    admin,
    body,
    civitai_detail,
    hf_detail,
    install_sources,
    model,
    principal,
    user,
)

BASE = "/api/plugins/model-browser-c"


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


@pytest.fixture
def viewer(monkeypatch):
    state = SimpleNamespace(ctx=context("allowed", "show"))
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda who: state.ctx)
    return state


@pytest.fixture
def civitai(monkeypatch, core, viewer):
    return install_sources(monkeypatch, FakeSource("civitai", civitai_detail()), FakeSource("huggingface", hf_detail()))[
        "civitai"
    ]


@pytest.fixture
def app():
    application = FastAPI()
    application.include_router(router)
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client_as(app):
    clients = []

    def make(who):
        for dependency in (require_download, require_browse, routes_downloads.require_downloads_access):
            app.dependency_overrides[dependency] = lambda who=who: who
        client = TestClient(app)
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


def _error(response):
    return response.json()["detail"]


def _insert(key, requested_by, **fields):
    detail = civitai_detail()
    version = detail.versions[0]
    values = dict(
        sha256=key,
        source="civitai",
        source_model_id="122359",
        source_version_id="135867",
        file_name="add-detail-xl.safetensors",
        requested_by=requested_by,
        assign_to=f"user:{requested_by}",
        info=downloads.pending_info(detail, version, version.files[0], "lora"),
    )
    values.update(fields)
    return pending.insert(PendingRow(**values))


def test_post_downloads_queues_the_file_and_answers_the_row_and_the_download(client_as, core, civitai):
    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert (data["already_installed"], data["already_queued"]) == (False, False)
    assert data["pending"]["sha256"] == SHA and data["pending"]["status"] == "queued"
    assert data["pending"]["download_id"] == "dl-1"
    assert data["pending"]["info"]["file_id"] == "1001"
    assert set(data["download"]) == set(downloads.PUBLIC_DOWNLOAD_FIELDS)
    assert core.queued[0]["created_by"] == "user-1"


def test_post_downloads_by_an_admin_answers_the_whole_download(client_as, core, civitai):
    data = client_as(admin()).post(f"{BASE}/downloads", json=body()).json()["data"]

    assert data["download"]["destination_path"] == "/app/models/lora/add-detail-xl.safetensors"
    assert data["pending"]["assign_to"] == "none"


def test_post_downloads_answers_an_installed_file_without_queuing_it(client_as, core, civitai):
    core.models_by_sha[SHA] = model("model-9")

    data = client_as(principal()).post(f"{BASE}/downloads", json=body()).json()["data"]

    assert (data["already_installed"], data["download"], data["pending"]["status"]) == (True, None, "indexed")
    assert core.queued == []


@pytest.mark.parametrize(
    "payload, where",
    [
        ({"source": "civitai", "model_id": "122359", "version_id": "135867"}, "file_id"),
        (body(subdir="../loras"), "subdir"),
        (body(filename="a/b.safetensors"), "filename"),
        (body(model_type="lycoris"), "model_type"),
        (["not", "an", "object"], "Invalid download"),
    ],
)
def test_post_downloads_refuses_an_invalid_body(client_as, core, civitai, payload, where):
    response = client_as(principal()).post(f"{BASE}/downloads", json=payload)

    assert response.status_code == 400
    error = _error(response)
    assert (error["success"], error["error"]) == (False, "invalid_request")
    assert where in error["message"]
    assert core.queued == []


@pytest.mark.parametrize(
    "who, payload, status, code",
    [
        (principal(sources=frozenset({"huggingface"})), body(), 403, "forbidden"),
        (principal(model_types=frozenset({"vae"})), body(), 403, "forbidden"),
        (principal(max_file_gb=0.01), body(), 403, "forbidden"),
        (principal(), body(destination_backend_id="worker-1"), 403, "forbidden"),
        (principal(), body(assign_group_id="artists"), 403, "forbidden"),
        (principal(), body(version_id="1"), 404, "not_found"),
        (admin(), body(source="civarchive"), 404, "unknown_source"),
    ],
    ids=["source", "type", "size", "remote", "group", "version", "unknown-source"],
)
def test_post_downloads_answers_each_refusal_with_its_status(client_as, core, civitai, who, payload, status, code):
    response = client_as(who).post(f"{BASE}/downloads", json=payload)

    assert response.status_code == status
    assert _error(response)["error"] == code
    assert _error(response)["message"]
    assert core.queued == [] and pending.list_rows() == []


def test_post_downloads_of_an_nsfw_model_is_a_404_for_a_blocked_user(client_as, core, civitai, viewer):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewer.ctx = context("blocked", "hide")

    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert (response.status_code, _error(response)["error"]) == (404, "not_found")
    assert core.queued == []


def test_post_downloads_over_a_plan_quota_answers_limit_exceeded_with_its_figures(client_as, core, civitai):
    from .download_fakes import limited

    core.limits[("user-1", "model-browser.downloads")] = limited("model-browser.downloads", 2, 2)

    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert response.status_code == 403
    assert _error(response) == {
        "success": False,
        "error": "limit_exceeded",
        "message": "You have reached your model downloads per month limit: 2 of 2 used. It resets on Nov 1 at 00:00 UTC. "
        "Ask your admin for more.",
        "kind": "model-browser.downloads",
        "label": "Model downloads per month",
        "used": 2,
        "limit": 2,
        "incoming": 1,
        "resets_at": "2026-11-01T00:00:00+00:00",
    }
    assert core.queued == [] and pending.list_rows() == []


@pytest.mark.parametrize("code, status", [("blocked", 409), ("invalid", 400)])
def test_post_downloads_passes_the_cores_refusal_on(client_as, core, civitai, code, status):
    core.queue_error = CoreBridgeError(code, "Blocked by the quota plugin")

    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert response.status_code == status
    assert _error(response) == {"success": False, "error": code, "message": "Blocked by the quota plugin"}
    assert pending.list_rows() == []


@pytest.mark.parametrize(
    "error, status, code",
    [
        (SourceNotFound("Model not found"), 404, "not_found"),
        (SourceRateLimited("CivitAI is rate limiting", retry_after=7), 429, "rate_limited"),
        (SourceUnavailable("CivitAI answered 503"), 502, "source_unavailable"),
    ],
)
def test_post_downloads_answers_source_errors_like_the_browsing_routes(client_as, core, civitai, error, status, code):
    civitai.error = error

    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert (response.status_code, _error(response)["error"]) == (status, code)
    if status == 429:
        assert response.headers["Retry-After"] == "7"


def test_post_downloads_answers_an_unexpected_failure_with_the_envelope(client_as, core, civitai, caplog):
    core.queue_error = RuntimeError("PotionUI's AppContainer no longer provides download_queue.queue_model_download")

    response = client_as(principal()).post(f"{BASE}/downloads", json=body())

    assert response.status_code == 500
    assert _error(response)["error"] == "internal_error"
    assert "AppContainer" not in _error(response)["message"]
    assert "Could not start the download" in caplog.text
    assert pending.list_rows() == []


@pytest.fixture
def real_checks(app, monkeypatch, core, civitai):
    state = SimpleNamespace(capabilities=frozenset())
    app.dependency_overrides[get_current_active_user] = lambda: user("user-1")
    monkeypatch.setattr(permissions, "resolve_grant", lambda who: replace(USER_GRANT, capabilities=state.capabilities))
    with TestClient(app) as client:
        yield client, state


def test_the_routes_check_their_capabilities(real_checks):
    client, state = real_checks

    state.capabilities = frozenset({"browse", "request"})
    response = client.post(f"{BASE}/downloads", json=body())
    assert (response.status_code, _error(response)["message"]) == (403, "This action needs the 'download' permission")
    assert client.get(f"{BASE}/downloads").status_code == 200
    assert client.post(f"{BASE}/installed", json={"sha256s": []}).status_code == 200

    state.capabilities = frozenset({"browse"})
    response = client.get(f"{BASE}/downloads")
    assert response.status_code == 403
    assert _error(response) == {
        "success": False,
        "error": "forbidden",
        "message": "This action needs one of these permissions: 'download', 'request'",
    }

    state.capabilities = frozenset({"download"})
    assert client.get(f"{BASE}/downloads").status_code == 200
    assert client.post(f"{BASE}/installed", json={"sha256s": []}).status_code == 403


def test_the_routes_need_a_signed_in_user(app, core):
    with TestClient(app) as anonymous:
        assert anonymous.post(f"{BASE}/downloads", json=body()).status_code in (401, 403)
        assert anonymous.get(f"{BASE}/downloads").status_code in (401, 403)
        assert anonymous.post(f"{BASE}/installed", json={"sha256s": []}).status_code in (401, 403)


@pytest.fixture
def rows(plugin_db, core):
    _insert(SHA, "user-1", download_id="dl-1")
    _insert(OTHER_SHA, "user-2", download_id="dl-2", status="enriched", model_id="model-2")
    _insert("c" * 64, "admin-1", assign_to="none", status="indexed", model_id="model-3")
    with plugin_db.get_cursor() as cursor:
        for key, stamp in ((SHA, "10"), (OTHER_SHA, "11"), ("c" * 64, "12")):
            cursor.execute(
                f"UPDATE {pending.TABLE} SET created_at = ? WHERE sha256 = ?", (f"2026-10-11 {stamp}:00:00", key)
            )
    core.add_download("dl-1", "downloading", destination_path="/app/models/loras/x.safetensors", progress=0.5)
    core.add_download("dl-2", "completed")


def _keys(response):
    return [entry["pending"]["sha256"][:1] for entry in response.json()["data"]]


def test_a_user_lists_their_own_downloads_only(client_as, rows, viewer):
    client = client_as(principal())

    for query in ("", "?all=true", "?all=false"):
        response = client.get(f"{BASE}/downloads{query}")
        assert response.status_code == 200
        assert _keys(response) == ["a"]
    [entry] = client.get(f"{BASE}/downloads").json()["data"]
    assert entry["download"]["status"] == "downloading" and entry["download"]["progress"] == 0.5
    assert "destination_path" not in entry["download"]


def test_an_admin_lists_every_download_unless_asking_for_their_own(client_as, rows, viewer):
    client = client_as(admin())

    response = client.get(f"{BASE}/downloads")
    assert _keys(response) == ["c", "b", "a"]
    entries = response.json()["data"]
    assert entries[0]["download"] is None
    assert entries[2]["download"]["destination_path"] == "/app/models/loras/x.safetensors"
    assert _keys(client.get(f"{BASE}/downloads?all=false")) == ["c"]
    assert _keys(client.get(f"{BASE}/downloads?limit=1")) == ["c"]


def test_an_approve_holder_lists_every_download_with_the_public_fields(client_as, rows, viewer):
    client = client_as(principal("user-3", capabilities=frozenset({"browse", "request", "approve"})))

    entries = client.get(f"{BASE}/downloads").json()["data"]

    assert [entry["pending"]["requested_by"] for entry in entries] == ["admin-1", "user-2", "user-1"]
    assert "destination_path" not in entries[2]["download"]


def test_listed_rows_show_their_images_as_the_viewer_may_see_them(client_as, rows, viewer):
    viewer.ctx = context("blocked", "hide")

    entries = client_as(admin()).get(f"{BASE}/downloads").json()["data"]

    for entry in entries:
        images = entry["pending"]["info"]["model"]["versions"][0]["images"]
        assert [image["url"] for image in images] == [SAFE_IMAGE]
    assert pending.get(SHA).info["model"]["versions"][0]["images"][1]["url"] == NSFW_IMAGE


def test_a_download_that_cannot_be_read_is_listed_without_it(client_as, rows, viewer, monkeypatch):
    def get_download(download_id):
        raise RuntimeError("download_queue.get_download moved")

    monkeypatch.setattr(core_bridge, "get_download", get_download)

    entries = client_as(principal()).get(f"{BASE}/downloads").json()["data"]

    assert [entry["download"] for entry in entries] == [None]


def test_installed_for_an_admin_is_every_indexed_file_with_its_file_present(client_as, core):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha["c" * 64] = model("model-3", "c" * 64, is_available=False)

    response = client_as(admin()).post(f"{BASE}/installed", json={"sha256s": [SHA.upper(), OTHER_SHA, " " + SHA, "c" * 64]})

    assert response.status_code == 200
    assert response.json() == {"success": True, "data": {"installed": {SHA: True, OTHER_SHA: False, "c" * 64: False}}}
    assert "visible_model_ids" not in core.names()


def test_installed_for_a_user_counts_only_the_models_they_can_see(client_as, core):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha[OTHER_SHA] = model("model-2", OTHER_SHA)
    core.visible["user-1"] = ["model-1"]

    response = client_as(principal()).post(f"{BASE}/installed", json={"sha256s": [SHA, OTHER_SHA]})

    assert response.json()["data"]["installed"] == {SHA: True, OTHER_SHA: False}
    assert core.calls_to("visible_model_ids") == [("user-1",)]


def test_installed_for_a_user_does_not_count_a_visible_model_whose_file_has_gone(client_as, core):
    core.models_by_sha[SHA] = model("model-1", is_available=False)
    core.visible["user-1"] = ["model-1"]

    response = client_as(principal()).post(f"{BASE}/installed", json={"sha256s": [SHA]})

    assert response.json()["data"]["installed"] == {SHA: False}


def test_installed_follows_the_visible_models_of_whoever_asks(client_as, core):
    core.models_by_sha[SHA] = model("model-1")
    core.visible["user-1"] = ["model-1"]
    core.visible["user-2"] = []

    first = client_as(principal("user-1")).post(f"{BASE}/installed", json={"sha256s": [SHA]})
    second = client_as(principal("user-2")).post(f"{BASE}/installed", json={"sha256s": [SHA]})

    assert first.json()["data"]["installed"] == {SHA: True}
    assert second.json()["data"]["installed"] == {SHA: False}


def test_installed_is_answered_by_the_helper_the_search_badge_uses(client_as, core, monkeypatch):
    asked = []

    async def installed_map(sha256s, who):
        asked.append((list(sha256s), who))
        return {sha256: True for sha256 in sha256s}

    monkeypatch.setattr(routes_downloads, "installed_map", installed_map)
    who = principal()

    response = client_as(who).post(f"{BASE}/installed", json={"sha256s": [SHA.upper(), OTHER_SHA, " " + SHA]})

    assert response.json() == {"success": True, "data": {"installed": {SHA: True, OTHER_SHA: True}}}
    assert asked == [([SHA, OTHER_SHA], who)]
    assert core.calls == []


def test_installed_fails_rather_than_claim_that_nothing_is_installed_when_the_core_cannot_answer(
    app, client_as, core, monkeypatch
):
    client_as(principal())

    def moved(sha256):
        raise RuntimeError("PotionUI AppContainer no longer provides model_repository.get_by_sha256")

    monkeypatch.setattr(core_bridge, "find_model_by_sha256", moved)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"{BASE}/installed", json={"sha256s": [SHA]})

    assert response.status_code == 500
    assert "installed" not in response.text


def test_installed_with_no_hash_asks_nothing(client_as, core):
    response = client_as(principal()).post(f"{BASE}/installed", json={"sha256s": []})

    assert response.json()["data"] == {"installed": {}}
    assert core.calls == []


@pytest.mark.parametrize(
    "payload",
    [{"sha256s": ["abc"]}, {"sha256s": ["g" * 64]}, {"sha256s": ["a" * 64] * 201}, {}, {"sha256s": "a" * 64}],
    ids=["short", "not-hex", "too-many", "missing", "not-a-list"],
)
def test_installed_refuses_what_is_not_a_list_of_sha256(client_as, core, payload):
    response = client_as(principal()).post(f"{BASE}/installed", json=payload)

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"
    assert core.calls == []
