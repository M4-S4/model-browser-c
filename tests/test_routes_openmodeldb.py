from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import cache as cache_module
from backend import http as http_module
from backend import nsfw, pending, routes_downloads
from backend import sources as sources_module
from backend.api import router
from backend.cache import Cache
from backend.http import SourceHttp
from backend.permissions import require_browse, require_download
from backend.security import is_allowed_download_url
from backend.tables import create_tables

from .civitai_fakes import FakeTransport, context, fake_settings
from .download_fakes import FakeCore, model, principal
from .huggingface_fakes import HuggingFaceTransport, combined_fetch
from .openmodeldb_fakes import (
    GITHUB_RAW,
    GOOGLE_DRIVE_ONLY,
    MEGA_ONLY,
    MODELS,
    ONNX_FIRST,
    THREE_FORMATS,
    OpenModelDBTransport,
)
from .openmodeldb_fakes import model as raw_model

BASE = "/api/plugins/model-browser-c"
OPENMODELDB_GRANT = {"sources": frozenset({"civitai", "huggingface", "openmodeldb"})}
NOMOS_SHA = raw_model(GITHUB_RAW)["resources"][0]["sha256"]


@pytest.fixture
def state():
    return SimpleNamespace(ctx=context("allowed", "show"), principal=principal(**OPENMODELDB_GRANT))


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


@pytest.fixture
def omdb(monkeypatch, state, core):
    fake_settings(monkeypatch)
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda user: state.ctx)
    monkeypatch.setattr(cache_module, "cache", Cache())
    monkeypatch.setattr(sources_module, "_instances", {})
    transport = OpenModelDBTransport()
    client = SourceHttp(
        fetch=combined_fetch(FakeTransport(), HuggingFaceTransport(), transport), limits={"openmodeldb": (0.0, 2)}
    )
    monkeypatch.setattr(http_module, "http", client)
    return transport


@pytest.fixture
def client(omdb, state):
    app = FastAPI()
    app.include_router(router)
    for dependency in (require_browse, require_download, routes_downloads.require_downloads_access):
        app.dependency_overrides[dependency] = lambda: state.principal
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _ids(response):
    return [found["id"] for found in response.json()["data"]["items"]]


def _error(response):
    return response.json()["detail"]


def test_search_pages_through_the_catalogue(client, omdb):
    first = client.get(f"{BASE}/openmodeldb/search")
    second = client.get(f"{BASE}/openmodeldb/search", params={"cursor": first.json()["data"]["next_cursor"]})

    assert first.status_code == second.status_code == 200
    page = first.json()["data"]
    assert (page["source"], len(page["items"]), page["next_cursor"]) == ("openmodeldb", 24, "24")
    assert (len(_ids(second)), second.json()["data"]["next_cursor"]) == (3, None)
    assert omdb.count(MODELS) == 1
    kinds = [found["mb_type"] for found in page["items"] + second.json()["data"]["items"]]
    assert kinds == ["upscaler"] * 17 + [None] * 10
    item = next(found for found in page["items"] if found["id"] == GITHUB_RAW)
    assert (item["type"], item["mb_type"], item["base_model"], item["creator"]) == ("ESRGAN", "upscaler", "ESRGAN 4x", "helaman")
    assert item["source_url"] == "https://openmodeldb.info/models/4x-Nomos8kSC"
    assert (item["thumbnail"]["url"], item["thumbnail"]["width"]) == (
        "https://openmodeldb.info/thumbs/79385d8f832b948a9ce1d339.jpg", 368,
    )


def test_search_with_the_box_filters_and_sort(client):
    author = client.get(f"{BASE}/openmodeldb/search", params={"q": "@kim2091"})
    filtered = client.get(f"{BASE}/openmodeldb/search", params={"types": "esrgan", "base_models": ["4"], "limit": 50})
    oldest = client.get(f"{BASE}/openmodeldb/search", params={"sort": "oldest", "limit": 2})
    link = client.get(f"{BASE}/openmodeldb/search", params={"q": "https://openmodeldb.info/models/4x-Nomos8kSC"})

    assert _ids(author) == ["4x-UltraSharpV2", "2x-GameUp-TSCUNet", "1x-BroadcastToStudio-SPAN", "4x-UltraSharp"]
    assert _ids(filtered) == ["4x-Nomos8kSC", "4x-Remacri", "4x-AnalogFrames-1-0", "4x-UltraSharp"]
    assert _ids(oldest) == ["16x-ESRGAN", "1x-KDM003-scans"]
    assert _ids(link) == [GITHUB_RAW]


def test_nothing_is_hidden_from_a_user_who_never_sees_nsfw(client, state):
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/openmodeldb/search", params={"limit": 50})

    assert len(_ids(response)) == 27
    assert all(found["thumbnail"] is None or found["thumbnail"]["url"] for found in response.json()["data"]["items"])


def test_search_marks_an_installed_upscaler(client, core, state):
    core.models_by_sha[NOMOS_SHA] = model("model-9", NOMOS_SHA)
    core.visible[state.principal.user.id] = ["model-9"]

    items = {found["id"]: found for found in client.get(f"{BASE}/openmodeldb/search", params={"limit": 50}).json()["data"]["items"]}

    assert (items[GITHUB_RAW]["primary_sha256"], items[GITHUB_RAW]["installed"]) == (NOMOS_SHA, True)
    assert items["16x-ESRGAN"]["installed"] is False
    assert (items[MEGA_ONLY]["primary_sha256"], items[MEGA_ONLY]["installed"], items[MEGA_ONLY]["mb_type"]) == (None, None, None)


@pytest.mark.parametrize(
    "params", [{"types": "not-an-architecture"}, {"base_models": "four"}, {"sort": "most_downloaded"}, {"cursor": "abc"}]
)
def test_bad_search_parameters_are_a_400(client, params):
    response = client.get(f"{BASE}/openmodeldb/search", params=params)

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"


def test_search_when_openmodeldb_is_down_is_a_502(client, omdb):
    omdb.responses[MODELS] = (503, {}, b"Service Unavailable")

    response = client.get(f"{BASE}/openmodeldb/search")

    assert response.status_code == 502
    assert _error(response)["error"] == "source_unavailable"


def test_detail_of_a_downloadable_upscaler(client):
    response = client.get(f"{BASE}/openmodeldb/models/{ONNX_FIRST}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert (data["source"], data["id"], data["type"], data["mb_type"]) == ("openmodeldb", ONNX_FIRST, "HAT", "upscaler")
    assert data["license"]["summary"] == "CC-BY-4.0"
    (version,) = data["versions"]
    assert (version["id"], version["name"], version["base_model"], version["partial"]) == (
        ONNX_FIRST, "Release", "HAT 4x", False,
    )
    assert [(file["name"], file["primary"], file["auth_provider_id"]) for file in version["files"]] == [
        ("4xNomos8kSCHAT-S.pth", True, None),
    ]
    assert len(version["images"]) == 7 and all(image["nsfw"] is False for image in version["images"])
    assert version["images"][0]["url"] == "https://openmodeldb.info/thumbs/124f13989b3543fa9edfa35e.jpg"
    assert data["description_html"].startswith("<p>Architecture: HAT · Scale: 4x")
    assert "OpenModelDB page</a>: ONNX</p>" in data["description_html"]
    assert data["description_html"].endswith(
        "<p>Note: PotionUI's built-in upscaler currently loads ESRGAN (.pth) models only; "
        "this HAT model will download and index but may not run in PotionUI.</p>"
    )


def test_a_model_whose_gallery_links_are_dead_shows_its_own_thumbnail(client):
    detail = client.get(f"{BASE}/openmodeldb/models/1x-Archiver-AntiLines").json()["data"]
    found = client.get(f"{BASE}/openmodeldb/search", params={"q": "https://openmodeldb.info/models/1x-Archiver-AntiLines"})

    own = "https://openmodeldb.info/thumbs/c431d6ede679511dd5bdcc54.jpg"
    assert [image["url"] for image in detail["versions"][0]["images"]] == [own]
    assert found.json()["data"]["items"][0]["thumbnail"]["url"] == own


def test_detail_of_a_model_only_the_openmodeldb_page_offers(client):
    data = client.get(f"{BASE}/openmodeldb/models/{GOOGLE_DRIVE_ONLY}").json()["data"]

    assert (data["mb_type"], data["versions"][0]["files"]) == (None, [])
    assert "Other downloads on the" in data["description_html"] and "Google Drive" in data["description_html"]


def test_an_unknown_model_is_404(client):
    response = client.get(f"{BASE}/openmodeldb/models/4x-Does-Not-Exist")

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"


def test_nsfw_media_is_empty_and_404_under_a_blocked_ceiling(client, state):
    assert client.get(f"{BASE}/openmodeldb/models/{GITHUB_RAW}/nsfw-media").json()["data"] == {"images": []}

    state.ctx = context("blocked", "hide")
    assert client.get(f"{BASE}/openmodeldb/models/{GITHUB_RAW}/nsfw-media").status_code == 404


def _body(model_id, file_id, **extra):
    return {"source": "openmodeldb", "model_id": model_id, "version_id": model_id, "file_id": file_id, **extra}


def test_downloading_an_openmodeldb_file_queues_its_direct_url_as_an_upscaler(client, core):
    response = client.post(f"{BASE}/downloads", json=_body(GITHUB_RAW, NOMOS_SHA))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    (queued,) = core.queued
    assert queued["url"] == "https://github.com/Phhofm/models/raw/main/4xNomos8kSC/4xNomos8kSC.pth"
    assert is_allowed_download_url("openmodeldb", queued["url"])
    assert (queued["filename"], queued["model_type"], queued["subdir"]) == ("4xNomos8kSC.pth", "upscaler", None)
    assert (queued["provider_id"], queued["checksum_sha256"]) == (None, NOMOS_SHA)
    assert queued["tags"] == ["model-browser", "model-browser:openmodeldb"]
    row = pending.get(NOMOS_SHA)
    assert (row.source, row.source_model_id, row.source_version_id, row.status) == ("openmodeldb", GITHUB_RAW, GITHUB_RAW, "queued")
    assert (row.info["mb_type"], row.info["file_id"]) == ("upscaler", NOMOS_SHA)
    assert row.info["model"]["versions"][0]["files"][0]["auth_provider_id"] is None
    assert data["pending"]["sha256"] == NOMOS_SHA


def test_the_second_file_of_a_model_can_be_chosen(client, core):
    safetensors = raw_model(THREE_FORMATS)["resources"][1]

    response = client.post(f"{BASE}/downloads", json=_body(THREE_FORMATS, safetensors["sha256"]))

    assert response.status_code == 200, response.text
    assert core.queued[0]["url"] == safetensors["urls"][0]
    assert (core.queued[0]["filename"], core.queued[0]["model_type"]) == ("1xDeNoise_realplksr_otf.safetensors", "upscaler")


def test_an_onnx_file_is_never_downloaded(client, core):
    onnx = raw_model(ONNX_FIRST)["resources"][0]

    response = client.post(f"{BASE}/downloads", json=_body(ONNX_FIRST, onnx["sha256"]))

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"
    assert core.queued == []


def test_a_model_only_the_openmodeldb_page_offers_has_no_file_to_download(client, core):
    sha = raw_model(GOOGLE_DRIVE_ONLY)["resources"][0]["sha256"]

    response = client.post(f"{BASE}/downloads", json=_body(GOOGLE_DRIVE_ONLY, sha))

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"
    assert core.queued == [] and pending.list_rows() == []


def test_a_grant_without_openmodeldb_refuses_its_files(client, core, state):
    state.principal = principal()

    response = client.post(f"{BASE}/downloads", json=_body(GITHUB_RAW, NOMOS_SHA))

    assert response.status_code == 403
    assert "openmodeldb" in _error(response)["message"]
    assert core.queued == []
