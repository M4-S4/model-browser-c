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

from .civarchive_fakes import (
    HTML_HEADERS,
    JSON_HEADERS,
    SHA_78291,
    CivArchiveTransport,
    data_body,
    full_page,
    model,
    props,
    results,
    search_body,
)
from .civitai_fakes import FakeTransport, context, fake_settings
from .download_fakes import FakeCore, principal
from .download_fakes import model as indexed_model
from .huggingface_fakes import HuggingFaceTransport, combined_fetch
from .openmodeldb_fakes import OpenModelDBTransport

BASE = "/api/plugins/model-browser-c"
CIVARCHIVE_GRANT = {"sources": frozenset({"civitai", "huggingface", "civarchive"})}
CIVITAI_URL = "https://civitai.com/api/download/models/83095"
LOWRA_SHA = "adf5b9e4c1f01243de32d5772216e1f6987f4097f4f23f53e2bed641d33a3ceb"
LOWRA_MIRROR = "https://huggingface.co/baycon/for_Colab/resolve/main/lowra_v10.safetensors"
NOTICE = (
    "Unofficial source: civarchive.com is a community index of CivitAI, not an official API. "
    "Deleted models may be unavailable, or come from third-party mirrors."
)


@pytest.fixture
def state():
    return SimpleNamespace(ctx=context("allowed", "show"), principal=principal(**CIVARCHIVE_GRANT))


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    return FakeCore().install(monkeypatch)


@pytest.fixture
def settings_values(monkeypatch):
    return fake_settings(monkeypatch, enable_civarchive=True)


@pytest.fixture
def archive(monkeypatch, state, core, settings_values):
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda user: state.ctx)
    monkeypatch.setattr(cache_module, "cache", Cache())
    monkeypatch.setattr(sources_module, "_instances", {})
    transport = CivArchiveTransport()
    client = SourceHttp(
        fetch=combined_fetch(FakeTransport().add("/enums", "enums"), HuggingFaceTransport(), OpenModelDBTransport(), transport),
        limits={"civitai": (0.0, 4), "huggingface": (0.0, 4), "openmodeldb": (0.0, 2), "civarchive": (0.0, 4)},
    )
    monkeypatch.setattr(http_module, "http", client)
    return transport


@pytest.fixture
def client(archive, state):
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


def _serve_search(archive, entries, total_hits=1000, **params):
    archive.clear("search.json").add("search.json", (200, JSON_HEADERS, search_body(entries, total_hits)), **params)


def test_sources_list_civarchive_last_with_its_filters_and_notice(client, archive):
    response = client.get(f"{BASE}/sources")

    assert response.status_code == 200
    listed = {source["id"]: source for source in response.json()["data"]}
    assert list(listed) == ["civitai", "huggingface", "openmodeldb", "civarchive"]
    civarchive = listed["civarchive"]
    assert civarchive["label"] == "CivArchive"
    assert (len(civarchive["types"]), len(civarchive["base_models"])) == (22, 104)
    assert {"value": "LORA", "label": "LORA"} in civarchive["types"]
    assert {"value": "SD 1.5", "label": "SD 1.5"} in civarchive["base_models"]
    assert [option["value"] for option in civarchive["sorts"]] == ["top", "newest", "downloads", "likes", "updated"]
    assert [option["value"] for option in civarchive["periods"]] == ["all_time", "year", "month", "week", "day"]
    assert civarchive["search_help"] == "Text, #tag, @user, or a civarchive.com / civitai.com model link"
    assert civarchive["notice"] == NOTICE
    assert [listed[source_id]["notice"] for source_id in ("civitai", "huggingface", "openmodeldb")] == [None, None, None]
    assert archive.data_calls("search.json") == [{"platform": "civitai"}]


def test_civarchive_is_left_out_while_its_setting_is_off(client, archive, settings_values):
    settings_values["enable_civarchive"] = False

    listed = [source["id"] for source in client.get(f"{BASE}/sources").json()["data"]]
    search = client.get(f"{BASE}/civarchive/search")
    detail = client.get(f"{BASE}/civarchive/models/78291")

    assert listed == ["civitai", "huggingface", "openmodeldb"]
    assert (search.status_code, _error(search)["error"]) == (404, "unknown_source")
    assert (detail.status_code, _error(detail)["error"]) == (404, "unknown_source")
    assert archive.calls == []


def test_sources_still_list_civarchive_when_it_is_down(client, archive):
    archive.home = (503, HTML_HEADERS, "<html>Service Unavailable</html>")

    civarchive = client.get(f"{BASE}/sources").json()["data"][-1]

    assert (civarchive["id"], civarchive["types"], civarchive["base_models"], civarchive["notice"]) == ("civarchive", [], [], NOTICE)


def test_search_through_the_route(client, archive):
    response = client.get(f"{BASE}/civarchive/search")

    assert response.status_code == 200
    page = response.json()["data"]
    assert (page["source"], _ids(response), page["next_cursor"]) == ("civarchive", ["4384", "257749", "6424", "4201", "7808"], None)
    easy = page["items"][-1]
    assert (easy["type"], easy["mb_type"], easy["nsfw"], easy["installed"], easy["primary_sha256"]) == (
        "TextualInversion", "embedding", False, None, None,
    )
    assert archive.data_calls("search.json")[0]["rating"] == "all"


def test_a_user_who_never_sees_nsfw_gets_neither_flagged_nor_deleted_models(client, archive, state):
    state.ctx = context("blocked", "hide")
    _serve_search(archive, results("search_safe") + results("search_deleted"), rating="safe")

    response = client.get(f"{BASE}/civarchive/search")

    assert response.status_code == 200
    assert _ids(response) == [raw["model_id"] for raw in results("search_safe")]
    assert archive.data_calls("search.json")[0]["rating"] == "safe"


def test_a_user_in_blur_mode_gets_flagged_thumbnails_without_their_url(client, archive, state):
    state.ctx = context("allowed", "blur")

    items = {found["id"]: found for found in client.get(f"{BASE}/civarchive/search").json()["data"]["items"]}

    assert (items["4384"]["nsfw"], items["4384"]["thumbnail"]["url"], items["4384"]["thumbnail"]["nsfw"]) == (True, None, True)
    assert items["4384"]["thumbnail"]["id"] == "1777041"
    assert items["7808"]["thumbnail"]["url"] == results("search_civitai_top")[4]["image_url"]


def test_search_pages_by_page_number(client, archive):
    first = full_page(results("search_civitai_top"), first_model=1_000)
    second = full_page(results("search_safe"), first_model=2_000)
    _serve_search(archive, first, page="1")
    archive.add("search.json", (200, JSON_HEADERS, search_body(second)), page="2")

    one = client.get(f"{BASE}/civarchive/search", params={"limit": 24})
    two = client.get(f"{BASE}/civarchive/search", params={"cursor": one.json()["data"]["next_cursor"]})

    assert (len(_ids(one)), one.json()["data"]["next_cursor"]) == (50, "2")
    assert (_ids(two)[0], two.json()["data"]["next_cursor"]) == ("2000", "3")
    assert [call["page"] for call in archive.data_calls("search.json")] == ["1", "2"]


def test_the_search_box_reaches_civarchive(client, archive):
    tag = client.get(f"{BASE}/civarchive/search", params={"q": "#anime"})
    user = client.get(f"{BASE}/civarchive/search", params={"q": "@TangBohu"})
    filtered = client.get(f"{BASE}/civarchive/search", params={"types": "LORA", "base_models": ["SD 1.5"], "sort": "newest"})

    assert _ids(tag) == [raw["model_id"] for raw in results("search_tag")]
    assert _ids(user) == [raw["model_id"] for raw in results("search_user")]
    assert _ids(filtered) == [raw["model_id"] for raw in results("search_civitai_lora_sd15_newest")]


def test_a_pasted_link_is_one_model_with_its_installed_badge(client, archive, core, state):
    core.models_by_sha[SHA_78291] = indexed_model("model-9", SHA_78291)
    core.visible[state.principal.user.id] = ["model-9"]

    response = client.get(f"{BASE}/civarchive/search", params={"q": "https://civarchive.com/models/78291?modelVersionId=83095"})

    (item,) = response.json()["data"]["items"]
    assert (item["id"], item["primary_sha256"], item["installed"]) == ("78291:83095", SHA_78291, True)
    assert response.json()["data"]["next_cursor"] is None
    assert client.get(f"{BASE}/civarchive/models/{item['id']}").json()["data"]["versions"][0]["id"] == "83095"


@pytest.mark.parametrize("params", [{"sort": "most_downloaded"}, {"period": "Week"}, {"cursor": "abc"}])
def test_bad_search_parameters_are_a_400(client, params):
    response = client.get(f"{BASE}/civarchive/search", params=params)

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"


def test_search_when_civarchive_is_down_is_a_502(client, archive):
    archive.home = OSError("network is unreachable")

    response = client.get(f"{BASE}/civarchive/search")

    assert response.status_code == 502
    assert _error(response)["error"] == "source_unavailable"


def test_detail_through_the_route(client, archive):
    latest = client.get(f"{BASE}/civarchive/models/78291")
    named = client.get(f"{BASE}/civarchive/models/78291:83095")

    assert latest.status_code == named.status_code == 200
    data = latest.json()["data"]
    assert data == named.json()["data"]
    assert (data["source"], data["id"], data["type"], data["mb_type"], data["license"]) == ("civarchive", "78291", "LORA", "lora", None)
    (version,) = data["versions"]
    assert version["partial"] is False
    assert [(file["id"], file["download_url"], file["auth_provider_id"]) for file in version["files"]] == [
        ("56837", CIVITAI_URL, "civitai"),
    ]
    assert archive.home_fetches == 1


def test_the_other_versions_are_partial_until_the_page_loads_them(client, archive):
    page = props("model_122359_v135867")
    older = page["model"]["version"]
    older.update(id="133241", name="v0.3", created_at=page["model"]["versions"][1]["created_at"])
    archive.add("models/122359.json", (200, JSON_HEADERS, data_body(page)), platform="civitai", modelVersionId=133241)

    shown = client.get(f"{BASE}/civarchive/models/122359:135867").json()["data"]
    loaded = client.get(f"{BASE}/civarchive/models/122359:133241").json()["data"]

    assert [(version["id"], version["partial"]) for version in shown["versions"]] == [("135867", False), ("133241", True)]
    assert shown["versions"][0]["files"] and shown["versions"][0]["images"]
    assert (shown["versions"][1]["files"], shown["versions"][1]["images"]) == ([], [])
    assert [(version["id"], version["partial"]) for version in loaded["versions"]] == [("133241", False), ("135867", True)]
    assert loaded["versions"][0]["files"] and loaded["versions"][0]["images"]
    assert [call["modelVersionId"] for call in archive.data_calls("models/122359.json")] == ["135867", "133241"]


@pytest.mark.parametrize("model_id", ["999999999", "78291:999999999", "78291:1", "abc", "42:43"])
def test_a_model_civarchive_does_not_have_is_404(client, model_id):
    response = client.get(f"{BASE}/civarchive/models/{model_id}")

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"


def test_civarchive_failing_on_a_detail_is_a_502(client, archive):
    archive.add("models/5.json", (500, HTML_HEADERS, "<title>Error 500: Internal Server Error</title>"))

    response = client.get(f"{BASE}/civarchive/models/5:6")

    assert response.status_code == 502
    assert _error(response) == {"success": False, "error": "source_unavailable", "message": "CivArchive answered HTTP 500"}


def test_an_nsfw_model_is_404_for_a_blocked_user_and_flagged_for_the_others(client, state):
    state.ctx = context("blocked", "hide")
    hidden = client.get(f"{BASE}/civarchive/models/122359:135867")
    state.ctx = context("allowed", "blur")
    shown = client.get(f"{BASE}/civarchive/models/122359:135867")

    assert hidden.status_code == 404
    assert (shown.status_code, shown.json()["data"]["nsfw"]) == (200, True)
    assert all(image["url"] for image in shown.json()["data"]["versions"][0]["images"])


def test_an_nsfw_image_is_blurred_then_revealed(client, state):
    state.ctx = context("allowed", "blur")

    detail = client.get(f"{BASE}/civarchive/models/298188:334947").json()["data"]
    revealed = client.get(f"{BASE}/civarchive/models/298188:334947/nsfw-media").json()["data"]["images"]

    blurred = [image for image in detail["versions"][0]["images"] if image["url"] is None]
    assert [(image["id"], image["nsfw_level"], bool(image["blurhash"])) for image in blurred] == [("6448716", 4, True)]
    assert [(image["id"], image["url"]) for image in revealed] == [
        ("6448716", next(raw["url"] for raw in model("model_298188_v334947")["version"]["images"] if raw["id"] == 6448716)),
    ]
    state.ctx = context("blocked", "hide")
    assert client.get(f"{BASE}/civarchive/models/298188:334947/nsfw-media").status_code == 404


def _body(model_id, version_id, file_id, **extra):
    return {"source": "civarchive", "model_id": model_id, "version_id": version_id, "file_id": file_id, **extra}


def test_a_civarchive_file_on_civitai_is_queued_with_the_civitai_provider(client, core):
    response = client.post(f"{BASE}/downloads", json=_body("78291", "83095", "56837"))

    assert response.status_code == 200, response.text
    (queued,) = core.queued
    assert (queued["url"], queued["provider_id"], queued["checksum_sha256"]) == (CIVITAI_URL, "civitai", SHA_78291)
    assert (queued["filename"], queued["model_type"], queued["subdir"]) == ("tangbohu_saturation_v1.0.safetensors", "lora", None)
    assert queued["tags"] == ["model-browser", "model-browser:civarchive"]
    assert is_allowed_download_url("civarchive", queued["url"])
    row = pending.get(SHA_78291)
    assert (row.source, row.source_model_id, row.source_version_id, row.status) == ("civarchive", "78291", "83095", "queued")
    assert row.info["model"]["versions"][0]["files"][0]["auth_provider_id"] == "civitai"
    assert row.info["model"]["source_url"] == "https://civarchive.com/models/78291?modelVersionId=83095"


def test_a_deleted_model_is_queued_from_its_hugging_face_mirror(client, core):
    response = client.post(f"{BASE}/downloads", json=_body("48139", "52753", "38399"))

    assert response.status_code == 200, response.text
    (queued,) = core.queued
    assert (queued["url"], queued["provider_id"], queued["checksum_sha256"]) == (LOWRA_MIRROR, "huggingface", LOWRA_SHA)
    assert (queued["filename"], queued["model_type"]) == ("LowRa.safetensors", "lora")
    assert is_allowed_download_url("civarchive", queued["url"])


def test_a_file_only_other_sites_have_is_refused(client, core):
    response = client.post(f"{BASE}/downloads", json=_body("298188", "334947", "266806"))

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"
    assert core.queued == [] and pending.list_rows() == []


def test_a_file_of_an_older_version_is_checked_against_that_version(client, archive, core):
    response = client.post(f"{BASE}/downloads", json=_body("195519", "225222", "173660"))

    assert response.status_code == 200, response.text
    assert core.queued[0]["url"] == "https://civitai.com/api/download/models/424706"
    assert [call.get("modelVersionId") for call in archive.data_calls("models/195519.json")] == ["225222"]


def test_a_grant_without_civarchive_refuses_its_files(client, core, state):
    state.principal = principal()

    response = client.post(f"{BASE}/downloads", json=_body("78291", "83095", "56837"))

    assert response.status_code == 403
    assert "civarchive" in _error(response)["message"]
    assert core.queued == []


def test_downloads_from_civarchive_need_its_setting(client, core, settings_values):
    settings_values["enable_civarchive"] = False

    response = client.post(f"{BASE}/downloads", json=_body("78291", "83095", "56837"))

    assert (response.status_code, _error(response)["error"]) == (404, "unknown_source")
    assert core.queued == []


def test_a_model_id_naming_its_version_is_recorded_as_the_model(client, core):
    response = client.post(f"{BASE}/downloads", json=_body("78291:83095", "83095", "56837"))

    assert response.status_code == 200, response.text
    row = pending.get(SHA_78291)
    assert (row.source_model_id, row.source_version_id) == ("78291", "83095")
