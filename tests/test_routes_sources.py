from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import cache as cache_module
from backend import core_bridge
from backend import http as http_module
from backend import nsfw, permissions
from backend import sources as sources_module
from backend.api import router
from backend.cache import Cache
from backend.http import SourceHttp
from backend.permissions import require_browse
from backend.routes_sources import MAX_UPSTREAM_PAGES
from backend.tables import create_tables
from src.plugin_api import get_current_active_user

from .civitai_fakes import ABSENT, FakeTransport, body, context, fake_settings, image_urls, item, page_body
from .download_fakes import USER_GRANT, FakeCore, admin, model, principal, user
from .huggingface_fakes import REPOS, HuggingFaceTransport, combined_fetch, cursor_of
from .huggingface_fakes import body as hf_body
from .openmodeldb_fakes import OpenModelDBTransport

BASE = "/api/plugins/model-browser-c"
PAGE_1_CURSOR = "409619|21294|82098"
PAGE_2_CURSOR = "287938|23768|12597"


@pytest.fixture
def state():
    return SimpleNamespace(ctx=context("allowed", "show"), principal=principal())


@pytest.fixture
def core(monkeypatch):
    return FakeCore().install(monkeypatch)


@pytest.fixture
def hf_transport():
    return HuggingFaceTransport()


@pytest.fixture
def omdb_transport():
    return OpenModelDBTransport()


@pytest.fixture
def transport(plugin_db, monkeypatch, state, hf_transport, omdb_transport, core):
    create_tables()
    fake_settings(monkeypatch)
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda user: state.ctx)
    monkeypatch.setattr(cache_module, "cache", Cache())
    monkeypatch.setattr(sources_module, "_instances", {})
    fake = FakeTransport().add("/enums", "enums")
    client = SourceHttp(
        fetch=combined_fetch(fake, hf_transport, omdb_transport),
        limits={"civitai": (0.0, 4), "huggingface": (0.0, 4), "openmodeldb": (0.0, 2)},
    )
    monkeypatch.setattr(http_module, "http", client)
    return fake


@pytest.fixture
def client(transport, state):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_browse] = lambda: state.principal
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _lora_pages(transport):
    transport.add("/models", "models_lora_page1", cursor=ABSENT)
    transport.add("/models", "models_lora_page2", cursor=PAGE_1_CURSOR)
    transport.add("/models", (200, {}, page_body([])), cursor=PAGE_2_CURSOR)


def _ids(response):
    return [found["id"] for found in response.json()["data"]["items"]]


def _error(response):
    return response.json()["detail"]


def test_sources_lists_civitai_hugging_face_and_openmodeldb_with_their_filters(client, hf_transport, omdb_transport):
    response = client.get(f"{BASE}/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    civitai, huggingface, openmodeldb = payload["data"]
    assert civitai["id"] == "civitai"
    assert [option["value"] for option in civitai["types"]] == body("enums")["ModelType"]
    assert [option["value"] for option in civitai["base_models"]] == body("enums")["ActiveBaseModel"]
    assert (huggingface["id"], huggingface["label"], huggingface["enabled"]) == ("huggingface", "Hugging Face", True)
    assert [option["value"] for option in huggingface["types"]][:2] == ["text-to-image", "image-to-image"]
    assert (huggingface["base_models"], huggingface["periods"]) == ([], [])
    assert [option["value"] for option in huggingface["sorts"]] == ["trending", "downloads", "likes", "updated", "newest"]
    assert hf_transport.calls == []
    for source in (civitai, huggingface):
        assert (source["types_label"], source["base_models_label"], source["base_model_label"]) == (
            "Types", "Base model", "Base model",
        )
    assert (openmodeldb["id"], openmodeldb["label"], openmodeldb["enabled"]) == ("openmodeldb", "OpenModelDB", True)
    assert (openmodeldb["types_label"], openmodeldb["base_models_label"], openmodeldb["base_model_label"]) == (
        "Architecture", "Scale", "Architecture",
    )
    assert openmodeldb["types"][:4] == [
        {"value": "cain", "label": "CAIN"},
        {"value": "compact", "label": "Compact"},
        {"value": "dat", "label": "DAT"},
        {"value": "esrgan", "label": "ESRGAN"},
    ]
    assert [option["label"] for option in openmodeldb["base_models"]] == ["1x", "2x", "3x", "4x", "8x", "16x"]
    assert openmodeldb["sorts"][0] == {"value": "downloadable", "label": "Downloadable first"}
    assert [option["value"] for option in openmodeldb["sorts"]] == ["downloadable", "newest", "oldest", "name"]
    assert openmodeldb["periods"] == []
    assert "@author" in openmodeldb["search_help"]
    assert sorted(omdb_transport.calls) == ["/api/v1/architectures.json", "/api/v1/models.json", "/api/v1/tags.json"]
    assert client.get(f"{BASE}/sources").json()["data"][2] == openmodeldb
    assert len(omdb_transport.calls) == 3


def test_sources_still_lists_openmodeldb_when_it_is_down(client, omdb_transport):
    omdb_transport.responses["/api/v1/models.json"] = (503, {}, b"")

    data = client.get(f"{BASE}/sources").json()["data"]

    assert [source["id"] for source in data] == ["civitai", "huggingface", "openmodeldb"]
    assert (data[2]["types"], data[2]["base_models"], data[2]["types_label"]) == ([], [], "Architecture")


def test_the_routes_need_a_signed_in_user(transport):
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as anonymous:
        for path in ("/sources", "/civitai/search", "/civitai/models/122359", "/civitai/models/122359/nsfw-media"):
            assert anonymous.get(f"{BASE}{path}").status_code in (401, 403), path
    assert transport.calls == []


def test_search_for_a_blocked_user_asks_for_safe_content_and_fills_the_page(client, transport, state):
    _lora_pages(transport)
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/civitai/search", params={"limit": 2})

    assert response.status_code == 200
    data = response.json()["data"]
    assert _ids(response) == ["58390", "82098"]
    assert all(found["nsfw"] is False for found in data["items"])
    assert data["next_cursor"] == PAGE_2_CURSOR
    calls = transport.calls_to("/models")
    assert [call["nsfw"] for call in calls] == ["false", "false"]
    assert [call.get("cursor") for call in calls] == [None, PAGE_1_CURSOR]
    assert all("civitai.red" not in found["source_url"] for found in data["items"])


def test_search_for_a_user_who_shows_nsfw(client, transport, state):
    _lora_pages(transport)
    state.ctx = context("allowed", "show")

    response = client.get(f"{BASE}/civitai/search", params={"limit": 2})

    assert _ids(response) == ["264290", "58390"]
    assert response.json()["data"]["next_cursor"] == PAGE_1_CURSOR
    assert [call["nsfw"] for call in transport.calls_to("/models")] == ["true"]
    items = response.json()["data"]["items"]
    assert items[0]["nsfw"] is True and items[0]["source_url"] == "https://civitai.red/models/264290"


@pytest.mark.parametrize(
    "ctx, nsfw_param",
    [
        (context("blocked", "hide"), "false"),
        (context("allowed", "hide"), "false"),
        (context("blur", "hide"), "false"),
        (context("blur", "blur"), "true"),
        (context("allowed", "blur"), "true"),
        (context("allowed", "show"), "true"),
    ],
)
def test_the_safe_hint_follows_the_ceiling_and_the_mode(client, transport, state, ctx, nsfw_param):
    transport.add("/models", "models_query")
    state.ctx = ctx

    client.get(f"{BASE}/civitai/search", params={"q": "detail tweaker", "limit": 1})

    assert transport.calls_to("/models")[0]["nsfw"] == nsfw_param


def test_search_in_blur_mode_sends_nsfw_thumbnails_without_their_url(client, transport, state):
    raw = body("models_query")
    nsfw_image = raw["items"][0]["modelVersions"][0]["images"][0]
    for version in raw["items"][0]["modelVersions"]:
        for image in version["images"]:
            image["nsfwLevel"] = 8
    transport.add("/models", (200, {}, raw))
    state.ctx = context("blur", "blur")

    response = client.get(f"{BASE}/civitai/search", params={"q": "detail tweaker", "limit": 2})

    first = response.json()["data"]["items"][0]
    assert first["nsfw"] is True
    assert first["thumbnail"]["url"] is None
    assert first["thumbnail"]["blurhash"] == nsfw_image["hash"]
    assert nsfw_image["url"] not in image_urls(response.json())


def test_search_passes_the_filters_through(client, transport):
    transport.add("/models", "models_flux_nsfwlevel")

    response = client.get(
        f"{BASE}/civitai/search",
        params=[
            ("q", "#flux"),
            ("types", "LORA"),
            ("types", "DoRA,LoCon"),
            ("base_models", "Flux.1 D"),
            ("sort", "most_downloaded"),
            ("period", "month"),
            ("limit", "10"),
        ],
    )

    assert response.status_code == 200
    assert transport.calls_to("/models")[0] == {
        "baseModels": "Flux.1 D",
        "limit": "30",
        "nsfw": "true",
        "period": "Month",
        "sort": "Most Downloaded",
        "tag": "flux",
        "types": ["DoRA", "LORA", "LoCon"],
    }


def test_an_ids_search_through_the_route(client, transport):
    transport.add("/models", "models_ids")
    transport.add("/models/58390", (200, {}, item("models_lora_page1", 58390)))

    response = client.get(f"{BASE}/civitai/search", params={"q": "$#122359,58390"})

    assert _ids(response) == ["122359", "58390"]
    assert response.json()["data"]["next_cursor"] is None


def test_page_filling_stops_after_ten_upstream_pages(client, transport, state):
    nsfw_only = item("models_flux_nsfwlevel", 667086)
    counter = {"n": 0}

    def endless(path, params):
        counter["n"] += 1
        return 200, {}, page_body([dict(nsfw_only, id=1000 + counter["n"])], f"cursor-{counter['n']}")

    transport.add("/models", endless)
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/civitai/search", params={"limit": 5})

    assert response.status_code == 200
    assert _ids(response) == []
    assert counter["n"] == MAX_UPSTREAM_PAGES
    assert response.json()["data"]["next_cursor"] == f"cursor-{MAX_UPSTREAM_PAGES}"


def test_a_cursor_that_does_not_move_ends_the_search(client, transport, state):
    transport.add("/models", (200, {}, page_body([], "same")), cursor="same")
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/civitai/search", params={"cursor": "same"})

    assert response.json()["data"] == {"source": "civitai", "items": [], "next_cursor": None}
    assert len(transport.calls_to("/models")) == 1


def test_a_failure_after_the_first_page_returns_what_was_found(client, transport, state):
    transport.add("/models", "models_lora_page1", cursor=ABSENT)
    transport.add("/models", (503, {}, {"error": "down"}), cursor=PAGE_1_CURSOR)
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/civitai/search", params={"limit": 2})

    assert response.status_code == 200
    assert _ids(response) == ["58390"]
    assert response.json()["data"]["next_cursor"] == PAGE_1_CURSOR


def test_two_users_share_one_cached_upstream_page(client, transport, state):
    transport.add("/models", "models_query")

    state.ctx = context("blur", "blur")
    blurred = client.get(f"{BASE}/civitai/search", params={"q": "detail tweaker", "limit": 2})
    state.ctx = context("allowed", "show")
    shown = client.get(f"{BASE}/civitai/search", params={"q": "detail tweaker", "limit": 2})

    assert len(transport.calls_to("/models")) == 1
    assert _ids(blurred) == _ids(shown) == ["122359", "58390"]


SHA_122359 = "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"
SHA_58390 = "47aaaf0d2945ca937151d61304946dd229b3f072140b85484bc93e38f2a6e2f7"


def _sha(name, model_id):
    return item(name, model_id)["modelVersions"][0]["files"][0]["hashes"]["SHA256"].lower()


def _marks(response):
    return {found["id"]: found["installed"] for found in response.json()["data"]["items"]}


def _query_search(client):
    return client.get(f"{BASE}/civitai/search", params={"q": "detail tweaker", "limit": 2})


def test_search_marks_each_item_installed_when_it_is_indexed_and_visible_to_the_user(client, transport, core):
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359)
    core.visible["user-1"] = ["model-122359"]

    response = _query_search(client)

    assert response.status_code == 200
    items = response.json()["data"]["items"]
    assert [(found["id"], found["primary_sha256"], found["installed"]) for found in items] == [
        ("122359", SHA_122359, True),
        ("58390", SHA_58390, False),
    ]


def test_an_indexed_model_the_user_cannot_see_is_not_installed_for_them(client, transport, core):
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359)
    core.visible["user-1"] = ["some-other-model"]

    assert _marks(_query_search(client)) == {"122359": False, "58390": False}


def test_an_admin_sees_every_indexed_model_as_installed(client, transport, core, state):
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359)
    state.principal = admin()

    assert _marks(_query_search(client)) == {"122359": True, "58390": False}
    assert "visible_model_ids" not in core.names()


def test_a_model_whose_file_has_gone_from_the_disk_is_not_installed(client, transport, core, state):
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359, is_available=False)
    state.principal = admin()

    assert _marks(_query_search(client)) == {"122359": False, "58390": False}


def test_the_badge_is_resolved_once_for_the_page_and_only_for_the_items_the_user_may_see(client, transport, core, state):
    _lora_pages(transport)
    state.ctx = context("blocked", "hide")
    sha_82098 = _sha("models_lora_page2", 82098)
    core.models_by_sha[sha_82098] = model("model-82098", sha_82098)
    core.visible["user-1"] = ["model-82098"]

    response = client.get(f"{BASE}/civitai/search", params={"limit": 2})

    assert _ids(response) == ["58390", "82098"]
    assert _marks(response) == {"58390": False, "82098": True}
    assert len(transport.calls_to("/models")) == 2
    assert core.names().count("visible_model_ids") == 1
    assert core.calls_to("find_model_by_sha256") == [(_sha("models_lora_page1", 58390),), (sha_82098,)]


def test_two_users_get_their_own_badges_from_one_cached_page(client, transport, core, state):
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359)
    core.visible["user-1"] = ["model-122359"]
    core.visible["user-2"] = []

    state.principal = principal("user-1")
    first = _query_search(client)
    state.principal = principal("user-2")
    second = _query_search(client)
    state.principal = admin()
    third = _query_search(client)

    assert len(transport.calls_to("/models")) == 1
    assert _marks(first) == {"122359": True, "58390": False}
    assert _marks(second) == {"122359": False, "58390": False}
    assert _marks(third) == {"122359": True, "58390": False}


def test_the_real_principal_reaches_the_badge_and_is_resolved_once_per_search(transport, core, monkeypatch):
    resolved = []

    def resolve_grant(who):
        resolved.append(who.id)
        return replace(USER_GRANT, capabilities=frozenset({"browse"}))

    monkeypatch.setattr(permissions, "resolve_grant", resolve_grant)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_active_user] = lambda: user("user-1")
    transport.add("/models", "models_query")
    core.models_by_sha[SHA_122359] = model("model-122359", SHA_122359)
    core.visible["user-1"] = ["model-122359"]

    with TestClient(app) as client:
        response = _query_search(client)

    assert response.status_code == 200
    assert resolved == ["user-1"]
    assert _marks(response) == {"122359": True, "58390": False}


def test_an_ids_search_carries_the_badge_too(client, transport, core, state):
    transport.add("/models", "models_ids")
    transport.add("/models/58390", (200, {}, item("models_lora_page1", 58390)))
    core.models_by_sha[SHA_58390] = model("model-58390", SHA_58390)
    state.principal = admin()

    response = client.get(f"{BASE}/civitai/search", params={"q": "$#122359,58390"})

    assert _marks(response) == {"122359": False, "58390": True}


def test_hugging_face_items_have_no_hash_so_no_badge_and_the_core_is_not_asked(client, hf_transport, core):
    hf_transport.add("/api/models", "models_search_lora")

    response = client.get(f"{BASE}/huggingface/search", params={"q": "lora", "limit": 3})

    items = response.json()["data"]["items"]
    assert len(items) == 3
    assert all(found["primary_sha256"] is None and found["installed"] is None for found in items)
    assert core.calls == []


@pytest.mark.parametrize("failing", ["find_model_by_sha256", "visible_model_ids"])
def test_a_core_failure_does_not_fail_the_search_and_leaves_the_badges_unset(
    client, transport, core, monkeypatch, failing
):
    transport.add("/models", "models_query")

    def moved(*args):
        raise RuntimeError("The AppContainer no longer provides model_repository.get_by_sha256")

    monkeypatch.setattr(core_bridge, failing, moved)

    response = _query_search(client)

    assert response.status_code == 200
    items = response.json()["data"]["items"]
    assert [found["id"] for found in items] == ["122359", "58390"]
    assert [found["installed"] for found in items] == [None, None]
    assert [found["primary_sha256"] for found in items] == [SHA_122359, SHA_58390]


def test_an_unknown_source_is_404(client, transport):
    for path in ("/nope/search", "/nope/models/1", "/nope/models/1/nsfw-media", "/civarchive/search"):
        response = client.get(f"{BASE}{path}")
        assert response.status_code == 404, path
        assert _error(response) == {
            "success": False,
            "error": "unknown_source",
            "message": f"Unknown source '{path.split('/')[1]}'",
        }
    assert transport.calls == []


def test_an_upstream_429_is_a_429_with_retry_after(client, transport):
    transport.add("/models", (429, {"Retry-After": "17"}, {"error": "Too Many Requests"}))

    response = client.get(f"{BASE}/civitai/search")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "17"
    assert _error(response)["error"] == "rate_limited"


def test_a_429_without_retry_after_gets_a_default(client, transport):
    transport.add("/models/122359", (429, {}, b""))

    response = client.get(f"{BASE}/civitai/models/122359")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_an_upstream_outage_is_a_502(client, transport):
    transport.add("/models", (503, {}, b"<html>maintenance</html>"))

    response = client.get(f"{BASE}/civitai/search")

    assert response.status_code == 502
    assert _error(response) == {"success": False, "error": "source_unavailable", "message": "CivitAI answered HTTP 503"}


def test_an_upstream_refusal_is_a_400_with_its_message(client, transport):
    transport.add("/models", "error_400_limit")

    response = client.get(f"{BASE}/civitai/search")

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"
    assert "Too big: expected number to be <=100" in _error(response)["message"]


@pytest.mark.parametrize("params", [{"sort": "best"}, {"period": "forever"}, {"q": "$#none"}])
def test_bad_search_parameters_are_a_400(client, transport, params):
    response = client.get(f"{BASE}/civitai/search", params=params)

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"
    assert transport.calls_to("/models") == []


@pytest.mark.parametrize("limit", [0, 51])
def test_the_limit_is_bounded(client, limit):
    assert client.get(f"{BASE}/civitai/search", params={"limit": limit}).status_code == 422


def test_detail_for_a_user_who_may_see_it(client, transport, state):
    transport.add("/models/122359", "model_122359")
    state.ctx = context("allowed", "show", max_level=4)

    response = client.get(f"{BASE}/civitai/models/122359")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "122359"
    assert data["nsfw"] is False
    assert data["source_url"] == "https://civitai.com/models/122359"
    assert data["license"]["summary"].startswith("Credit required")
    assert [version["id"] for version in data["versions"]] == ["135867", "133241"]
    assert [version["partial"] for version in data["versions"]] == [False, False]


def test_detail_of_an_nsfw_model_is_404_for_a_blocked_user(client, transport, state):
    transport.add("/models/122359", "model_122359")
    state.ctx = context("blocked", "hide", max_level=2)

    response = client.get(f"{BASE}/civitai/models/122359")

    assert response.status_code == 404
    assert _error(response) == {"success": False, "error": "not_found", "message": "Model not found"}


def test_detail_in_blur_mode_keeps_the_model_without_nsfw_urls(client, transport, state):
    raw = body("model_122359")
    hidden = raw["modelVersions"][0]["images"][1]
    hidden["nsfwLevel"] = 16
    transport.add("/models/122359", (200, {}, raw))
    state.ctx = context("blur", "blur", max_level=2)

    response = client.get(f"{BASE}/civitai/models/122359")

    assert response.status_code == 200
    images = response.json()["data"]["versions"][0]["images"]
    assert [image["url"] is None for image in images] == [False, True, False]
    assert images[1]["blurhash"] == hidden["hash"]
    assert images[1]["id"] == "1917133"
    assert hidden["url"] not in image_urls(response.json())


def test_detail_of_an_unknown_model_is_404(client):
    response = client.get(f"{BASE}/civitai/models/999999")

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"


def test_detail_hides_a_real_person_model_when_the_admin_says_so(client, transport, state):
    raw = body("model_122359")
    raw["poi"] = True
    transport.add("/models/122359", (200, {}, raw))
    state.ctx = context("allowed", "show", max_level=4, hide_poi=True)

    assert client.get(f"{BASE}/civitai/models/122359").status_code == 404


def test_nsfw_media_is_404_for_a_blocked_user_without_asking_civitai(client, transport, state):
    transport.add("/models/122359", "model_122359")
    state.ctx = context("blocked", "hide", max_level=8)

    response = client.get(f"{BASE}/civitai/models/122359/nsfw-media")

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"
    assert transport.calls_to("/models/122359") == []


def test_nsfw_media_gives_the_urls_in_blur_mode(client, transport, state):
    raw = body("model_122359")
    hidden = [raw["modelVersions"][0]["images"][1], raw["modelVersions"][1]["images"][0]]
    for image in hidden:
        image["nsfwLevel"] = 8
    transport.add("/models/122359", (200, {}, raw))
    state.ctx = context("blur", "blur", max_level=2)

    response = client.get(f"{BASE}/civitai/models/122359/nsfw-media")

    assert response.status_code == 200
    images = response.json()["data"]["images"]
    assert [image["url"] for image in images] == [image["url"] for image in hidden]
    assert all(image["nsfw"] and image["blurhash"] for image in images)


def test_nsfw_media_is_404_when_the_model_itself_is_hidden(client, transport, state):
    transport.add("/models/122359", "model_122359")
    state.ctx = context("allowed", "hide", max_level=2)

    assert client.get(f"{BASE}/civitai/models/122359/nsfw-media").status_code == 404


LCM = REPOS["lcm_lora_sdxl"]
LORA_IDS = [
    "sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA",
    "sample-lab/MiniMax-H3-Character-Sample-LoRA",
    "pablodawson/MiniMax-H3-360-Orbit-LoRA",
]


def _nsfw_demo_repo(hf_transport):
    raw = hf_body("model_lcm_lora_sdxl")
    raw["id"] = raw["modelId"] = "org/demo"
    raw["tags"].append("not-for-all-audiences")
    tree = [
        {"type": "file", "path": "README.md", "size": 60},
        {"type": "file", "path": "images/preview.png", "size": 10},
        {"type": "file", "path": "demo.safetensors", "size": 99, "lfs": {"oid": "ab" * 32, "size": 99}},
    ]
    hf_transport.add("/api/models/org/demo", (200, {}, raw))
    hf_transport.add("/api/models/org/demo/tree/main", (200, {}, tree))
    hf_transport.add(
        "/org/demo/raw/main/README.md", (200, {"Content-Type": "text/plain"}, "# Demo\n\n![p](images/preview.png)\n")
    )
    return "https://huggingface.co/org/demo/resolve/main/images/preview.png"


def _hf_paths(hf_transport):
    return [path for path, _ in hf_transport.calls]


def test_the_hugging_face_routes_need_a_signed_in_user(transport, hf_transport):
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as anonymous:
        for path in ("/huggingface/search", f"/huggingface/models/{LCM}", f"/huggingface/models/{LCM}/nsfw-media"):
            assert anonymous.get(f"{BASE}{path}").status_code in (401, 403), path
    assert hf_transport.calls == []


def test_hugging_face_search_through_the_route(client, hf_transport):
    hf_transport.add("/api/models", "models_search_lora")

    response = client.get(f"{BASE}/huggingface/search", params={"q": "lora", "limit": 3})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["source"] == "huggingface"
    assert _ids(response) == LORA_IDS
    assert data["next_cursor"] == cursor_of("models_search_lora")
    assert data["items"][0]["mb_type"] == "lora" and data["items"][0]["thumbnail"] is None
    assert hf_transport.calls_to("/api/models") == [
        {"direction": "-1", "full": "true", "limit": "3", "search": "lora", "sort": "trendingScore"}
    ]


def test_hugging_face_cursors_go_back_upstream_whole(client, hf_transport):
    hf_transport.add("/api/models", "models_search_lora_page2", cursor=cursor_of("models_search_lora"))
    long_cursor = "x" * 2048

    page2 = client.get(
        f"{BASE}/huggingface/search", params={"q": "lora", "limit": 3, "cursor": cursor_of("models_search_lora")}
    )
    long = client.get(f"{BASE}/huggingface/search", params={"limit": 3, "cursor": long_cursor})
    too_long = client.get(f"{BASE}/huggingface/search", params={"cursor": long_cursor + "x"})

    assert page2.status_code == 200
    assert _ids(page2)[0] == "RunningHubAI/rh-qwen-image-2.1-lora-2104918997757157378"
    assert page2.json()["data"]["next_cursor"] == cursor_of("models_search_lora_page2")
    assert long.status_code == 404
    assert hf_transport.calls_to("/api/models")[-1]["cursor"] == long_cursor
    assert too_long.status_code == 422


def test_a_blocked_user_gets_the_page_refilled_from_the_link_cursor(client, hf_transport, state):
    nsfw_cursor = cursor_of("models_filter_nsfw_tag")
    hf_transport.add("/api/models", "models_filter_nsfw_tag", cursor=ABSENT)
    hf_transport.add("/api/models", "models_search_lora", cursor=nsfw_cursor)
    state.ctx = context("blocked", "hide")

    response = client.get(f"{BASE}/huggingface/search", params={"limit": 2})

    assert response.status_code == 200
    assert _ids(response) == LORA_IDS
    assert response.json()["data"]["next_cursor"] == cursor_of("models_search_lora")
    assert [call.get("cursor") for call in hf_transport.calls_to("/api/models")] == [None, nsfw_cursor]


def test_a_user_who_shows_nsfw_gets_the_tagged_repos(client, hf_transport, state):
    hf_transport.add("/api/models", "models_filter_nsfw_tag")
    state.ctx = context("allowed", "show")

    response = client.get(f"{BASE}/huggingface/search", params={"q": "#not-for-all-audiences", "limit": 2})

    assert [found["nsfw"] for found in response.json()["data"]["items"]] == [True, True]
    assert response.json()["data"]["next_cursor"] == cursor_of("models_filter_nsfw_tag")


def test_hugging_face_detail_with_a_slash_in_its_id(client, hf_transport):
    hf_transport.add_repo("lcm_lora_sdxl")

    response = client.get(f"{BASE}/huggingface/models/{LCM}")
    encoded = client.get(f"{BASE}/huggingface/models/latent-consistency%2Flcm-lora-sdxl")

    assert response.status_code == 200
    data = response.json()["data"]
    assert (data["id"], data["mb_type"], data["gated"]) == (LCM, "lora", False)
    assert data["license"]["summary"] == "openrail++"
    [version] = data["versions"]
    assert version["partial"] is False
    assert [file["name"] for file in version["files"]] == ["pytorch_lora_weights.safetensors"]
    assert version["files"][0]["download_url"] == (
        f"https://huggingface.co/{LCM}/resolve/main/pytorch_lora_weights.safetensors"
    )
    assert data["description_html"].startswith("<h3>")
    assert encoded.json()["data"] == data


def test_an_unknown_hugging_face_repo_is_404(client, hf_transport):
    hf_transport.add("/api/models/no-such-org/no-such-repo", "error_401_unknown_repo")
    hf_transport.add("/api/models/no-such-org/no-such-repo/tree/main", "error_401_unknown_tree")

    for path in ("/huggingface/models/no-such-org/no-such-repo", "/huggingface/models/no-such-org/no-such-repo/nsfw-media"):
        response = client.get(f"{BASE}{path}")
        assert response.status_code == 404, path
        assert _error(response)["error"] == "not_found"


def test_an_nsfw_hugging_face_repo_in_blur_mode(client, hf_transport, state):
    url = _nsfw_demo_repo(hf_transport)
    state.ctx = context("blur", "blur")

    detail = client.get(f"{BASE}/huggingface/models/org/demo")
    media = client.get(f"{BASE}/huggingface/models/org/demo/nsfw-media")

    assert detail.status_code == 200
    data = detail.json()["data"]
    assert data["nsfw"] is True and data["versions"][0]["nsfw"] is True
    [image] = data["versions"][0]["images"]
    assert (image["url"], image["nsfw"], image["nsfw_level"]) == (None, True, 8)
    assert url not in detail.text
    assert media.status_code == 200
    [revealed] = media.json()["data"]["images"]
    assert (revealed["url"], revealed["id"]) == (url, image["id"])


def test_an_nsfw_hugging_face_repo_for_a_user_who_shows_it(client, hf_transport, state):
    url = _nsfw_demo_repo(hf_transport)
    state.ctx = context("allowed", "show")

    data = client.get(f"{BASE}/huggingface/models/org/demo").json()["data"]

    assert [image["url"] for image in data["versions"][0]["images"]] == [url]


@pytest.mark.parametrize("ctx", [context("blocked", "hide"), context("allowed", "hide")])
def test_an_nsfw_hugging_face_repo_is_404_for_a_user_who_hides_nsfw(client, hf_transport, state, ctx):
    _nsfw_demo_repo(hf_transport)
    state.ctx = ctx

    detail = client.get(f"{BASE}/huggingface/models/org/demo")
    media = client.get(f"{BASE}/huggingface/models/org/demo/nsfw-media")

    assert (detail.status_code, media.status_code) == (404, 404)
    assert _hf_paths(hf_transport).count("/api/models/org/demo") == 1


@pytest.mark.parametrize(
    "path",
    [
        "/huggingface/models/org/%2E%2E/secret",
        "/huggingface/models/%2E%2E%2Fsecret",
        "/huggingface/models/org/%2E%2E/secret/nsfw-media",
        "/huggingface/models//etc/passwd",
        "/huggingface/models//etc/passwd/nsfw-media",
        "/huggingface/models/",
        "/huggingface/models/org%5Crepo",
        "/huggingface/models/org/repo%00",
        "/huggingface/models/" + "a" * 201,
        "/civitai/models/%2E%2E",
    ],
)
def test_model_ids_that_could_escape_are_refused(client, transport, hf_transport, path):
    response = client.get(f"{BASE}{path}")

    assert response.status_code == 400, path
    assert _error(response) == {"success": False, "error": "invalid_request", "message": "Invalid model id"}
    assert transport.calls == [] and hf_transport.calls == []


def test_a_200_character_id_reaches_the_source(client, hf_transport):
    response = client.get(f"{BASE}/huggingface/models/" + "a" * 200)

    assert response.status_code == 404
    assert _error(response)["error"] == "not_found"
    assert hf_transport.calls == []


def test_civitai_ids_still_work_with_the_path_routes(client, transport, state):
    transport.add("/models/122359", "model_122359")
    state.ctx = context("allowed", "show", max_level=4)

    assert client.get(f"{BASE}/civitai/models/122359").json()["data"]["id"] == "122359"
    assert client.get(f"{BASE}/civitai/models/122359/nsfw-media").json()["data"] == {"images": []}
    nested = client.get(f"{BASE}/civitai/models/122359/versions")
    assert nested.status_code == 404 and _error(nested)["error"] == "not_found"
    assert [path for path, _ in transport.calls] == ["/models/122359"]
