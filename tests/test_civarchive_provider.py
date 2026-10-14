import copy
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend import cache as cache_module
from backend import http as http_module
from backend import sources as sources_module
from backend.http import SourceHttp
from backend.security import html_to_text, sanitize_html
from provider import civarchive_provider, pending_provider
from provider.civarchive_provider import PROVIDER_ID, CivArchiveProvider, model_page_url, pick_model
from src.plugin_api import MarketplaceProviderBase, ProviderCapability, ProviderConnectionError, ProviderModelInfo

from .civarchive_fakes import (
    HTML_HEADERS,
    JSON_HEADERS,
    SHA_78291,
    SHA_UNKNOWN,
    CivArchiveTransport,
    data_body,
    make_source,
    model,
    props,
)
from .civitai_fakes import MemoryCache, fake_settings

CIVITAI_URL = "https://civitai.com/api/download/models/83095"
LOWRA_SHA = "adf5b9e4c1f01243de32d5772216e1f6987f4097f4f23f53e2bed641d33a3ceb"
TWEAKER_SHA = "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"


@pytest.fixture
def settings_values(monkeypatch):
    return fake_settings(monkeypatch, enable_civarchive=True)


@pytest.fixture
def transport():
    return CivArchiveTransport()


@pytest.fixture
def provider(settings_values, transport):
    return CivArchiveProvider(source=make_source(transport))


def _hash_page(transport, sha256, *models, files=()):
    page = {"id": sha256, "files": list(files), "models": list(models), "meta": {}, "isNsfw": False}
    transport.add(f"sha256/{sha256}.json", (200, JSON_HEADERS, data_body(page)))
    return page


def test_metadata_declares_a_hash_lookup_provider_without_credentials(provider):
    metadata = provider.get_metadata()

    assert isinstance(provider, MarketplaceProviderBase)
    assert (metadata.id, metadata.name, metadata.website) == ("civarchive", "CivArchive", "https://civarchive.com")
    assert metadata.capabilities == [
        ProviderCapability.HASH_LOOKUP,
        ProviderCapability.MODEL_INFO,
        ProviderCapability.MEDIA_DOWNLOAD,
    ]
    assert not metadata.has_capability(ProviderCapability.API_KEY_REQUIRED)
    assert provider.provider_id == PROVIDER_ID == "civarchive"
    assert "unofficial" in metadata.description


async def test_it_is_ready_without_settings_and_never_handles_a_download(provider, transport):
    assert await provider.initialize({}) is True
    assert await provider.shutdown() is None
    assert provider.get_settings_schema() == {"type": "object", "properties": {}, "required": []}
    assert provider.get_download_headers() == {}
    assert provider.matches_download_url(CIVITAI_URL) is False
    assert provider.matches_download_url("https://civarchive.com/models/78291") is False
    assert transport.calls == []


async def test_the_connection_test_reads_the_build_id_while_the_source_is_on(settings_values, transport):
    assert await CivArchiveProvider(source=make_source(transport)).test_connection() is True
    assert transport.paths() == ["/"]

    down = CivArchiveTransport()
    down.home = (503, HTML_HEADERS, "<html>down</html>")
    assert await CivArchiveProvider(source=make_source(down)).test_connection() is False

    settings_values["enable_civarchive"] = False
    off = CivArchiveTransport()
    assert await CivArchiveProvider(source=make_source(off)).test_connection() is False
    assert off.calls == []


async def test_while_the_source_is_off_a_hash_is_none_without_any_request(monkeypatch, transport):
    fake_settings(monkeypatch)

    assert await CivArchiveProvider(source=make_source(transport)).get_model_by_hash(SHA_78291) is None
    assert transport.calls == []


async def test_an_unreadable_setting_keeps_the_provider_off(monkeypatch, transport):
    from backend import settings

    def broken(key):
        raise RuntimeError("no database")

    monkeypatch.setattr(settings, "get_setting", broken)

    assert await CivArchiveProvider(source=make_source(transport)).get_model_by_hash(SHA_78291) is None
    assert transport.calls == []


async def test_the_setting_is_read_on_every_lookup(settings_values, provider, transport):
    assert await provider.get_model_by_hash(SHA_78291) is not None
    settings_values["enable_civarchive"] = False

    assert await provider.get_model_by_hash(SHA_78291) is None
    assert transport.paths() == ["/", f"sha256/{SHA_78291}.json"]


async def test_a_known_file_gives_the_expected_provider_model_info(provider, transport):
    info = await provider.get_model_by_hash(f" {SHA_78291.upper()} ")

    raw = model("model_78291_v83095")
    assert info == ProviderModelInfo(
        provider_id="civarchive",
        provider_model_id="civarchive:78291",
        provider_version_id="83095",
        name="Tangbohu Saturation Tweaker LoRA (唐伯虎饱和度调整LoRA)",
        description=html_to_text(sanitize_html(raw["description"]), limit=4000),
        tags=["style"],
        trigger_words=[],
        nsfw=False,
        download_url=CIVITAI_URL,
        media_urls=[image["url"] for image in raw["version"]["images"]],
        model_type="lora",
        base_model="SD 1.5",
        created_at=datetime(2023, 5, 27, 20, 28, 45, 27000, tzinfo=timezone.utc),
        updated_at=None,
        extra_data={
            "source": "civarchive",
            "source_url": "https://civarchive.com/models/78291?modelVersionId=83095",
            "mb_base_model": "SD 1.5",
        },
    )
    assert info.description.startswith("这是一个LoRA") and "<" not in info.description
    json.dumps(info.to_dict())
    assert transport.paths() == ["/", f"sha256/{SHA_78291}.json"]


async def test_an_unknown_hash_is_none(provider, transport):
    assert await provider.get_model_by_hash(SHA_UNKNOWN) is None
    assert transport.paths() == ["/", f"sha256/{SHA_UNKNOWN}.json"]


async def test_a_hash_page_that_is_gone_is_none(provider, transport):
    assert await provider.get_model_by_hash("ab" * 32) is None
    assert transport.paths() == ["/", "sha256/" + "ab" * 32 + ".json", "/", "sha256/" + "ab" * 32 + ".json"]


@pytest.mark.parametrize("value", [None, 42, "", "  ", "abc", "g" * 64, SHA_78291 + "0"])
async def test_a_value_that_is_not_a_sha256_is_none_without_a_request(provider, transport, value):
    assert await provider.get_model_by_hash(value) is None
    assert transport.calls == []


@pytest.mark.parametrize(
    "where, failure",
    [
        ("home", OSError("network is unreachable")),
        ("home", (503, HTML_HEADERS, "<html>down</html>")),
        ("page", (502, HTML_HEADERS, "<html>bad gateway</html>")),
        ("page", (500, HTML_HEADERS, "<title>Error 500: Internal Server Error</title>")),
    ],
    ids=["unreachable", "home-503", "page-502", "page-500"],
)
async def test_civarchive_unreachable_is_a_connection_error(settings_values, caplog, where, failure):
    down = CivArchiveTransport()
    if where == "home":
        down.home = failure
    else:
        down.clear(f"sha256/{SHA_78291}.json").add(f"sha256/{SHA_78291}.json", failure)

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"), pytest.raises(ProviderConnectionError) as error:
        await CivArchiveProvider(source=make_source(down)).get_model_by_hash(SHA_78291)

    assert "Could not read CivArchive" in str(error.value)
    assert "CivArchive unavailable for a hash lookup" in caplog.text


async def test_a_model_that_cannot_be_read_is_none(provider, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ValueError("unexpected shape")

    monkeypatch.setattr(civarchive_provider, "build_model_info", broken)

    with caplog.at_level(logging.ERROR, logger="plugins.model-browser"):
        assert await provider.get_model_by_hash(SHA_78291) is None
    assert "Could not build the CivArchive details" in caplog.text


def test_the_civitai_model_of_a_page_comes_first():
    civitai_model = model("model_78291_v83095")
    other = dict(copy.deepcopy(civitai_model), id="tensorart_1", platform="tensorart")
    no_version = {"id": "5", "platform": "civitai"}

    assert pick_model({"models": [other, no_version, civitai_model]}) is civitai_model
    assert pick_model({"models": [other]}) is other
    assert pick_model({"models": []}) is None
    assert pick_model({"models": "not a list"}) is None


async def test_nsfw_by_the_model_bitmask_and_at_most_six_previews(provider, transport):
    _hash_page(transport, TWEAKER_SHA, model("model_122359_v135867"))

    info = await provider.get_model_by_hash(TWEAKER_SHA)

    assert (info.provider_model_id, info.provider_version_id, info.nsfw) == ("civarchive:122359", "135867", True)
    assert info.media_urls == [image["url"] for image in model("model_122359_v135867")["version"]["images"][:6]]
    assert (info.model_type, info.base_model) == ("lora", "SDXL 1.0")


async def test_a_file_above_the_safe_level_is_nsfw(provider, transport):
    raw = model("model_78291_v83095")
    raw["version"]["files"][0]["nsfw_level"] = 4
    _hash_page(transport, SHA_78291, raw)

    assert (await provider.get_model_by_hash(SHA_78291)).nsfw is True


async def test_previews_leave_out_nsfw_images_and_real_people_when_the_admin_hides_them(settings_values, provider, transport):
    raw = model("model_78291_v83095")
    images = raw["version"]["images"]
    images[0]["nsfwLevel"] = 4
    images[1]["poi"] = True
    _hash_page(transport, SHA_78291, raw)

    shown = await provider.get_model_by_hash(SHA_78291)
    settings_values["hide_poi"] = True
    hidden = await CivArchiveProvider(source=make_source(transport)).get_model_by_hash(SHA_78291)

    assert shown.media_urls == [images[1]["url"], images[2]["url"]]
    assert hidden.media_urls == [images[2]["url"]]


async def test_a_model_deleted_from_civitai_is_found_with_its_hugging_face_mirror(provider, transport):
    _hash_page(transport, LOWRA_SHA, model("model_48139_v52753"))

    info = await provider.get_model_by_hash(LOWRA_SHA)

    assert (info.provider_model_id, info.name, info.trigger_words) == ("civarchive:48139", "LowRA", ["dark theme"])
    assert info.download_url == "https://huggingface.co/baycon/for_Colab/resolve/main/lowra_v10.safetensors"
    assert (info.nsfw, info.media_urls) == (True, [])
    assert info.description.endswith(
        "Deleted from CivitAI on 2025-08-07: its files come from Hugging Face mirrors uploaded by other people."
    )


async def test_a_file_only_torrents_have_is_found_without_a_download_url(provider, transport):
    raw = model("model_298188_v334947")
    sha256 = raw["version"]["files"][0]["sha256"]
    _hash_page(transport, sha256, raw)

    info = await provider.get_model_by_hash(sha256)

    assert (info.provider_model_id, info.download_url, info.model_type) == ("civarchive:298188", None, "lora")


async def test_a_file_missing_from_the_version_downloads_from_the_pages_own_mirrors(provider, transport):
    raw = model("model_78291_v83095")
    raw["version"]["files"] = []
    page = props("sha256_e47d")
    _hash_page(transport, SHA_78291, raw, files=page["files"])

    info = await provider.get_model_by_hash(SHA_78291)

    assert info.download_url == CIVITAI_URL
    page["files"][0]["deleted_at"] = "2026-01-01T00:00:00.000Z"
    _hash_page(transport, "cd" * 32, raw, files=page["files"])
    assert (await provider.get_model_by_hash("cd" * 32)).download_url == (
        "https://huggingface.co/idolt/Lora4/resolve/main/tangbohu_saturation_v1.0.safetensors"
    )


async def test_the_provider_shares_the_registry_source(settings_values, monkeypatch, transport):
    monkeypatch.setattr(sources_module, "_instances", {})
    monkeypatch.setattr(http_module, "http", SourceHttp(fetch=transport, limits={"civarchive": (0.0, 4)}))
    monkeypatch.setattr(cache_module, "cache", MemoryCache())
    provider = CivArchiveProvider()

    info = await provider.get_model_by_hash(SHA_78291)
    detail = await sources_module.get_source("civarchive").detail("78291:83095")

    assert info.name == detail.name
    assert transport.home_fetches == 1


async def test_the_core_fetcher_enriches_a_model_already_on_disk(provider, monkeypatch):
    import src.features.providers.registry as provider_registry
    from src.features.models.provider_info import ProviderInfoFetcher
    from src.plugin_api import HookContext

    stored, fetched = {}, []

    async def get_model_by_hash(provider_id, sha256, raise_errors=False):
        assert provider_id == PROVIDER_ID
        return await provider.get_model_by_hash(sha256)

    async def fetch_media_bytes(self, url):
        fetched.append(url)
        return b"\xff\xd8\xff\xe0", "image/webp"

    class Editor:
        def __init__(self):
            self.metadata, self.previews = {}, []

        def notify_metadata_changed(self, model_id, fields):
            pass

        def update_model_metadata(self, model_id, values):
            self.metadata = values

        def list_model_previews(self, model_id):
            return list(self.previews)

        def add_model_preview(self, model_id, preview):
            self.previews.append(preview)

    on_disk = SimpleNamespace(id="model-7", sha256=SHA_78291, description=None, model_metadata={}, is_directory=False)
    repository = SimpleNamespace(
        get_by_id=lambda model_id, include_providers=False: on_disk if model_id == "model-7" else None,
        upsert_provider=lambda model_id, row: stored.__setitem__("provider", row),
        update_description=lambda model_id, text: stored.__setitem__("description", text),
    )
    plugins = SimpleNamespace(
        execute_hook=lambda hook, initial_data=None: (HookContext(hook_name=hook, plugin_id="system", data=initial_data or {}), True)
    )
    monkeypatch.setattr(provider_registry, "get_provider_registry", lambda: SimpleNamespace(get_model_by_hash=get_model_by_hash))
    monkeypatch.setattr(ProviderInfoFetcher, "_fetch_media_bytes", fetch_media_bytes)
    editor = Editor()
    fetcher = ProviderInfoFetcher(repository, plugins, metadata_editor=editor, storage_driver=SimpleNamespace(put_bytes=lambda key, data: None))

    assert await fetcher.run_provider_fetch(PROVIDER_ID, ["model-7"]) == {"successful": 1, "failed": 0}

    row = stored["provider"]
    assert (row.model_id, row.provider, row.provider_model_id, row.provider_version_id) == (
        "model-7", "civarchive", "civarchive:78291", "83095",
    )
    assert (row.nsfw, row.download_url) == (False, CIVITAI_URL)
    assert stored["description"].startswith("这是一个LoRA")
    assert fetched == [image["url"] for image in model("model_78291_v83095")["version"]["images"]]
    assert len(editor.previews) == 3


@pytest.mark.parametrize(
    "model_id, version, url",
    [
        ("civarchive:78291", "83095", "https://civarchive.com/models/78291?modelVersionId=83095"),
        ("civarchive:78291", 83095, "https://civarchive.com/models/78291?modelVersionId=83095"),
        ("civarchive:78291", None, "https://civarchive.com/models/78291"),
        ("civarchive:78291", "v1.0", "https://civarchive.com/models/78291"),
        ("civarchive:78291:83095", None, "https://civarchive.com/models/78291?modelVersionId=83095"),
    ],
)
def test_the_mirror_link_is_the_civarchive_page(provider, transport, monkeypatch, model_id, version, url):
    from backend import settings

    def unexpected(key):
        raise AssertionError("get_model_page_url must not read anything")

    monkeypatch.setattr(settings, "get_setting", unexpected)

    assert provider.get_model_page_url(model_id, version) == url
    assert model_page_url(model_id, version) == url
    assert transport.calls == []


@pytest.mark.parametrize(
    "model_id",
    [None, 42, "", "78291", "civarchive:", "civarchive:abc", "civarchive:78291:x", "civarchive:0",
     "civarchive:../78291", "civarchive:78291?modelVersionId=1", "CivArchive:78291", "civitai:78291"],
)
def test_anything_else_has_no_mirror_link(provider, model_id):
    assert provider.get_model_page_url(model_id, "83095") is None


def test_the_model_browser_provider_links_civarchive_downloads_to_the_same_page():
    assert pending_provider.model_page_url("civarchive:78291", "83095") == (
        "https://civarchive.com/models/78291?modelVersionId=83095"
    )
    assert pending_provider.model_page_url("civarchive:78291") == "https://civarchive.com/models/78291"
    assert pending_provider.model_page_url("civarchive:not-an-id", "1") is None
