import importlib
import json
import logging
from datetime import datetime, timezone

import pytest

from backend import PLUGIN_ID
from backend import cache as cache_module
from backend import http as http_module
from backend import sources as sources_module
from backend.http import SourceHttp
from hooks import provider_hooks
from provider import openmodeldb_provider
from provider.openmodeldb_provider import PROVIDER_ID, OpenModelDBProvider, model_page_url
from src.plugin_api import HookContext, MarketplaceProviderBase, ProviderCapability, ProviderConnectionError, ProviderModelInfo

from .civitai_fakes import MemoryCache, fake_settings
from .openmodeldb_fakes import (
    GITHUB_RAW,
    MEGA_ONLY,
    MODELS,
    NO_LICENSE,
    RAW_GITHUB,
    THREE_FORMATS,
    TWO_AUTHORS,
    OpenModelDBTransport,
    make_source,
    model,
    models,
)

NOMOS_SHA = model(GITHUB_RAW)["resources"][0]["sha256"]
NOMOS_URL = "https://github.com/Phhofm/models/raw/main/4xNomos8kSC/4xNomos8kSC.pth"
NOMOS_IMAGES = ["https://openmodeldb.info/thumbs/79385d8f832b948a9ce1d339.jpg"]


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    return fake_settings(monkeypatch)


@pytest.fixture
def transport():
    return OpenModelDBTransport()


@pytest.fixture
def provider(transport):
    return OpenModelDBProvider(source=make_source(transport))


def test_metadata_declares_a_hash_lookup_provider_without_credentials(provider):
    metadata = provider.get_metadata()

    assert isinstance(provider, MarketplaceProviderBase)
    assert (metadata.id, metadata.name, metadata.website) == ("openmodeldb", "OpenModelDB", "https://openmodeldb.info")
    assert metadata.capabilities == [
        ProviderCapability.HASH_LOOKUP,
        ProviderCapability.MODEL_INFO,
        ProviderCapability.MEDIA_DOWNLOAD,
    ]
    assert not metadata.has_capability(ProviderCapability.API_KEY_REQUIRED)
    assert not metadata.has_capability(ProviderCapability.SEARCH)
    assert provider.provider_id == PROVIDER_ID == "openmodeldb"
    assert metadata.description


async def test_it_is_ready_without_settings_and_never_handles_a_download(provider, transport):
    assert await provider.initialize({}) is True
    assert await provider.shutdown() is None
    assert provider.get_settings_schema() == {"type": "object", "properties": {}, "required": []}
    assert provider.get_download_headers() == {}
    assert provider.matches_download_url(NOMOS_URL) is False
    assert provider.matches_download_url("https://openmodeldb.info/models/4x-Nomos8kSC") is False
    assert transport.calls == []


async def test_the_connection_test_loads_the_catalogue(transport):
    assert await OpenModelDBProvider(source=make_source(transport)).test_connection() is True

    down = OpenModelDBTransport()
    down.responses[MODELS] = (503, {}, b"")
    assert await OpenModelDBProvider(source=make_source(down)).test_connection() is False


async def test_a_known_file_gives_the_expected_provider_model_info(provider):
    info = await provider.get_model_by_hash(f" {NOMOS_SHA.upper()} ")

    assert info == ProviderModelInfo(
        provider_id="openmodeldb",
        provider_model_id="openmodeldb:4x-Nomos8kSC",
        provider_version_id=None,
        name="Nomos8kSC",
        description=(
            "Architecture: ESRGAN · Scale: 4x · Input: image · Channels: 3 → 3 · Dataset: Nomos8k_sfw · "
            "Training: 85,000 iterations, 137 epochs 4x photo upscaler with otf jpg compression and blur, "
            "trained on musl's Nomos8k_sfw dataset for realisic sr"
        ),
        tags=["Compression Removal", "JPEG", "Realistic", "Restoration"],
        trigger_words=[],
        nsfw=False,
        download_url=NOMOS_URL,
        media_urls=NOMOS_IMAGES,
        model_type="upscaler",
        base_model="ESRGAN 4x",
        created_at=datetime(2023, 5, 10, tzinfo=timezone.utc),
        updated_at=None,
        extra_data={
            "source": "openmodeldb",
            "source_url": "https://openmodeldb.info/models/4x-Nomos8kSC",
            "license": "CC-BY-4.0",
            "license_url": "https://spdx.org/licenses/CC-BY-4.0.html",
            "architecture": "ESRGAN",
            "scale": 4,
            "mb_base_model": "ESRGAN 4x",
            "mb_license": "CC-BY-4.0",
        },
    )
    json.dumps(info.to_dict())


async def test_each_file_of_a_model_answers_with_its_own_url_but_an_onnx_with_none(provider):
    pth, safetensors, onnx = model(THREE_FORMATS)["resources"]

    found = [await provider.get_model_by_hash(resource["sha256"]) for resource in (pth, safetensors, onnx)]

    assert [info.download_url for info in found] == [pth["urls"][0], safetensors["urls"][0], None]
    assert {info.provider_model_id for info in found} == {"openmodeldb:1x-DeNoise-realplksr-otf"}


async def test_the_description_has_no_closing_notes(provider):
    info = await provider.get_model_by_hash(model(RAW_GITHUB)["resources"][0]["sha256"])

    assert info.description.startswith("Architecture: SPAN · Scale: 2x")
    assert "Note: PotionUI" not in info.description and "Other downloads" not in info.description


async def test_a_file_only_the_openmodeldb_page_offers_is_found_without_a_download_url(provider):
    info = await provider.get_model_by_hash(model(MEGA_ONLY)["resources"][0]["sha256"])

    assert (info.provider_model_id, info.download_url, info.model_type) == ("openmodeldb:4x-UltraSharp", None, "upscaler")
    assert "Other downloads" not in info.description


async def test_a_model_without_licence(provider):
    info = await provider.get_model_by_hash(model(NO_LICENSE)["resources"][0]["sha256"])

    assert {key: info.extra_data[key] for key in ("license", "license_url", "mb_license")} == {
        "license": None, "license_url": None, "mb_license": None,
    }


async def test_previews_are_the_versions_at_most_six_https_images_on_public_hosts(transport):
    body = models()
    gallery = [{"type": "standalone", "url": f"https://i.slow.pics/{index}.webp"} for index in range(9)]
    body[TWO_AUTHORS]["images"] = [
        {"type": "standalone", "url": "https://127.0.0.1/local.jpg"},
        {"type": "standalone", "url": "https://localhost/local.jpg"},
        {"type": "paired", "LR": "https://i.slow.pics/lr.webp", "SR": "http://i.slow.pics/plain.webp"},
        {"type": "paired", "LR": "https://imgsli.com/i/lr.jpg", "SR": "https://imgsli.com/i/sr.jpg"},
        {"type": "standalone", "url": "https://github.com/o/r/assets/1/x"},
        *gallery,
    ]
    transport.responses[MODELS] = (200, {}, body)
    provider = OpenModelDBProvider(source=make_source(transport))

    info = await provider.get_model_by_hash(body[TWO_AUTHORS]["resources"][0]["sha256"])

    assert info.media_urls == ["https://openmodeldb.info/thumbs/12c55df0bf2bdbcda120944a.jpg"] + [
        image["url"] for image in gallery[:5]
    ]


async def test_the_description_is_plain_text_of_at_most_4000_characters(transport):
    body = models()
    body[GITHUB_RAW]["description"] = "# Title\n\n" + ("word " * 50 + "[link](https://example.com)\n\n") * 40
    transport.responses[MODELS] = (200, {}, body)

    description = (await OpenModelDBProvider(source=make_source(transport)).get_model_by_hash(NOMOS_SHA)).description

    assert 3900 < len(description) <= 4000
    assert description.startswith("Architecture: ESRGAN · Scale: 4x") and " Title word word" in description
    assert description.endswith("…") and "<" not in description and "\n" not in description


async def test_an_unknown_hash_is_none(provider):
    assert await provider.get_model_by_hash("0" * 64) is None


@pytest.mark.parametrize("value", [None, 42, "", "  ", "abc", "g" * 64, NOMOS_SHA + "0"])
async def test_a_value_that_is_not_a_sha256_is_none_without_loading_anything(provider, transport, value):
    assert await provider.get_model_by_hash(value) is None
    assert transport.calls == []


async def test_a_catalogue_that_cannot_be_loaded_is_a_connection_error(caplog):
    down = OpenModelDBTransport()
    down.responses[MODELS] = OSError("network is unreachable")

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"), pytest.raises(ProviderConnectionError) as error:
        await OpenModelDBProvider(source=make_source(down)).get_model_by_hash(NOMOS_SHA)

    assert "Could not load the OpenModelDB catalogue" in str(error.value)
    assert "catalogue unavailable for a hash lookup" in caplog.text


async def test_an_entry_that_cannot_be_read_is_none(provider, monkeypatch, caplog):
    def broken(*args):
        raise ValueError("unexpected shape")

    monkeypatch.setattr(openmodeldb_provider, "build_model_info", broken)

    with caplog.at_level(logging.ERROR, logger="plugins.model-browser"):
        assert await provider.get_model_by_hash(NOMOS_SHA) is None
    assert "Could not build the OpenModelDB details" in caplog.text


async def test_the_provider_shares_the_registry_source_and_its_catalogue(monkeypatch, transport):
    monkeypatch.setattr(sources_module, "_instances", {})
    monkeypatch.setattr(http_module, "http", SourceHttp(fetch=transport, limits={"openmodeldb": (0.0, 2)}))
    monkeypatch.setattr(cache_module, "cache", MemoryCache())
    provider = OpenModelDBProvider()

    info = await provider.get_model_by_hash(NOMOS_SHA)
    page = await sources_module.get_source("openmodeldb").search(sources_module.SearchQuery(q="nomos8k"))

    assert info.name == "Nomos8kSC" and GITHUB_RAW in [found.id for found in page.items]
    assert transport.count(MODELS) == 1


@pytest.mark.parametrize(
    "model_id, url",
    [
        ("openmodeldb:4x-UltraSharp", "https://openmodeldb.info/models/4x-UltraSharp"),
        ("openmodeldb:2x-AnimeJaNai-Standard-v1-SuperUltraCompact",
         "https://openmodeldb.info/models/2x-AnimeJaNai-Standard-v1-SuperUltraCompact"),
    ],
)
def test_the_mirror_link_is_the_model_page(provider, transport, model_id, url):
    assert provider.get_model_page_url(model_id, None) == url
    assert provider.get_model_page_url(model_id, "Release") == url
    assert model_page_url(model_id) == url
    assert transport.calls == []


@pytest.mark.parametrize(
    "model_id",
    [None, 42, "", "4x-UltraSharp", "openmodeldb:", "openmodeldb:../x", "openmodeldb:a/b", "openmodeldb:4x x",
     "OpenModelDB:4x-UltraSharp", "civitai:122359", "huggingface:org/repo"],
)
def test_anything_else_has_no_mirror_link(provider, model_id):
    assert provider.get_model_page_url(model_id) is None


def test_a_broken_openmodeldb_provider_does_not_keep_the_others_out(monkeypatch, caplog):
    real = importlib.import_module

    def failing(name, package=None):
        if name.endswith("openmodeldb_provider"):
            raise ImportError("broken")
        return real(name, package)

    monkeypatch.setattr(provider_hooks.importlib, "import_module", failing)
    context = HookContext(hook_name="provider.register", plugin_id=PLUGIN_ID, data={"providers": {}})

    with caplog.at_level(logging.ERROR, logger="plugins.model-browser"):
        assert provider_hooks.register_providers(context) is context

    assert list(context.data["providers"]) == ["model-browser", "civarchive"]
    assert "Could not register the OpenModelDBProvider marketplace provider" in caplog.text
