import json
import sqlite3
from datetime import datetime, timezone

import pytest

from backend import pending
from backend.pending import PendingRow
from backend.schema import File, Image, License, ModelDetail, Version
from backend.sources.civitai import normalize_detail
from backend.tables import create_tables
from provider import pending_provider
from provider.pending_provider import PROVIDER_ID, PendingProvider, build_model_info, model_page_url, preview_policy
from src.plugin_api import MarketplaceProviderBase, ProviderCapability, ProviderConnectionError, ProviderModelInfo

from .civitai_fakes import body

SHA = "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7"
OTHER_SHA = "a5e633c9afd9d12da2d34a2fd691e74d3e5fda19df8ad4b73b7f94e65f1fb168"
DOWNLOAD_URL = "https://civitai.com/api/download/models/135867?fileId=99264"
SAFE_IMAGE = "https://image.civitai.com/xG1/safe-1/original=true/"
SAFE_VIDEO = "https://image.civitai.com/xG1/safe-2/original=true/"
NSFW_IMAGE = "https://image.civitai.com/xG1/nsfw-1/original=true/"


@pytest.fixture
def table(plugin_db):
    create_tables()
    return plugin_db


@pytest.fixture(autouse=True)
def admin_settings(monkeypatch):
    values = {"nsfw_max_level": 2, "hide_poi": False}
    monkeypatch.setattr(pending_provider.plugin_settings, "get_setting", lambda key: values[key])
    return values


@pytest.fixture
def provider():
    return PendingProvider()


def lora(**changes) -> ModelDetail:
    version = Version(
        id="135867",
        name="v1.0",
        base_model="SDXL 1.0",
        trigger_words=["add detail", "detailed"],
        created_at="2023-08-07T14:55:02.627Z",
        images=[
            Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1, width=832, height=1216, id="11"),
            Image(url=NSFW_IMAGE, nsfw=True, nsfw_level=8, id="12"),
            Image(url=None, blurhash="UBF$:LxuIUIo", nsfw=True, nsfw_level=16, id="13"),
            Image(url=SAFE_VIDEO, nsfw=False, nsfw_level=2, kind="video", id="14"),
        ],
        files=[
            File(
                id="99264",
                name="add-detail-xl.safetensors",
                size_bytes=228_455_604,
                sha256=SHA.upper(),
                format="SafeTensor",
                primary=True,
                download_url=DOWNLOAD_URL,
                auth_provider_id="civitai",
            )
        ],
    )
    model = ModelDetail(
        source="civitai",
        id="122359",
        name="Detail Tweaker XL",
        type="LORA",
        mb_type="lora",
        creator="nocrypt",
        description_html="<p>Adds <strong>detail</strong>.</p><p>Weight&nbsp;0.5 to 1.5</p>",
        tags=["concept", "detailed"],
        license=License(summary="Commercial use allowed, credit required", commercial_use=["Image", "Sell"]),
        source_url="https://civitai.red/models/122359",
        nsfw=False,
        versions=[version],
    )
    return model.model_copy(update=changes)


def info_of(model: ModelDetail, *, version_id="135867", file_id="99264", mb_type="lora") -> dict:
    return {"model": model.model_dump(mode="json"), "version_id": version_id, "file_id": file_id, "mb_type": mb_type}


def store(info, *, key=SHA, status="assigned", source="civitai", source_model_id="122359") -> PendingRow:
    return pending.insert(
        PendingRow(
            sha256=key,
            source=source,
            source_model_id=source_model_id,
            source_version_id="135867",
            file_name="add-detail-xl.safetensors",
            requested_by="user-1",
            assign_to="user:user-1",
            info=info,
            status=status,
            model_id="model-1",
        )
    )


def with_images(*images: Image) -> ModelDetail:
    model = lora()
    return model.model_copy(update={"versions": [model.versions[0].model_copy(update={"images": list(images)})]})


def test_metadata_declares_a_hash_lookup_provider_without_credentials(provider):
    metadata = provider.get_metadata()

    assert isinstance(provider, MarketplaceProviderBase)
    assert (metadata.id, metadata.name, metadata.website) == (
        "model-browser", "Model Browser", "https://github.com/PotionUI/PotionUI",
    )
    assert metadata.capabilities == [
        ProviderCapability.HASH_LOOKUP,
        ProviderCapability.MODEL_INFO,
        ProviderCapability.MEDIA_DOWNLOAD,
    ]
    assert not metadata.has_capability(ProviderCapability.API_KEY_REQUIRED)
    assert not metadata.has_capability(ProviderCapability.SEARCH)
    assert not metadata.has_capability(ProviderCapability.DOWNLOAD_URL)
    assert provider.provider_id == PROVIDER_ID == "model-browser"
    assert metadata.description


async def test_it_is_ready_without_settings_and_never_handles_a_download(provider):
    assert await provider.initialize({}) is True
    assert await provider.initialize({"api_key": "ignored"}) is True
    assert await provider.test_connection() is True
    assert await provider.shutdown() is None
    assert provider.get_settings_schema() == {"type": "object", "properties": {}, "required": []}
    assert provider.get_download_headers() == {}
    assert provider.matches_download_url(DOWNLOAD_URL) is False
    assert provider.matches_download_url("https://huggingface.co/a/b/resolve/main/c.safetensors") is False


async def test_a_downloaded_file_gives_the_expected_provider_model_info(table, provider):
    store(info_of(lora()))

    info = await provider.get_model_by_hash(f" {SHA.upper()} ")

    assert info == ProviderModelInfo(
        provider_id="model-browser",
        provider_model_id="civitai:122359",
        provider_version_id="135867",
        name="Detail Tweaker XL",
        description="Adds detail. Weight 0.5 to 1.5",
        tags=["concept", "detailed"],
        trigger_words=["add detail", "detailed"],
        nsfw=False,
        download_url=DOWNLOAD_URL,
        media_urls=[SAFE_IMAGE, SAFE_VIDEO],
        model_type="lora",
        base_model="SDXL 1.0",
        created_at=datetime(2023, 8, 7, 14, 55, 2, 627000, tzinfo=timezone.utc),
        updated_at=None,
        extra_data={
            "source": "civitai",
            "source_url": "https://civitai.red/models/122359",
            "license": "Commercial use allowed, credit required",
            "license_url": None,
            "mb_base_model": "SDXL 1.0",
            "mb_license": "Commercial use allowed, credit required",
        },
    )
    json.dumps(info.to_dict())


async def test_nsfw_images_never_become_previews(table, provider):
    store(info_of(with_images(
        Image(url=NSFW_IMAGE, nsfw=True, nsfw_level=8),
        Image(url=NSFW_IMAGE + "x", nsfw=True, nsfw_level=None),
        Image(url=NSFW_IMAGE + "y", nsfw=True, nsfw_level=1),
        Image(url=None, blurhash="UBF$", nsfw=True, nsfw_level=4),
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1),
    )))

    info = await provider.get_model_by_hash(SHA)

    assert info.media_urls == [SAFE_IMAGE]


async def test_an_image_above_the_admins_current_level_is_left_out(table, provider, admin_settings):
    store(info_of(with_images(
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1),
        Image(url=SAFE_VIDEO, nsfw=False, nsfw_level=4),
    )))

    assert (await provider.get_model_by_hash(SHA)).media_urls == [SAFE_IMAGE]
    admin_settings["nsfw_max_level"] = 4
    assert (await provider.get_model_by_hash(SHA)).media_urls == [SAFE_IMAGE, SAFE_VIDEO]


async def test_hide_poi_leaves_out_images_of_real_people(table, provider, admin_settings):
    store(info_of(with_images(
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1),
        Image(url=SAFE_VIDEO, nsfw=False, nsfw_level=1, poi=True),
    )))

    assert (await provider.get_model_by_hash(SHA)).media_urls == [SAFE_IMAGE, SAFE_VIDEO]
    admin_settings["hide_poi"] = True
    assert (await provider.get_model_by_hash(SHA)).media_urls == [SAFE_IMAGE]


async def test_previews_are_only_https_urls_on_public_hosts_without_duplicates(table, provider):
    odd = [
        "http://image.civitai.com/a.jpeg",
        "https://127.0.0.1/a.jpeg",
        "https://[::1]/a.jpeg",
        "https://localhost/a.jpeg",
        "https://media.localhost/a.jpeg",
        "https://2130706433/a.jpeg",
        "https://image.civitai.com:8443/a.jpeg",
        "https://user:secret@image.civitai.com/a.jpeg",
        "https://image.civitai.com/a b.jpeg",
        "https:///a.jpeg",
        "file:///etc/passwd",
        "data:image/png;base64,AAAA",
    ]
    store(info_of(with_images(
        *(Image(url=url, nsfw=False, nsfw_level=1) for url in odd),
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1),
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=1),
        Image(url="https://huggingface.co/org/repo/resolve/main/sample.png", nsfw=False, nsfw_level=1),
    )))

    assert (await provider.get_model_by_hash(SHA)).media_urls == [
        SAFE_IMAGE,
        "https://huggingface.co/org/repo/resolve/main/sample.png",
    ]


async def test_a_model_with_only_nsfw_images_has_no_previews(table, provider):
    model = with_images(
        Image(url=NSFW_IMAGE, nsfw=True, nsfw_level=8),
        Image(url=None, blurhash="UBF$", nsfw=True, nsfw_level=16),
    ).model_copy(update={"nsfw": True})
    store(info_of(model))

    info = await provider.get_model_by_hash(SHA)

    assert info.media_urls == []
    assert info.nsfw is True


async def test_an_nsfw_version_marks_the_model_nsfw(table, provider):
    model = lora()
    model = model.model_copy(update={"versions": [model.versions[0].model_copy(update={"nsfw": True})]})
    store(info_of(model))

    assert (await provider.get_model_by_hash(SHA)).nsfw is True


async def test_unreadable_settings_fall_back_to_the_defaults(table, provider, monkeypatch):
    def broken(key):
        raise sqlite3.OperationalError("no such table: plugin_settings")

    monkeypatch.setattr(pending_provider.plugin_settings, "get_setting", broken)
    store(info_of(with_images(
        Image(url=SAFE_IMAGE, nsfw=False, nsfw_level=2),
        Image(url=SAFE_VIDEO, nsfw=False, nsfw_level=4),
    )))

    assert preview_policy() == (2, False)
    assert (await provider.get_model_by_hash(SHA)).media_urls == [SAFE_IMAGE]


@pytest.mark.parametrize("html", [None, "", "<p> </p>", "<script>alert(1)</script>"])
async def test_an_empty_description_is_none(table, provider, html):
    store(info_of(lora(description_html=html)))

    assert (await provider.get_model_by_hash(SHA)).description is None


async def test_the_description_is_plain_text_of_at_most_4000_characters(table, provider):
    paragraph = "<p>" + " ".join(["word"] * 50) + " <a href='https://example.com'>link</a></p>"
    store(info_of(lora(description_html="<h1>Title</h1>" + paragraph * 40)))

    description = (await provider.get_model_by_hash(SHA)).description

    assert 3900 < len(description) <= 4000
    assert description.startswith("Title word word")
    assert description.endswith("…")
    assert "<" not in description and "\n" not in description


async def test_an_unknown_hash_is_none(table, provider):
    store(info_of(lora()))

    assert await provider.get_model_by_hash(OTHER_SHA) is None


@pytest.mark.parametrize("value", [None, 42, "", "  ", "abc", "g" * 64, SHA + "0", f"nohash:{DOWNLOAD_URL}"])
async def test_a_value_that_is_not_a_sha256_is_none_without_a_lookup(provider, monkeypatch, value):
    def unexpected(key):
        raise AssertionError("the pending table must not be read")

    monkeypatch.setattr(pending, "get", unexpected)

    assert await provider.get_model_by_hash(value) is None


async def test_a_failed_download_is_none(table, provider):
    store(info_of(lora()), status="failed")

    assert await provider.get_model_by_hash(SHA) is None


@pytest.mark.parametrize("status", ["queued", "indexed", "assigned", "enriched"])
async def test_every_other_status_answers(table, provider, status):
    store(info_of(lora()), status=status)

    assert (await provider.get_model_by_hash(SHA)).provider_model_id == "civitai:122359"


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"model": None},
        {"model": "Detail Tweaker XL"},
        {"model": ["civitai", "122359"]},
        {"model": {"id": "122359", "name": "Detail Tweaker XL"}},
        {"model": {**lora().model_dump(mode="json"), "source": "somewhere-else"}},
        {"model": {**lora().model_dump(mode="json"), "versions": "135867"}},
        {"model": {**lora().model_dump(mode="json"), "versions": [{"id": "135867"}]}},
    ],
    ids=["empty", "null", "text", "list", "partial", "unknown-source", "versions-text", "version-partial"],
)
async def test_malformed_info_is_none(table, provider, info):
    store(info)

    assert await provider.get_model_by_hash(SHA) is None


async def test_an_info_column_that_is_not_json_is_none(table, provider):
    store(info_of(lora()))
    with table.get_cursor() as cursor:
        cursor.execute("UPDATE model_browser_pending SET info = ? WHERE sha256 = ?", ("{not json", SHA))

    assert await provider.get_model_by_hash(SHA) is None


async def test_a_database_failure_is_reported_not_hidden(provider, monkeypatch):
    def locked(key):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pending, "get", locked)

    with pytest.raises(ProviderConnectionError, match="database is locked"):
        await provider.get_model_by_hash(SHA)


async def test_the_chosen_version_and_file_are_picked_by_id(table, provider):
    model = lora()
    first = model.versions[0]
    second = first.model_copy(update={
        "id": "133241",
        "base_model": "SDXL 0.9",
        "trigger_words": ["tweak"],
        "images": [Image(url=SAFE_VIDEO, nsfw=False, nsfw_level=1)],
        "files": [
            first.files[0].model_copy(update={"id": "97112", "download_url": DOWNLOAD_URL + "-other"}),
            first.files[0].model_copy(update={"id": "97113", "download_url": DOWNLOAD_URL + "-chosen"}),
        ],
    })
    model = model.model_copy(update={"versions": [first, second]})
    store(info_of(model, version_id="133241", file_id="97113"))
    store(info_of(model, version_id=None, file_id=None), key=OTHER_SHA)

    chosen = await provider.get_model_by_hash(SHA)
    fallback = await provider.get_model_by_hash(OTHER_SHA)

    assert (chosen.provider_version_id, chosen.download_url, chosen.trigger_words, chosen.base_model) == (
        "133241", DOWNLOAD_URL + "-chosen", ["tweak"], "SDXL 0.9",
    )
    assert chosen.media_urls == [SAFE_VIDEO]
    assert (fallback.provider_version_id, fallback.download_url) == ("135867", DOWNLOAD_URL)


async def test_numeric_ids_in_info_still_pick_the_version_and_file(table, provider):
    model = lora()
    second = model.versions[0].model_copy(update={"id": "133241"})
    store(info_of(model.model_copy(update={"versions": [second, model.versions[0]]}), version_id=135867, file_id=99264))

    assert (await provider.get_model_by_hash(SHA)).provider_version_id == "135867"


async def test_a_model_without_a_version_still_gives_its_model_fields(table, provider):
    store(info_of(lora(versions=[])))

    info = await provider.get_model_by_hash(SHA)

    assert (info.provider_model_id, info.name, info.tags) == ("civitai:122359", "Detail Tweaker XL", ["concept", "detailed"])
    assert (info.provider_version_id, info.download_url, info.base_model, info.created_at) == (None, None, None, None)
    assert (info.trigger_words, info.media_urls) == ([], [])
    assert info.extra_data["mb_base_model"] is None


async def test_a_hugging_face_file(table, provider):
    repo = "latent-consistency/lcm-lora-sdxl"
    weights = f"https://huggingface.co/{repo}/resolve/main/pytorch_lora_weights.safetensors"
    sample = f"https://huggingface.co/{repo}/resolve/main/image.png"
    model = ModelDetail(
        source="huggingface",
        id=repo,
        name="lcm-lora-sdxl",
        type="LoRA",
        mb_type=None,
        tags=["lora", "license:openrail++"],
        license=License(summary="openrail++", url=f"https://huggingface.co/{repo}/blob/main/LICENSE"),
        source_url=f"https://huggingface.co/{repo}",
        versions=[
            Version(
                id="main",
                name="main",
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
                images=[Image(url=sample, nsfw=False, nsfw_level=1)],
                files=[
                    File(
                        id="pytorch_lora_weights.safetensors",
                        name="pytorch_lora_weights.safetensors",
                        sha256=OTHER_SHA,
                        primary=True,
                        download_url=weights,
                        auth_provider_id="huggingface",
                    )
                ],
            )
        ],
    )
    store(
        info_of(model, version_id="main", file_id="pytorch_lora_weights.safetensors", mb_type="lora"),
        key=OTHER_SHA,
        source="huggingface",
        source_model_id=repo,
    )

    info = await provider.get_model_by_hash(OTHER_SHA)

    assert (info.provider_model_id, info.provider_version_id) == (f"huggingface:{repo}", "main")
    assert (info.download_url, info.media_urls, info.model_type) == (weights, [sample], "lora")
    assert (info.description, info.created_at, info.nsfw) == (None, None, False)
    assert info.extra_data == {
        "source": "huggingface",
        "source_url": f"https://huggingface.co/{repo}",
        "license": "openrail++",
        "license_url": f"https://huggingface.co/{repo}/blob/main/LICENSE",
        "mb_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "mb_license": "openrail++",
    }
    assert provider.get_model_page_url(info.provider_model_id, info.provider_version_id) == f"https://huggingface.co/{repo}"


async def test_the_type_chosen_for_the_download_wins_over_the_proposed_one(table, provider):
    store(info_of(lora(), mb_type="embedding"))
    store(info_of(lora(), mb_type=None), key=OTHER_SHA)

    assert (await provider.get_model_by_hash(SHA)).model_type == "embedding"
    assert (await provider.get_model_by_hash(OTHER_SHA)).model_type == "lora"


async def test_a_model_without_licence_has_no_licence_values(table, provider):
    store(info_of(lora(license=None)))

    extra = (await provider.get_model_by_hash(SHA)).extra_data

    assert (extra["license"], extra["license_url"], extra["mb_license"]) == (None, None, None)


async def test_a_real_civitai_detail_round_trips(table, provider):
    detail = normalize_detail(body("model_122359"), 2)
    version = next(version for version in detail.versions if version.id == "135867")
    file = next(file for file in version.files if file.primary)
    cut = detail.model_copy(update={"versions": [version.model_copy(update={"files": [file]})]})
    store(info_of(cut, version_id=version.id, file_id=file.id), key=file.sha256.lower())

    info = await provider.get_model_by_hash(file.sha256)

    safe = [image.url for image in version.images if not image.nsfw and image.url]
    assert safe and info.media_urls == safe
    assert (info.provider_model_id, info.provider_version_id, info.name) == ("civitai:122359", "135867", "Detail Tweaker XL")
    assert (info.download_url, info.base_model, info.model_type) == (file.download_url, "SDXL 1.0", "lora")
    assert info.nsfw is True
    assert info.created_at == datetime(2023, 8, 7, 14, 55, 2, 627000, tzinfo=timezone.utc)
    assert info.description and "<" not in info.description
    assert info.extra_data["source_url"] == "https://civitai.red/models/122359"
    assert info.extra_data["mb_license"] == cut.license.summary
    assert provider.get_model_page_url(info.provider_model_id, info.provider_version_id) == (
        "https://civitai.com/models/122359?modelVersionId=135867"
    )


def test_build_model_info_is_usable_on_its_own():
    row = PendingRow(
        sha256=SHA, source="civitai", source_model_id="122359", file_name="add-detail-xl.safetensors",
        requested_by="user-1", assign_to="none", info=info_of(lora()),
    )

    info = build_model_info(row, max_level=1, hide_poi=False)

    assert info.media_urls == [SAFE_IMAGE]
    assert (info.extra_data["mb_base_model"], info.extra_data["mb_license"]) == (
        "SDXL 1.0", "Commercial use allowed, credit required",
    )


@pytest.mark.parametrize(
    "model_id, version_id, url",
    [
        ("civitai:122359", None, "https://civitai.com/models/122359"),
        ("civitai:122359", "135867", "https://civitai.com/models/122359?modelVersionId=135867"),
        ("civitai:122359", 135867, "https://civitai.com/models/122359?modelVersionId=135867"),
        ("civitai:122359", " 135867 ", "https://civitai.com/models/122359?modelVersionId=135867"),
        ("civitai:122359", "", "https://civitai.com/models/122359"),
        ("civitai:122359", "v1.0", "https://civitai.com/models/122359"),
        ("civitai:122359", "1&x=2", "https://civitai.com/models/122359"),
        ("civitai:122359", True, "https://civitai.com/models/122359"),
        ("huggingface:latent-consistency/lcm-lora-sdxl", None, "https://huggingface.co/latent-consistency/lcm-lora-sdxl"),
        ("huggingface:latent-consistency/lcm-lora-sdxl", "main", "https://huggingface.co/latent-consistency/lcm-lora-sdxl"),
        ("huggingface:gpt2", None, "https://huggingface.co/gpt2"),
        ("huggingface:stabilityai/stable-diffusion-3.5-large", None, "https://huggingface.co/stabilityai/stable-diffusion-3.5-large"),
        ("openmodeldb:4x-UltraSharp", None, "https://openmodeldb.info/models/4x-UltraSharp"),
        ("openmodeldb:4x-Nomos8kSC", "4x-Nomos8kSC", "https://openmodeldb.info/models/4x-Nomos8kSC"),
        ("civarchive:78291", "83095", "https://civarchive.com/models/78291?modelVersionId=83095"),
        ("civarchive:78291", None, "https://civarchive.com/models/78291"),
        ("civarchive:78291", "1&x=2", "https://civarchive.com/models/78291"),
    ],
)
def test_the_mirror_link_points_at_the_source_page(provider, model_id, version_id, url):
    assert provider.get_model_page_url(model_id, version_id) == url
    assert model_page_url(model_id, version_id) == url


@pytest.mark.parametrize(
    "model_id",
    [
        None,
        122359,
        "",
        "122359",
        "civitai",
        "civitai:",
        ":122359",
        "civitai:abc",
        "civitai:-1",
        "civitai:12 3",
        "civitai:122359/../1",
        "civitai:١٢٣",
        "CivitAI:122359",
        "huggingface:",
        "huggingface:../etc/passwd",
        "huggingface:/abs/path",
        "huggingface:org/repo/extra",
        "huggingface:org/repo?x=1",
        "huggingface:org/re..po",
        "huggingface:org/-repo",
        "openmodeldb:",
        "openmodeldb:../x",
        "openmodeldb:4x x",
        "openmodeldb:org/repo",
        "civarchive:",
        "civarchive:abc",
        "civarchive:12 3",
        "civarchive:١٢٣",
        "unknown:1",
    ],
)
def test_an_unknown_source_or_a_malformed_id_has_no_mirror_link(provider, model_id):
    assert provider.get_model_page_url(model_id, "135867") is None


def test_the_mirror_link_reads_no_database(provider, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("get_model_page_url must not read anything")

    monkeypatch.setattr(pending, "get", unexpected)
    monkeypatch.setattr(pending_provider.plugin_settings, "get_setting", unexpected)

    assert provider.get_model_page_url("civitai:1", "2") == "https://civitai.com/models/1?modelVersionId=2"
