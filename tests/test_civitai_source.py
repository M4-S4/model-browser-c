import json
import re

import pytest

from backend.http import SourceInvalidRequest, SourceNotFound, SourceUnavailable
from backend.sources import SearchQuery, enabled_sources, get_source, UnknownSource
from backend.sources.civitai import (
    CivitaiSource,
    ParsedQuery,
    license_summary,
    normalize_detail,
    normalize_item,
    parse_ids,
    parse_model_url,
    parse_query,
    type_label,
    version_early_access,
)

from .civitai_fakes import ABSENT, FakeTransport, MemoryCache, body, fake_settings, item, make_source, page_body


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    return fake_settings(monkeypatch)


def _items(name, max_level=2):
    return {raw["id"]: normalize_item(raw, max_level) for raw in body(name)["items"]}


def test_lora_page_items_are_normalised():
    items = _items("models_lora_page1")
    styles = items[264290]

    assert styles.source == "civitai"
    assert styles.id == "264290"
    assert styles.name == "Not Artists Styles for Pony Diffusion V6 XL"
    assert (styles.type, styles.mb_type) == ("LORA", "lora")
    assert styles.base_model == "Pony"
    assert styles.base_models == ["Pony"]
    assert styles.creator == "obuyb404"
    assert styles.stats.model_dump() == {"downloads": 663330, "likes": 26764, "rating": None}
    assert styles.early_access is False
    assert styles.poi is False
    assert styles.installed is None
    assert styles.thumbnail.model_dump() == {
        "url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/7c79113b-a6d7-4e22-a3d4-b06c9bb2cfa4/original=true/64739376.jpeg",
        "blurhash": "U6E:3yEU?K~VMckspKI.0Jr;QRDiFyDiRf?b",
        "nsfw": False,
        "nsfw_level": 1,
        "width": 768,
        "height": 1152,
        "kind": "image",
        "poi": False,
        "id": "64739376",
    }


def test_the_bitmask_makes_a_model_nsfw_despite_its_flag():
    items = _items("models_lora_page1")

    assert items[264290].nsfw is True
    assert items[264290].source_url == "https://civitai.red/models/264290"
    assert items[58390].nsfw is False
    assert items[58390].source_url == "https://civitai.com/models/58390"


def test_flux_items_at_level_29_and_31_are_nsfw_at_pg13():
    flux = _items("models_flux_nsfwlevel", max_level=2)
    page2 = _items("models_lora_page2", max_level=2)

    assert body("models_flux_nsfwlevel")["items"][0]["nsfw"] is False
    assert flux[667086].nsfw is True
    assert page2[200255].nsfw is True
    assert flux[796382].nsfw is True
    assert page2[82098].nsfw is False


@pytest.mark.parametrize(
    "max_level, expected",
    [
        (1, {667086: True, 796382: True, 200255: True, 82098: True}),
        (2, {667086: True, 796382: True, 200255: True, 82098: False}),
        (4, {667086: True, 796382: False, 200255: True, 82098: False}),
        (8, {667086: True, 796382: False, 200255: True, 82098: False}),
    ],
)
def test_which_models_flip_with_the_safe_level(max_level, expected):
    items = {**_items("models_flux_nsfwlevel", max_level), **_items("models_lora_page2", max_level)}

    assert {model_id: items[model_id].nsfw for model_id in expected} == expected


def test_thumbnails_are_safe_images():
    for name in ("models_lora_page1", "models_lora_page2", "models_flux_nsfwlevel", "models_query"):
        for normalised in _items(name).values():
            assert normalised.thumbnail is not None
            assert normalised.thumbnail.nsfw is False


def test_the_thumbnail_skips_nsfw_images_and_falls_back_to_the_first_image():
    raw = item("models_lora_page1", 58390)
    images = raw["modelVersions"][0]["images"]
    images[0]["nsfwLevel"] = 16
    images[1]["nsfwLevel"] = 8

    assert normalize_item(raw, 2).thumbnail.id == "692332"

    images[2]["nsfwLevel"] = 4
    only_nsfw = normalize_item(raw, 2)
    assert only_nsfw.nsfw is True
    assert only_nsfw.thumbnail.id == "692411"
    assert only_nsfw.thumbnail.nsfw is True


def test_an_image_of_unknown_level_is_nsfw():
    raw = item("models_lora_page1", 58390)
    for image in raw["modelVersions"][0]["images"]:
        image["nsfwLevel"] = 0

    normalised = normalize_item(raw, 8)

    assert normalised.thumbnail.nsfw is True
    assert normalised.nsfw is True


def test_a_creator_flagged_model_is_nsfw_whatever_its_images():
    raw = item("models_lora_page1", 58390)
    raw["nsfw"] = True

    assert normalize_item(raw, 8).nsfw is True


def test_videos_and_poi_are_marked():
    raw = item("models_lora_page1", 58390)
    raw["poi"] = True
    raw["modelVersions"][0]["images"][0]["type"] = "video"
    raw["modelVersions"][0]["images"][1]["poi"] = True

    normalised = normalize_item(raw, 2)

    assert normalised.poi is True
    assert normalised.thumbnail.id == "692332"


def _file(file_id, sha256, primary=False):
    return {
        "id": file_id,
        "name": f"file-{file_id}.safetensors",
        "primary": primary,
        "hashes": {"SHA256": sha256} if sha256 else {},
        "downloadUrl": f"https://civitai.com/api/download/models/1?fileId={file_id}",
    }


def test_primary_sha256_is_the_lower_cased_hash_of_the_first_versions_primary_file():
    items = _items("models_lora_page1")

    assert items[264290].primary_sha256 == "df7c757437ef3696e76ee5cc18c063681478639132547b6292b0b4d773814ba5"
    assert items[58390].primary_sha256 == "47aaaf0d2945ca937151d61304946dd229b3f072140b85484bc93e38f2a6e2f7"
    assert items[264290].installed is None


@pytest.mark.parametrize(
    "name", ["models_lora_page1", "models_lora_page2", "models_flux_nsfwlevel", "models_query", "models_ids"]
)
def test_every_recorded_list_item_has_the_hash_its_detail_gives(name):
    for raw in body(name)["items"]:
        listed = normalize_item(raw, 2).primary_sha256

        assert re.fullmatch(r"[0-9a-f]{64}", listed)
        assert listed == normalize_detail(raw, 2).versions[0].files[0].sha256


def test_the_primary_file_is_preferred_to_a_file_listed_before_it():
    raw = item("models_lora_page1", 58390)
    raw["modelVersions"][0]["files"].insert(0, _file(1, "1" * 64))

    assert normalize_item(raw, 2).primary_sha256 == "47aaaf0d2945ca937151d61304946dd229b3f072140b85484bc93e38f2a6e2f7"


def test_without_a_usable_primary_hash_the_first_file_with_one_is_used():
    raw = item("models_lora_page1", 58390)
    files = raw["modelVersions"][0]["files"]
    files[0]["hashes"]["SHA256"] = "not-a-hash"
    files.extend([_file(2, None), _file(3, "AB" * 32), _file(4, "cd" * 32)])

    assert normalize_item(raw, 2).primary_sha256 == "ab" * 32


def test_without_a_primary_file_the_first_file_with_a_hash_is_used():
    raw = item("models_lora_page1", 58390)
    files = raw["modelVersions"][0]["files"]
    files[0]["primary"] = False
    files.insert(0, _file(2, None))

    assert normalize_item(raw, 2).primary_sha256 == "47aaaf0d2945ca937151d61304946dd229b3f072140b85484bc93e38f2a6e2f7"


def test_only_the_listed_first_version_is_read():
    raw = item("models_lora_page1", 264290)
    for file in raw["modelVersions"][0]["files"]:
        file.pop("hashes")

    assert raw["modelVersions"][1]["files"][0]["hashes"]["SHA256"]
    assert normalize_item(raw, 2).primary_sha256 is None


@pytest.mark.parametrize(
    "change",
    [
        lambda version: version.pop("files"),
        lambda version: version.update(files=[]),
        lambda version: version.update(files="not a list"),
        lambda version: version.update(files=["not a file", None, 7]),
        lambda version: [file.pop("hashes") for file in version["files"]],
        lambda version: [file.update(hashes=["SHA256"]) for file in version["files"]],
        lambda version: [file.update(hashes={"AutoV2": "abcdef0123"}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": None}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": 1234}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": "not-a-hash"}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": "ab" * 31}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": "ab" * 33}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": " " + "ab" * 32}) for file in version["files"]],
        lambda version: [file.update(hashes={"SHA256": "ab" * 32 + chr(10)}) for file in version["files"]],
    ],
    ids=[
        "no-files-key",
        "no-files",
        "files-not-a-list",
        "files-not-objects",
        "no-hashes",
        "hashes-not-an-object",
        "no-sha256",
        "sha256-null",
        "sha256-number",
        "sha256-not-hex",
        "sha256-too-short",
        "sha256-too-long",
        "sha256-padded",
        "sha256-newline",
    ],
)
def test_an_item_without_a_usable_file_hash_has_no_primary_sha256(change):
    raw = item("models_lora_page1", 58390)
    change(raw["modelVersions"][0])

    normalised = normalize_item(raw, 2)

    assert normalised.primary_sha256 is None
    assert normalised.installed is None


@pytest.mark.parametrize("versions", [[], None, "not a list", [None, 3]], ids=["empty", "missing", "not-a-list", "not-objects"])
def test_an_item_without_versions_has_no_primary_sha256(versions):
    raw = item("models_lora_page1", 58390)
    if versions is None:
        raw.pop("modelVersions")
    else:
        raw["modelVersions"] = versions

    assert normalize_item(raw, 2).primary_sha256 is None


def test_detail_is_normalised():
    detail = normalize_detail(body("model_122359"), 4)

    assert (detail.id, detail.name, detail.type, detail.mb_type) == ("122359", "Detail Tweaker XL", "LORA", "lora")
    assert detail.creator == "w4r10ck"
    assert detail.tags == ["detailed", "enhancer", "undetailed", "detail", "concept"]
    assert detail.description_html.startswith("<p>Detail tweaker for SDXL.</p><p>Works with weights [-3, 3]</p>")
    assert detail.nsfw is False
    assert detail.source_url == "https://civitai.com/models/122359"
    assert detail.stats.downloads == 457351
    assert [version.id for version in detail.versions] == ["135867", "133241"]
    assert [version.partial for version in detail.versions] == [False, False]

    latest = detail.versions[0]
    assert (latest.name, latest.base_model, latest.created_at) == ("v1.0", "SDXL 1.0", "2023-08-07T14:55:02.627Z")
    assert latest.trigger_words == []
    assert latest.early_access is False
    assert [image.id for image in latest.images] == ["1917130", "1917133", "1917136"]
    assert latest.files[0].model_dump() == {
        "id": "99264",
        "name": "add-detail-xl.safetensors",
        "size_bytes": 228452344,
        "sha256": "0d9bd1b873a7863e128b4672e3e245838858f71469a3cec58123c16c06f83bd7",
        "format": "SafeTensor",
        "precision": None,
        "file_type": "Model",
        "primary": True,
        "download_url": "https://civitai.com/api/download/models/135867?fileId=99264",
        "auth_provider_id": "civitai",
    }


def test_detail_nsfw_at_pg13_follows_the_bitmask_of_the_model_and_each_version():
    detail = normalize_detail(body("model_122359"), 2)

    assert detail.nsfw is True
    assert detail.source_url == "https://civitai.red/models/122359"
    assert [version.nsfw for version in detail.versions] == [False, True]
    assert all(not image.nsfw for version in detail.versions for image in version.images)


def test_file_precision_and_trigger_words():
    page2 = {raw["id"]: raw for raw in body("models_lora_page2")["items"]}
    detail = normalize_detail(page2[82098], 2)
    page1 = normalize_detail(item("models_lora_page1", 264290), 2)

    assert detail.versions[0].files[0].precision == "fp16"
    assert detail.versions[0].files[0].size_bytes == round(9324.0185546875 * 1024)
    assert page1.versions[0].trigger_words == ["abstractionism, brush stroke, traditional media"]
    assert page1.versions[1].trigger_words == ["sketch", "monochrome, greyscale"]


def test_the_description_is_sanitised():
    raw = body("model_122359")
    raw["description"] = '<p onclick="x()">Hi<script>alert(1)</script><img src=x onerror=alert(1)></p>'

    assert normalize_detail(raw, 2).description_html == "<p>Hi</p>"


def test_a_model_without_creator_or_description_still_normalises():
    raw = body("error_404_model")

    detail = normalize_detail(raw, 2)

    assert detail.creator is None
    assert (detail.type, detail.mb_type) == ("Checkpoint", "checkpoint")
    assert detail.versions[0].files[0].format == "PickleTensor"
    assert detail.versions[0].files[0].precision == "fp16"


def test_files_without_a_download_url_are_left_out_and_bad_hashes_dropped():
    raw = body("model_122359")
    files = raw["modelVersions"][0]["files"]
    files.append(dict(files[0], id=2, downloadUrl=None))
    files[0]["hashes"]["SHA256"] = "not-a-hash"

    version = normalize_detail(raw, 2).versions[0]

    assert [file.id for file in version.files] == ["99264"]
    assert version.files[0].sha256 is None


def test_license_from_the_four_flags():
    detail = normalize_detail(body("model_122359"), 2)

    assert detail.license.model_dump() == {
        "summary": (
            "Credit required; Commercial use: selling generated images, generation on CivitAI, "
            "generation services; Merges allowed, with the same permissions"
        ),
        "commercial_use": ["Image", "RentCivit", "Rent"],
        "no_credit": False,
        "derivatives": True,
        "different_license": False,
        "url": None,
    }


def test_license_variants():
    page1 = normalize_detail(item("models_lora_page1", 58390), 2).license
    page2 = normalize_detail(item("models_lora_page2", 200255), 2).license
    flux = normalize_detail(item("models_flux_nsfwlevel", 796382), 2).license

    assert page1.summary == "No credit required; No commercial use; Merges allowed, under any license"
    assert page2.summary == "Credit required; Commercial use: generation on CivitAI; No merges"
    assert flux.commercial_use == ["Image", "RentCivit", "Rent", "Sell", "SellMerge"]
    assert flux.summary.startswith(
        "No credit required; Commercial use: selling generated images, generation on CivitAI, "
        "generation services, selling the model, selling merges"
    )


def test_license_edge_cases():
    raw = body("model_122359")
    raw["allowCommercialUse"] = "None"
    assert normalize_detail(raw, 2).license.commercial_use == []

    for flag in ("allowCommercialUse", "allowNoCredit", "allowDerivatives", "allowDifferentLicense"):
        raw.pop(flag)
    assert normalize_detail(raw, 2).license is None
    assert license_summary(None, None, None, None) == "No license information"
    assert license_summary(["Custom"], True, True, None) == "No credit required; Commercial use: Custom; Merges allowed"


def test_early_access_from_availability_and_paid_access():
    raw = item("models_lora_page1", 58390)
    assert normalize_item(raw, 2).early_access is False

    raw["modelVersions"][0]["availability"] = "EarlyAccess"
    assert normalize_item(raw, 2).early_access is True
    assert normalize_detail(raw, 2).versions[0].early_access is True

    raw = item("models_lora_page1", 58390)
    raw["availability"] = "EarlyAccess"
    assert normalize_item(raw, 2).early_access is True

    raw = item("models_lora_page1", 58390)
    raw["availability"] = "Unsearchable"
    assert normalize_item(raw, 2).early_access is False

    raw["modelVersions"][0]["paidAccess"] = {"price": 500}
    assert normalize_item(raw, 2).early_access is True


def test_early_access_dates():
    assert version_early_access({"earlyAccessEndsAt": "2999-01-01T00:00:00.000Z"}) is True
    assert version_early_access({"earlyAccessEndsAt": "2020-01-01T00:00:00.000Z"}) is False
    assert version_early_access({"earlyAccessDeadline": "not a date"}) is False
    assert version_early_access({"availability": "Public", "paidAccess": None}) is False


def test_only_the_latest_version_decides_the_item_badge():
    raw = item("models_lora_page1", 264290)
    raw["modelVersions"][1]["availability"] = "EarlyAccess"

    assert normalize_item(raw, 2).early_access is False
    assert [version.early_access for version in normalize_detail(raw, 2).versions] == [False, True]


@pytest.mark.parametrize(
    "q, expected",
    [
        ("", ParsedQuery()),
        ("   ", ParsedQuery()),
        ("detail tweaker", ParsedQuery(kind="text", value="detail tweaker")),
        ("#anime", ParsedQuery(kind="tag", value="anime")),
        ("# western art ", ParsedQuery(kind="tag", value="western art")),
        ("#", ParsedQuery()),
        ("@obuyb404", ParsedQuery(kind="username", value="obuyb404")),
        ("@", ParsedQuery()),
        ("$#122359", ParsedQuery(kind="ids", ids=(122359,))),
        ("$#122359, 58390;82098 122359", ParsedQuery(kind="ids", ids=(122359, 58390, 82098))),
        ("https://civitai.com/models/122359/detail-tweaker-xl", ParsedQuery(kind="ids", ids=(122359,))),
        (
            "https://civitai.red/models/122359?modelVersionId=135867",
            ParsedQuery(kind="ids", ids=(122359,), version_id=135867),
        ),
        ("civitai.com/models/122359", ParsedQuery(kind="ids", ids=(122359,))),
        ("https://www.civitai.com/models/122359/", ParsedQuery(kind="ids", ids=(122359,))),
        ("http://CIVITAI.RED/models/122359#gallery", ParsedQuery(kind="ids", ids=(122359,))),
        ("https://evil.example/models/122359", ParsedQuery(kind="text", value="https://evil.example/models/122359")),
        ("https://civitai.com/images/123", ParsedQuery(kind="text", value="https://civitai.com/images/123")),
        ("https://civitai.green/models/122359", ParsedQuery(kind="text", value="https://civitai.green/models/122359")),
    ],
)
def test_parse_query(q, expected):
    assert parse_query(q) == expected


@pytest.mark.parametrize("q", ["$#", "$#abc", "$# , ;"])
def test_ids_syntax_without_ids_is_an_invalid_request(q):
    with pytest.raises(SourceInvalidRequest):
        parse_query(q)


def test_too_many_ids_is_an_invalid_request():
    with pytest.raises(SourceInvalidRequest):
        parse_query("$#" + ",".join(str(n) for n in range(1, 1002)))


def test_parse_helpers():
    assert parse_ids("1, 2,2 x 0 3٣") == (1, 2)
    assert parse_model_url("https://civitai.com/models/12?modelVersionId=x") == (12, None)
    assert parse_model_url("https://civitai.com/models/12/a/b") is None
    assert parse_model_url("ftp://civitai.com/models/12") is None
    assert parse_model_url("https://civitai.com/models/12 extra") is None


def test_type_labels():
    assert [type_label(value) for value in ("LORA", "TextualInversion", "AestheticGradient", "Checkpoint", "MotionModule")] == [
        "LoRA",
        "Embedding",
        "Aesthetic gradient",
        "Checkpoint",
        "Motion module",
    ]


def _source(transport=None, cache=None):
    transport = transport or FakeTransport()
    transport.add("/enums", "enums")
    return make_source(transport, cache), transport


async def test_a_plain_search_sends_the_page_size_plus_20_and_nsfw_true():
    source, transport = _source(FakeTransport().add("/models", "models_lora_page1"))

    page = await source.search(SearchQuery(sort="most_downloaded", period="all_time", types=["LORA"]), max_level=2)

    assert transport.calls_to("/models") == [
        {"limit": "44", "nsfw": "true", "period": "AllTime", "sort": "Most Downloaded", "types": "LORA"}
    ]
    assert [normalised.id for normalised in page.items] == ["264290", "58390"]
    assert page.next_cursor == "409619|21294|82098"


async def test_safe_queries_ask_for_nsfw_false():
    source, transport = _source(FakeTransport().add("/models", "models_lora_page1"))

    await source.search(SearchQuery(safe=True), max_level=2)

    assert transport.calls_to("/models")[0]["nsfw"] == "false"


@pytest.mark.parametrize(
    "q, expected",
    [
        ("detail tweaker", {"query": "detail tweaker"}),
        ("#anime", {"tag": "anime"}),
        ("@obuyb404", {"username": "obuyb404"}),
    ],
)
async def test_the_search_box_syntax_becomes_civitai_parameters(q, expected):
    source, transport = _source(FakeTransport().add("/models", "models_query"))

    await source.search(SearchQuery(q=q, limit=10, types=["LORA", "Checkpoint", "LORA"]), max_level=2)

    sent = transport.calls_to("/models")[0]
    assert {key: sent[key] for key in expected} == expected
    assert sent["limit"] == "30"
    assert sent["types"] == ["Checkpoint", "LORA"]


async def test_cursor_chaining_from_page_1_to_page_2():
    transport = FakeTransport()
    transport.add("/models", "models_lora_page1", cursor=ABSENT)
    transport.add("/models", "models_lora_page2", cursor="409619|21294|82098")
    source, _ = _source(transport)

    first = await source.search(SearchQuery(limit=2), max_level=2)
    second = await source.search(SearchQuery(limit=2, cursor=first.next_cursor), max_level=2)

    assert [normalised.id for normalised in second.items] == ["82098", "200255"]
    assert second.next_cursor == "287938|23768|12597"
    assert [call.get("cursor") for call in transport.calls_to("/models")] == [None, "409619|21294|82098"]


async def test_the_query_search_cursor_is_kept_as_text():
    source, _ = _source(FakeTransport().add("/models", "models_query"))

    page = await source.search(SearchQuery(q="detail tweaker", limit=2), max_level=2)

    assert page.next_cursor == "2"


async def test_base_models_are_sent_only_when_the_selection_is_partial():
    source, transport = _source(FakeTransport().add("/models", "models_flux_nsfwlevel"))
    every_active = body("enums")["ActiveBaseModel"]

    await source.search(SearchQuery(base_models=["Flux.1 D", "SDXL 1.0"]), max_level=2)
    await source.search(SearchQuery(base_models=every_active), max_level=2)

    calls = transport.calls_to("/models")
    assert calls[0]["baseModels"] == ["Flux.1 D", "SDXL 1.0"]
    assert "baseModels" not in calls[1]


async def test_unknown_sorts_and_periods_are_invalid_requests():
    source, transport = _source(FakeTransport().add("/models", "models_query"))

    with pytest.raises(SourceInvalidRequest, match="sort"):
        await source.search(SearchQuery(sort="Most Downloaded"), max_level=2)
    with pytest.raises(SourceInvalidRequest, match="period"):
        await source.search(SearchQuery(period="forever"), max_level=2)
    assert transport.calls == []


async def test_the_cache_key_includes_the_nsfw_parameter():
    cache = MemoryCache()
    source, transport = _source(FakeTransport().add("/models", "models_lora_page1"), cache=cache)

    await source.search(SearchQuery(safe=True), max_level=2)
    await source.search(SearchQuery(safe=False), max_level=2)
    await source.search(SearchQuery(safe=True), max_level=2)
    await source.search(SearchQuery(safe=True), max_level=8)

    assert [call["nsfw"] for call in transport.calls_to("/models")] == ["false", "true"]
    assert len(cache.entries) == 2


async def test_the_api_base_comes_from_the_settings(_settings):
    _settings["civitai_api_base"] = "https://civitai.red/api/v1"
    seen = []

    async def fetch(url, params):
        seen.append(url)
        return 200, {}, b'{"items": [], "metadata": {}}'

    from backend.http import SourceHttp

    source = CivitaiSource(http_client=SourceHttp(fetch=fetch, limits={"civitai": (0, 4)}), cache_store=MemoryCache())
    await source.search(SearchQuery(), max_level=2)

    assert seen == ["https://civitai.red/api/v1/models"]


async def test_the_nsfw_level_setting_is_the_default(_settings):
    _settings["nsfw_max_level"] = 8
    source, _ = _source(FakeTransport().add("/models", "models_flux_nsfwlevel"))

    page = await source.search(SearchQuery())

    assert {normalised.id: normalised.nsfw for normalised in page.items} == {"667086": True, "796382": False}


async def test_upstream_errors_propagate():
    source, _ = _source(FakeTransport().add("/models", "error_400_limit"))

    with pytest.raises(SourceInvalidRequest, match="Too big"):
        await source.search(SearchQuery(), max_level=2)


async def test_ids_are_sent_in_chunks_of_100_with_our_own_cursor():
    ids = list(range(1, 251))
    transport = FakeTransport()
    transport.add("/models", lambda path, params: (200, {}, page_body([])))
    source, _ = _source(transport)
    query = SearchQuery(q="$#" + ",".join(map(str, ids)), types=["LORA"], base_models=["Pony"], period="day", sort="newest")

    cursors = []
    cursor = None
    for _ in range(3):
        page = await source.search(query.model_copy(update={"cursor": cursor}), max_level=2)
        cursors.append(page.next_cursor)
        cursor = page.next_cursor

    lists = [call for call in transport.calls_to("/models")]
    assert cursors == ["ids:1", "ids:2", None]
    assert [call["ids"] for call in lists] == [
        ",".join(map(str, ids[:100])),
        ",".join(map(str, ids[100:200])),
        ",".join(map(str, ids[200:])),
    ]
    for call in lists:
        assert call["limit"] == "100"
        assert call["period"] == "AllTime"
        assert call["sort"] == "Newest"
        assert "types" not in call and "baseModels" not in call and "query" not in call
    singles = [path for path, _ in transport.calls if path.startswith("/models/")]
    assert len(singles) == 30


async def test_a_pasted_link_searches_that_model():
    source, transport = _source(FakeTransport().add("/models", "models_ids"))

    page = await source.search(SearchQuery(q="https://civitai.red/models/122359?modelVersionId=135867"), max_level=4)

    assert transport.calls_to("/models")[0]["ids"] == "122359"
    assert [normalised.id for normalised in page.items] == ["122359"]
    assert page.next_cursor is None


async def test_ids_left_out_of_the_list_are_fetched_one_by_one_in_the_asked_order():
    transport = FakeTransport()
    transport.add("/models", "models_ids")
    transport.add("/models/58390", (200, {}, item("models_lora_page1", 58390)))
    source, _ = _source(transport)

    page = await source.search(SearchQuery(q="$#58390,999999,122359"), max_level=2)

    assert [normalised.id for normalised in page.items] == ["58390", "122359"]
    assert [path for path, _ in transport.calls if path != "/enums"] == ["/models", "/models/58390", "/models/999999"]


async def test_a_model_listed_without_versions_is_fetched_by_id():
    listed = item("models_ids", 122359)
    listed["modelVersions"] = []
    transport = FakeTransport()
    transport.add("/models", (200, {}, page_body([listed])))
    transport.add("/models/122359", "model_122359")
    source, _ = _source(transport)

    page = await source.search(SearchQuery(), max_level=4)

    assert page.items[0].thumbnail is not None
    assert page.items[0].base_model == "SDXL 1.0"
    assert [path for path, _ in transport.calls] == ["/models", "/models/122359"]


async def test_a_refetch_outage_keeps_the_list_as_it_is():
    listed = item("models_ids", 122359)
    listed["modelVersions"] = []
    transport = FakeTransport()
    transport.add("/models", (200, {}, page_body([listed])))
    transport.add("/models/122359", (503, {}, {"error": "down"}))
    source, _ = _source(transport)

    page = await source.search(SearchQuery(), max_level=2)

    assert [normalised.id for normalised in page.items] == ["122359"]
    assert page.items[0].nsfw is True


async def test_archived_and_taken_down_models_are_dropped():
    archived = dict(item("models_lora_page1", 264290), mode="Archived")
    taken_down = dict(item("models_lora_page2", 82098), mode="TakenDown")
    kept = item("models_lora_page1", 58390)
    transport = FakeTransport().add("/models", (200, {}, page_body([archived, taken_down, kept])))
    source, _ = _source(transport)

    page = await source.search(SearchQuery(), max_level=2)

    assert [normalised.id for normalised in page.items] == ["58390"]


async def test_unreadable_and_duplicate_items_are_skipped():
    kept = item("models_lora_page1", 58390)
    transport = FakeTransport().add(
        "/models", (200, {}, page_body([{"name": "no id"}, kept, kept, "not a model", {"id": 5, "modelVersions": "x"}]))
    )
    source, _ = _source(transport)

    page = await source.search(SearchQuery(), max_level=2)

    assert [normalised.id for normalised in page.items] == ["58390", "5"]


def test_fields_of_the_wrong_shape_are_ignored():
    raw = item("models_lora_page1", 58390)
    raw["tags"] = "lora"
    raw["stats"] = None
    raw["creator"] = "someone"
    raw["modelVersions"][0]["trainedWords"] = "add_detail"
    raw["modelVersions"][0]["files"] = {"id": 1}
    raw["modelVersions"][0]["images"].append("not an image")

    detail = normalize_detail(raw, 2)

    assert (detail.tags, detail.creator, detail.stats.downloads) == ([], None, None)
    assert (detail.versions[0].trigger_words, detail.versions[0].files) == ([], [])
    assert len(detail.versions[0].images) == 3


async def test_an_unexpected_answer_is_unavailable():
    source, transport = _source(FakeTransport().add("/models", (200, {}, ["not", "a", "page"])))

    with pytest.raises(SourceUnavailable):
        await source.search(SearchQuery(), max_level=2)
    with pytest.raises(SourceUnavailable):
        await source.search(SearchQuery(q="$#1,2"), max_level=2)
    assert [path for path, _ in transport.calls if path != "/enums"] == ["/models", "/models"]


async def test_detail_fetches_the_model():
    source, transport = _source(FakeTransport().add("/models/122359", "model_122359"))

    detail = await source.detail("122359", max_level=4)

    assert detail.id == "122359"
    assert detail.nsfw is False
    assert transport.calls == [("/models/122359", [])]


@pytest.mark.parametrize("model_id", ["abc", "12a", "-1", "0", "٣", "1" * 13, ""])
async def test_detail_of_a_non_numeric_id_is_not_found_without_asking(model_id):
    source, transport = _source()

    with pytest.raises(SourceNotFound):
        await source.detail(model_id)

    assert transport.calls == []


async def test_detail_of_an_unknown_model_is_not_found():
    source, _ = _source(FakeTransport())

    with pytest.raises(SourceNotFound):
        await source.detail("999999", max_level=2)


@pytest.mark.parametrize("mode", ["Archived", "TakenDown"])
async def test_detail_of_a_removed_model_is_not_found(mode):
    raw = body("model_122359")
    raw["mode"] = mode
    source, _ = _source(FakeTransport().add("/models/122359", (200, {}, raw)))

    with pytest.raises(SourceNotFound):
        await source.detail("122359", max_level=2)


async def test_info_lists_the_enums_and_caches_them_for_a_day():
    cache = MemoryCache()
    source, transport = _source(cache=cache)
    enums = body("enums")

    info = await source.info()
    again = await source.info()

    assert info == again
    assert (info.id, info.label) == ("civitai", "CivitAI")
    assert [option.value for option in info.types] == enums["ModelType"]
    assert info.types[enums["ModelType"].index("LORA")].label == "LoRA"
    assert [option.value for option in info.base_models] == enums["ActiveBaseModel"]
    assert [option.value for option in info.sorts] == [
        "highest_rated", "most_downloaded", "most_liked", "most_discussed", "most_collected", "most_images",
        "newest", "oldest",
    ]
    assert [option.value for option in info.periods] == ["all_time", "year", "month", "week", "day"]
    assert "$#" in info.search_help
    assert [path for path, _ in transport.calls] == ["/enums"]
    assert list(cache.ttls.values()) == [86400]


async def test_info_without_civitai_still_lists_the_source():
    transport = FakeTransport().add("/enums", (502, {}, b"<html>bad gateway</html>"))
    source = make_source(transport)

    info = await source.info()

    assert (info.id, info.types, info.base_models) == ("civitai", [], [])
    assert info.sorts


def test_the_registry_has_civarchive_only_while_its_setting_is_on(_settings):
    assert [source.id for source in enabled_sources()] == ["civitai", "huggingface", "openmodeldb"]
    assert get_source("civitai") is get_source("civitai")
    assert get_source("openmodeldb") is get_source("openmodeldb")
    for unknown in ("civarchive", "nope"):
        with pytest.raises(UnknownSource):
            get_source(unknown)

    _settings["enable_civarchive"] = True

    assert [source.id for source in enabled_sources()] == ["civitai", "huggingface", "openmodeldb", "civarchive"]
    assert get_source("civarchive") is get_source("civarchive")
    with pytest.raises(UnknownSource):
        get_source("nope")


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://civitai.red/api/download/models/135867?type=Model&format=SafeTensor",
         "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"),
        ("https://www.civitai.red/api/download/models/1", "https://civitai.com/api/download/models/1"),
        ("https://CIVITAI.RED/api/download/models/1", "https://civitai.com/api/download/models/1"),
        ("https://civitai.com/api/download/models/1", "https://civitai.com/api/download/models/1"),
        ("https://civitai.red:8443/api/download/models/1", "https://civitai.red:8443/api/download/models/1"),
        ("https://user@civitai.red/api/download/models/1", "https://user@civitai.red/api/download/models/1"),
        ("https://civitai.red.evil.example/x", "https://civitai.red.evil.example/x"),
        ("https://[broken/x", "https://[broken/x"),
    ],
)
def test_civitai_red_download_urls_move_to_civitai_com(url, expected):
    from backend.sources.civitai import com_download_url

    assert com_download_url(url) == expected


async def test_a_red_api_base_still_gives_downloads_the_allowlist_and_the_provider_accept(_settings):
    from backend.http import SourceHttp
    from backend.security import is_allowed_download_url

    _settings["civitai_api_base"] = "https://civitai.red/api/v1"
    raw = body("model_122359")
    for version in raw["modelVersions"]:
        for file in version["files"]:
            file["downloadUrl"] = file["downloadUrl"].replace("https://civitai.com/", "https://civitai.red/")
    seen = []

    async def fetch(url, params):
        seen.append(url)
        return 200, {}, json.dumps(raw).encode("utf-8")

    source = CivitaiSource(http_client=SourceHttp(fetch=fetch, limits={"civitai": (0, 4)}), cache_store=MemoryCache())
    detail = await source.detail("122359", max_level=2)

    files = [file for version in detail.versions for file in version.files]
    assert seen == ["https://civitai.red/api/v1/models/122359"]
    assert files and all(file.download_url.startswith("https://civitai.com/api/download/models/") for file in files)
    assert all(is_allowed_download_url("civitai", file.download_url) for file in files)
    assert {file.auth_provider_id for file in files} == {"civitai"}
    assert files[0].download_url == "https://civitai.com/api/download/models/135867?fileId=99264"
