import asyncio
import copy
import json

import pytest

from backend import http as http_module
from backend.cache import cache_key
from pydantic import ValidationError

from backend.http import SourceHttp, SourceInvalidRequest, SourceNotFound, SourceUnavailable
from backend.schema import File, Image, SearchItem, Stats, Version
from backend.security import DOWNLOAD_HOSTS, is_allowed_download_url, sanitize_html
from backend.sources import SearchQuery
from backend.sources import civarchive
from backend.sources.civarchive import (
    BUILD_ID_TTL_SECONDS,
    CivArchiveSource,
    download_for,
    huggingface_download_url,
    normalize_detail,
    normalize_item,
    parse_model_ref,
    parse_query,
)

from .civarchive_fakes import (
    CAPTURED_BUILD_ID,
    HTML_HEADERS,
    JSON_HEADERS,
    SHA_78291,
    SHA_UNKNOWN,
    CivArchiveTransport,
    data_body,
    full_page,
    make_source,
    model,
    props,
    results,
    search_body,
)
from .civitai_fakes import MemoryCache, fake_settings

NEW_BUILD_ID = "newDeployment_2"
LOWRA_MIRROR = "https://huggingface.co/baycon/for_Colab/resolve/main/lowra_v10.safetensors"
TOP_IDS = ["4384", "257749", "6424", "4201", "7808"]


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    return fake_settings(monkeypatch)


@pytest.fixture
def transport():
    return CivArchiveTransport()


@pytest.fixture
def cache():
    return MemoryCache()


@pytest.fixture
def source(transport, cache):
    return make_source(transport, cache)


def _with_model(transport, model_id, version_id, change):
    page = props(f"model_{model_id}_v{version_id}")
    change(page["model"])
    transport.add(f"models/{model_id}.json", (200, JSON_HEADERS, data_body(page)), platform="civitai", modelVersionId=version_id)
    return page


def _search_route(transport, entries, total_hits=1000, **params):
    transport.clear("search.json").add("search.json", (200, JSON_HEADERS, search_body(entries, total_hits)), **params)


async def test_the_build_id_is_read_from_the_home_page_before_the_first_data_request(transport, source):
    await source.search(SearchQuery())

    assert transport.paths() == ["/", "search.json"]
    assert transport.build_ids == [CAPTURED_BUILD_ID]
    assert await source.build_id() == CAPTURED_BUILD_ID
    assert transport.home_fetches == 1


async def test_one_home_page_serves_every_request_after_it(transport, source):
    await source.search(SearchQuery())
    await source.detail("78291:83095")
    await source.hash_page(SHA_78291)
    await source.info()
    await source.detail("122359:135867")

    assert transport.home_fetches == 1
    assert set(transport.build_ids) == {CAPTURED_BUILD_ID}


async def test_the_build_id_is_cached_for_an_hour_so_a_restart_does_not_read_the_home_page(transport, cache, source):
    await source.search(SearchQuery())

    key = cache_key("civarchive", {"build_id": True})
    assert cache.entries[key]["build_id"] == CAPTURED_BUILD_ID
    assert cache.ttls[key] == BUILD_ID_TTL_SECONDS == 3600
    restarted = make_source(transport, cache)
    await restarted.detail("78291:83095")
    assert transport.paths() == ["/", "search.json", "models/78291.json"]


async def test_the_build_id_is_read_again_after_an_hour(transport, cache):
    now = [1_000_000.0]
    source = make_source(transport, cache, clock=lambda: now[0])
    await source.search(SearchQuery())

    now[0] += BUILD_ID_TTL_SECONDS - 1
    await source.detail("78291:83095")
    assert transport.home_fetches == 1
    now[0] += 2
    await source.hash_page(SHA_78291)
    assert transport.home_fetches == 2
    restarted = make_source(transport, cache, clock=lambda: now[0] + BUILD_ID_TTL_SECONDS)
    await restarted.detail("122359:135867")
    assert transport.home_fetches == 3


async def test_a_new_deployment_renews_the_build_id_and_repeats_the_request_once(transport, cache, source):
    await source.search(SearchQuery())
    transport.redeploy(NEW_BUILD_ID)

    detail = await source.detail("78291:83095")

    assert detail.id == "78291"
    assert transport.paths() == ["/", "search.json", "models/78291.json", "/", "models/78291.json"]
    assert transport.build_ids == [CAPTURED_BUILD_ID, CAPTURED_BUILD_ID, NEW_BUILD_ID]
    assert await source.build_id() == NEW_BUILD_ID
    assert cache.entries[cache_key("civarchive", {"build_id": True})]["build_id"] == NEW_BUILD_ID
    await source.detail("122359:135867")
    assert transport.build_ids[-1] == NEW_BUILD_ID and transport.home_fetches == 2


async def test_a_stale_build_id_in_the_cache_is_renewed_after_a_restart(transport, cache, source):
    await source.build_id()
    transport.redeploy(NEW_BUILD_ID)
    restarted = make_source(transport, cache)

    page = await restarted.search(SearchQuery())

    assert [item.id for item in page.items] == TOP_IDS
    assert transport.paths() == ["/", "search.json", "/", "search.json"]
    assert transport.build_ids == [CAPTURED_BUILD_ID, NEW_BUILD_ID]


async def test_a_second_404_is_not_found_for_a_detail(transport, source):
    with pytest.raises(SourceNotFound):
        await source.detail("42:43")

    assert transport.paths() == ["/", "models/42.json", "/", "models/42.json"]


async def test_a_second_404_is_an_empty_search_page(transport, source):
    transport.add("search.json", (404, HTML_HEADERS, "<!DOCTYPE html><title>Error 404</title>"), q="gone")

    page = await source.search(SearchQuery(q="gone"))

    assert (page.items, page.next_cursor) == ([], None)
    assert transport.paths() == ["/", "search.json", "/", "search.json"]


async def test_a_home_page_without_a_build_id_is_unavailable(transport, source):
    transport.build_id = None

    with pytest.raises(SourceUnavailable, match="CivArchive changed its page layout: no build id"):
        await source.search(SearchQuery())

    assert transport.paths() == ["/"]


async def test_a_home_page_that_does_not_answer_is_unavailable(transport, source):
    transport.home = (503, HTML_HEADERS, "<html>Service Unavailable</html>")

    with pytest.raises(SourceUnavailable):
        await source.detail("78291:83095")

    assert transport.paths() == ["/"]


async def test_concurrent_first_requests_read_the_home_page_once(transport, source):
    await asyncio.gather(
        source.search(SearchQuery()),
        source.detail("78291:83095"),
        source.hash_page(SHA_78291),
        source.detail("122359:135867"),
        source.info(),
    )

    assert transport.home_fetches == 1
    assert len(transport.build_ids) == 5


async def test_concurrent_requests_under_a_stale_build_id_renew_it_once(transport, source):
    await source.build_id()
    transport.redeploy(NEW_BUILD_ID)

    details = await asyncio.gather(
        source.detail("78291:83095"), source.detail("122359:135867"), source.hash_page(SHA_78291)
    )

    assert [details[0].id, details[1].id] == ["78291", "122359"]
    assert transport.home_fetches == 2
    assert transport.build_ids.count(CAPTURED_BUILD_ID) == 3 and transport.build_ids.count(NEW_BUILD_ID) == 3


async def test_a_home_page_still_giving_the_stale_id_is_read_once_for_the_requests_that_saw_it(transport, source):
    answers = await asyncio.gather(source.detail("41:1"), source.detail("42:1"), return_exceptions=True)

    assert all(isinstance(answer, SourceNotFound) for answer in answers)
    assert transport.home_fetches == 2
    assert transport.paths().count("models/41.json") == transport.paths().count("models/42.json") == 2


async def test_html_answers_are_never_cached(transport, cache, source):
    await source.search(SearchQuery())
    await source.detail("78291")

    stored = json.dumps(list(cache.entries.values()))
    assert "<html" not in stored and "<!DOCTYPE" not in stored
    assert cache.entries[cache_key("civarchive", {"build_id": True})].keys() == {"build_id", "resolved_at"}


async def test_an_unknown_model_is_not_found_without_renewing_the_build_id(transport, source):
    with pytest.raises(SourceNotFound):
        await source.detail("999999999:1")
    with pytest.raises(SourceNotFound):
        await source.detail("78291:999999999")

    assert transport.home_fetches == 1
    assert transport.paths() == ["/", "models/999999999.json", "models/78291.json"]


async def test_any_other_500_is_unavailable(transport, source):
    transport.add("models/5.json", (500, HTML_HEADERS, "<title>Error 500: Internal Server Error</title>"))

    with pytest.raises(SourceUnavailable) as caught:
        await source.detail("5:6")

    assert caught.value.status == 500
    assert transport.home_fetches == 1


async def test_an_answer_without_page_props_is_unavailable(transport, source):
    transport.add("models/5.json", (200, JSON_HEADERS, {"notFound": True}))

    with pytest.raises(SourceUnavailable, match="cannot read"):
        await source.detail("5:6")


async def test_a_version_of_another_model_is_not_found(transport, source):
    assert model("model_78291_v1")["version"]["model_id"] == 1

    with pytest.raises(SourceNotFound, match="no version 1 of model 78291"):
        await source.detail("78291:1")


@pytest.mark.parametrize("model_id", ["", "abc", "0", "78291:", "78291:x", "78291:0", "-5", "78291/1", "1e3", "x" * 300])
async def test_a_malformed_model_id_is_not_found_without_a_request(transport, source, model_id):
    with pytest.raises(SourceNotFound):
        await source.detail(model_id)

    assert transport.calls == []


async def test_answers_are_cached_without_their_comments_for_their_own_time(transport, cache, source):
    page = props("model_78291_v83095")
    page.update(comments=[{"id": 1, "content": "nice"}], faqSections=[{"q": "?"}])
    transport.add("models/78291.json", (200, JSON_HEADERS, data_body(page)), platform="civitai", modelVersionId=83095)
    await source.search(SearchQuery())
    await source.search(SearchQuery())
    await source.detail("78291:83095")
    await source.detail("78291:83095")
    await source.hash_page(SHA_78291)

    assert transport.paths() == ["/", "search.json", "models/78291.json", "sha256/" + SHA_78291 + ".json"]
    detail_key = cache_key(
        "civarchive", {"path": "models/78291.json", "params": [["modelVersionId", "83095"], ["platform", "civitai"]]}
    )
    search_key = cache_key(
        "civarchive",
        {"path": "search.json", "params": sorted([["kind", "version"], ["page", "1"], ["period", "all"], ["platform", "civitai"], ["rating", "all"], ["sort", "top"]])},
    )
    hash_key = cache_key("civarchive", {"path": f"sha256/{SHA_78291}.json", "params": []})
    assert (cache.ttls[search_key], cache.ttls[detail_key], cache.ttls[hash_key]) == (120, 300, 300)
    assert "model" in cache.entries[detail_key]
    assert "comments" not in cache.entries[detail_key] and "faqSections" not in cache.entries[detail_key]


async def test_requests_are_a_second_apart_one_at_a_time_by_default(transport):
    now = [100.0]
    starts = []

    async def sleep(seconds):
        now[0] += seconds

    async def fetch(url, params):
        starts.append(now[0])
        return await transport(url, params)

    client = SourceHttp(fetch=fetch, clock=lambda: now[0], sleep=sleep)
    source = CivArchiveSource(http_client=client, cache_store=MemoryCache())
    await source.search(SearchQuery())
    await source.detail("78291:83095")

    assert http_module.DEFAULT_LIMITS["civarchive"] == (1.0, 1)
    assert starts == [100.0, 101.0, 102.0]


async def test_the_default_search_asks_for_civitai_versions_of_any_rating(transport, source):
    page = await source.search(SearchQuery())

    assert transport.data_calls("search.json") == [
        {"platform": "civitai", "kind": "version", "rating": "all", "sort": "top", "period": "all", "page": "1"}
    ]
    assert page.source == "civarchive"
    assert [item.id for item in page.items] == TOP_IDS
    assert page.next_cursor is None


async def test_a_safe_query_asks_for_rating_safe(transport, source):
    page = await source.search(SearchQuery(safe=True))

    assert transport.data_calls("search.json")[0]["rating"] == "safe"
    assert [item.id for item in page.items] == [raw["model_id"] for raw in results("search_safe")]


@pytest.mark.parametrize(
    "query, sent, fixture",
    [
        (SearchQuery(q="#anime"), {"tags": "anime"}, "search_tag"),
        (SearchQuery(q="  @TangBohu "), {"username": "TangBohu"}, "search_user"),
        (SearchQuery(q="zzqxjv-no-such-model-0"), {"q": "zzqxjv-no-such-model-0"}, "search_empty"),
        (
            SearchQuery(types=["LORA"], base_models=["SD 1.5"], sort="newest"),
            {"type": "LORA", "base_model": "SD 1.5", "sort": "newest"},
            "search_civitai_lora_sd15_newest",
        ),
    ],
    ids=["tag", "user", "text", "filters"],
)
async def test_the_search_box_and_the_filters(transport, source, query, sent, fixture):
    page = await source.search(query)

    (params,) = transport.data_calls("search.json")
    assert {name: params[name] for name in sent} == sent
    assert (params["platform"], params["kind"]) == ("civitai", "version")
    assert [item.id for item in page.items] == list(dict.fromkeys(raw["model_id"] for raw in results(fixture)))


async def test_types_and_base_models_are_repeated_and_sorted(transport, source):
    _search_route(transport, [], total_hits=0)

    await source.search(SearchQuery(types=["LORA", "Checkpoint", " LORA "], base_models=["SDXL 1.0", "SD 1.5"]))

    (params,) = transport.data_calls("search.json")
    assert (params["type"], params["base_model"]) == (["Checkpoint", "LORA"], ["SD 1.5", "SDXL 1.0"])


@pytest.mark.parametrize(
    "sort, period, sent",
    [
        (None, None, ("top", "all")),
        ("newest", "week", ("newest", "week")),
        ("downloads", "month", ("downloads", "month")),
        ("likes", "year", ("likes", "year")),
        ("updated", "day", ("updated", "day")),
        ("top", "all_time", ("top", "all")),
    ],
)
async def test_sorts_and_periods(transport, source, sort, period, sent):
    _search_route(transport, [], total_hits=0)

    await source.search(SearchQuery(sort=sort, period=period))

    (params,) = transport.data_calls("search.json")
    assert (params["sort"], params["period"]) == sent


@pytest.mark.parametrize(
    "query",
    [
        SearchQuery(sort="most_downloaded"),
        SearchQuery(period="Week"),
        SearchQuery(cursor="abc"),
        SearchQuery(cursor="0"),
        SearchQuery(cursor="-1"),
        SearchQuery(q="https://civarchive.com/models/78291", sort="oldest"),
    ],
    ids=["civitai-sort", "period-case", "cursor-text", "cursor-zero", "cursor-negative", "link-with-bad-sort"],
)
async def test_unknown_sorts_periods_and_cursors_are_invalid(transport, source, query):
    with pytest.raises(SourceInvalidRequest):
        await source.search(query)

    assert transport.calls == []


async def test_the_cursor_is_the_next_page_number(transport, source):
    _search_route(transport, full_page(results("search_civitai_top")), page="3")

    page = await source.search(SearchQuery(cursor="3"))

    assert transport.data_calls("search.json")[0]["page"] == "3"
    assert (len(page.items), page.next_cursor) == (50, "4")


@pytest.mark.parametrize(
    "size, total, cursor, expected",
    [
        (50, 1000, None, "2"),
        (50, 1000, "19", "20"),
        (50, 1000, "20", None),
        (50, 100, "2", None),
        (49, 1000, None, None),
        (50, None, "7", "8"),
        (0, 0, None, None),
    ],
    ids=["first", "before-the-cap", "the-cap", "total-reached", "short-page", "no-total", "empty"],
)
async def test_the_next_cursor_rules(transport, source, size, total, cursor, expected):
    entries = full_page(results("search_civitai_top"), size) if size else []
    _search_route(transport, entries, total_hits=total)

    page = await source.search(SearchQuery(cursor=cursor))

    assert page.next_cursor == expected


async def test_one_item_per_model_with_its_newest_version_at_the_models_first_place(transport, source):
    entries = results("search_civitai_top")
    older = entries[4]
    newer = dict(copy.deepcopy(older), version_id="999001", created_at=older["created_at"] + 100, name="EasyNegative v2")
    oldest = dict(copy.deepcopy(older), version_id="1", created_at=older["created_at"] - 100, name="EasyNegative v0")
    _search_route(transport, [entries[0], older, entries[1], newer, oldest])

    page = await source.search(SearchQuery())

    assert [item.id for item in page.items] == ["4384", "7808", "257749"]
    easy = page.items[1]
    assert (easy.name, easy.source_url) == ("EasyNegative v2", "https://civarchive.com/models/7808?modelVersionId=999001")


async def test_results_that_are_not_civitai_versions_are_left_out(transport, source):
    entries = results("search_civitai_top")
    pixai = dict(copy.deepcopy(entries[0]), platform="pixai", model_id="1611246784314739674")
    article = dict(copy.deepcopy(entries[1]), kind="article", model_id=None)
    file = dict(copy.deepcopy(entries[2]), kind="file")
    nameless = {key: value for key, value in entries[3].items() if key != "model_id"}
    _search_route(transport, [pixai, article, file, nameless, entries[4], "not an object"])

    page = await source.search(SearchQuery())

    assert [item.id for item in page.items] == ["7808"]


def test_a_list_result_as_an_item():
    raw = results("search_safe")[0]

    item = normalize_item(raw)

    assert item == SearchItem(
        source="civarchive",
        id="7808",
        name="EasyNegative EasyNegative",
        type="TextualInversion",
        mb_type="embedding",
        base_model="SD 1.5",
        base_models=["SD 1.5"],
        creator="rqdwdw",
        nsfw=False,
        poi=False,
        early_access=False,
        gated=False,
        thumbnail=Image(url=raw["image_url"], nsfw=False, nsfw_level=None, kind="image", id="88246"),
        stats=Stats(downloads=raw["download_count"]),
        source_url="https://civarchive.com/models/7808?modelVersionId=9208",
        primary_sha256=None,
    )


def test_a_list_result_is_nsfw_by_is_nsfw_and_its_thumbnail_with_it():
    dreamshaper = normalize_item(results("search_civitai_top")[0])

    assert dreamshaper.nsfw is True and dreamshaper.thumbnail.nsfw is True
    assert (dreamshaper.type, dreamshaper.mb_type) == ("Checkpoint", "checkpoint")
    assert (dreamshaper.thumbnail.nsfw_level, dreamshaper.thumbnail.blurhash) == (None, None)


def test_a_model_deleted_from_civitai_is_nsfw_in_a_list_whatever_is_nsfw_says():
    entries = results("search_deleted")
    assert all(raw["is_nsfw"] is False and raw["is_deleted"] is True for raw in entries)

    items = [normalize_item(raw) for raw in entries]

    assert all(item.nsfw and item.thumbnail.nsfw for item in items)
    only_dated = dict(entries[0], is_deleted=False)
    assert normalize_item(only_dated).nsfw is True


def test_a_list_result_with_a_video_or_without_media():
    raw = results("search_safe")[0]
    video = dict(raw, image_url=None, video_url="https://c.genur.art/0b1c-video.mp4")
    nothing = dict(raw, image_url=None, video_url=None)
    odd = dict(raw, image_url="javascript:alert(1)")

    assert normalize_item(video).thumbnail.kind == "video"
    assert normalize_item(video).thumbnail.url == "https://c.genur.art/0b1c-video.mp4"
    assert normalize_item(nothing).thumbnail is None
    assert normalize_item(odd).thumbnail is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ("all", "", None, None)),
        ("  ", ("all", "", None, None)),
        ("#anime", ("tag", "anime", None, None)),
        ("#", ("all", "", None, None)),
        ("@TangBohu", ("username", "TangBohu", None, None)),
        ("detail tweaker", ("text", "detail tweaker", None, None)),
        ("https://civarchive.com/models/78291?modelVersionId=83095", ("model", "", 78291, 83095)),
        ("civarchive.com/models/78291", ("model", "", 78291, None)),
        ("https://www.civarchive.com/models/78291/slug", ("model", "", 78291, None)),
        ("https://civitai.com/models/122359/detail-tweaker-xl?modelVersionId=135867", ("model", "", 122359, 135867)),
        ("https://civitai.red/models/122359", ("model", "", 122359, None)),
        ("https://civarchive.com/sha256/" + SHA_78291, ("text", "https://civarchive.com/sha256/" + SHA_78291, None, None)),
        ("https://evil.example/models/78291", ("text", "https://evil.example/models/78291", None, None)),
    ],
)
def test_parse_query(text, expected):
    parsed = parse_query(text)

    assert (parsed.kind, parsed.value, parsed.model_id, parsed.version_id) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("78291", (78291, None)),
        (" 78291 ", (78291, None)),
        ("78291:83095", (78291, 83095)),
        ("78291?modelVersionId=83095", (78291, 83095)),
        ("78291?modelVersionId=", (78291, None)),
        ("https://civarchive.com/models/78291?modelVersionId=83095", (78291, 83095)),
        ("https://civitai.com/models/78291/slug?modelVersionId=83095", (78291, 83095)),
        ("abc", None),
        ("0", None),
        ("78291:0", None),
        ("78291:abc", None),
        ("78291/83095", None),
        (None, None),
    ],
)
def test_parse_model_ref(value, expected):
    assert parse_model_ref(value) == expected


async def test_a_pasted_civarchive_link_is_that_model_and_version(transport, source):
    page = await source.search(SearchQuery(q="https://civarchive.com/models/78291?modelVersionId=83095", types=["VAE"]))

    (item,) = page.items
    assert page.next_cursor is None
    assert (item.id, item.name, item.mb_type, item.base_model) == ("78291:83095", model("model_78291_v83095")["name"], "lora", "SD 1.5")
    assert item.primary_sha256 == SHA_78291
    assert item.thumbnail.id == "936343" and item.thumbnail.blurhash
    assert item.source_url == "https://civarchive.com/models/78291?modelVersionId=83095"
    assert transport.paths() == ["/", "models/78291.json"]


async def test_a_pasted_civitai_link_without_a_version_is_the_latest_one(transport, source):
    page = await source.search(SearchQuery(q="https://civitai.com/models/78291/tangbohu-saturation-tweaker-lora"))

    assert [item.id for item in page.items] == ["78291"]
    assert [call.get("modelVersionId") for call in transport.data_calls("models/78291.json")] == [None, "83095"]


async def test_a_pasted_link_to_a_model_civarchive_does_not_know_finds_nothing(transport, source):
    page = await source.search(SearchQuery(q="https://civarchive.com/models/999999999?modelVersionId=1"))

    assert (page.items, page.next_cursor) == ([], None)


async def test_a_pasted_link_has_no_second_page(transport, source):
    page = await source.search(SearchQuery(q="https://civarchive.com/models/78291", cursor="2"))

    assert (page.items, page.next_cursor) == ([], None)
    assert transport.calls == []


async def test_info_lists_the_civitai_types_and_base_models_and_the_notice(transport, cache, source):
    info = await source.info()
    again = await source.info()

    options = props("search_filters")["filterOptions"]
    assert (info.id, info.label) == ("civarchive", "CivArchive")
    assert [option.value for option in info.types] == options["modelTypes"] and len(info.types) == 22
    assert all(option.label == option.value for option in info.types + info.base_models)
    assert [option.value for option in info.base_models] == options["baseModels"] and len(info.base_models) == 104
    assert [(option.value, option.label) for option in info.sorts] == [
        ("top", "Top"),
        ("newest", "Newest"),
        ("downloads", "Most downloaded"),
        ("likes", "Most liked"),
        ("updated", "Recently updated"),
    ]
    assert [(option.value, option.label) for option in info.periods] == [
        ("all_time", "All time"),
        ("year", "Past year"),
        ("month", "Past month"),
        ("week", "Past week"),
        ("day", "Past day"),
    ]
    assert info.search_help == "Text, #tag, @user, or a civarchive.com / civitai.com model link"
    assert info.notice == (
        "Unofficial source: civarchive.com is a community index of CivitAI, not an official API. "
        "Deleted models may be unavailable, or come from third-party mirrors."
    )
    assert (info.types_label, info.base_models_label, info.base_model_label) == ("Types", "Base model", "Base model")
    assert again == info
    assert transport.data_calls("search.json") == [{"platform": "civitai"}]
    assert cache.ttls[cache_key("civarchive", {"path": "search.json", "params": [["platform", "civitai"]]})] == 3600


async def test_info_without_civarchive_still_lists_the_source_and_waits_before_asking_again(transport):
    now = [1000.0]
    transport.home = (503, HTML_HEADERS, "<html>Service Unavailable</html>")
    source = make_source(transport, clock=lambda: now[0])

    info = await source.info()
    await source.info()

    assert (info.types, info.base_models) == ([], [])
    assert info.sorts and info.periods and info.notice
    assert transport.home_fetches == 1
    now[0] += 61
    transport.home = "index"
    assert len((await source.info()).types) == 22
    assert transport.home_fetches == 2


async def test_a_detail_with_its_version(transport, source):
    detail = await source.detail("78291:83095")

    raw = model("model_78291_v83095")
    assert (detail.source, detail.id, detail.name, detail.type, detail.mb_type) == ("civarchive", "78291", raw["name"], "LORA", "lora")
    assert (detail.creator, detail.tags, detail.license) == ("TangBohu", ["style"], None)
    assert detail.source_url == "https://civarchive.com/models/78291?modelVersionId=83095"
    assert (raw["nsfw_level"], raw["is_nsfw"]) == (3, True)
    assert (detail.nsfw, detail.poi, detail.gated) == (False, False, False)
    assert detail.stats == Stats(downloads=raw["download_count"])
    assert detail.description_html == sanitize_html(raw["description"])
    (version,) = detail.versions
    images = raw["version"]["images"]
    assert version == Version(
        id="83095",
        name="v1.0",
        base_model="SD 1.5",
        trigger_words=[],
        created_at="2023-05-27T20:28:45.027Z",
        early_access=False,
        nsfw=False,
        partial=False,
        images=[
            Image(
                url=image["url"],
                blurhash=image["hash"],
                nsfw=False,
                nsfw_level=1,
                width=image["width"],
                height=image["height"],
                kind="image",
                poi=False,
                id=str(image["id"]),
            )
            for image in images
        ],
        files=[
            File(
                id="56837",
                name="tangbohu_saturation_v1.0.safetensors",
                size_bytes=27215728,
                sha256=SHA_78291,
                format="SafeTensor",
                precision=None,
                file_type="Model",
                primary=True,
                download_url="https://civitai.com/api/download/models/83095",
                auth_provider_id="civitai",
            )
        ],
    )
    assert is_allowed_download_url("civarchive", version.files[0].download_url)


async def test_a_detail_without_a_version_follows_the_redirect_to_the_latest(transport, source):
    detail = await source.detail("78291")

    assert [call.get("modelVersionId") for call in transport.data_calls("models/78291.json")] == [None, "83095"]
    assert [version.id for version in detail.versions] == ["83095"]
    assert detail.source_url == "https://civarchive.com/models/78291?modelVersionId=83095"


async def test_a_detail_id_may_name_its_version_in_every_form(transport, source):
    for model_id in ("78291:83095", "78291?modelVersionId=83095", "https://civarchive.com/models/78291?modelVersionId=83095"):
        assert (await source.detail(model_id)).versions[0].id == "83095"

    assert [call["modelVersionId"] for call in transport.data_calls("models/78291.json")] == ["83095"]


async def test_a_redirect_elsewhere_is_not_found(transport, source):
    transport.add("models/7.json", (200, JSON_HEADERS, data_body({"__N_REDIRECT": "/", "__N_REDIRECT_STATUS": 307})))

    with pytest.raises(SourceNotFound):
        await source.detail("7")


async def test_the_shown_version_comes_first_and_the_others_have_no_files_or_images(transport, source):
    tweaker = await source.detail("122359:135867")
    lcm = await source.detail("195519:225222")

    assert [version.id for version in tweaker.versions] == ["135867", "133241"]
    listed = model("model_122359_v135867")["versions"][1]
    assert tweaker.versions[1] == Version(
        id="133241", name="v0.3", created_at=listed["created_at"], nsfw=True, partial=True
    )
    assert tweaker.versions[0].partial is False
    assert [version.id for version in lcm.versions] == ["225222", "424720", "424706", "219782"]
    assert all(version.partial and version.files == [] and version.images == [] for version in lcm.versions[1:])
    assert [version.name for version in lcm.versions] == [
        "LCM for SD 1.5", "LCM for SDXL (Latest)", "LCM for SD 1.5", "LCM for SDXL",
    ]


@pytest.mark.parametrize(
    "model_id", ["78291:83095", "122359:135867", "298188:334947", "195519:225222", "48139:52753", "1879:2026"]
)
async def test_only_the_version_asked_for_is_whole(transport, source, model_id):
    detail = await source.detail(model_id)

    shown, *others = detail.versions
    assert (shown.id, shown.partial) == (model_id.split(":")[1], False)
    assert len(others) == len(model(f"model_{model_id.replace(':', '_v')}")["versions"]) - 1
    assert all(version.partial and version.files == [] and version.images == [] for version in others)
    assert all(version.trigger_words == [] and version.base_model is None for version in others)


def test_a_partial_version_never_carries_files_or_images():
    file = File(id="1", name="x.safetensors", download_url="https://civitai.com/api/download/models/1")
    image = Image(url="https://c.genur.art/x_small.webp", nsfw_level=1)

    assert Version(id="1", name="v1", partial=True).files == []
    assert Version(id="1", name="v1", files=[file], images=[image]).partial is False
    for content in ({"files": [file]}, {"images": [image]}):
        with pytest.raises(ValidationError, match="a partial version carries no files or images"):
            Version(id="1", name="v1", partial=True, **content)


async def test_nsfw_by_the_model_bitmask(transport, source):
    raw = model("model_122359_v135867")
    assert (raw["nsfw_level"], raw["version"]["nsfw_level"]) == (5, 1)

    at_pg13 = await source.detail("122359:135867", max_level=2)
    at_r = await source.detail("122359:135867", max_level=4)

    assert (at_pg13.nsfw, at_pg13.versions[0].nsfw, at_pg13.versions[1].nsfw) == (True, False, True)
    assert not any(image.nsfw for image in at_pg13.versions[0].images)
    assert (at_r.nsfw, at_r.versions[0].nsfw, at_r.versions[1].nsfw) == (False, False, False)


async def test_nsfw_images_by_their_level(transport, source):
    at_pg13 = await source.detail("298188:334947", max_level=2)
    at_r = await source.detail("298188:334947", max_level=4)

    levels = {image.id: image.nsfw_level for image in at_pg13.versions[0].images}
    assert sorted(levels.values()) == [1] * 18 + [2, 4]
    assert [image.id for image in at_pg13.versions[0].images if image.nsfw] == ["6448716"]
    assert levels["6448716"] == 4 and all(image.blurhash for image in at_pg13.versions[0].images)
    assert (at_pg13.nsfw, at_pg13.versions[0].nsfw) == (True, True)
    assert not any(image.nsfw for image in at_r.versions[0].images)
    assert (at_r.nsfw, at_r.versions[0].nsfw) == (False, False)


async def test_images_without_a_level_are_nsfw_and_so_is_a_model_with_no_safe_image(transport, source):
    for max_level in (2, 8):
        detail = await source.detail("48139:52753", max_level=max_level)

        images = detail.versions[0].images
        assert len(images) == 10 and all(image.nsfw and image.nsfw_level is None and image.blurhash for image in images)
        assert (detail.nsfw, detail.versions[0].nsfw) == (True, True)


async def test_an_image_flagged_minor_is_dropped_and_one_of_a_real_person_makes_the_model_poi(transport, source):
    def change(raw):
        raw["version"]["images"][0]["minor"] = True
        raw["version"]["images"][1]["poi"] = True

    _with_model(transport, 78291, 83095, change)

    detail = await source.detail("78291:83095")

    images = detail.versions[0].images
    assert [image.id for image in images] == ["936312", "936357"]
    assert [image.poi for image in images] == [True, False]
    assert detail.poi is True


async def test_previews_without_an_id_type_or_level(transport, source):
    detail = await source.detail("1879:2026")
    again = await make_source(CivArchiveTransport()).detail("1879:2026")

    images = detail.versions[0].images
    assert len(images) == 8 and all(image.kind == "video" for image in images)
    assert all(len(image.id) == 16 for image in images) and len({image.id for image in images}) == 8
    assert [image.id for image in images] == [image.id for image in again.versions[0].images]
    assert (detail.type, detail.mb_type, detail.versions[0].trigger_words) == ("TextualInversion", "embedding", ["HeartArt"])


def test_an_image_id_comes_from_the_civitai_url_when_the_answer_has_none():
    image = civarchive.normalize_image(
        {"url": "https://imagecache.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/4ec2/width=450/80150", "hash": "U9AJD,xZ0g0g", "nsfwLevel": None},
        2,
    )

    assert (image.id, image.kind, image.nsfw) == ("80150", "image", True)
    assert civarchive.normalize_image({"url": "http://x.example/a.jpg", "nsfwLevel": 1}, 2) is None
    assert civarchive.normalize_image("not an image", 2) is None


async def test_a_file_on_civitai_under_another_version_keeps_its_civitai_url(transport, source):
    detail = await source.detail("195519:225222")

    (file,) = detail.versions[0].files
    assert (file.download_url, file.auth_provider_id) == ("https://civitai.com/api/download/models/424706", "civitai")
    assert file.sha256 == "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4"
    assert detail.description_html.endswith("<p>This version was deleted from CivitAI on 2025-04-24.</p>")


async def test_a_model_deleted_from_civitai_downloads_from_its_first_hugging_face_mirror(transport, source):
    detail = await source.detail("48139:52753")

    (file,) = detail.versions[0].files
    assert (file.name, file.download_url, file.auth_provider_id, file.primary) == (
        "LowRa.safetensors", LOWRA_MIRROR, "huggingface", True,
    )
    assert (detail.mb_type, detail.gated, detail.versions[0].trigger_words) == ("lora", False, ["dark theme"])
    assert is_allowed_download_url("civarchive", file.download_url)
    assert detail.description_html.endswith(
        "<p>Deleted from CivitAI on 2025-08-07: its files come from Hugging Face mirrors uploaded by other people.</p>"
    )


async def test_a_model_whose_files_are_only_on_other_sites_is_not_downloadable(transport, source):
    detail = await source.detail("298188:334947")

    assert (detail.versions[0].files, detail.mb_type) == ([], None)
    assert detail.description_html.endswith(
        "<p>Deleted from CivitAI on 2025-04-24: none of its files is on a host this browser downloads from.</p>"
    )


async def test_gated_mirrors_come_last_and_mark_the_model(transport, source):
    def all_gated(raw):
        for mirror in raw["version"]["files"][0]["mirrors"]:
            mirror["is_gated"] = mirror["source"] == "huggingface"

    def first_gated(raw):
        raw["version"]["files"][0]["mirrors"][1]["is_gated"] = True

    _with_model(transport, 48139, 52753, all_gated)
    gated = await source.detail("48139:52753")
    _with_model(transport, 48139, 52753, first_gated)
    mixed = await make_source(transport).detail("48139:52753")

    assert (gated.versions[0].files[0].download_url, gated.gated) == (LOWRA_MIRROR, True)
    assert (mixed.versions[0].files[0].download_url, mixed.gated) == (
        "https://huggingface.co/TenAI/sd/resolve/main/models/Lora/lowra_v10.safetensors", False,
    )


async def test_a_mirror_url_with_spaces_or_other_scripts_is_percent_encoded(transport, source):
    def only_odd_mirrors(raw):
        mirrors = raw["version"]["files"][0]["mirrors"]
        mirrors[0]["deleted_at"] = "2026-01-01T00:00:00.000Z"
        del mirrors[1]

    _with_model(transport, 78291, 83095, only_odd_mirrors)

    detail = await source.detail("78291:83095")

    (file,) = detail.versions[0].files
    assert file.download_url == "https://huggingface.co/QEEWDFHGFH/052723/resolve/main/tangbohu_saturation_v1.0%20(1).safetensors"
    assert file.auth_provider_id == "huggingface" and is_allowed_download_url("civarchive", file.download_url)
    assert huggingface_download_url(
        "https://huggingface.co/tim9292654/lora/resolve/main/细化/tangbohu_saturation_v1.0.safetensors"
    ) == "https://huggingface.co/tim9292654/lora/resolve/main/%E7%BB%86%E5%8C%96/tangbohu_saturation_v1.0.safetensors"
    assert huggingface_download_url("https://huggingface.co/a/b/resolve/main/x%20y.safetensors") == (
        "https://huggingface.co/a/b/resolve/main/x%20y.safetensors"
    )


async def test_a_paid_civitai_version_is_early_access(transport, source):
    _with_model(transport, 78291, 83095, lambda raw: raw["version"]["files"][0]["mirrors"][0].update(is_paid=True))

    detail = await source.detail("78291:83095")
    page = await source.search(SearchQuery(q="civarchive.com/models/78291?modelVersionId=83095"))

    assert detail.versions[0].early_access is True
    assert detail.versions[0].files[0].auth_provider_id == "civitai"
    assert page.items[0].early_access is True


def _file(mirrors, **extra):
    return {"id": 1, "name": "x.safetensors", "download_url": "https://civitai.com/api/download/models/9", "mirrors": mirrors, **extra}


def _mirror(source, url, **extra):
    return {"source": source, "url": url, "deleted_at": None, "is_gated": False, "is_paid": False, **extra}


CIVITAI_9 = "https://civitai.com/api/download/models/9"
HF_FILE = "https://huggingface.co/org/repo/resolve/main/x.safetensors"
DELETED = "2025-04-24T17:42:03.780Z"


@pytest.mark.parametrize(
    "raw_file, served, expected",
    [
        (_file([_mirror("civitai", CIVITAI_9), _mirror("huggingface", HF_FILE)]), False, (CIVITAI_9, "civitai")),
        (_file([_mirror("civitai", CIVITAI_9, deleted_at=DELETED), _mirror("huggingface", HF_FILE)]), True, (HF_FILE, "huggingface")),
        (_file([]), True, (CIVITAI_9, "civitai")),
        (_file([_mirror("huggingface", HF_FILE)]), True, (CIVITAI_9, "civitai")),
        (_file([_mirror("huggingface", HF_FILE)]), False, (HF_FILE, "huggingface")),
        (_file([]), False, None),
        (_file([], download_url="/api/download/models/9"), True, (CIVITAI_9, "civitai")),
        (_file([_mirror("civitai", "https://civitai.red/api/download/models/9")]), False, (CIVITAI_9, "civitai")),
        (_file([_mirror("civitai", "https://civitai.com/models/9")]), False, None),
        (_file([_mirror("civitai", "https://evil.example/api/download/models/9")]), True, None),
        (_file([_mirror("huggingface", "https://huggingface.co/org/repo/blob/main/x.safetensors")]), False, None),
        (_file([_mirror("huggingface", HF_FILE, deleted_at=1747958400)]), False, None),
        (_file([_mirror("huggingface", "http://huggingface.co/org/repo/resolve/main/x.safetensors")]), False, None),
        (_file([_mirror("huggingface", "https://cdn.example/org/repo/resolve/main/x.safetensors")]), False, None),
        (_file([_mirror("civitasbay", "https://civitasbay.org/torrents/3A59"), _mirror("tensorfiles", "https://tensordrive.app/files/fc6d")]), False, None),
        (_file([_mirror("tensorart", "/tensorart/models/631052174908116492/versions/631052174908116492")]), True, (CIVITAI_9, "civitai")),
    ],
    ids=[
        "civitai-mirror", "civitai-mirror-deleted", "own-url", "own-url-before-hf", "not-served-hf", "nothing",
        "relative-own-url", "red-mirror", "civitai-page", "civitai-mirror-elsewhere", "hf-blob", "hf-deleted",
        "hf-http", "hf-other-host", "torrent-and-tensordrive", "tensorart",
    ],
)
def test_the_download_policy(raw_file, served, expected):
    found = download_for(raw_file, served_by_civitai=served)

    assert (None if found is None else (found.url, found.auth_provider_id)) == expected
    if found is not None:
        assert is_allowed_download_url("civarchive", found.url)


def test_files_without_an_id_or_a_url_are_left_out_and_unusable_values_are_dropped():
    raw_version = {
        "id": 5,
        "name": "v1",
        "files": [
            {"id": 7, "name": "data.zip", "type": "Training Data", "size_kb": 0, "sha256": "", "mirrors": [], "download_url": CIVITAI_9 + "?type=Training%20Data"},
            {"name": "no-id.safetensors", "mirrors": [_mirror("civitai", CIVITAI_9)]},
            {"id": 8, "name": "model.gguf", "type": "Model", "size_kb": 1.5, "sha256": "ABC", "mirrors": [_mirror("civitai", CIVITAI_9)]},
            "not a file",
        ],
        "images": [],
    }

    files = civarchive.normalize_version({"id": 4}, raw_version, 2).version.files

    assert [(file.id, file.size_bytes, file.sha256, file.format, file.primary) for file in files] == [
        ("7", None, None, None, False),
        ("8", 1536, None, "GGUF", True),
    ]
    assert files[0].download_url == "https://civitai.com/api/download/models/9?type=Training%20Data"


def test_civarchive_downloads_only_from_civitai_and_hugging_face():
    assert DOWNLOAD_HOSTS["civarchive"] == frozenset({"civitai.com", "huggingface.co"})


def test_a_model_without_a_version_cannot_be_read():
    with pytest.raises(ValueError):
        normalize_detail({"id": 1, "name": "x"}, 2)


async def test_version_detail_loads_the_version_a_download_names(transport, source):
    direct = await source.version_detail("78291", "83095")
    named = await source.version_detail("78291:1", "83095")
    latest = await source.version_detail("78291", "")

    assert [detail.versions[0].id for detail in (direct, named, latest)] == ["83095"] * 3
    assert [call.get("modelVersionId") for call in transport.data_calls("models/78291.json")] == ["83095", None]


async def test_version_detail_of_a_malformed_id_is_not_found(transport, source):
    with pytest.raises(SourceNotFound):
        await source.version_detail("abc", "1")

    assert transport.calls == []


async def test_a_hash_page_lists_the_models_using_the_file(transport, source):
    page = await source.hash_page(f" {SHA_78291.upper()} ")

    assert [raw["id"] for raw in page["models"]] == ["78291"]
    assert page["models"][0]["version"]["files"][0]["sha256"] == SHA_78291
    assert transport.paths() == ["/", f"sha256/{SHA_78291}.json"]


async def test_an_unknown_hash_answers_a_page_without_models(transport, source):
    page = await source.hash_page(SHA_UNKNOWN)

    assert (page["models"], page["files"]) == ([], [])


async def test_a_hash_that_is_not_a_sha256_is_invalid_without_a_request(transport, source):
    with pytest.raises(SourceInvalidRequest):
        await source.hash_page("e47d")

    assert transport.calls == []
