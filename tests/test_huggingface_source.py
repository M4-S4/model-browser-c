import re
import sys
import types


import pytest

from backend.http import SourceInvalidRequest, SourceNotFound, SourceRateLimited, SourceUnavailable
from backend.security import is_allowed_download_url
from backend.sources import SearchQuery, get_source
from backend.sources import huggingface as hf
from backend.sources.huggingface import (
    HuggingFaceSource,
    ParsedQuery,
    base_models_from_tags,
    display_tags,
    file_precision,
    file_type_from_path,
    is_gated,
    license_from,
    next_cursor_from_link,
    normalize_detail,
    normalize_item,
    parse_query,
    readme_image_url,
    strip_frontmatter,
    weight_files,
)

from .civitai_fakes import ABSENT, MemoryCache, fake_settings
from .huggingface_fakes import REPOS, HuggingFaceTransport, body, item, link_header, load, make_source
from .huggingface_fakes import cursor_of as _cursor

_SANITISED_TAGS = {"p", "br", "b", "strong", "i", "em", "u", "s", "ul", "ol", "li", "blockquote", "code", "pre", "hr", "a", "h3", "h4"}

LCM = REPOS["lcm_lora_sdxl"]
FLUX = REPOS["flux_dev_gated"]
SDXL = REPOS["sdxl_base"]
ESRGAN = REPOS["real_esrgan"]
SD35 = REPOS["sd35_large"]


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    return fake_settings(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_render_memo():
    hf._render_memo.clear()
    yield
    hf._render_memo.clear()


def _items(name):
    return {raw["id"]: normalize_item(raw) for raw in body(name)}


def _detail(key, readme=None, **kwargs):
    raw = body(f"model_{key}")
    tree = body(f"tree_{key}")
    return normalize_detail(raw, tree, readme, **kwargs)


def _lcm_readme():
    return body("readme_lcm_lora_sdxl")


def _source(transport=None, cache=None):
    transport = transport or HuggingFaceTransport()
    return make_source(transport, cache), transport


def _demo_repo(transport, *, tags=(), tree=(), readme=None, repo="org/demo"):
    raw = body("model_lcm_lora_sdxl")
    raw["id"] = raw["modelId"] = repo
    raw["tags"] = raw["tags"] + list(tags)
    transport.add(f"/api/models/{repo}", (200, {}, raw))
    transport.add(f"/api/models/{repo}/tree/main", (200, {}, list(tree)))
    if readme is not None:
        transport.add(f"/{repo}/raw/main/README.md", (200, {"Content-Type": "text/plain; charset=utf-8"}, readme))
    return raw


def _file(path, size=10, lfs_oid=None):
    entry = {"type": "file", "oid": "0" * 40, "size": size, "path": path}
    if lfs_oid:
        entry["lfs"] = {"oid": lfs_oid, "size": size, "pointerSize": 134}
    return entry


def test_search_items_are_normalised():
    items = _items("models_search_lora")
    outfit = items["sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA"]

    assert list(items) == [
        "sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA",
        "sample-lab/MiniMax-H3-Character-Sample-LoRA",
        "pablodawson/MiniMax-H3-360-Orbit-LoRA",
    ]
    assert outfit.model_dump() == {
        "source": "huggingface",
        "id": "sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA",
        "name": "sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA",
        "type": "image-to-image",
        "mb_type": "lora",
        "base_model": "Qwen/Qwen-Image-2.1",
        "base_models": ["Qwen/Qwen-Image-2.1"],
        "creator": "sample-lab",
        "nsfw": False,
        "poi": False,
        "early_access": False,
        "gated": False,
        "thumbnail": None,
        "stats": {"downloads": 0, "likes": 116, "rating": None},
        "source_url": "https://huggingface.co/sample-lab/Qwen-Image-2.1-Outfit-Sample-Consistency-LoRA",
        "primary_sha256": None,
        "installed": None,
    }
    assert [(found.type, found.mb_type) for found in items.values()] == [
        ("image-to-image", "lora"),
        ("video-to-video", "lora"),
        ("image-text-to-video", "lora"),
    ]
    character = items["sample-lab/MiniMax-H3-Character-Sample-LoRA"]
    assert (character.creator, character.stats.downloads, character.base_models) == (
        "sample-lab", 40640, ["Comfy-Org/MiniMax-H3"],
    )
    [raw_character] = [raw for raw in body("models_search_lora") if raw["id"] == character.id]
    assert "dataset:sample-lab/H3-Character-Sample-v1" in display_tags(raw_character["tags"])


def test_list_answers_carry_no_file_hash():
    for name in ("models_search_lora", "models_search_lora_page2", "models_text_to_image_top", "models_author"):
        for normalised in _items(name).values():
            assert normalised.primary_sha256 is None
            assert normalised.installed is None


def test_license_base_model_and_region_tags_are_not_listed():
    for name in ("models_search_lora", "models_search_lora_page2", "models_text_to_image_top", "models_author"):
        for raw in body(name):
            assert not [tag for tag in display_tags(raw["tags"]) if tag.startswith(("license:", "base_model:", "region:"))]


def test_gated_comes_as_a_string():
    page2 = _items("models_search_lora_page2")

    assert body("models_search_lora_page2")[1]["gated"] == "auto"
    assert page2["Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"].gated is True
    assert page2["akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA"].gated is False


@pytest.mark.parametrize(
    "value, expected",
    [("auto", True), ("manual", True), (True, True), (False, False), (None, False), ("false", False), ("", False), (0, False)],
)
def test_is_gated(value, expected):
    assert is_gated(value) is expected


def test_the_not_for_all_audiences_tag_marks_nsfw():
    items = _items("models_filter_nsfw_tag")

    assert {repo: found.nsfw for repo, found in items.items()} == {
        "sample-org/Sample-Q-Mature-Qwen-Image-2.1": True,
        "sample-labs/Flux-Sample-Mature": True,
    }
    assert all(not found.nsfw for found in _items("models_search_lora").values())
    assert items["sample-org/Sample-Q-Mature-Qwen-Image-2.1"].base_models == ["Qwen/Qwen-Image-2.1"]
    assert items["sample-labs/Flux-Sample-Mature"].base_models == ["black-forest-labs/FLUX.1-dev"]


def test_types_proposed_for_the_text_to_image_top():
    items = _items("models_text_to_image_top")

    assert {repo: (found.type, found.mb_type) for repo, found in items.items()} == {
        "stabilityai/stable-diffusion-xl-base-1.0": ("text-to-image", "checkpoint"),
        "sample-quant/Qwen-Image-2.1-Sample-GGUF": ("text-to-image", "diffusion_model"),
        "stable-diffusion-v1-5/stable-diffusion-v1-5": ("text-to-image", None),
    }


def test_lora_tags_of_language_models_are_not_potionui_loras():
    items = _items("models_filter_lora_tag")

    assert {repo: (found.type, found.mb_type) for repo, found in items.items()} == {
        "autotrust/JEV-27B-VL": ("image-text-to-text", None),
        "Edge0/Edge0-35B-A3B-preview": ("text-generation", None),
        "autotrust/GEV-26B-Decide": ("text-classification", None),
    }


def test_author_listing():
    items = _items("models_author")

    assert {repo: found.mb_type for repo, found in items.items()} == {
        "latent-consistency/lcm-lora-sdv1-5": "lora",
        "latent-consistency/lcm-sdxl": "checkpoint",
        "latent-consistency/lcm-lora-sdxl": "lora",
    }
    assert all(found.creator == "latent-consistency" for found in items.values())


def test_the_type_falls_back_to_kind_tags_then_the_library():
    raw = item("models_search_lora", "sample-lab/MiniMax-H3-Character-Sample-LoRA")
    raw.pop("pipeline_tag")
    assert normalize_item(raw).type == "lora"

    raw["tags"] = ["comfyui"]
    assert normalize_item(raw).type == "diffusion-single-file"

    raw.pop("library_name")
    assert normalize_item(raw).type == "model"


def test_unreadable_items_are_skipped_and_duplicates_listed_once():
    kept = item("models_author", LCM)
    items = HuggingFaceSource._normalise_items([{"id": "no/../repo"}, {"name": "no id"}, "x", kept, kept, {"id": 3}])

    assert [found.id for found in items] == [LCM]


def test_base_models_and_display_tags():
    tags = [
        "lora",
        "base_model:stabilityai/sdxl",
        "base_model:adapter:stabilityai/sdxl",
        "base_model:quantized:Qwen/Qwen-Image",
        "base_model:merge:a/b",
        "base_model:",
        "license:mit",
        "region:us",
        "arxiv:1234",
    ]

    assert base_models_from_tags(tags) == ["stabilityai/sdxl", "Qwen/Qwen-Image", "a/b"]
    assert display_tags(tags) == ["lora", "arxiv:1234"]


@pytest.mark.parametrize(
    "q, expected",
    [
        ("", ParsedQuery()),
        ("   ", ParsedQuery()),
        ("lora", ParsedQuery(kind="text", value="lora")),
        ("stable diffusion", ParsedQuery(kind="text", value="stable diffusion")),
        ("@latent-consistency", ParsedQuery(kind="author", value="latent-consistency")),
        ("@", ParsedQuery()),
        ("#lora", ParsedQuery(kind="tag", value="lora")),
        ("# not-for-all-audiences ", ParsedQuery(kind="tag", value="not-for-all-audiences")),
        ("#", ParsedQuery()),
        ("latent-consistency/lcm-lora-sdxl", ParsedQuery(kind="repo", value="latent-consistency/lcm-lora-sdxl")),
        ("black-forest-labs/FLUX.1-dev", ParsedQuery(kind="repo", value="black-forest-labs/FLUX.1-dev")),
        ("https://huggingface.co/latent-consistency/lcm-lora-sdxl", ParsedQuery(kind="repo", value=LCM)),
        (
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors",
            ParsedQuery(kind="repo", value=FLUX),
        ),
        ("huggingface.co/latent-consistency/lcm-lora-sdxl/tree/main", ParsedQuery(kind="repo", value=LCM)),
        ("https://www.huggingface.co/latent-consistency/lcm-lora-sdxl?x=1#y", ParsedQuery(kind="repo", value=LCM)),
        ("https://hf.co/latent-consistency/lcm-lora-sdxl", ParsedQuery(kind="repo", value=LCM)),
        ("http://HUGGINGFACE.CO/latent-consistency/lcm-lora-sdxl", ParsedQuery(kind="repo", value=LCM)),
        ("https://huggingface.co/latent-consistency", ParsedQuery(kind="author", value="latent-consistency")),
        ("https://huggingface.co/datasets/org/data", ParsedQuery(kind="text", value="https://huggingface.co/datasets/org/data")),
        ("https://huggingface.co/spaces/org/app", ParsedQuery(kind="text", value="https://huggingface.co/spaces/org/app")),
        ("https://huggingface.co/models?search=x", ParsedQuery(kind="text", value="https://huggingface.co/models?search=x")),
        ("https://evil.example/org/repo", ParsedQuery(kind="text", value="https://evil.example/org/repo")),
        ("org/repo/extra", ParsedQuery(kind="text", value="org/repo/extra")),
        ("org/../repo", ParsedQuery(kind="text", value="org/../repo")),
        ("-org/repo", ParsedQuery(kind="text", value="-org/repo")),
        ("sd 1.5/lora", ParsedQuery(kind="text", value="sd 1.5/lora")),
    ],
)
def test_parse_query(q, expected):
    assert parse_query(q) == expected


def test_next_cursor_from_the_link_header():
    cursor = next_cursor_from_link(link_header("models_search_lora"))

    assert cursor == _cursor("models_search_lora")
    assert cursor.startswith("eyIkb3IiOlt7") and cursor.endswith("=")
    assert next_cursor_from_link('<https://huggingface.co/api/models?cursor=a%2Bb%2Fc>; rel="next"') == "a+b/c"
    assert next_cursor_from_link(
        '<https://huggingface.co/api/models?cursor=p>; rel="prev", <https://huggingface.co/api/models?limit=2&cursor=n>; rel=next'
    ) == "n"
    assert next_cursor_from_link('<https://huggingface.co/api/models?cursor=p>; rel="prev"') is None
    assert next_cursor_from_link('<https://huggingface.co/api/models?limit=2>; rel="next"') is None
    assert next_cursor_from_link("garbage") is None
    assert next_cursor_from_link("") is None
    assert next_cursor_from_link(None) is None


async def test_a_plain_search_asks_for_full_answers_sorted_by_trend():
    source, transport = _source(HuggingFaceTransport().add("/api/models", "models_search_lora"))

    page = await source.search(SearchQuery(limit=3))

    assert transport.calls_to("/api/models") == [
        {"direction": "-1", "full": "true", "limit": "3", "sort": "trendingScore"}
    ]
    assert [found.id for found in page.items] == list(_items("models_search_lora"))
    assert page.source == "huggingface"
    assert page.next_cursor == _cursor("models_search_lora")


@pytest.mark.parametrize(
    "q, expected",
    [
        ("lora", {"search": "lora"}),
        ("  stable diffusion  ", {"search": "stable diffusion"}),
        ("@latent-consistency", {"author": "latent-consistency"}),
        ("https://huggingface.co/latent-consistency", {"author": "latent-consistency"}),
        ("#lora", {"filter": "lora"}),
    ],
)
async def test_the_search_box_syntax_becomes_hugging_face_parameters(q, expected):
    source, transport = _source(HuggingFaceTransport().add("/api/models", "models_author"))

    await source.search(SearchQuery(q=q, limit=10))

    sent = transport.calls_to("/api/models")[0]
    assert sent == {"direction": "-1", "full": "true", "limit": "10", "sort": "trendingScore", **expected}


async def test_an_exact_repo_id_is_fetched_through_the_detail_endpoint():
    transport = HuggingFaceTransport().add(f"/api/models/{LCM}", "model_lcm_lora_sdxl")
    source, _ = _source(transport)

    page = await source.search(SearchQuery(q=LCM, types=["lora"], sort="likes"))

    assert transport.paths() == [f"/api/models/{LCM}"]
    assert transport.calls == [(f"/api/models/{LCM}", [])]
    [found] = page.items
    assert (found.id, found.type, found.mb_type) == (LCM, "text-to-image", "lora")
    assert found.base_model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert page.next_cursor is None


async def test_a_pasted_link_is_fetched_through_the_detail_endpoint():
    transport = HuggingFaceTransport().add(f"/api/models/{FLUX}", "model_flux_dev_gated")
    source, _ = _source(transport)

    page = await source.search(SearchQuery(q=f"https://huggingface.co/{FLUX}/blob/main/flux1-dev.safetensors"))

    assert transport.paths() == [f"/api/models/{FLUX}"]
    [found] = page.items
    assert (found.id, found.gated, found.mb_type) == (FLUX, True, "checkpoint")


async def test_an_exact_repo_that_does_not_exist_falls_back_to_a_text_search():
    transport = HuggingFaceTransport()
    transport.add("/api/models/no-such-org/no-such-repo", "error_401_unknown_repo")
    transport.add("/api/models", "models_search_lora")
    source, _ = _source(transport)

    page = await source.search(SearchQuery(q="no-such-org/no-such-repo", limit=3))
    later = await source.search(SearchQuery(q="no-such-org/no-such-repo", limit=3, cursor="c2"))

    assert transport.paths() == ["/api/models/no-such-org/no-such-repo", "/api/models", "/api/models"]
    assert [call["search"] for call in transport.calls_to("/api/models")] == ["no-such-org/no-such-repo"] * 2
    assert transport.calls_to("/api/models")[1]["cursor"] == "c2"
    assert len(page.items) == 3 and page.next_cursor == _cursor("models_search_lora")
    assert len(later.items) == 3


async def test_an_exact_nsfw_repo_is_dropped_from_a_safe_search():
    raw = body("model_lcm_lora_sdxl")
    raw["tags"].append("not-for-all-audiences")
    transport = HuggingFaceTransport().add(f"/api/models/{LCM}", (200, {}, raw))
    source, _ = _source(transport)

    assert (await source.search(SearchQuery(q=LCM, safe=True))).items == []
    assert [found.nsfw for found in (await source.search(SearchQuery(q=LCM))).items] == [True]


async def test_the_next_cursor_comes_from_the_link_header_and_goes_back_as_cursor():
    page1_cursor = _cursor("models_search_lora")
    transport = HuggingFaceTransport()
    transport.add("/api/models", "models_search_lora", cursor=ABSENT)
    transport.add("/api/models", "models_search_lora_page2", cursor=page1_cursor)
    source, _ = _source(transport)

    first = await source.search(SearchQuery(q="lora", limit=3))
    second = await source.search(SearchQuery(q="lora", limit=3, cursor=first.next_cursor))

    assert first.next_cursor == page1_cursor
    assert [found.id for found in second.items] == [
        "RunningHubAI/rh-qwen-image-2.1-lora-2104918997757157378",
        "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen",
        "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA",
    ]
    assert second.next_cursor == _cursor("models_search_lora_page2")
    calls = transport.calls_to("/api/models")
    assert [call.get("cursor") for call in calls] == [None, page1_cursor]
    assert {key: value for key, value in calls[1].items() if key != "cursor"} == calls[0]


async def test_a_page_without_a_next_link_ends_the_search():
    transport = HuggingFaceTransport().add("/api/models", (200, {}, body("models_author")))
    source, _ = _source(transport)

    assert (await source.search(SearchQuery())).next_cursor is None


async def test_safe_searches_drop_the_not_for_all_audiences_repos():
    transport = HuggingFaceTransport().add("/api/models", "models_filter_nsfw_tag")
    source, _ = _source(transport)

    safe = await source.search(SearchQuery(q="#not-for-all-audiences", limit=2, safe=True))
    unsafe = await source.search(SearchQuery(q="#not-for-all-audiences", limit=2))

    assert safe.items == []
    assert safe.next_cursor == _cursor("models_filter_nsfw_tag")
    assert [found.nsfw for found in unsafe.items] == [True, True]
    assert transport.calls_to("/api/models") == [
        {"direction": "-1", "filter": "not-for-all-audiences", "full": "true", "limit": "2", "sort": "trendingScore"}
    ]


async def test_types_and_base_models_become_pipeline_tag_and_filters():
    source, transport = _source(HuggingFaceTransport().add("/api/models", "models_text_to_image_top"))

    await source.search(
        SearchQuery(
            q="#sdxl",
            types=["safetensors", "text-to-image", "LoRA", "lora"],
            base_models=["stabilityai/stable-diffusion-xl-base-1.0", "base_model:a/b"],
            sort="downloads",
        )
    )

    assert transport.calls == [
        (
            "/api/models",
            [
                ("direction", "-1"),
                ("filter", "base_model:a/b"),
                ("filter", "base_model:stabilityai/stable-diffusion-xl-base-1.0"),
                ("filter", "lora"),
                ("filter", "safetensors"),
                ("filter", "sdxl"),
                ("full", "true"),
                ("limit", "24"),
                ("pipeline_tag", "text-to-image"),
                ("sort", "downloads"),
            ],
        )
    ]


@pytest.mark.parametrize(
    "sort, value",
    [
        (None, "trendingScore"),
        ("trending", "trendingScore"),
        ("downloads", "downloads"),
        ("likes", "likes"),
        ("updated", "lastModified"),
        ("newest", "createdAt"),
    ],
)
async def test_sorts(sort, value):
    source, transport = _source(HuggingFaceTransport().add("/api/models", "models_text_to_image_top"))

    await source.search(SearchQuery(sort=sort, period="month"))

    sent = transport.calls_to("/api/models")[0]
    assert (sent["sort"], sent["direction"]) == (value, "-1")
    assert "period" not in sent


@pytest.mark.parametrize(
    "query, message",
    [
        (SearchQuery(sort="most_downloaded"), "sort"),
        (SearchQuery(types=["text-to-image", "image-to-video"]), "one task at a time"),
        (SearchQuery(types=["Checkpoint"]), "Unknown Hugging Face type"),
        (SearchQuery(base_models=["not a repo"]), "base model"),
        (SearchQuery(q=LCM, types=["nope"]), "Unknown Hugging Face type"),
    ],
)
async def test_bad_parameters_are_invalid_requests_without_asking(query, message):
    source, transport = _source()

    with pytest.raises(SourceInvalidRequest, match=message):
        await source.search(query)

    assert transport.calls == []


async def test_every_answer_is_cached_with_every_parameter():
    cache = MemoryCache()
    transport = HuggingFaceTransport().add("/api/models", "models_search_lora")
    transport.add_repo("lcm_lora_sdxl")
    source, _ = _source(transport, cache)

    await source.search(SearchQuery(q="lora"))
    await source.search(SearchQuery(q="lora"))
    await source.search(SearchQuery(q="lora", cursor="c2"))
    await source.search(SearchQuery(q="lora", sort="likes"))
    first = await source.detail(LCM)
    again = await source.detail(LCM)

    assert first == again
    assert transport.paths() == [
        "/api/models",
        "/api/models",
        "/api/models",
        f"/api/models/{LCM}",
        f"/api/models/{LCM}/tree/main",
        f"/{LCM}/raw/main/README.md",
    ]
    assert len(cache.entries) == 6
    payloads = list(cache.entries.values())
    assert payloads[0] == {"body": body("models_search_lora"), "link": link_header("models_search_lora")}
    assert payloads[-1] == {"text": _lcm_readme()}


async def test_the_nsfw_level_does_not_change_hugging_face_results():
    source, _ = _source(HuggingFaceTransport().add("/api/models", "models_filter_nsfw_tag"))

    strict = await source.search(SearchQuery(), max_level=1)
    loose = await source.search(SearchQuery(), max_level=8)

    assert strict == loose


@pytest.mark.parametrize(
    "response, error",
    [
        ((429, {"Retry-After": "30"}, {"error": "Too many requests"}), SourceRateLimited),
        ((503, {}, b"<html>down</html>"), SourceUnavailable),
        ((400, {}, {"error": "Invalid sort"}), SourceInvalidRequest),
        ((200, {}, {"not": "a list"}), SourceUnavailable),
        ((200, {"Content-Type": "text/html"}, "<html>challenge</html>"), SourceUnavailable),
    ],
)
async def test_upstream_errors_on_search(response, error):
    source, _ = _source(HuggingFaceTransport().add("/api/models", response))

    with pytest.raises(error):
        await source.search(SearchQuery())


async def test_detail_of_a_public_lora():
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    source, _ = _source(transport)

    detail = await source.detail(LCM)

    assert transport.calls == [
        (f"/api/models/{LCM}", []),
        (f"/api/models/{LCM}/tree/main", [("recursive", "1")]),
        (f"/{LCM}/raw/main/README.md", []),
    ]
    assert (detail.source, detail.id, detail.name) == ("huggingface", LCM, LCM)
    assert (detail.type, detail.mb_type, detail.creator) == ("text-to-image", "lora", "latent-consistency")
    assert (detail.nsfw, detail.gated, detail.poi) == (False, False, False)
    assert [version.partial for version in detail.versions] == [False]
    assert detail.source_url == f"https://huggingface.co/{LCM}"
    assert detail.tags == ["diffusers", "lora", "text-to-image", "arxiv:2311.05556"]
    assert detail.stats.model_dump() == {"downloads": 42591, "likes": 774, "rating": None}
    assert detail.license.model_dump() == {
        "summary": "openrail++",
        "commercial_use": None,
        "no_credit": None,
        "derivatives": None,
        "different_license": None,
        "url": None,
    }
    [version] = detail.versions
    assert (version.id, version.name) == ("main", "main")
    assert version.base_model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert (version.trigger_words, version.created_at, version.early_access, version.nsfw) == (
        [], "2023-11-09T00:34:02.000Z", False, False,
    )
    assert version.images == []
    assert [file.model_dump() for file in version.files] == [
        {
            "id": "pytorch_lora_weights.safetensors",
            "name": "pytorch_lora_weights.safetensors",
            "size_bytes": 393855224,
            "sha256": "a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141",
            "format": "SafeTensor",
            "precision": None,
            "file_type": "LoRA",
            "primary": True,
            "download_url": f"https://huggingface.co/{LCM}/resolve/main/pytorch_lora_weights.safetensors",
            "auth_provider_id": "huggingface",
        }
    ]


async def test_the_readme_becomes_sanitised_html_without_its_frontmatter():
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    source, _ = _source(transport)

    description = (await source.detail(LCM)).description_html

    assert description.startswith("<h3>Latent Consistency Model (LCM) LoRA: SDXL</h3>")
    for frontmatter in ("library_name", "inference: false", "license: openrail++", "---"):
        assert frontmatter not in description
    assert "<strong>2 - 8 steps</strong>" in description
    assert '<a href="https://arxiv.org/abs/2311.05556" rel="noopener noreferrer" target="_blank">' in description
    assert "<pre><code>pip install --upgrade pip" in description
    lowered = description.lower()
    for needle in ("<img", "<script", "<table", "<div", "<span", "class="):
        assert needle not in lowered
    assert set(re.findall(r"</?([a-z0-9]+)", description)) <= _SANITISED_TAGS
    assert "load_image, make_image_grid" in description


async def test_detail_of_a_gated_repo():
    transport = HuggingFaceTransport()
    transport.add_repo("flux_dev_gated")
    source, _ = _source(transport)

    detail = await source.detail(FLUX)
    again = await source.detail(FLUX)

    assert detail == again
    assert detail.gated is True
    assert detail.description_html is None
    assert transport.paths().count(f"/{FLUX}/raw/main/README.md") == 1
    assert detail.license.summary == "flux-1-dev-non-commercial-license"
    assert detail.license.url == f"https://huggingface.co/{FLUX}/blob/main/LICENSE.md"
    assert (detail.type, detail.mb_type) == ("text-to-image", "checkpoint")
    [version] = detail.versions
    assert version.base_model is None
    files = {file.name: file for file in version.files}
    assert list(files) == [
        "flux1-dev.safetensors",
        "ae.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    ]
    assert [file.name for file in version.files if file.primary] == ["flux1-dev.safetensors"]
    assert files["flux1-dev.safetensors"].size_bytes == 23802932552
    assert files["ae.safetensors"].size_bytes == 335304388
    assert body("tree_flux_dev_gated")[9]["lfs"]["oid"] == "*" * 64
    assert all(file.sha256 is None for file in version.files)
    assert {name: file.file_type for name, file in files.items()} == {
        "flux1-dev.safetensors": "Model",
        "ae.safetensors": "VAE",
        "text_encoder/model.safetensors": "Text Encoder",
        "text_encoder_2/model-00001-of-00002.safetensors": "Text Encoder",
        "text_encoder_2/model-00002-of-00002.safetensors": "Text Encoder",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "UNet",
    }
    assert files["transformer/diffusion_pytorch_model-00001-of-00003.safetensors"].download_url == (
        f"https://huggingface.co/{FLUX}/resolve/main/transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
    )


async def test_detail_of_a_diffusers_folder_repo():
    transport = HuggingFaceTransport()
    transport.add_repo("sdxl_base")
    source, _ = _source(transport)

    detail = await source.detail(SDXL)

    [version] = detail.versions
    names = [file.name for file in version.files]
    assert names == [
        "sd_xl_base_1.0.safetensors",
        "sd_xl_base_1.0_0.9vae.safetensors",
        "sd_xl_offset_example-lora_1.0.safetensors",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/model.onnx",
        "text_encoder/model.safetensors",
    ]
    for left_out in ("comparison.png", "01.png", "model_index.json", "LICENSE.md", "README.md",
                     "text_encoder/flax_model.msgpack", "scheduler", "text_encoder", "tokenizer"):
        assert left_out not in names
    assert [name for name in names if "/" not in name and name.endswith(".safetensors")] == names[:3]
    primary = version.files[0]
    assert (primary.name, primary.primary, primary.size_bytes) == ("sd_xl_base_1.0.safetensors", True, 6938078334)
    assert primary.sha256 == "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
    assert sum(file.primary for file in version.files) == 1
    files = {file.name: file for file in version.files}
    assert (files["text_encoder/model.onnx"].format, files["text_encoder/model.onnx"].file_type) == ("ONNX", "Text Encoder")
    assert files["text_encoder/model.fp16.safetensors"].precision == "fp16"
    assert files["sd_xl_offset_example-lora_1.0.safetensors"].file_type == "LoRA"
    assert files["sd_xl_base_1.0_0.9vae.safetensors"].file_type == "Model"
    assert detail.mb_type == "checkpoint"
    assert detail.license.model_dump(include={"summary", "url"}) == {
        "summary": "openrail++",
        "url": f"https://huggingface.co/{SDXL}/blob/main/LICENSE.md",
    }
    assert detail.description_html is None


async def test_detail_of_an_upscaler_repo():
    transport = HuggingFaceTransport()
    transport.add_repo("real_esrgan")
    source, _ = _source(transport)

    detail = await source.detail(ESRGAN)

    assert (detail.type, detail.mb_type, detail.license) == ("model", "upscaler", None)
    assert detail.tags == ["PyTorch", "ru", "en", "arxiv:2107.10833"]
    [version] = detail.versions
    assert [(file.name, file.format, file.primary, file.size_bytes) for file in version.files] == [
        ("RealESRGAN_x8.pth", "PickleTensor", True, 67189359),
        ("RealESRGAN_x2.pth", "PickleTensor", False, 67061725),
        ("RealESRGAN_x4.pth", "PickleTensor", False, 67040989),
    ]
    assert version.files[0].sha256 == "8b72fb469d12f05a4770813d2603eb1b550f40df6fb8b37d6c7bc2db3d2bff5e"


async def test_detail_of_another_gated_repo_with_a_named_licence():
    transport = HuggingFaceTransport()
    transport.add_repo("sd35_large", readme=(401, {"Content-Type": "text/plain"}, "Access restricted"))
    source, _ = _source(transport)

    detail = await source.detail(SD35)

    assert (detail.gated, detail.description_html) == (True, None)
    assert detail.license.summary == "stabilityai-ai-community"
    files = {file.name: file for file in detail.versions[0].files}
    assert [name for name, file in files.items() if file.primary] == ["sd3.5_large.safetensors"]
    assert files["text_encoder_2/model.fp16.safetensors"].precision == "fp16"
    assert files["text_encoder_3/model-00001-of-00002.safetensors"].file_type == "Text Encoder"


async def test_an_unknown_repo_is_not_found():
    transport = HuggingFaceTransport()
    transport.add("/api/models/no-such-org/no-such-repo", "error_401_unknown_repo")
    transport.add("/api/models/no-such-org/no-such-repo/tree/main", "error_401_unknown_tree")
    source, _ = _source(transport)

    with pytest.raises(SourceNotFound) as caught:
        await source.detail("no-such-org/no-such-repo")

    assert caught.value.status == 401
    assert "no-such-org/no-such-repo" in caught.value.message
    assert "/no-such-org/no-such-repo/raw/main/README.md" not in transport.paths()


@pytest.mark.parametrize(
    "model_id",
    ["", " ", "a/b/c", "../x", "org/..", "org/../repo", "-org/repo", "org/repo-", ".org/repo", "org/re po",
     "org/repo/", "/org/repo", "org//repo", "a" * 97 + "/b", "org/rép"],
)
async def test_ids_that_are_not_repo_ids_are_not_found_without_asking(model_id):
    source, transport = _source()

    with pytest.raises(SourceNotFound):
        await source.detail(model_id)

    assert transport.calls == []


async def test_a_legacy_id_without_an_organisation_is_asked_for():
    raw = body("model_real_esrgan")
    raw["id"] = raw["modelId"] = "gpt2"
    transport = HuggingFaceTransport().add("/api/models/gpt2", (200, {}, raw))
    transport.add("/api/models/gpt2/tree/main", (200, {}, []))
    source, _ = _source(transport)

    detail = await source.detail("gpt2")

    assert (detail.id, detail.creator, detail.versions[0].files) == ("gpt2", "ai-forever", [])


async def test_a_disabled_repo_is_not_found():
    raw = body("model_lcm_lora_sdxl")
    raw["disabled"] = True
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    transport.add(f"/api/models/{LCM}", (200, {}, raw))
    transport.add("/api/models", (200, {}, []))
    source, _ = _source(transport)

    with pytest.raises(SourceNotFound, match="disabled"):
        await source.detail(LCM)
    assert (await source.search(SearchQuery(q=LCM))).items == []
    assert transport.calls_to("/api/models")[0]["search"] == LCM


@pytest.mark.parametrize(
    "response, error",
    [
        ((429, {"Retry-After": "5"}, {"error": "slow down"}), SourceRateLimited),
        ((502, {}, b""), SourceUnavailable),
        ((200, {}, ["not", "a", "model"]), SourceUnavailable),
    ],
)
async def test_model_errors_win_over_the_tree(response, error):
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    transport.add(f"/api/models/{LCM}", response)
    source, _ = _source(transport)

    with pytest.raises(error):
        await source.detail(LCM)


async def test_a_tree_failure_fails_the_detail_but_a_missing_revision_does_not():
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    transport.add(f"/api/models/{LCM}/tree/main", (503, {}, b"down"))
    source, _ = _source(transport)
    with pytest.raises(SourceUnavailable):
        await source.detail(LCM)

    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl")
    transport.add(f"/api/models/{LCM}/tree/main", (404, {}, {"error": "Revision Not Found"}))
    source, _ = _source(transport)
    detail = await source.detail(LCM)

    assert detail.versions[0].files == []
    assert detail.description_html.startswith("<h3>")
    assert detail.mb_type == "lora"


async def test_the_tree_follows_its_link_pages():
    second = [_file("b/model.safetensors", 20)]
    transport = HuggingFaceTransport()
    _demo_repo(transport)
    transport.add(
        "/api/models/org/demo/tree/main",
        (200, {"Link": '<https://huggingface.co/api/models/org/demo/tree/main?recursive=1&cursor=page%3D2>; rel="next"'},
         [_file("a.safetensors", 10)]),
        cursor=ABSENT,
    )
    transport.add("/api/models/org/demo/tree/main", (200, {}, second), cursor="page=2")
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    assert [file.name for file in detail.versions[0].files] == ["a.safetensors", "b/model.safetensors"]
    assert [call.get("cursor") for call in transport.calls_to("/api/models/org/demo/tree/main")] == [None, "page=2"]


async def test_the_tree_stops_after_five_pages():
    counter = {"n": 0}

    def endless(path, params):
        counter["n"] += 1
        link = f'<https://huggingface.co/api/models/org/demo/tree/main?cursor=c{counter["n"]}>; rel="next"'
        return 200, {"Link": link}, [_file(f"part{counter['n']}.safetensors", counter["n"])]

    transport = HuggingFaceTransport()
    _demo_repo(transport, readme="![preview](images/not-listed.png)\n\nText")
    transport.add("/api/models/org/demo/tree/main", endless)
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    assert counter["n"] == hf.MAX_TREE_PAGES
    assert len(detail.versions[0].files) == hf.MAX_TREE_PAGES
    assert [image.url for image in detail.versions[0].images] == [
        "https://huggingface.co/org/demo/resolve/main/images/not-listed.png"
    ]


_IMAGES_README = """---
license: mit
tags: [x]
---
# Title <script>alert(1)</script>

Inline ![a](images/a.png) and <img src="https://cdn-uploads.huggingface.co/production/uploads/x/b.jpg" width=300>.

![badge](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)
![external](https://github.com/x/y/raw/main/c.png)
![blob](https://huggingface.co/org/other/blob/main/d.webp)
![root](/org/other/resolve/main/e.gif)
![missing](./not-there.png)
![escape](../../secret.png)
![data](data:image/png;base64,AAAA)
<img src="javascript:alert(1)//x.png">
<IMG SRC="images/space name.png">
![again](images/a.png)

```html
<img src="images/in-code.png">
```
"""
_IMAGES_TREE = [_file("README.md"), _file("images/a.png"), _file("images/space name.png"), _file("images/in-code.png")]


async def test_readme_images_on_huggingface_co_become_previews():
    transport = HuggingFaceTransport()
    _demo_repo(transport, tree=_IMAGES_TREE, readme=_IMAGES_README)
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    images = detail.versions[0].images
    assert [image.url for image in images] == [
        "https://huggingface.co/org/demo/resolve/main/images/a.png",
        "https://cdn-uploads.huggingface.co/production/uploads/x/b.jpg",
        "https://huggingface.co/org/other/resolve/main/d.webp",
        "https://huggingface.co/org/other/resolve/main/e.gif",
        "https://huggingface.co/org/demo/resolve/main/images/space%20name.png",
    ]
    assert all((image.nsfw, image.nsfw_level, image.kind) == (False, 1, "image") for image in images)
    assert len({image.id for image in images}) == len(images)
    assert detail.description_html.startswith("<h3>Title </h3>")
    assert "alert" not in detail.description_html and "<img" not in detail.description_html
    assert '<pre><code>&lt;img src="images/in-code.png"&gt;' in detail.description_html


async def test_readme_images_of_an_nsfw_repo_are_nsfw_and_at_most_six():
    readme = "\n".join(f"![{n}](https://huggingface.co/org/demo/resolve/main/{n}.png)" for n in range(9))
    transport = HuggingFaceTransport()
    _demo_repo(transport, tags=["not-for-all-audiences"], tree=[_file("README.md")], readme=readme)
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    images = detail.versions[0].images
    assert len(images) == hf.MAX_IMAGES
    assert all((image.nsfw, image.nsfw_level) == (True, 8) for image in images)
    assert (detail.nsfw, detail.versions[0].nsfw) == (True, True)


async def test_a_readme_too_large_is_not_read():
    transport = HuggingFaceTransport()
    _demo_repo(transport, tree=[_file("README.md", hf.MAX_README_BYTES + 1)], readme="# Big")
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    assert detail.description_html is None
    assert "/org/demo/raw/main/README.md" not in transport.paths()


async def test_a_readme_answer_over_the_cap_is_dropped_and_remembered():
    cache = MemoryCache()
    transport = HuggingFaceTransport()
    _demo_repo(transport, tree=[_file("README.md", 100)], readme="x" * (hf.MAX_README_BYTES + 1))
    source, _ = _source(transport, cache)

    first = await source.detail("org/demo")
    second = await source.detail("org/demo")

    assert first.description_html is None and second.description_html is None
    assert transport.paths().count("/org/demo/raw/main/README.md") == 1
    assert {"text": None} in cache.entries.values()


async def test_no_readme_in_the_tree_means_no_request():
    transport = HuggingFaceTransport()
    _demo_repo(transport, tree=[_file("model.safetensors", 100)], readme="# Never read")
    source, _ = _source(transport)

    assert (await source.detail("org/demo")).description_html is None
    assert "/org/demo/raw/main/README.md" not in transport.paths()


async def test_a_readme_outage_leaves_the_description_out_without_failing():
    transport = HuggingFaceTransport()
    transport.add_repo("lcm_lora_sdxl", readme=(503, {}, b"down"))
    cache = MemoryCache()
    source, _ = _source(transport, cache)

    detail = await source.detail(LCM)
    await source.detail(LCM)

    assert detail.description_html is None
    assert transport.paths().count(f"/{LCM}/raw/main/README.md") == 2
    assert detail.versions[0].files


async def test_without_markdown2_the_readme_is_escaped_paragraphs(monkeypatch):
    monkeypatch.setitem(sys.modules, "markdown2", None)
    readme = "---\nlicense: mit\n---\n# Title\n\nSome <b>bold</b> & text\n![img](images/a.png)\n\n"
    transport = HuggingFaceTransport()
    _demo_repo(transport, tree=[_file("README.md"), _file("images/a.png")], readme=readme)
    source, _ = _source(transport)

    detail = await source.detail("org/demo")

    assert detail.description_html == (
        "<p># Title</p><p>Some &lt;b&gt;bold&lt;/b&gt; &amp; text\n![img](images/a.png)</p>"
    )
    assert [image.url for image in detail.versions[0].images] == [
        "https://huggingface.co/org/demo/resolve/main/images/a.png"
    ]


def test_a_markdown2_failure_falls_back_to_paragraphs(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "markdown2", types.SimpleNamespace(markdown=broken))

    assert hf.render_markdown("one\n\ntwo <x>") == ("<p>one</p><p>two &lt;x&gt;</p>", [])


def test_a_readme_is_converted_once(monkeypatch):
    import markdown2

    calls = []

    def counting(text, **kwargs):
        calls.append(text)
        return markdown2.markdown(text, **kwargs)

    monkeypatch.setitem(sys.modules, "markdown2", types.SimpleNamespace(markdown=counting))

    first = hf.render_markdown("# Hello ![x](a.png)")
    second = hf.render_markdown("# Hello ![x](a.png)")

    assert first == second == ("<h3>Hello </h3>", ["a.png"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("---\nlicense: mit\n---\n# Title\n", "# Title\n"),
        ("\N{ZERO WIDTH NO-BREAK SPACE}---\na: 1\n---\r\nBody", "Body"),
        ("---\n---\nBody", "Body"),
        ("---\na: 1\n...\nBody", "Body"),
        ("# No frontmatter\n---\nnot: yaml\n---\n", "# No frontmatter\n---\nnot: yaml\n---\n"),
        ("---\nunterminated: true\n# Title", "---\nunterminated: true\n# Title"),
        ("----\na: 1\n----\nBody", "----\na: 1\n----\nBody"),
        ("", ""),
    ],
)
def test_strip_frontmatter(text, expected):
    assert strip_frontmatter(text) == expected


@pytest.mark.parametrize(
    "src, expected",
    [
        ("images/a.png", "https://huggingface.co/org/demo/resolve/main/images/a.png"),
        ("./images/a.png", "https://huggingface.co/org/demo/resolve/main/images/a.png"),
        ("images/../images/a.png", "https://huggingface.co/org/demo/resolve/main/images/a.png"),
        ("images/my%20pic.JPG", "https://huggingface.co/org/demo/resolve/main/images/my%20pic.JPG"),
        ("https://huggingface.co/org/demo/resolve/main/x.webp?download=true",
         "https://huggingface.co/org/demo/resolve/main/x.webp?download=true"),
        ("http://huggingface.co/org/demo/blob/main/x.avif", "https://huggingface.co/org/demo/resolve/main/x.avif"),
        ("//cdn-uploads.huggingface.co/u/x.gif", "https://cdn-uploads.huggingface.co/u/x.gif"),
        ("/org/demo/resolve/main/x.jpeg#frag", "https://huggingface.co/org/demo/resolve/main/x.jpeg"),
        ("../x.png", None),
        ("..", None),
        ("images/a.svg", None),
        ("images/a", None),
        ("https://huggingface.co.evil.example/x.png", None),
        ("https://evilhuggingface.co/x.png", None),
        ("https://user@huggingface.co/x.png", None),
        ("https://huggingface.co:8443/x.png", None),
        ("ftp://huggingface.co/x.png", None),
        ("javascript:alert(1)//x.png", None),
        ("data:image/png;base64,AAAA", None),
        ("https://huggingface.co/x y.png", None),
        ('https://huggingface.co/x"onerror=y.png', None),
        ("images/<b>.png", None),
        ("images/it's.png", None),
        ("images/`x`.png", None),
        ("https://huggingface.co/x\\y.png", None),
        ("images/a.png\n", "https://huggingface.co/org/demo/resolve/main/images/a.png"),
        ("", None),
        (None, None),
    ],
)
def test_readme_image_url(src, expected):
    assert readme_image_url("org/demo", src) == expected


def test_readme_image_url_checks_relative_paths_against_the_files():
    assert readme_image_url("org/demo", "a.png", {"a.png"}) == "https://huggingface.co/org/demo/resolve/main/a.png"
    assert readme_image_url("org/demo", "b.png", {"a.png"}) is None
    assert readme_image_url("org/demo", "https://huggingface.co/org/demo/resolve/main/b.png", {"a.png"}) is not None


@pytest.mark.parametrize(
    "path, expected",
    [
        ("qwen-image-2.1-UC-Q4_K_M.gguf", "q4_k_m"),
        ("qwen-image-2.1-UC-Q8_0.gguf", "q8_0"),
        ("model-IQ4_XS.gguf", "iq4_xs"),
        ("qwen-image-2.1-UC-BF16.gguf", "bf16"),
        ("model-F16.gguf", "fp16"),
        ("text_encoder/model.fp16.safetensors", "fp16"),
        ("text_encoder_3/model.fp16-00001-of-00002.safetensors", "fp16"),
        ("flux1-dev-fp8-e4m3fn.safetensors", "fp8"),
        ("flux1-dev_fp8_e4m3fn.safetensors", "fp8"),
        ("t5xxl_fp32.safetensors", "fp32"),
        ("qwen3vl_8b_bf16.safetensors", "bf16"),
        ("SampleQ_V4_int8_convrot.safetensors", "int8"),
        ("qwen-image-2.1-UC-NVFP4.safetensors", "nvfp4"),
        ("sd_xl_base_1.0.safetensors", None),
        ("qwen_q4_k_m.safetensors", None),
        ("fp16fix.safetensors", None),
        ("RealESRGAN_x4.pth", None),
    ],
)
def test_file_precision(path, expected):
    assert file_precision(path) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("vae/diffusion_pytorch_model.safetensors", "VAE"),
        ("vae_1_0/diffusion_pytorch_model.fp16.safetensors", "VAE"),
        ("ae.safetensors", "VAE"),
        ("sdxl_vae.safetensors", "VAE"),
        ("sd_xl_base_1.0_0.9vae.safetensors", "Model"),
        ("text_encoder/model.safetensors", "Text Encoder"),
        ("text_encoders/qwen3vl_8b_bf16.safetensors", "Text Encoder"),
        ("clip_l.safetensors", "Text Encoder"),
        ("t5xxl_fp16.safetensors", "Text Encoder"),
        ("unet/diffusion_pytorch_model.safetensors", "UNet"),
        ("transformer_2/diffusion_pytorch_model-00001-of-00002.safetensors", "UNet"),
        ("pytorch_lora_weights.safetensors", "LoRA"),
        ("loras/style.safetensors", "LoRA"),
        ("flux1-dev.safetensors", "Model"),
        ("RealESRGAN_x4.pth", "Model"),
    ],
)
def test_file_type_from_path(path, expected):
    assert file_type_from_path(path) == expected


def test_odd_tree_entries_are_left_out():
    sha = "AB" * 32
    entries = [
        "not an entry",
        {"type": "directory", "path": "folder.safetensors"},
        _file("../escape.safetensors"),
        _file("/absolute.safetensors"),
        _file("a//b.safetensors"),
        _file("a/./b.safetensors"),
        _file("bad\\path.safetensors"),
        _file("bad\npath.safetensors"),
        {"type": "file", "path": None},
        _file("notes.txt"),
        _file("nested/small.safetensors", 999),
        _file("ok.ckpt", 5, lfs_oid=sha),
        _file("ok.ckpt", 7),
        _file("weights.SFT", 6),
        {"type": "file", "path": "nosize.bin", "lfs": {"oid": "short", "size": "12"}},
    ]

    files = weight_files("org/demo", entries)

    assert [(file.name, file.size_bytes, file.sha256, file.format, file.primary) for file in files] == [
        ("weights.SFT", 6, None, "SafeTensor", True),
        ("nosize.bin", None, None, "PickleTensor", False),
        ("ok.ckpt", 5, sha.lower(), "PickleTensor", False),
        ("nested/small.safetensors", 999, None, "SafeTensor", False),
    ]


def test_without_a_root_weight_file_nothing_is_primary():
    files = weight_files("org/demo", [_file("unet/a.safetensors", 5), _file("vae/b.safetensors", 9)])

    assert [file.primary for file in files] == [False, False]


def test_every_download_url_passes_the_allowlist():
    files = []
    for key in REPOS:
        files += _detail(key).versions[0].files
    files += weight_files("org/demo", [_file("sub dir/my model+v2.safetensors"), _file("été.gguf")])

    assert len(files) > 20
    for file in files:
        assert is_allowed_download_url("huggingface", file.download_url), file.download_url
    urls = {file.name: file.download_url for file in files}
    assert urls["sub dir/my model+v2.safetensors"] == (
        "https://huggingface.co/org/demo/resolve/main/sub%20dir/my%20model%2Bv2.safetensors"
    )
    assert urls["été.gguf"] == "https://huggingface.co/org/demo/resolve/main/%C3%A9t%C3%A9.gguf"


def test_license_variants():
    raw = body("model_flux_dev_gated")
    assert license_from(raw, FLUX).model_dump(include={"summary", "url"}) == {
        "summary": "flux-1-dev-non-commercial-license",
        "url": f"https://huggingface.co/{FLUX}/blob/main/LICENSE.md",
    }

    raw["cardData"]["license_link"] = "https://ai.example/terms"
    assert license_from(raw, FLUX).url == "https://ai.example/terms"
    raw["cardData"]["license_link"] = "javascript:alert(1)"
    assert license_from(raw, FLUX).url is None
    raw["cardData"]["license_link"] = "./docs/LICENSE.txt"
    assert license_from(raw, FLUX).url == f"https://huggingface.co/{FLUX}/blob/main/docs/LICENSE.txt"
    raw["cardData"]["license_link"] = "/black-forest-labs/FLUX.1-dev/raw/main/LICENSE.md"
    assert license_from(raw, FLUX).url == f"https://huggingface.co/{FLUX}/raw/main/LICENSE.md"

    del raw["cardData"]["license_name"]
    assert license_from(raw, FLUX).summary == "other"

    raw["tags"] = [tag for tag in raw["tags"] if not tag.startswith("license:")]
    raw["cardData"] = {"license": ["apache-2.0", "mit"]}
    assert license_from(raw, FLUX).model_dump(include={"summary", "url"}) == {"summary": "apache-2.0", "url": None}
    assert license_from(raw, FLUX, ["LICENSE.txt", "sub/LICENSE"]).url == f"https://huggingface.co/{FLUX}/blob/main/LICENSE.txt"

    raw.pop("cardData")
    assert license_from(raw, FLUX) is None


def test_trigger_words_and_the_card_base_model():
    raw = body("model_lcm_lora_sdxl")
    raw["cardData"]["instance_prompt"] = " a photo of sks dog "
    raw["cardData"]["base_model"] = ["first/base", "second/base"]
    raw["tags"] = [tag for tag in raw["tags"] if not tag.startswith("base_model:")]

    detail = normalize_detail(raw, body("tree_lcm_lora_sdxl"))
    listed = normalize_item(raw)

    assert detail.versions[0].trigger_words == ["a photo of sks dog"]
    assert detail.versions[0].base_model == "first/base"
    assert (listed.base_model, listed.base_models) == ("first/base", ["first/base"])


def test_normalize_detail_without_a_tree_has_no_files():
    detail = normalize_detail(body("model_lcm_lora_sdxl"))

    assert detail.versions[0].files == []
    assert detail.mb_type == "lora"
    assert detail.description_html is None


async def test_info_lists_fixed_filters_without_asking():
    source, transport = _source()

    info = await source.info()

    assert (info.id, info.label) == ("huggingface", "Hugging Face")
    assert [option.value for option in info.types] == [
        "text-to-image", "image-to-image", "text-to-video", "image-to-video", "text-to-audio", "text-to-3d",
        "image-to-3d", "lora", "controlnet", "gguf", "safetensors", "diffusers",
    ]
    assert info.types[7].label == "LoRA"
    assert [option.value for option in info.sorts] == ["trending", "downloads", "likes", "updated", "newest"]
    assert (info.periods, info.base_models) == ([], [])
    assert info.search_help == "org/repo, @author, #tag, free text, or a pasted huggingface.co link"
    assert transport.calls == []


def test_the_registry_serves_hugging_face():
    source = get_source("huggingface")

    assert isinstance(source, HuggingFaceSource)
    assert (source.id, source.label) == ("huggingface", "Hugging Face")


def test_the_fixtures_are_the_recorded_shapes():
    assert load("error_401_unknown_repo")["_status"] == 401
    assert load("readme_flux_dev_gated")["_status"] == 401
    assert body("model_flux_dev_gated")["gated"] == "auto"
    assert 'rel="next"' in link_header("models_search_lora")
