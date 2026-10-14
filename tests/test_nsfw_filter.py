import pytest

from backend.nsfw_filter import (
    NsfwForbidden,
    display_mode,
    filter_detail,
    filter_image,
    filter_item,
    filter_page,
    nsfw_media,
)
from backend.schema import Image, ModelDetail, SearchItem, SearchPage, Version
from backend.sources.civitai import normalize_detail, normalize_item

from .civitai_fakes import body, context, image_urls

SAFE = Image(id="1", url="https://image.civitai.com/a/safe.jpeg", blurhash="Usafe", nsfw=False, nsfw_level=1, width=512, height=768)
SAFE_POI = Image(id="2", url="https://image.civitai.com/a/poi.jpeg", blurhash="Upoi", nsfw=False, nsfw_level=2, poi=True)
NSFW_X = Image(id="3", url="https://image.civitai.com/a/x.jpeg", blurhash="Ux", nsfw=True, nsfw_level=8, width=832, height=1216)
NSFW_XXX = Image(id="4", url="https://image.civitai.com/a/xxx.jpeg", blurhash="Uxxx", nsfw=True, nsfw_level=16)
NSFW_URLS = {NSFW_X.url, NSFW_XXX.url}


def _item(model_id, *, nsfw=False, poi=False, thumbnail=None):
    return SearchItem(
        source="civitai",
        id=model_id,
        name=model_id,
        type="LORA",
        mb_type="lora",
        nsfw=nsfw,
        poi=poi,
        thumbnail=thumbnail,
        source_url=f"https://civitai.{'red' if nsfw else 'com'}/models/{model_id}",
    )


PAGE = SearchPage(
    source="civitai",
    items=[
        _item("safe", thumbnail=SAFE),
        _item("nsfw", nsfw=True, thumbnail=NSFW_XXX),
        _item("nsfw-safe-thumb", nsfw=True, thumbnail=SAFE),
        _item("poi", poi=True, thumbnail=SAFE_POI),
        _item("no-thumb"),
    ],
    next_cursor="409619|21294|82098",
)


def _detail(*, nsfw=False, poi=False):
    return ModelDetail(
        source="civitai",
        id="122359",
        name="Detail",
        type="LORA",
        mb_type="lora",
        source_url="https://civitai.com/models/122359",
        nsfw=nsfw,
        poi=poi,
        versions=[
            Version(id="10", name="v1", images=[SAFE, NSFW_X, SAFE_POI, NSFW_X]),
            Version(id="11", name="v2", nsfw=True, images=[NSFW_XXX]),
        ],
    )


BLOCKED = context("blocked", "hide")
HIDE = context("allowed", "hide")
BLUR = context("blur", "blur")
BLUR_BY_CHOICE = context("allowed", "blur")
SHOW = context("allowed", "show")


def _ids(page):
    return [item.id for item in page.items]


@pytest.mark.parametrize(
    "ceiling, mode, expected",
    [
        ("allowed", "show", "show"),
        ("allowed", "blur", "blur"),
        ("allowed", "hide", "hide"),
        ("blur", "blur", "blur"),
        ("blur", "hide", "hide"),
        ("blur", "show", "blur"),
        ("blocked", "hide", "hide"),
        ("blocked", "show", "hide"),
        ("blocked", "blur", "hide"),
    ],
)
def test_display_mode_never_exceeds_the_ceiling(ceiling, mode, expected):
    ctx = context(ceiling, "hide").model_copy(update={"mode": mode})

    assert display_mode(ctx) == expected


@pytest.mark.parametrize("ctx", [BLOCKED, HIDE], ids=["blocked", "hide"])
def test_blocked_and_hide_drop_nsfw_items(ctx):
    page = filter_page(PAGE, ctx)

    assert _ids(page) == ["safe", "poi", "no-thumb"]
    assert page.next_cursor == PAGE.next_cursor
    assert not NSFW_URLS & set(image_urls(page.model_dump()))


@pytest.mark.parametrize("ctx", [BLUR, BLUR_BY_CHOICE], ids=["blur-ceiling", "blur-mode"])
def test_blur_keeps_items_but_strips_nsfw_urls(ctx):
    page = filter_page(PAGE, ctx)

    assert _ids(page) == _ids(PAGE)
    nsfw_thumb = page.items[1].thumbnail
    assert nsfw_thumb.url is None
    assert (nsfw_thumb.blurhash, nsfw_thumb.nsfw, nsfw_thumb.nsfw_level, nsfw_thumb.id) == ("Uxxx", True, 16, "4")
    assert page.items[2].thumbnail.url == SAFE.url
    assert page.items[0].thumbnail.url == SAFE.url
    assert not NSFW_URLS & set(image_urls(page.model_dump()))


def test_show_keeps_everything():
    assert filter_page(PAGE, SHOW) == PAGE


def test_a_blur_ceiling_wins_over_a_show_preference():
    ctx = context("blur", "blur").model_copy(update={"mode": "show"})

    assert filter_page(PAGE, ctx).items[1].thumbnail.url is None


def test_a_blocked_ceiling_wins_over_any_preference():
    for mode in ("show", "blur"):
        ctx = context("blocked", "hide").model_copy(update={"mode": mode})
        assert _ids(filter_page(PAGE, ctx)) == ["safe", "poi", "no-thumb"]


@pytest.mark.parametrize("ctx", [BLOCKED, HIDE, BLUR, SHOW], ids=["blocked", "hide", "blur", "show"])
def test_hide_poi_drops_real_people_in_every_mode(ctx):
    page = filter_page(PAGE, ctx.model_copy(update={"hide_poi": True}))

    assert "poi" not in _ids(page)
    assert SAFE_POI.url not in image_urls(page.model_dump())


def test_inputs_are_not_modified():
    before = PAGE.model_dump()

    filter_page(PAGE, BLUR)
    filter_page(PAGE, BLOCKED)

    assert PAGE.model_dump() == before


def test_filter_item_and_filter_image():
    assert filter_item(_item("x", nsfw=True, thumbnail=NSFW_X), BLOCKED) is None
    assert filter_item(_item("x", thumbnail=SAFE), BLOCKED).thumbnail is SAFE
    assert filter_image(NSFW_X, HIDE) is None
    assert filter_image(NSFW_X, BLUR).url is None
    assert filter_image(NSFW_X, SHOW) is NSFW_X
    assert filter_image(SAFE, BLOCKED) is SAFE


def test_a_defensive_nsfw_thumbnail_on_a_safe_item_is_dropped_when_hiding():
    item = filter_item(_item("odd", thumbnail=NSFW_X), BLOCKED)

    assert item is not None and item.thumbnail is None


@pytest.mark.parametrize("ctx", [BLOCKED, HIDE], ids=["blocked", "hide"])
def test_blocked_and_hide_drop_nsfw_images_and_versions(ctx):
    detail = filter_detail(_detail(), ctx)

    assert [version.id for version in detail.versions] == ["10"]
    assert detail.versions[0].images == [SAFE, SAFE_POI]
    assert not NSFW_URLS & set(image_urls(detail.model_dump()))


@pytest.mark.parametrize("ctx", [BLOCKED, HIDE], ids=["blocked", "hide"])
def test_blocked_and_hide_do_not_show_an_nsfw_model(ctx):
    assert filter_detail(_detail(nsfw=True), ctx) is None


def test_blur_keeps_the_nsfw_model_without_nsfw_urls():
    detail = filter_detail(_detail(nsfw=True), BLUR)

    assert [version.id for version in detail.versions] == ["10", "11"]
    assert [image.url for image in detail.versions[0].images] == [SAFE.url, None, SAFE_POI.url, None]
    assert detail.versions[1].images[0].blurhash == "Uxxx"
    assert not NSFW_URLS & set(image_urls(detail.model_dump()))


def test_show_keeps_the_detail_as_is():
    assert filter_detail(_detail(nsfw=True), SHOW) == _detail(nsfw=True)


@pytest.mark.parametrize("ctx", [BLOCKED, HIDE, BLUR, SHOW], ids=["blocked", "hide", "blur", "show"])
def test_hide_poi_hides_a_real_person_model_and_its_images(ctx):
    ctx = ctx.model_copy(update={"hide_poi": True})

    assert filter_detail(_detail(poi=True), ctx) is None
    assert SAFE_POI.url not in image_urls(filter_detail(_detail(), ctx).model_dump())


def test_nsfw_media_is_forbidden_under_a_blocked_ceiling():
    with pytest.raises(NsfwForbidden):
        nsfw_media(_detail(), BLOCKED)


@pytest.mark.parametrize("ctx", [BLUR, BLUR_BY_CHOICE, SHOW, HIDE], ids=["blur", "blur-mode", "show", "hide"])
def test_nsfw_media_lists_the_nsfw_images_with_urls_once(ctx):
    images = nsfw_media(_detail(), ctx)

    assert images == [NSFW_X, NSFW_XXX]


def test_nsfw_media_leaves_real_people_out_with_hide_poi():
    poi_nsfw = NSFW_X.model_copy(update={"poi": True, "url": "https://image.civitai.com/a/poi-x.jpeg"})
    detail = _detail().model_copy(update={"versions": [Version(id="1", name="v", images=[poi_nsfw, NSFW_XXX])]})

    assert nsfw_media(detail, BLUR.model_copy(update={"hide_poi": True})) == [NSFW_XXX]


@pytest.mark.parametrize("fixture", ["models_lora_page1", "models_lora_page2", "models_flux_nsfwlevel", "models_query"])
@pytest.mark.parametrize("ctx", [BLOCKED, HIDE, BLUR], ids=["blocked", "hide", "blur"])
def test_no_nsfw_url_reaches_a_user_who_does_not_show_nsfw(fixture, ctx):
    raw = body(fixture)
    for model in raw["items"]:
        for version in model["modelVersions"]:
            for index, image in enumerate(version["images"]):
                if index % 2:
                    image["nsfwLevel"] = 16
    page = SearchPage(source="civitai", items=[normalize_item(model, 2) for model in raw["items"]])
    details = [normalize_detail(model, 2) for model in raw["items"]]
    nsfw_urls = {
        image["url"]
        for model in raw["items"]
        for version in model["modelVersions"]
        for image in version["images"]
        if image["nsfwLevel"] > 2
    }

    leaked = set(image_urls(filter_page(page, ctx).model_dump()))
    for detail in details:
        visible = filter_detail(detail, ctx)
        if visible is not None:
            leaked |= set(image_urls(visible.model_dump()))

    assert nsfw_urls
    assert not nsfw_urls & leaked
