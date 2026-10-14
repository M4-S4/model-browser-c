from typing import List, Optional

from .schema import Image, ModelDetail, NsfwContext, NsfwMode, SearchItem, SearchPage, Version


class NsfwForbidden(Exception):
    pass


def display_mode(ctx: NsfwContext) -> NsfwMode:
    if ctx.ceiling == "blocked":
        return "hide"
    if ctx.ceiling == "blur" and ctx.mode == "show":
        return "blur"
    if ctx.mode in ("show", "blur", "hide"):
        return ctx.mode
    return "hide"


def filter_image(image: Image, ctx: NsfwContext) -> Optional[Image]:
    if ctx.hide_poi and image.poi:
        return None
    if not image.nsfw:
        return image
    mode = display_mode(ctx)
    if mode == "hide":
        return None
    if mode == "blur":
        return image if image.url is None else image.model_copy(update={"url": None})
    return image


def _filter_images(images: List[Image], ctx: NsfwContext) -> List[Image]:
    return [kept for kept in (filter_image(image, ctx) for image in images) if kept is not None]


def filter_item(item: SearchItem, ctx: NsfwContext) -> Optional[SearchItem]:
    if ctx.hide_poi and item.poi:
        return None
    if item.nsfw and display_mode(ctx) == "hide":
        return None
    if item.thumbnail is None:
        return item
    thumbnail = filter_image(item.thumbnail, ctx)
    if thumbnail is item.thumbnail:
        return item
    return item.model_copy(update={"thumbnail": thumbnail})


def filter_page(page: SearchPage, ctx: NsfwContext) -> SearchPage:
    items = [kept for kept in (filter_item(item, ctx) for item in page.items) if kept is not None]
    return page.model_copy(update={"items": items})


def filter_version(version: Version, ctx: NsfwContext) -> Optional[Version]:
    if version.nsfw and display_mode(ctx) == "hide":
        return None
    return version.model_copy(update={"images": _filter_images(version.images, ctx)})


def filter_detail(detail: ModelDetail, ctx: NsfwContext) -> Optional[ModelDetail]:
    if ctx.hide_poi and detail.poi:
        return None
    if detail.nsfw and display_mode(ctx) == "hide":
        return None
    versions = [kept for kept in (filter_version(version, ctx) for version in detail.versions) if kept is not None]
    return detail.model_copy(update={"versions": versions})


def nsfw_media(detail: ModelDetail, ctx: NsfwContext) -> List[Image]:
    if ctx.ceiling == "blocked":
        raise NsfwForbidden("NSFW media is not available to this user")
    seen = set()
    images: List[Image] = []
    for version in detail.versions:
        for image in version.images:
            if not image.nsfw or not image.url or image.url in seen:
                continue
            if ctx.hide_poi and image.poi:
                continue
            seen.add(image.url)
            images.append(image)
    return images
