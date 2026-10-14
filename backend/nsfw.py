from typing import Tuple

from src.plugin_api import User

from . import core_bridge, settings
from .schema import NsfwCeiling, NsfwContext, NsfwMode

_CEILING_RANK = {"allowed": 0, "blur": 1, "blocked": 2}

ALLOWED_MODES = {
    "allowed": ("blur", "show", "hide"),
    "blur": ("blur", "hide"),
    "blocked": ("hide",),
}


def effective_ceiling(core_policy: str, ceiling_setting: str) -> NsfwCeiling:
    core = core_policy if core_policy in _CEILING_RANK else "blocked"
    if ceiling_setting in ("blur", "blocked") and _CEILING_RANK[ceiling_setting] > _CEILING_RANK[core]:
        return ceiling_setting  # type: ignore[return-value]
    return core  # type: ignore[return-value]


def bounded_mode(ceiling: NsfwCeiling, user_mode: str) -> NsfwMode:
    allowed: Tuple[str, ...] = ALLOWED_MODES[ceiling]
    if user_mode in allowed:
        return user_mode  # type: ignore[return-value]
    if ceiling == "blocked":
        return "hide"
    return "blur"


def compute_nsfw(
    core_policy: str,
    ceiling_setting: str,
    user_mode: str,
    max_level: int,
    hide_poi: bool,
) -> NsfwContext:
    ceiling = effective_ceiling(core_policy, ceiling_setting)
    return NsfwContext(
        ceiling=ceiling,
        mode=bounded_mode(ceiling, user_mode),
        allowed_modes=ALLOWED_MODES[ceiling],
        max_level=max_level,
        hide_poi=hide_poi,
    )


def resolve_nsfw(user: User) -> NsfwContext:
    return compute_nsfw(
        core_policy=core_bridge.effective_policy(user.id),
        ceiling_setting=settings.get_setting("nsfw_ceiling"),
        user_mode=settings.user_nsfw_mode(user.id),
        max_level=settings.get_setting("nsfw_max_level"),
        hide_poi=settings.get_setting("hide_poi"),
    )
