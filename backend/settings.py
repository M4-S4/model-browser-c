from typing import Any, Callable, Dict, List
from urllib.parse import urlsplit

from . import PLUGIN_ID

CORE_USER_NSFW_SETTING = "media_nsfw_filter_mode"
USER_NSFW_MODES = ("show", "blur", "hide")

DEFAULTS: Dict[str, Any] = {
    "nsfw_ceiling": "inherit",
    "nsfw_max_level": 2,
    "hide_poi": False,
    "civitai_api_base": "https://civitai.com/api/v1",
    "enable_civarchive": False,
    "cache_ttl_minutes": 10,
}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_choice(*choices: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        value = str(value).strip().lower()
        if value not in choices:
            raise ValueError(value)
        return value

    return parse


def _parse_int(lo: int, hi: int, allowed: tuple = ()) -> Callable[[str], int]:
    def parse(value: str) -> int:
        number = int(str(value).strip())
        if allowed and number not in allowed:
            raise ValueError(value)
        return min(max(number, lo), hi)

    return parse


def _parse_api_base(value: str) -> str:
    value = str(value).strip().rstrip("/")
    if not value.startswith("https://") or not urlsplit(value).netloc:
        raise ValueError(value)
    return value


BOOLEAN_SETTINGS = ("hide_poi", "enable_civarchive")

_PARSERS: Dict[str, Callable[[str], Any]] = {
    "nsfw_ceiling": _parse_choice("inherit", "blur", "blocked"),
    "nsfw_max_level": _parse_int(1, 8, allowed=(1, 2, 4, 8)),
    "hide_poi": _parse_bool,
    "civitai_api_base": _parse_api_base,
    "enable_civarchive": _parse_bool,
    "cache_ttl_minutes": _parse_int(1, 1440),
}


def parse_setting(key: str, raw: Any) -> Any:
    if key not in DEFAULTS:
        raise KeyError(f"model-browser-c has no setting {key!r}")
    if raw is None or str(raw).strip() == "":
        return DEFAULTS[key]
    try:
        return _PARSERS[key](raw)
    except (ValueError, TypeError):
        return DEFAULTS[key]


def get_setting(key: str) -> Any:
    from src.plugin_api import PluginRepository

    row = PluginRepository().get_plugin_setting(PLUGIN_ID, key)
    return parse_setting(key, None if row is None else row.setting_value)


def normalise_stored_booleans() -> List[str]:
    from src.plugin_api import PluginRepository

    repository = PluginRepository()
    rewritten: List[str] = []
    for key in BOOLEAN_SETTINGS:
        row = repository.get_plugin_setting(PLUGIN_ID, key)
        if row is None or row.setting_value in ("true", "false"):
            continue
        value = "true" if parse_setting(key, row.setting_value) else "false"
        repository.set_plugin_setting(plugin_id=PLUGIN_ID, setting_key=key, setting_value=value)
        rewritten.append(key)
    return rewritten


def user_nsfw_mode(user_id: str) -> str:
    from src.plugin_api import SettingRepository

    value = SettingRepository().get_user_setting_by_key(user_id, CORE_USER_NSFW_SETTING)
    value = str(value).strip().lower() if value is not None else ""
    return value if value in USER_NSFW_MODES else "blur"
