from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.settings import (
    CORE_USER_NSFW_SETTING,
    DEFAULTS,
    get_setting,
    parse_setting,
    user_nsfw_mode,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://civitai.com/api/v1"


def test_the_defaults_and_options_match_the_manifest():
    manifest = yaml.safe_load((PLUGIN_ROOT / "manifest.yml").read_text(encoding="utf-8"))
    specs = {spec["name"]: spec for spec in manifest["settings"] if spec["type"] != "info"}

    assert set(specs) == set(DEFAULTS)
    for key, spec in specs.items():
        assert parse_setting(key, spec["default"]) == DEFAULTS[key], key
        for option in spec.get("options", []):
            assert str(parse_setting(key, option["value"])).lower() == option["value"], (key, option)


def test_the_yes_no_settings_are_manifest_booleans():
    manifest = yaml.safe_load((PLUGIN_ROOT / "manifest.yml").read_text(encoding="utf-8"))
    specs = {spec["name"]: spec for spec in manifest["settings"]}

    booleans = {key for key, value in DEFAULTS.items() if isinstance(value, bool)}
    assert booleans == {key for key, spec in specs.items() if spec["type"] == "boolean"}
    for key in booleans:
        assert specs[key]["default"] is DEFAULTS[key], key
        assert (parse_setting(key, "true"), parse_setting(key, "false")) == (True, False), key


@pytest.mark.parametrize("raw", [None, "", "   "])
@pytest.mark.parametrize("key", sorted(DEFAULTS))
def test_an_unset_setting_is_its_default(key, raw):
    assert parse_setting(key, raw) == DEFAULTS[key]


@pytest.mark.parametrize(
    "key, raw, value",
    [
        ("nsfw_ceiling", "inherit", "inherit"),
        ("nsfw_ceiling", "blur", "blur"),
        ("nsfw_ceiling", "blocked", "blocked"),
        ("nsfw_ceiling", " Blocked ", "blocked"),
        ("nsfw_ceiling", "allowed", "inherit"),
        ("nsfw_ceiling", "hide", "inherit"),
        ("nsfw_max_level", "1", 1),
        ("nsfw_max_level", "4", 4),
        ("nsfw_max_level", " 8 ", 8),
        ("nsfw_max_level", 2, 2),
        ("nsfw_max_level", "3", 2),
        ("nsfw_max_level", "16", 2),
        ("nsfw_max_level", "0", 2),
        ("nsfw_max_level", "-1", 2),
        ("nsfw_max_level", "R", 2),
        ("nsfw_max_level", "4.0", 2),
        ("hide_poi", "True", True),
        ("hide_poi", "true", True),
        ("hide_poi", "1", True),
        ("hide_poi", "yes", True),
        ("hide_poi", " on ", True),
        ("hide_poi", True, True),
        ("hide_poi", "False", False),
        ("hide_poi", "0", False),
        ("hide_poi", "off", False),
        ("hide_poi", False, False),
        ("hide_poi", "maybe", False),
        ("enable_civarchive", "True", True),
        ("enable_civarchive", "false", False),
        ("enable_civarchive", "maybe", False),
        ("civitai_api_base", "https://civitai.red/api/v1", "https://civitai.red/api/v1"),
        ("civitai_api_base", "https://civitai.red/api/v1/", "https://civitai.red/api/v1"),
        ("civitai_api_base", "  https://civitai.com/api/v1//  ", "https://civitai.com/api/v1"),
        ("civitai_api_base", "http://civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "ftp://civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "https://", DEFAULT_BASE),
        ("civitai_api_base", "https:///api/v1", DEFAULT_BASE),
        ("civitai_api_base", "https://[::1/api/v1", DEFAULT_BASE),
        ("cache_ttl_minutes", "30", 30),
        ("cache_ttl_minutes", 45, 45),
        ("cache_ttl_minutes", "1", 1),
        ("cache_ttl_minutes", "1440", 1440),
        ("cache_ttl_minutes", "0", 1),
        ("cache_ttl_minutes", "-5", 1),
        ("cache_ttl_minutes", "5000", 1440),
        ("cache_ttl_minutes", "ten", 10),
        ("cache_ttl_minutes", "7.5", 10),
    ],
)
def test_parse_setting(key, raw, value):
    assert parse_setting(key, raw) == value


def test_parse_setting_refuses_a_key_the_plugin_does_not_have():
    with pytest.raises(KeyError):
        parse_setting("nsfw_filter", "true")


class _Repository:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def __call__(self):
        return self

    def get_user_setting_by_key(self, user_id, key):
        self.asked.append((user_id, key))
        return self.answer(key)

    def get_plugin_setting(self, plugin_id, key):
        self.asked.append((plugin_id, key))
        return self.answer(key)


@pytest.mark.parametrize(
    "stored, mode",
    [
        ("show", "show"),
        ("blur", "blur"),
        ("hide", "hide"),
        (" SHOW ", "show"),
        (None, "blur"),
        ("", "blur"),
        ("explicit", "blur"),
    ],
)
def test_user_nsfw_mode_reads_the_cores_user_setting(monkeypatch, stored, mode):
    repository = _Repository(lambda key: stored)
    monkeypatch.setattr("src.plugin_api.SettingRepository", repository)

    assert user_nsfw_mode("user-1") == mode
    assert repository.asked == [("user-1", CORE_USER_NSFW_SETTING)]
    assert CORE_USER_NSFW_SETTING == "media_nsfw_filter_mode"


def test_get_setting_parses_what_the_core_stored_for_the_plugin(monkeypatch):
    stored = {
        "nsfw_ceiling": SimpleNamespace(setting_value="blocked"),
        "cache_ttl_minutes": SimpleNamespace(setting_value="99999"),
        "hide_poi": SimpleNamespace(setting_value="True"),
        "nsfw_max_level": SimpleNamespace(setting_value=None),
    }
    repository = _Repository(stored.get)
    monkeypatch.setattr("src.plugin_api.PluginRepository", repository)

    assert get_setting("nsfw_ceiling") == "blocked"
    assert get_setting("cache_ttl_minutes") == 1440
    assert get_setting("hide_poi") is True
    assert get_setting("nsfw_max_level") == 2
    assert get_setting("civitai_api_base") == DEFAULT_BASE
    assert repository.asked == [
        ("model-browser-c", "nsfw_ceiling"),
        ("model-browser-c", "cache_ttl_minutes"),
        ("model-browser-c", "hide_poi"),
        ("model-browser-c", "nsfw_max_level"),
        ("model-browser-c", "civitai_api_base"),
    ]
