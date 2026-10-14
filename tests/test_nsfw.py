import pytest

from backend import core_bridge, settings
from backend.nsfw import bounded_mode, compute_nsfw, effective_ceiling, resolve_nsfw
from backend.schema import NsfwContext
from src.plugin_api import AccountType, User

POLICIES = ("allowed", "blur", "blocked")
CEILING_SETTINGS = ("inherit", "blur", "blocked")
USER_MODES = ("show", "blur", "hide")

CEILING = {
    ("allowed", "inherit"): "allowed",
    ("allowed", "blur"): "blur",
    ("allowed", "blocked"): "blocked",
    ("blur", "inherit"): "blur",
    ("blur", "blur"): "blur",
    ("blur", "blocked"): "blocked",
    ("blocked", "inherit"): "blocked",
    ("blocked", "blur"): "blocked",
    ("blocked", "blocked"): "blocked",
}

MODE = {
    ("allowed", "show"): "show",
    ("allowed", "blur"): "blur",
    ("allowed", "hide"): "hide",
    ("blur", "show"): "blur",
    ("blur", "blur"): "blur",
    ("blur", "hide"): "hide",
    ("blocked", "show"): "hide",
    ("blocked", "blur"): "hide",
    ("blocked", "hide"): "hide",
}

ALLOWED = {
    "allowed": ("blur", "show", "hide"),
    "blur": ("blur", "hide"),
    "blocked": ("hide",),
}

MATRIX = [(policy, ceiling, mode) for policy in POLICIES for ceiling in CEILING_SETTINGS for mode in USER_MODES]


def _user():
    return User(
        id="user-1",
        username="user",
        email="user@test.com",
        password_hash="hash",
        account_type=AccountType.USER,
    )


@pytest.mark.parametrize("core_policy, ceiling_setting, user_mode", MATRIX, ids=["-".join(case) for case in MATRIX])
def test_the_matrix(core_policy, ceiling_setting, user_mode):
    context = compute_nsfw(core_policy, ceiling_setting, user_mode, max_level=2, hide_poi=False)

    ceiling = CEILING[(core_policy, ceiling_setting)]
    assert context.ceiling == ceiling
    assert context.mode == MODE[(ceiling, user_mode)]
    assert context.allowed_modes == ALLOWED[ceiling]
    assert context.mode in context.allowed_modes
    assert context.blocked is (ceiling == "blocked")


@pytest.mark.parametrize("core_policy", ["restricted", "ALLOWED", "", None])
@pytest.mark.parametrize("ceiling_setting", CEILING_SETTINGS)
def test_a_core_policy_the_plugin_does_not_know_is_blocked(core_policy, ceiling_setting):
    context = compute_nsfw(core_policy, ceiling_setting, "show", max_level=2, hide_poi=False)

    assert (context.ceiling, context.mode, context.allowed_modes) == ("blocked", "hide", ("hide",))


@pytest.mark.parametrize("ceiling_setting", ["allowed", "", "never"])
@pytest.mark.parametrize("core_policy", POLICIES)
def test_a_ceiling_setting_that_cannot_tighten_follows_the_core(core_policy, ceiling_setting):
    assert effective_ceiling(core_policy, ceiling_setting) == core_policy


@pytest.mark.parametrize("ceiling, mode", [("allowed", "blur"), ("blur", "blur"), ("blocked", "hide")])
def test_a_user_mode_the_plugin_does_not_know_falls_back(ceiling, mode):
    assert bounded_mode(ceiling, "explicit") == mode
    assert bounded_mode(ceiling, "") == mode


def test_the_level_and_poi_settings_pass_through():
    context = compute_nsfw("allowed", "inherit", "show", max_level=8, hide_poi=True)

    assert (context.max_level, context.hide_poi) == (8, True)


@pytest.mark.parametrize(
    "core_policy, ceiling_setting, user_mode, expected",
    [
        ("allowed", "inherit", "show", ("allowed", "show")),
        ("allowed", "blocked", "show", ("blocked", "hide")),
        ("blur", "inherit", "show", ("blur", "blur")),
        ("blocked", "inherit", "blur", ("blocked", "hide")),
    ],
)
def test_resolve_nsfw_reads_the_core_policy_the_settings_and_the_users_preference(
    monkeypatch, core_policy, ceiling_setting, user_mode, expected
):
    asked = []
    values = {"nsfw_ceiling": ceiling_setting, "nsfw_max_level": 4, "hide_poi": True}

    def effective_policy(user_id):
        asked.append(("policy", user_id))
        return core_policy

    def get_setting(key):
        asked.append(("setting", key))
        return values[key]

    def user_nsfw_mode(user_id):
        asked.append(("mode", user_id))
        return user_mode

    monkeypatch.setattr(core_bridge, "effective_policy", effective_policy)
    monkeypatch.setattr(settings, "get_setting", get_setting)
    monkeypatch.setattr(settings, "user_nsfw_mode", user_nsfw_mode)

    context = resolve_nsfw(_user())

    ceiling, mode = expected
    assert context == NsfwContext(
        ceiling=ceiling, mode=mode, allowed_modes=ALLOWED[ceiling], max_level=4, hide_poi=True
    )
    assert sorted(asked) == [
        ("mode", "user-1"),
        ("policy", "user-1"),
        ("setting", "hide_poi"),
        ("setting", "nsfw_ceiling"),
        ("setting", "nsfw_max_level"),
    ]
