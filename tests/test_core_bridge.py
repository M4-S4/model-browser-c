import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from backend import core_bridge
from backend.core_bridge import CoreBridgeError
from src.plugin_api import (
    AccountType,
    DownloadQueueException,
    InvalidModelMetadataException,
    LimitStatus,
    ModelAssignmentException,
    ModelNotFoundException,
    NotificationTypeRegistry,
    NotificationTypeSpec,
    ProviderFetchException,
    UnsafeFilenameException,
    User,
)

_ADMIN = User(id="admin-1", username="admin", email="admin@test.com", password_hash="hash", account_type=AccountType.ADMIN)
_URL = "https://civitai.com/api/download/models/135867"


def _stand_in(real, answer=None):
    signature = inspect.signature(real)
    calls = []

    def respond(args, kwargs):
        arguments = dict(signature.bind(*args, **kwargs).arguments)
        calls.append(arguments)
        return answer(**arguments) if callable(answer) else answer

    if inspect.iscoroutinefunction(real):
        async def stand_in(*args, **kwargs):
            return respond(args, kwargs)

    else:
        def stand_in(*args, **kwargs):
            return respond(args, kwargs)

    stand_in.calls = calls
    return stand_in


def _replace(monkeypatch, module, name, answer=None):
    stand_in = _stand_in(getattr(module, name), answer)
    monkeypatch.setattr(module, name, stand_in)
    return stand_in


def _raise(exception):
    def answer(**arguments):
        raise exception

    return answer


@pytest.mark.parametrize(
    "message, code",
    [
        ("Quota plugin: monthly download limit reached", "blocked"),
        ("No model root resolver is configured", "blocked"),
        ("Subfolder '../x' must be a relative path with no '..' segments", "invalid"),
        ("Destination directory 'x' escapes the configured directory '/app/models'", "invalid"),
        ("No backend configured with id 'worker-9'", "invalid"),
        ("Backend 'worker-1' is not a native.remote backend", "invalid"),
        ("Backend 'worker-1' is not fully configured", "invalid"),
    ],
)
async def test_queue_model_download_turns_the_cores_refusals_into_codes(monkeypatch, message, code):
    _replace(monkeypatch, core_bridge.api_downloads, "queue_model_download", _raise(DownloadQueueException(message)))

    with pytest.raises(CoreBridgeError) as raised:
        await core_bridge.queue_model_download(_URL, model_type="lora")

    assert (raised.value.code, raised.value.message) == (code, message)


async def test_an_unsafe_file_name_is_invalid(monkeypatch):
    refusal = UnsafeFilenameException("Filename '..' would escape its directory")
    _replace(monkeypatch, core_bridge.api_downloads, "queue_model_download", _raise(refusal))

    with pytest.raises(CoreBridgeError) as raised:
        await core_bridge.queue_model_download(_URL, filename="..")

    assert raised.value.code == "invalid"


def test_find_model_by_id_lets_any_other_failure_through(monkeypatch):
    _replace(
        monkeypatch, core_bridge.api_models, "resolve_model_file", _raise(RuntimeError("AppContainer not initialized"))
    )

    with pytest.raises(RuntimeError, match="AppContainer"):
        core_bridge.find_model_by_id("model-1")


def test_visible_model_ids_are_the_exports(monkeypatch):
    visible = _replace(monkeypatch, core_bridge.api_models, "visible_model_ids", ["model-1", "model-2"])

    assert core_bridge.visible_model_ids("user-1") == ["model-1", "model-2"]
    assert visible.calls == [{"user_id": "user-1"}]


def test_assign_to_user_says_whether_the_model_was_newly_assigned(monkeypatch):
    assigned = set()

    def assign(model_id, user_id):
        if user_id == "ghost":
            raise ModelAssignmentException(f"Could not assign model '{model_id}' to user '{user_id}' - no such user")
        newly = (model_id, user_id) not in assigned
        assigned.add((model_id, user_id))
        return newly

    _replace(monkeypatch, core_bridge.api_models, "assign_model_to_user", assign)

    assert core_bridge.assign_to_user("model-1", "user-1") is True
    assert core_bridge.assign_to_user("model-1", "user-1") is False
    with pytest.raises(CoreBridgeError) as raised:
        core_bridge.assign_to_user("model-1", "ghost")
    assert raised.value.code == "refused" and "no such user" in raised.value.message


def _groups(monkeypatch, answer=lambda group_id, model_ids, acting_user: list(model_ids)):
    _replace(
        monkeypatch,
        core_bridge.api_groups,
        "list_groups",
        [
            {"id": "all_users", "name": "All Users", "is_system": True, "content_policy": None},
            {"id": "artists", "name": "Artists", "is_system": False, "content_policy": "blur"},
        ],
    )
    return _replace(monkeypatch, core_bridge.api_groups, "assign_models_to_group", answer)


def test_assign_to_group_goes_through_the_cores_group_operation_in_the_acting_users_name(monkeypatch):
    operation = _groups(monkeypatch, lambda group_id, model_ids, acting_user: model_ids[:1])

    assert core_bridge.assign_to_group("artists", ["model-1", "", "model-3"], _ADMIN) == ["model-1"]
    assert operation.calls == [{"group_id": "artists", "model_ids": ["model-1", "model-3"], "acting_user": _ADMIN}]


def test_assign_to_group_refuses_a_group_that_does_not_exist_without_calling_the_operation(monkeypatch):
    operation = _groups(monkeypatch)

    for model_ids in (["model-1"], []):
        with pytest.raises(CoreBridgeError) as raised:
            core_bridge.assign_to_group("ghosts", model_ids, _ADMIN)
        assert raised.value.code == "not_found"
    assert operation.calls == []


def test_assign_to_group_turns_the_cores_value_error_into_refused(monkeypatch):
    _groups(monkeypatch, _raise(ValueError("Admin access required")))
    user = User(id="user-1", username="user", email="user@test.com", password_hash="hash", account_type=AccountType.USER)

    with pytest.raises(CoreBridgeError) as raised:
        core_bridge.assign_to_group("artists", ["model-1"], user)

    assert (raised.value.code, raised.value.message) == ("refused", "Admin access required")


@pytest.mark.parametrize("acting_user", [None, "admin-1"], ids=["none", "an-id-not-a-user"])
def test_assign_to_group_refuses_without_an_acting_user(monkeypatch, acting_user):
    operation = _groups(monkeypatch)

    with pytest.raises(CoreBridgeError) as raised:
        core_bridge.assign_to_group("artists", ["model-1"], acting_user)

    assert raised.value.code == "refused" and "acting admin" in raised.value.message
    assert operation.calls == []


def test_assign_to_group_without_a_model_asks_the_operation_nothing(monkeypatch):
    operation = _groups(monkeypatch)

    assert core_bridge.assign_to_group("artists", [], _ADMIN) == []
    assert core_bridge.assign_to_group("artists", ["", ""], _ADMIN) == []
    assert operation.calls == []


async def test_run_provider_fetch_asks_for_these_models_only(monkeypatch):
    fetch = _replace(
        monkeypatch, core_bridge.api_models, "run_provider_fetch", {"successful": 1, "failed": 0, "errors": []}
    )

    assert await core_bridge.run_provider_fetch("model-browser", ("model-1", "")) == {
        "successful": 1,
        "failed": 0,
        "errors": [],
    }
    assert fetch.calls == [{"provider": "model-browser", "model_ids": ["model-1"]}]
    assert await core_bridge.run_provider_fetch("model-browser", [""]) == {"successful": 0, "failed": 0}
    assert len(fetch.calls) == 1


async def test_run_provider_fetch_lets_the_cores_refusal_through(monkeypatch):
    refusal = ProviderFetchException("Provider 'model-browser' is not initialised")
    _replace(monkeypatch, core_bridge.api_models, "run_provider_fetch", _raise(refusal))

    with pytest.raises(ProviderFetchException):
        await core_bridge.run_provider_fetch("model-browser", ["model-1"])


def test_write_attributes_hands_the_values_to_the_core(monkeypatch):
    update = _replace(monkeypatch, core_bridge.api_models, "update_model_metadata", {"mb_license": "MIT"})

    assert core_bridge.write_attributes("model-1", {"mb_base_model": "SDXL 1.0", "mb_license": "MIT"}) is None
    assert update.calls == [{"model_id": "model-1", "values": {"mb_base_model": "SDXL 1.0", "mb_license": "MIT"}}]
    assert type(update.calls[0]["values"]) is dict


@pytest.mark.parametrize(
    "refusal, code",
    [
        (InvalidModelMetadataException("'unknown' is not a declared attribute for model type 'lora'"), "invalid"),
        (ModelNotFoundException("Model 'model-gone' not found"), "not_found"),
    ],
    ids=["invalid", "not-found"],
)
def test_write_attributes_turns_the_cores_refusals_into_codes(monkeypatch, refusal, code):
    _replace(monkeypatch, core_bridge.api_models, "update_model_metadata", _raise(refusal))

    with pytest.raises(CoreBridgeError) as raised:
        core_bridge.write_attributes("model-1", {"mb_license": "MIT"})

    assert (raised.value.code, raised.value.message) == (code, str(refusal))


@pytest.mark.parametrize("mode", ["allowed", "blur", "blocked"])
def test_effective_policy_is_the_policy_the_core_enforces_for_the_user(monkeypatch, mode):
    policy = _replace(monkeypatch, core_bridge.api_groups, "effective_content_policy", mode)

    assert core_bridge.effective_policy("user-1") == mode
    assert policy.calls == [{"user_id": "user-1"}]


@pytest.mark.parametrize("mode", ["restricted", "ALLOWED", "", None])
def test_effective_policy_counts_a_mode_it_does_not_know_as_blocked(monkeypatch, mode):
    _replace(monkeypatch, core_bridge.api_groups, "effective_content_policy", mode)

    assert core_bridge.effective_policy("user-1") == "blocked"


_GROUP_DICTS = [
    {"id": "all_admins", "name": "All Admins", "is_system": True, "content_policy": None},
    {"id": "artists", "name": "Artists", "is_system": False, "content_policy": "blur"},
]


def test_user_groups_list_groups_and_members_are_the_exports(monkeypatch):
    mine = _replace(monkeypatch, core_bridge.api_groups, "get_user_groups", _GROUP_DICTS[1:])
    _replace(monkeypatch, core_bridge.api_groups, "list_groups", _GROUP_DICTS)
    members = _replace(monkeypatch, core_bridge.api_groups, "group_member_ids", ["user-2", "user-1"])

    assert core_bridge.user_groups("user-1") == _GROUP_DICTS[1:]
    assert core_bridge.list_groups() == _GROUP_DICTS
    assert core_bridge.group_member_ids("approvers") == ["user-2", "user-1"]
    assert mine.calls == [{"user_id": "user-1"}]
    assert members.calls == [{"group_id": "approvers"}]


def test_notify_sends_one_users_notification_from_the_plugin(monkeypatch):
    sent = _replace(monkeypatch, core_bridge.api_notifications, "notify", 1)

    core_bridge.notify(
        user_id="user-1",
        notification_type="model-browser.download_done",
        title="Model downloaded",
        message="Detail Tweaker XL is ready",
        level="success",
        metadata={"model_id": "model-1"},
    )

    assert sent.calls == [
        {
            "source": "model-browser",
            "title": "Model downloaded",
            "message": "Detail Tweaker XL is ready",
            "user_id": "user-1",
            "type": "model-browser.download_done",
            "level": "success",
            "category": "plugin",
            "metadata": {"model_id": "model-1"},
        }
    ]


@pytest.mark.parametrize("user_id", ["", None])
def test_notify_never_broadcasts(monkeypatch, caplog, user_id):
    sent = _replace(monkeypatch, core_bridge.api_notifications, "notify", 3)

    core_bridge.notify(user_id=user_id, notification_type="model-browser.download_done", title="x")

    assert sent.calls == []
    assert "no user to notify" in caplog.text


def test_notify_never_raises(monkeypatch, caplog):
    sent = _replace(
        monkeypatch, core_bridge.api_notifications, "notify", _raise(ValueError("'loud' is not a valid NotificationLevel"))
    )

    core_bridge.notify(user_id="user-1", notification_type="model-browser.download_done", title="x", level="loud")

    assert len(sent.calls) == 1 and sent.calls[0]["metadata"] is None
    assert "Could not send" in caplog.text


def test_register_notification_types_declares_each_type_once_on_the_cores_registry():
    registry = NotificationTypeRegistry()
    registry.register(NotificationTypeSpec(key="model-browser.download_done", label="Already there"))
    types = [
        SimpleNamespace(key="model-browser.download_done", label="Model ready", description="d", category="plugin"),
        SimpleNamespace(key="model-browser.request_received", label="New request", description="r", category="plugin"),
    ]

    assert core_bridge.register_notification_types(registry, types) == ["model-browser.request_received"]
    assert core_bridge.register_notification_types(registry, types) == []
    spec = registry.get("model-browser.request_received")
    assert isinstance(spec, NotificationTypeSpec)
    assert (spec.label, spec.description, spec.category, spec.default_enabled, spec.admin_only) == (
        "New request", "r", "plugin", True, False,
    )
    assert registry.get("model-browser.download_done").label == "Already there"


_KIND = "model-browser.downloads"


@pytest.mark.parametrize(
    "status, expected",
    [
        (
            LimitStatus(kind=_KIND),
            {"kind": _KIND, "limit": None, "used": None, "exempt": False, "resets_at": None, "contact_line": ""},
        ),
        (
            LimitStatus(kind=_KIND, limit=1, exempt=True, contact_line="Ask your admin."),
            {"kind": _KIND, "limit": 1, "used": None, "exempt": True, "resets_at": None, "contact_line": "Ask your admin."},
        ),
        (
            LimitStatus(
                kind=_KIND,
                limit=5,
                used=2.0,
                resets_at=datetime(2026, 10, 31, 23, 0, tzinfo=timezone.utc),
                contact_line="Ask your admin.",
            ),
            {
                "kind": _KIND,
                "limit": 5,
                "used": 2.0,
                "exempt": False,
                "resets_at": "2026-10-31T23:00:00+00:00",
                "contact_line": "Ask your admin.",
            },
        ),
        (
            LimitStatus(kind=_KIND, limit=5, used=5.0, resets_at=datetime(2026, 11, 1, tzinfo=ZoneInfo("Europe/Paris"))),
            {
                "kind": _KIND,
                "limit": 5,
                "used": 5.0,
                "exempt": False,
                "resets_at": "2026-10-31T23:00:00+00:00",
                "contact_line": "",
            },
        ),
    ],
    ids=["unlimited", "exempt", "measured", "measured-in-another-zone"],
)
def test_limit_status_is_the_cores_status_as_a_dict_with_the_reset_in_utc(monkeypatch, status, expected):
    read = _replace(monkeypatch, core_bridge.api_limits, "limit_status", status)

    assert core_bridge.limit_status("user-1", _KIND) == expected
    assert read.calls == [{"user_id": "user-1", "kind": _KIND}]


def test_limit_status_reads_unlimited_when_the_plans_cannot_be_read(monkeypatch, caplog):
    _replace(monkeypatch, core_bridge.api_limits, "limit_status", _raise(RuntimeError("no such table: plans")))

    status = core_bridge.limit_status("user-1", _KIND)

    assert status == {"kind": _KIND, "limit": None, "used": None, "exempt": False, "resets_at": None, "contact_line": ""}
    assert "treating it as unlimited" in caplog.text
