import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import core_bridge, permissions
from backend.permissions import (
    ADMIN_GRANT,
    CAPABILITIES,
    DEFAULT_SOURCES,
    NO_PARAMS,
    Grant,
    Params,
    Principal,
    current_principal,
    merge_rows,
    require,
    require_any,
    require_browse,
    require_download,
    resolve_grant,
)
from backend.permissions_store import Row, default_row, load_rows, save_rows
from backend.tables import create_tables
from src.plugin_api import AccountType, User, get_current_active_user

_FORBIDDEN_DOWNLOAD = {
    "success": False,
    "error": "forbidden",
    "message": "This action needs the 'download' permission",
}


def _user(user_id="user-1", account_type=AccountType.USER):
    return User(
        id=user_id,
        username=user_id,
        email=f"{user_id}@test.com",
        password_hash="hash",
        account_type=account_type,
    )


@pytest.fixture
def membership(plugin_db, monkeypatch):
    create_tables()
    members = {}

    def user_groups(user_id):
        return [
            {
                "id": group_id,
                "name": group_id,
                "is_system": group_id in ("all_users", "all_admins"),
                "content_policy": None,
            }
            for group_id in members.get(user_id, [])
        ]

    monkeypatch.setattr(core_bridge, "user_groups", user_groups)
    return members


def _download_row(group_id, **params):
    return Row(group_id=group_id, capabilities=frozenset({"download"}), **params)


def _request_row(group_id, **params):
    return Row(group_id=group_id, capabilities=frozenset({"request"}), **params)


def _view(params):
    return (params.sources, params.model_types, params.max_file_gb, params.assign_to, params.allow_remote)


def test_the_public_names_stay_importable_from_permissions():
    assert permissions.CAPABILITIES == ("browse", "request", "download", "approve")
    assert permissions.DEFAULT_CAPABILITIES["all_users"] == {"browse", "request"}
    assert permissions.DEFAULT_CAPABILITIES["all_admins"] == set(CAPABILITIES)
    assert permissions.DEFAULT_SOURCES == {"civitai", "huggingface", "openmodeldb"}


def test_an_admin_gets_the_admin_grant_without_a_group_lookup(monkeypatch):
    def user_groups(user_id):
        raise AssertionError("an admin's groups are not read")

    monkeypatch.setattr(core_bridge, "user_groups", user_groups)

    grant = resolve_grant(_user("admin-1", AccountType.ADMIN))

    assert grant is ADMIN_GRANT
    assert grant.unrestricted
    assert all(grant.has(capability) for capability in CAPABILITIES)
    for purpose in ("download", "request"):
        assert grant.allows_source("civarchive", purpose) and grant.allows_type("checkpoint", purpose)


def test_a_user_in_all_users_only_may_browse_and_request_by_default(membership):
    membership["user-1"] = ["all_users"]

    grant = resolve_grant(_user())

    assert grant.capabilities == {"browse", "request"}
    assert not grant.has("download") and not grant.has("approve")
    assert not grant.unrestricted
    assert (grant.sources, grant.model_types, grant.max_file_gb, grant.assign_to, grant.allow_remote) == (
        frozenset(),
        frozenset(),
        None,
        "user",
        False,
    )
    assert not grant.allows_source("civitai")
    assert _view(grant.request_params) == (DEFAULT_SOURCES, frozenset(), None, "user", False)
    assert grant.allows_source("civitai", "request") and not grant.allows_source("civarchive", "request")


def test_a_non_admin_account_in_all_admins_gets_every_capability_with_the_default_parameters(membership):
    membership["user-1"] = ["all_users", "all_admins"]

    grant = resolve_grant(_user())

    assert grant.capabilities == set(CAPABILITIES)
    assert not grant.unrestricted
    assert grant.sources == DEFAULT_SOURCES
    assert not grant.allows_source("civarchive")
    assert grant.model_types == frozenset()
    assert (grant.max_file_gb, grant.assign_to, grant.allow_remote) == (None, "user", False)
    assert _view(grant.request_params) == (DEFAULT_SOURCES, frozenset(), None, "user", False)


def test_a_stored_row_replaces_the_groups_default(membership):
    membership["user-1"] = ["all_users"]
    save_rows([Row(group_id="all_users", capabilities=frozenset({"browse"}))])

    assert resolve_grant(_user()).capabilities == {"browse"}


def test_two_downloading_groups_merge_their_parameters(membership):
    membership["user-1"] = ["all_users", "lora-makers", "hf-users"]
    save_rows(
        [
            _download_row("lora-makers", sources={"civitai"}, model_types={"lora"}, max_file_gb=2.0),
            Row(
                group_id="hf-users",
                capabilities=frozenset({"browse", "download"}),
                sources=frozenset({"huggingface"}),
                model_types=frozenset(),
                max_file_gb=10.0,
                assign_to="group",
                allow_remote=True,
            ),
        ]
    )

    grant = resolve_grant(_user())

    assert grant.capabilities == {"browse", "request", "download"}
    assert grant.sources == {"civitai", "huggingface"}
    assert grant.model_types == frozenset()
    assert grant.allows_type("checkpoint")
    assert grant.max_file_gb == 10.0
    assert grant.assign_to == "group"
    assert grant.allow_remote is True


def test_the_default_all_users_row_does_not_widen_a_stricter_download_group(membership):
    membership["user-1"] = ["all_users", "lora-makers"]
    save_rows(
        [_download_row("lora-makers", sources={"civitai"}, model_types={"lora", "embedding"}, max_file_gb=2.0)]
    )

    grant = resolve_grant(_user())

    assert grant.capabilities == {"browse", "request", "download"}
    assert _view(grant.download_params) == ({"civitai"}, {"lora", "embedding"}, 2.0, "user", False)
    assert not grant.allows_type("checkpoint") and not grant.allows_source("huggingface")
    assert _view(grant.request_params) == (DEFAULT_SOURCES, frozenset(), None, "user", False)
    assert grant.allows_type("checkpoint", "request") and grant.allows_source("huggingface", "request")


def test_a_user_with_request_only_has_an_empty_download_view_and_the_request_view(membership):
    membership["user-1"] = ["requesters"]
    save_rows(
        [
            Row(
                group_id="requesters",
                capabilities=frozenset({"browse", "request"}),
                sources=frozenset({"huggingface"}),
                model_types=frozenset({"embedding"}),
                max_file_gb=0.5,
                assign_to="group",
                allow_remote=True,
            )
        ]
    )

    grant = resolve_grant(_user())

    assert grant.download_params == NO_PARAMS
    assert not grant.allows_source("huggingface")
    assert _view(grant.request_params) == ({"huggingface"}, {"embedding"}, 0.5, "group", False)


def test_a_user_with_download_only_has_an_empty_request_view(membership):
    membership["user-1"] = ["downloaders"]
    save_rows([_download_row("downloaders", sources={"civitai"}, model_types={"lora"}, max_file_gb=1.5)])

    grant = resolve_grant(_user())

    assert grant.request_params == NO_PARAMS
    assert not grant.allows_source("civitai", "request")
    assert _view(grant.download_params) == ({"civitai"}, {"lora"}, 1.5, "user", False)


def test_a_group_without_a_row_that_has_no_default_grants_nothing(membership):
    membership["user-1"] = ["artists"]

    grant = resolve_grant(_user())

    assert grant.capabilities == frozenset()
    assert not any(grant.has(capability) for capability in CAPABILITIES)


def test_a_user_in_no_group_may_do_nothing(membership):
    grant = resolve_grant(_user("loner"))

    assert grant.capabilities == frozenset()
    assert not any(grant.has(capability) for capability in CAPABILITIES)


def test_the_row_of_a_deleted_group_is_ignored_and_left_for_the_admin_routes(membership):
    membership["user-1"] = ["all_users"]
    save_rows([Row(group_id="deleted-group", capabilities=frozenset(CAPABILITIES))])

    assert not resolve_grant(_user()).has("download")
    assert "deleted-group" in load_rows()


def test_capabilities_are_united_across_groups():
    grant = merge_rows(
        [Row(group_id="a", capabilities=frozenset({"browse"})), Row(group_id="b", capabilities=frozenset({"approve"}))]
    )

    assert grant.capabilities == {"browse", "approve"}


def test_model_types_are_united_when_every_downloading_group_lists_some():
    grant = merge_rows(
        [_download_row("a", model_types={"lora"}), _download_row("b", model_types={"embedding", "vae"})]
    )

    assert grant.model_types == {"lora", "embedding", "vae"}
    assert not grant.allows_type("checkpoint")


@pytest.mark.parametrize(
    "sizes, merged",
    [((2.0, 8.5), 8.5), ((2.0, None), None), ((None,), None), ((4.0,), 4.0)],
)
def test_the_largest_size_wins_and_no_limit_beats_any_limit(sizes, merged):
    grant = merge_rows([_download_row(f"g{index}", max_file_gb=size) for index, size in enumerate(sizes)])

    assert grant.max_file_gb == merged


@pytest.mark.parametrize(
    "assign_tos, merged",
    [(("user", "user"), "user"), (("user", "group"), "group"), (("group",), "group")],
)
def test_assign_to_is_group_as_soon_as_one_group_says_so(assign_tos, merged):
    grant = merge_rows([_download_row(f"g{index}", assign_to=value) for index, value in enumerate(assign_tos)])

    assert grant.assign_to == merged


@pytest.mark.parametrize("remotes, merged", [((False, False), False), ((False, True), True)])
def test_allow_remote_when_one_group_allows_it(remotes, merged):
    grant = merge_rows([_download_row(f"g{index}", allow_remote=value) for index, value in enumerate(remotes)])

    assert grant.allow_remote is merged


def test_parameters_come_only_from_the_groups_that_grant_download():
    browse_only = Row(
        group_id="browsers",
        capabilities=frozenset({"browse", "request"}),
        sources=frozenset({"civarchive"}),
        model_types=frozenset(),
        max_file_gb=None,
        assign_to="group",
        allow_remote=True,
    )
    downloader = _download_row("downloaders", sources={"civitai"}, model_types={"lora"}, max_file_gb=1.5)

    grant = merge_rows([browse_only, downloader])

    assert grant.capabilities == {"browse", "request", "download"}
    assert grant.sources == {"civitai"}
    assert grant.model_types == {"lora"}
    assert grant.max_file_gb == 1.5
    assert grant.assign_to == "user"
    assert grant.allow_remote is False
    assert _view(grant.request_params) == ({"civarchive"}, frozenset(), None, "group", False)


def test_the_request_view_merges_the_groups_that_grant_request_by_the_same_rules():
    rows = [
        _request_row("a", sources={"civitai"}, model_types={"lora"}, max_file_gb=1.0),
        _request_row("b", sources={"huggingface"}, model_types={"embedding"}, max_file_gb=4.0, assign_to="group"),
        _download_row("c", sources={"civarchive"}, model_types={"vae"}, max_file_gb=None),
        Row(group_id="d", capabilities=frozenset({"browse"}), model_types=frozenset()),
    ]

    view = merge_rows(rows).request_params

    assert _view(view) == ({"civitai", "huggingface"}, {"lora", "embedding"}, 4.0, "group", False)
    assert merge_rows(rows + [_request_row("e", max_file_gb=None)]).request_params.max_file_gb is None
    assert merge_rows(rows + [_request_row("e", model_types=set())]).request_params.model_types == frozenset()


def test_a_group_granting_both_counts_in_both_views():
    both = Row(
        group_id="both",
        capabilities=frozenset({"request", "download"}),
        sources=frozenset({"civitai"}),
        model_types=frozenset({"lora"}),
        max_file_gb=3.0,
        allow_remote=True,
    )

    grant = merge_rows([both])

    assert _view(grant.download_params) == ({"civitai"}, {"lora"}, 3.0, "user", True)
    assert _view(grant.request_params) == ({"civitai"}, {"lora"}, 3.0, "user", False)


def test_merging_no_row_grants_nothing():
    assert merge_rows([]) == merge_rows([default_row("artists")])
    assert merge_rows([]).capabilities == frozenset()
    assert merge_rows([]).request_params == NO_PARAMS


def _grant(**changes):
    values = dict(
        capabilities=frozenset({"browse", "download"}),
        sources=frozenset({"civitai"}),
        model_types=frozenset({"lora"}),
        max_file_gb=None,
        assign_to="user",
        allow_remote=False,
        unrestricted=False,
    )
    values.update(changes)
    return Grant(**values)


def test_allows_source_is_membership_and_no_source_allows_none():
    assert _grant().allows_source("civitai")
    assert not _grant().allows_source("huggingface")
    assert not _grant(sources=frozenset()).allows_source("civitai")


def test_allows_type_is_membership_and_no_type_allows_every_type():
    assert _grant().allows_type("lora")
    assert not _grant().allows_type("checkpoint")
    assert _grant(model_types=frozenset()).allows_type("checkpoint")


def test_an_unrestricted_grant_allows_everything():
    assert ADMIN_GRANT.allows_source("civarchive")
    assert ADMIN_GRANT.allows_type("checkpoint")
    assert ADMIN_GRANT.allows_source("civarchive", "request")
    assert ADMIN_GRANT.allows_type("checkpoint", "request")
    assert _grant(sources=frozenset(), unrestricted=True).allows_source("huggingface")
    assert _grant(capabilities=frozenset(), unrestricted=True).has("approve")


def test_the_helpers_check_the_view_of_the_purpose_asked():
    grant = _grant(request_params=Params(sources={"huggingface"}, model_types={"embedding"}))

    assert grant.allows_source("civitai") and not grant.allows_source("civitai", "request")
    assert grant.allows_source("huggingface", "request") and not grant.allows_source("huggingface")
    assert grant.allows_type("lora") and not grant.allows_type("lora", "request")
    assert grant.allows_type("embedding", "request")
    assert grant.params("download") == grant.download_params == Params(sources={"civitai"}, model_types={"lora"})
    assert grant.params("request") is grant.request_params
    with pytest.raises(ValueError):
        grant.params("upload")


def test_a_grant_without_a_request_view_allows_no_request():
    grant = _grant()

    assert grant.request_params == NO_PARAMS
    assert not grant.allows_source("civitai", "request")
    assert grant.allows_type("lora", "request")


def test_require_download_refuses_a_browse_only_user_with_the_envelope(membership):
    membership["user-1"] = ["all_users"]

    with pytest.raises(HTTPException) as raised:
        require_download(user=_user())

    assert raised.value.status_code == 403
    assert raised.value.detail == _FORBIDDEN_DOWNLOAD


def test_require_yields_the_principal_when_the_capability_is_granted(membership):
    membership["user-1"] = ["all_users"]

    principal = require_browse(user=_user())

    assert isinstance(principal, Principal)
    assert principal.user.id == "user-1"
    assert principal.grant.capabilities == {"browse", "request"}
    assert not principal.is_admin


def test_require_rejects_an_unknown_capability():
    with pytest.raises(ValueError):
        require("manage")


def test_require_any_yields_the_principal_when_one_of_the_capabilities_is_granted(membership):
    membership["user-1"] = ["all_users"]
    either = require_any("download", "request")

    principal = either(user=_user())

    assert isinstance(principal, Principal)
    assert principal.grant.capabilities == {"browse", "request"}
    assert either.__name__ == "require_any_download_request"


def test_require_any_refuses_a_user_with_none_of_the_capabilities(membership):
    membership["user-1"] = ["all_users"]

    with pytest.raises(HTTPException) as raised:
        require_any("download", "approve")(user=_user())

    assert raised.value.status_code == 403
    assert raised.value.detail == {
        "success": False,
        "error": "forbidden",
        "message": "This action needs one of these permissions: 'download', 'approve'",
    }
    assert require_any("download", "approve")(user=_user("admin-1", AccountType.ADMIN)).is_admin


@pytest.mark.parametrize("capabilities", [(), ("download", "manage")])
def test_require_any_rejects_no_capability_or_an_unknown_one(capabilities):
    with pytest.raises(ValueError):
        require_any(*capabilities)


def test_current_principal_needs_no_capability(membership):
    assert current_principal(user=_user("loner")).grant.capabilities == frozenset()
    assert current_principal(user=_user("admin-1", AccountType.ADMIN)).is_admin


def test_a_route_behind_require_download_answers_403_then_200(membership):
    app = FastAPI()

    @app.get("/download-only")
    def download_only(principal: Principal = Depends(require_download)):
        return {"user": principal.user.id}

    app.dependency_overrides[get_current_active_user] = lambda: _user()
    membership["user-1"] = ["all_users"]

    with TestClient(app) as client:
        refused = client.get("/download-only")
        save_rows([Row(group_id="all_users", capabilities=frozenset({"browse", "download"}))])
        allowed = client.get("/download-only")

    assert refused.status_code == 403
    assert refused.json() == {"detail": _FORBIDDEN_DOWNLOAD}
    assert allowed.status_code == 200
    assert allowed.json() == {"user": "user-1"}


@pytest.fixture
def directory(membership, monkeypatch):
    users = {
        "admin-1": _user("admin-1", AccountType.ADMIN),
        "admin-2": _user("admin-2", AccountType.ADMIN),
        "user-1": _user("user-1"),
        "approver-1": _user("approver-1"),
        "user-2": _user("user-2"),
    }
    monkeypatch.setattr(permissions, "get_user", users.get)
    monkeypatch.setattr(permissions, "list_user_ids", lambda: list(users))
    monkeypatch.setattr(
        core_bridge,
        "list_groups",
        lambda: [{"id": group_id, "name": group_id} for group_id in ("all_admins", "all_users", "reviewers")],
    )
    monkeypatch.setattr(
        core_bridge,
        "group_member_ids",
        lambda group_id: [user_id for user_id, groups in membership.items() if group_id in groups],
    )
    membership.update(
        {
            "admin-1": ["all_admins", "all_users"],
            "admin-2": ["all_users"],
            "user-1": ["all_users"],
            "approver-1": ["all_users", "reviewers"],
            "user-2": ["all_users", "reviewers"],
        }
    )
    return users


def test_approvers_are_every_admin_and_the_members_of_groups_granting_approve(directory):
    save_rows([Row(group_id="reviewers", capabilities=frozenset({"browse", "approve"}))])

    assert permissions.approver_ids() == {"admin-1", "admin-2", "approver-1", "user-2"}


def test_without_a_group_granting_approve_only_admins_approve(directory):
    assert permissions.approver_ids() == {"admin-1", "admin-2"}

    save_rows([Row(group_id="all_admins", capabilities=frozenset({"browse"}))])
    assert permissions.approver_ids() == {"admin-1", "admin-2"}


def test_user_by_id_never_raises_and_admin_by_id_answers_admins_only(directory, monkeypatch, caplog):
    assert permissions.user_by_id("user-1").username == "user-1"
    assert permissions.user_by_id("ghost") is None
    assert permissions.user_by_id(None) is None
    assert permissions.admin_by_id("admin-1").id == "admin-1"
    assert permissions.admin_by_id("user-1") is None
    assert permissions.admin_by_id(None) is None

    def broken(user_id):
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(permissions, "get_user", broken)
    assert permissions.user_by_id("user-1") is None
    assert "Could not look user user-1 up" in caplog.text


_REVIEWERS = Row(group_id="reviewers", capabilities=frozenset({"browse", "approve"}))


def test_the_acting_admin_is_the_requester_when_they_are_an_admin(directory):
    save_rows([_REVIEWERS], "admin-2")

    assert permissions.acting_admin("admin-1", "admin-2", "reviewers") is directory["admin-1"]


def test_otherwise_the_acting_admin_is_the_decider_when_they_are_an_admin(directory):
    save_rows([_REVIEWERS], "admin-1")

    assert permissions.acting_admin("user-1", "admin-2", "reviewers") is directory["admin-2"]


def test_otherwise_the_acting_admin_is_the_admin_who_last_saved_the_groups_row(directory):
    save_rows([_REVIEWERS, Row(group_id="all_users", capabilities=frozenset({"browse"}))], "admin-2")

    assert permissions.acting_admin("user-1", "approver-1", "reviewers") is directory["admin-2"]
    assert permissions.acting_admin("user-1", None, "all_users") is directory["admin-2"]


@pytest.mark.parametrize(
    "row, saved_by",
    [
        (_REVIEWERS, None),
        (_REVIEWERS, "user-2"),
        (_REVIEWERS, "ghost"),
        (Row(group_id="all_users", capabilities=frozenset({"browse"})), "admin-1"),
    ],
    ids=["row-stored-before-it-was-recorded", "saved-by-a-non-admin", "saved-by-a-deleted-user", "default-row"],
)
def test_without_an_admin_to_name_there_is_no_acting_admin(directory, row, saved_by):
    save_rows([row], saved_by)

    assert permissions.acting_admin("user-1", "approver-1", "reviewers") is None


def test_acting_admin_never_raises(directory, monkeypatch, caplog):
    def broken():
        raise RuntimeError("no such table: model_browser_group_permissions")

    monkeypatch.setattr(permissions.permissions_store, "load_rows", broken)

    assert permissions.acting_admin("user-1", "approver-1", "reviewers") is None
    assert "Could not read the permissions of user group reviewers" in caplog.text
    assert permissions.acting_admin(None, "admin-1", "reviewers") is directory["admin-1"]


def test_principal_for_resolves_another_users_grant(directory):
    save_rows([Row(group_id="reviewers", capabilities=frozenset({"browse", "approve"}))])

    requester = permissions.principal_for(directory["user-1"])
    approver = permissions.principal_for(directory["approver-1"])

    assert requester.user.id == "user-1" and requester.grant.capabilities == {"browse", "request"}
    assert not permissions.can_approve(requester)
    assert permissions.can_approve(approver)
    assert permissions.can_approve(permissions.principal_for(directory["admin-2"]))
