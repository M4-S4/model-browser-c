from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import downloads, nsfw, pending, permissions, requests, routes_downloads, routes_requests
from backend.api import router
from backend.permissions import (
    current_principal,
    require_approve,
    require_browse,
    require_download,
    require_request,
)
from backend.permissions_store import Row, save_rows
from backend.tables import create_tables
from src.plugin_api import get_current_active_user

from .civitai_fakes import context
from .download_fakes import (
    NSFW_IMAGE,
    SAFE_IMAGE,
    SHA,
    USER_GRANT,
    FakeCore,
    FakeSource,
    admin,
    civitai_detail,
    hf_detail,
    install_sources,
    limited,
    model,
    principal,
    requesting,
    user,
)

BASE = "/api/plugins/model-browser-c"
REQUESTER = principal("user-1", capabilities=frozenset({"browse", "request"}))
OTHER = principal("user-2", capabilities=frozenset({"browse", "request"}))
ADMIN = admin("admin-1")
APPROVER = principal("approver-1", capabilities=frozenset({"browse", "approve"}))
DEPENDENCIES = (
    require_request,
    require_approve,
    require_download,
    require_browse,
    routes_requests.require_requests_access,
    routes_downloads.require_downloads_access,
    current_principal,
)


@pytest.fixture
def core(plugin_db, monkeypatch):
    create_tables()
    fake = FakeCore().install(monkeypatch)
    fake.group_ids = ["all_admins", "all_users", "artists", "reviewers"]
    fake.add_user("admin-1", admin=True, groups=["all_admins", "all_users"])
    fake.add_user("user-1", groups=["all_users", "artists"])
    fake.add_user("user-2", groups=["all_users"])
    fake.add_user("approver-1", groups=["all_users", "reviewers"])
    return fake


@pytest.fixture
def viewers(monkeypatch):
    contexts = {}
    monkeypatch.setattr(nsfw, "resolve_nsfw", lambda who: contexts.get(who.id, context("allowed", "show")))
    return contexts


@pytest.fixture
def civitai(monkeypatch, core, viewers):
    return install_sources(monkeypatch, FakeSource("civitai", civitai_detail()), FakeSource("huggingface", hf_detail()))[
        "civitai"
    ]


@pytest.fixture
def app():
    application = FastAPI()
    application.include_router(router)
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def api(app):
    client = TestClient(app)

    def call(who, method, path, **kwargs):
        for dependency in DEPENDENCIES:
            app.dependency_overrides[dependency] = lambda who=who: who
        return client.request(method, f"{BASE}{path}", **kwargs)

    yield call
    client.close()


def _body(**overrides):
    values = {"source": "civitai", "model_id": "122359", "version_id": "135867", "file_id": "1001"}
    values.update(overrides)
    return values


def _error(response):
    return response.json()["detail"]


def _create(api, who=REQUESTER, **overrides):
    response = api(who, "POST", "/requests", json=_body(**overrides))
    assert response.status_code == 200, response.text
    return response.json()["data"]["request"]


def _approve(api, request_id, who=ADMIN, **body):
    return api(who, "POST", f"/requests/{request_id}/approve", json=body)


def _types(core):
    return [notification["notification_type"] for notification in core.notifications]


def test_a_request_is_created_open_with_a_snapshot_of_the_file(api, core, civitai):
    response = api(REQUESTER, "POST", "/requests", json=_body(note="  For the SDXL workflow  "))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["already_requested"] is False
    request = data["request"]
    assert (request["status"], request["user_id"], request["username"]) == ("open", "user-1", "user-1")
    assert (request["source"], request["source_model_id"], request["source_version_id"], request["file_id"]) == (
        "civitai", "122359", "135867", "1001",
    )
    assert request["note"] == "For the SDXL workflow"
    assert (request["decided_by"], request["decided_by_username"], request["sha256"], request["assign_to"]) == (
        None, None, None, None,
    )
    assert request["snapshot"] == {
        "name": "Detail Tweaker XL",
        "type": "LORA",
        "mb_type": "lora",
        "version_name": "v1.0",
        "base_model": "SDXL 1.0",
        "file_name": "add-detail-xl.safetensors",
        "size_bytes": 228_000_000,
        "nsfw": False,
        "poi": False,
        "license": "CreativeML Open RAIL++-M; no selling",
        "thumbnail_url": SAFE_IMAGE,
        "source_url": "https://civitai.com/models/122359",
        "sha256": SHA,
        "already_installed": False,
    }
    assert core.queued == [] and pending.list_rows() == []
    assert requests.get(request["id"]).status == "open"


def test_a_new_request_notifies_the_admins_and_approve_holders_but_never_the_requester(api, core, civitai):
    save_rows([Row(group_id="reviewers", capabilities=frozenset({"browse", "approve"}))])
    core.add_user("admin-2", admin=True)

    request = _create(api)

    received = [n for n in core.notifications if n["notification_type"] == "model-browser.request_received"]
    assert sorted(n["user_id"] for n in received) == ["admin-1", "admin-2", "approver-1"]
    assert received[0]["message"] == "user-1 asked for Detail Tweaker XL (add-detail-xl.safetensors)."
    assert received[0]["metadata"]["request_id"] == request["id"]
    assert received[0]["metadata"]["link"] == "/plugins/model-browser-c?tab=requests"


def test_an_approver_who_requests_is_not_notified_of_their_own_request(api, core, civitai):
    save_rows([Row(group_id="reviewers", capabilities=frozenset({"browse", "request", "approve"}))])

    _create(api, who=principal("approver-1", capabilities=frozenset({"browse", "request", "approve"})))

    assert [n["user_id"] for n in core.notifications] == ["admin-1"]


def test_asking_twice_for_the_same_file_answers_the_same_request(api, core, civitai):
    first = _create(api)
    sent = len(core.notifications)

    again = api(REQUESTER, "POST", "/requests", json=_body(note="again"))

    assert again.status_code == 200
    assert again.json()["data"]["already_requested"] is True
    assert again.json()["data"]["request"]["id"] == first["id"]
    assert len(core.notifications) == sent
    assert len(requests.list_rows()) == 1


def test_another_user_or_another_file_gets_its_own_request(api, core, civitai):
    first = _create(api)
    other_user = _create(api, who=OTHER)
    other_file = _create(api, file_id="1002")

    assert len({first["id"], other_user["id"], other_file["id"]}) == 3


def test_a_cancelled_request_does_not_stop_a_new_one(api, core, civitai):
    first = _create(api)
    api(REQUESTER, "DELETE", f"/requests/{first['id']}").raise_for_status()

    again = api(REQUESTER, "POST", "/requests", json=_body())

    assert again.json()["data"]["already_requested"] is False
    assert again.json()["data"]["request"]["id"] != first["id"]


def test_a_file_installed_and_visible_to_the_requester_is_already_available(api, core, civitai):
    core.models_by_sha[SHA] = model("model-9")
    core.visible["user-1"] = ["model-9"]

    response = api(REQUESTER, "POST", "/requests", json=_body())

    assert response.status_code == 409
    assert _error(response) == {
        "success": False,
        "error": "already_available",
        "message": "This file is already installed and in your models",
    }
    assert requests.list_rows() == [] and core.notifications == []


def test_a_file_installed_but_not_visible_to_the_requester_is_requested_as_already_installed(api, core, civitai):
    core.models_by_sha[SHA] = model("model-9")

    request = _create(api)

    assert request["snapshot"]["already_installed"] is True
    assert request["status"] == "open"


def test_a_file_whose_model_file_is_gone_is_requested_as_not_installed(api, core, civitai):
    core.models_by_sha[SHA] = model("model-9", is_available=False)
    core.visible["user-1"] = ["model-9"]

    assert _create(api)["snapshot"]["already_installed"] is False


def test_a_model_the_requester_may_not_see_is_not_found(api, core, civitai, viewers):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewers["user-1"] = context("blocked", "hide")

    response = api(REQUESTER, "POST", "/requests", json=_body())

    assert (response.status_code, _error(response)["error"]) == (404, "not_found")
    assert requests.list_rows() == []


def _requester_with(**view):
    return replace(REQUESTER, grant=replace(REQUESTER.grant, request_params=requesting(**view)))


@pytest.mark.parametrize(
    "view, overrides, status, code, message",
    [
        (
            {"sources": frozenset({"huggingface"})},
            {},
            403,
            "forbidden",
            "Your permissions do not allow requesting models from 'civitai'",
        ),
        ({"model_types": frozenset({"embedding"})}, {}, 403, "forbidden", "Your permissions do not allow requesting 'lora' models"),
        (
            {"max_file_gb": 0.1},
            {},
            403,
            "forbidden",
            "This file is 0.21 GB: your permissions allow requesting files of at most 0.1 GB",
        ),
        ({}, {"version_id": "999"}, 404, "not_found", "Version not found"),
        (
            {},
            {"source": "civarchive"},
            403,
            "forbidden",
            "Your permissions do not allow requesting models from 'civarchive'",
        ),
    ],
    ids=["source", "type", "size", "version", "source-not-allowed"],
)
def test_the_requesters_request_parameters_bind_a_request(api, core, civitai, view, overrides, status, code, message):
    response = api(_requester_with(**view), "POST", "/requests", json=_body(**overrides))

    assert response.status_code == status
    assert _error(response) == {"success": False, "error": code, "message": message}
    assert requests.list_rows() == []


def test_the_download_view_does_not_bind_a_request(api, core, civitai):
    may_not_download = replace(REQUESTER, grant=replace(REQUESTER.grant, sources=frozenset(), max_file_gb=0.01))

    request = _create(api, who=may_not_download)

    assert request["status"] == "open"


def test_a_type_the_requester_may_choose_is_kept_in_the_snapshot(api, core, civitai):
    request = _create(api, who=_requester_with(model_types=frozenset({"embedding"})), model_type="embedding")

    assert request["snapshot"]["mb_type"] == "embedding"


@pytest.mark.parametrize(
    "payload, where",
    [
        (_body(note="x" * 1001), "note"),
        ({"source": "civitai", "model_id": "122359", "version_id": "135867"}, "file_id"),
        (_body(model_type="lycoris"), "model_type"),
        (_body(model_id="../1"), "model_id"),
        (["not", "an", "object"], "Invalid request"),
    ],
)
def test_an_invalid_body_is_refused(api, core, civitai, payload, where):
    response = api(REQUESTER, "POST", "/requests", json=payload)

    assert response.status_code == 400
    assert _error(response)["error"] == "invalid_request"
    assert where in _error(response)["message"]


def test_a_source_failure_is_answered_like_the_browsing_routes(api, core, civitai):
    from backend.http import SourceUnavailable

    civitai.error = SourceUnavailable("CivitAI answered 503")

    response = api(REQUESTER, "POST", "/requests", json=_body())

    assert (response.status_code, _error(response)["error"]) == (502, "source_unavailable")


@pytest.fixture
def listed(api, core, civitai, plugin_db):
    first = _create(api)
    second = _create(api, who=OTHER)
    third = _create(api, file_id="1002")
    api(ADMIN, "POST", f"/requests/{third['id']}/reject", json={}).raise_for_status()
    with plugin_db.get_cursor() as cursor:
        for request_id, stamp in ((first["id"], "09"), (second["id"], "10"), (third["id"], "11")):
            cursor.execute(
                f"UPDATE {requests.TABLE} SET created_at = ? WHERE id = ?", (f"2026-10-11 {stamp}:00:00", request_id)
            )
    return SimpleNamespace(first=first["id"], second=second["id"], third=third["id"])


def _ids(response):
    return [request["id"] for request in response.json()["data"]["requests"]]


def test_a_requester_lists_their_own_requests_only(api, listed):
    for query in ("", "?all=true", "?all=false"):
        response = api(REQUESTER, "GET", f"/requests{query}")
        assert response.status_code == 200
        assert _ids(response) == [listed.third, listed.first]
    data = api(REQUESTER, "GET", "/requests").json()["data"]
    assert (data["open_count"], data["can_approve"]) == (0, False)


def test_an_approver_lists_every_request_unless_asking_for_their_own(api, listed):
    for who in (ADMIN, APPROVER):
        data = api(who, "GET", "/requests").json()["data"]
        assert [request["id"] for request in data["requests"]] == [listed.third, listed.second, listed.first]
        assert (data["open_count"], data["can_approve"]) == (2, True)
        assert _ids(api(who, "GET", "/requests?all=false")) == []
    declined = api(ADMIN, "GET", "/requests").json()["data"]["requests"][0]
    assert (declined["status"], declined["decided_by"], declined["decided_by_username"]) == ("rejected", "admin-1", "admin-1")


def test_requests_can_be_filtered_by_status(api, listed):
    assert _ids(api(ADMIN, "GET", "/requests?status=open")) == [listed.second, listed.first]
    assert _ids(api(ADMIN, "GET", "/requests?status=rejected,open&limit=2")) == [listed.third, listed.second]
    assert _ids(api(ADMIN, "GET", "/requests?status=rejected&status=cancelled")) == [listed.third]

    response = api(ADMIN, "GET", "/requests?status=lost")
    assert response.status_code == 400
    assert "Unknown request status 'lost'" in _error(response)["message"]


def test_the_snapshot_is_listed_as_each_viewer_may_see_it(api, core, civitai, viewers):
    civitai.detail_value = civitai_detail(nsfw=True)
    _create(api)
    viewers["admin-1"] = context("blocked", "hide")

    [shown] = api(ADMIN, "GET", "/requests").json()["data"]["requests"]
    [own] = api(REQUESTER, "GET", "/requests").json()["data"]["requests"]

    assert (shown["snapshot"]["thumbnail_url"], shown["snapshot"]["hidden"]) == (None, True)
    assert shown["snapshot"]["name"] == "Detail Tweaker XL"
    assert own["snapshot"]["thumbnail_url"] == SAFE_IMAGE and "hidden" not in own["snapshot"]


def test_approving_queues_the_file_for_the_requester_in_the_approvers_name(api, core, civitai):
    request = _create(api)

    response = _approve(api, request["id"], note="Enjoy")

    assert response.status_code == 200
    data = response.json()["data"]
    assert (data["already_installed"], data["already_queued"]) == (False, False)
    approved = data["request"]
    assert (approved["status"], approved["decided_by"], approved["decided_by_username"]) == ("approved", "admin-1", "admin-1")
    assert (approved["decision_note"], approved["sha256"], approved["assign_to"]) == ("Enjoy", SHA, "user:user-1")
    [queued] = core.queued
    assert (queued["created_by"], queued["model_type"], queued["checksum_sha256"]) == ("admin-1", "lora", SHA)
    row = pending.get(SHA)
    assert (row.requested_by, row.assign_to, row.request_id, row.status) == ("user-1", "user:user-1", request["id"], "queued")
    assert data["pending"]["sha256"] == SHA and data["download"]["id"] == "dl-1"
    assert data["download"]["destination_path"]
    assert [(event[0], event[1]) for event in core.limit_events] == [
        ("user-1", "model-browser.downloads"),
        ("user-1", "model-browser.download_bytes"),
    ]
    approved_notice = [n for n in core.notifications if n["notification_type"] == "model-browser.request_approved"]
    assert [n["user_id"] for n in approved_notice] == ["user-1"]
    assert approved_notice[0]["message"] == (
        "admin-1 approved your request for Detail Tweaker XL (add-detail-xl.safetensors): it is being downloaded "
        'for you. Note: "Enjoy"'
    )


def test_a_non_admin_approver_gets_the_public_fields_of_the_download(api, core, civitai):
    request = _create(api)

    data = _approve(api, request["id"], who=APPROVER).json()["data"]

    assert set(data["download"]) == set(downloads.PUBLIC_DOWNLOAD_FIELDS)
    assert core.queued[0]["created_by"] == "approver-1"
    assert data["request"]["decided_by"] == "approver-1"


def test_the_approver_sees_the_row_as_their_own_context_allows(api, core, civitai, viewers):
    request = _create(api)
    viewers["admin-1"] = context("blocked", "hide")

    data = _approve(api, request["id"]).json()["data"]

    images = data["pending"]["info"]["model"]["versions"][0]["images"]
    assert [image["url"] for image in images] == [SAFE_IMAGE]
    stored = pending.get(SHA).info["model"]["versions"][0]["images"]
    assert [image["url"] for image in stored] == [SAFE_IMAGE, NSFW_IMAGE]


def test_approving_with_one_of_the_requesters_groups(api, core, civitai):
    save_rows([Row(group_id="all_users", capabilities=frozenset({"browse", "request"}), assign_to="group")])
    request = _create(api)

    data = _approve(api, request["id"], assign_group_id="artists").json()["data"]

    assert data["request"]["assign_to"] == "group:artists"
    assert pending.get(SHA).assign_to == "group:artists"


def test_a_group_the_requester_is_not_in_is_refused_and_the_request_stays_open(api, core, civitai):
    save_rows([Row(group_id="all_users", capabilities=frozenset({"browse", "request"}), assign_to="group")])
    request = _create(api)

    response = _approve(api, request["id"], assign_group_id="reviewers")

    assert response.status_code == 403
    assert _error(response)["message"] == "A request can only be assigned to one of the requester's groups"
    stored = requests.get(request["id"])
    assert (stored.status, stored.decided_by, stored.decision_note) == ("open", None, None)
    assert core.queued == []


def test_an_approver_may_change_the_type_within_the_requesters_types(api, core, civitai):
    save_rows([Row(group_id="all_users", capabilities=frozenset({"browse", "request"}), model_types=frozenset({"lora", "embedding"}))])
    request = _create(api)

    assert _approve(api, request["id"], model_type="embedding").status_code == 200
    assert core.queued[0]["model_type"] == "embedding"


def test_approving_an_installed_file_only_assigns_it(api, core, civitai):
    core.models_by_sha[SHA] = model("model-9")
    request = _create(api)

    data = _approve(api, request["id"]).json()["data"]

    assert (data["already_installed"], data["already_queued"], data["download"]) == (True, False, None)
    row = pending.get(SHA)
    assert (row.status, row.model_id, row.requested_by, row.request_id) == ("indexed", "model-9", "user-1", request["id"])
    assert core.queued == [] and core.limit_events == []
    assert data["request"]["sha256"] == SHA
    notice = [n for n in core.notifications if n["notification_type"] == "model-browser.request_approved"][0]
    assert "already installed and is being added to your models" in notice["message"]


def test_approving_a_file_already_on_its_way_for_someone_else_links_the_request(api, core, civitai):
    api(ADMIN, "POST", "/downloads", json=_body()).raise_for_status()
    request = _create(api)

    data = _approve(api, request["id"]).json()["data"]

    assert (data["already_queued"], data["already_installed"]) == (True, False)
    assert len(core.queued) == 1
    assert data["pending"]["requested_by"] == "admin-1"
    stored = requests.get(request["id"])
    assert (stored.status, stored.sha256, stored.assign_to) == ("approved", SHA, "user:user-1")
    assert [request.id for request in requests.list_approved_for_sha256(SHA)] == [request["id"]]
    notice = [n for n in core.notifications if n["notification_type"] == "model-browser.request_approved"][0]
    assert "already being downloaded" in notice["message"]


def test_approval_checks_the_requesters_rules_again(api, core, civitai):
    request = _create(api)
    save_rows([Row(group_id="all_users", capabilities=frozenset({"browse", "request"}), model_types=frozenset({"embedding"}))])

    response = _approve(api, request["id"])

    assert response.status_code == 403
    assert _error(response) == {
        "success": False,
        "error": "forbidden",
        "message": "The requester's permissions do not allow requesting 'lora' models",
    }
    assert requests.get(request["id"]).status == "open"
    assert core.queued == [] and pending.list_rows() == []
    assert "model-browser.request_approved" not in _types(core)


def test_approval_ignores_the_requesters_download_view(api, core, civitai):
    request = _create(api)
    save_rows(
        [
            Row(group_id="all_users", capabilities=frozenset({"browse", "request"}), model_types=frozenset({"embedding"})),
            Row(group_id="artists", capabilities=frozenset({"browse", "download"})),
        ]
    )

    response = _approve(api, request["id"])

    assert response.status_code == 403
    assert _error(response)["message"] == "The requester's permissions do not allow requesting 'lora' models"


def test_approval_of_a_model_the_requester_may_no_longer_see_is_refused(api, core, civitai, viewers):
    request = _create(api)
    civitai.detail_value = civitai_detail(nsfw=True)
    viewers["user-1"] = context("blocked", "hide")

    response = _approve(api, request["id"])

    assert response.status_code == 403
    assert _error(response)["error"] == "not_visible_to_requester"
    assert requests.get(request["id"]).status == "open"
    assert core.queued == []


def test_approval_follows_the_requesters_context_not_the_approvers(api, core, civitai, viewers):
    civitai.detail_value = civitai_detail(nsfw=True)
    viewers["user-1"] = context("blur", "blur")
    request = _create(api)
    viewers["admin-1"] = context("blocked", "hide")

    response = _approve(api, request["id"])

    assert response.status_code == 200
    assert response.json()["data"]["request"]["snapshot"]["hidden"] is True


def test_approval_over_the_requesters_quota_is_refused(api, core, civitai):
    core.limits[("user-1", "model-browser.downloads")] = limited("model-browser.downloads", 1, 1)
    request = _create(api)

    response = _approve(api, request["id"])

    assert response.status_code == 403
    error = _error(response)
    assert (error["error"], error["kind"], error["used"], error["limit"]) == (
        "limit_exceeded", "model-browser.downloads", 1, 1,
    )
    assert error["message"] == "user-1 has reached their model downloads per month limit: 1 of 1 used. It resets on Nov 1 at 00:00 UTC."
    assert requests.get(request["id"]).status == "open"
    assert core.queued == []


def test_a_request_whose_author_is_gone_cannot_be_approved(api, core, civitai):
    request = _create(api)
    del core.users["user-1"]

    response = _approve(api, request["id"])

    assert (response.status_code, _error(response)["error"]) == (409, "requester_gone")
    assert requests.get(request["id"]).status == "open"


def test_an_unexpected_failure_reopens_the_request(api, core, civitai, caplog):
    core.queue_error = RuntimeError("download_queue.queue_model_download moved")
    request = _create(api)

    response = _approve(api, request["id"])

    assert (response.status_code, _error(response)["error"]) == (500, "internal_error")
    assert "moved" not in _error(response)["message"]
    assert requests.get(request["id"]).status == "open"
    assert pending.list_rows() == []


def test_only_an_open_request_can_be_approved(api, core, civitai):
    request = _create(api)
    api(ADMIN, "POST", f"/requests/{request['id']}/reject", json={}).raise_for_status()

    response = _approve(api, request["id"])

    assert response.status_code == 409
    assert _error(response) == {
        "success": False,
        "error": "not_open",
        "message": "This request is no longer open (it is rejected)",
    }
    assert core.queued == []


def test_a_request_another_approver_claimed_meanwhile_is_not_approved_twice(api, core, civitai, monkeypatch):
    request = _create(api)
    stale = requests.get(request["id"])
    requests.set_status(request["id"], "approved", decided_by="approver-1")
    monkeypatch.setattr(routes_requests, "_get_request", lambda request_id: stale)

    response = _approve(api, request["id"])

    assert (response.status_code, _error(response)["error"]) == (409, "not_open")
    assert requests.get(request["id"]).decided_by == "approver-1"
    assert core.queued == []


SELF_APPROVER = principal("approver-1", capabilities=frozenset({"browse", "request", "approve"}))


def test_an_approver_cannot_decide_on_their_own_request(api, core, civitai):
    request = _create(api, who=SELF_APPROVER)

    for action in ("approve", "reject"):
        response = api(SELF_APPROVER, "POST", f"/requests/{request['id']}/{action}", json={"note": "mine"})
        assert response.status_code == 403
        assert _error(response) == {
            "success": False,
            "error": "forbidden",
            "message": "You cannot decide on your own request: cancel it instead",
        }
    stored = requests.get(request["id"])
    assert (stored.status, stored.decided_by, stored.decision_note) == ("open", None, None)
    assert core.queued == []
    assert api(SELF_APPROVER, "DELETE", f"/requests/{request['id']}").status_code == 200


def test_an_approver_decides_on_other_users_requests(api, core, civitai):
    request = _create(api)

    assert _approve(api, request["id"], who=SELF_APPROVER).status_code == 200


def test_an_admin_may_decide_on_their_own_request(api, core, civitai):
    approved = _create(api, who=ADMIN)
    rejected = _create(api, who=ADMIN, file_id="1002")

    response = _approve(api, approved["id"])
    assert response.status_code == 200
    assert (response.json()["data"]["request"]["decided_by"], pending.get(SHA).assign_to) == ("admin-1", "none")
    assert api(ADMIN, "POST", f"/requests/{rejected['id']}/reject", json={}).json()["data"]["request"]["status"] == (
        "rejected"
    )


def test_an_approvers_count_leaves_out_their_own_open_requests(api, core, civitai):
    _create(api, who=SELF_APPROVER)
    _create(api)

    data = api(SELF_APPROVER, "GET", "/requests").json()["data"]

    assert (len(data["requests"]), data["open_count"], data["can_approve"]) == (2, 1, True)
    assert api(ADMIN, "GET", "/requests").json()["data"]["open_count"] == 2


def test_approving_an_unknown_request_is_not_found(api, core, civitai):
    response = _approve(api, "01JUNKJUNKJUNKJUNKJUNKJUNK")

    assert (response.status_code, _error(response)["error"]) == (404, "not_found")


@pytest.mark.parametrize(
    "payload, where",
    [({"model_type": "lycoris"}, "model_type"), ({"note": "x" * 1001}, "note"), ({"assign_group_id": "a\\b"}, "assign_group_id")],
)
def test_the_approval_body_is_validated(api, core, civitai, payload, where):
    request = _create(api)

    response = _approve(api, request["id"], **payload)

    assert response.status_code == 400
    assert where in _error(response)["message"]
    assert requests.get(request["id"]).status == "open"


def test_an_approval_without_a_body_is_accepted(api, core, civitai):
    request = _create(api)

    response = api(ADMIN, "POST", f"/requests/{request['id']}/approve")

    assert response.status_code == 200
    assert response.json()["data"]["request"]["decision_note"] is None


def test_rejecting_closes_the_request_and_tells_the_requester(api, core, civitai):
    request = _create(api)

    response = api(APPROVER, "POST", f"/requests/{request['id']}/reject", json={"note": "Too large"})

    assert response.status_code == 200
    rejected = response.json()["data"]["request"]
    assert (rejected["status"], rejected["decided_by"], rejected["decision_note"]) == ("rejected", "approver-1", "Too large")
    [notice] = [n for n in core.notifications if n["notification_type"] == "model-browser.request_rejected"]
    assert notice["user_id"] == "user-1"
    assert notice["message"] == (
        'approver-1 declined your request for Detail Tweaker XL (add-detail-xl.safetensors). Note: "Too large"'
    )
    assert core.queued == []


def test_only_an_open_request_can_be_rejected(api, core, civitai):
    request = _create(api)
    _approve(api, request["id"]).raise_for_status()

    response = api(ADMIN, "POST", f"/requests/{request['id']}/reject", json={})

    assert (response.status_code, _error(response)["error"]) == (409, "not_open")
    assert requests.get(request["id"]).status == "approved"


def test_rejecting_an_unknown_request_is_not_found(api, core, civitai):
    response = api(ADMIN, "POST", "/requests/nope/reject", json={})

    assert (response.status_code, _error(response)["error"]) == (404, "not_found")


def test_the_author_cancels_their_open_request(api, core, civitai):
    request = _create(api)

    response = api(REQUESTER, "DELETE", f"/requests/{request['id']}")

    assert response.status_code == 200
    assert response.json()["data"]["request"]["status"] == "cancelled"
    assert requests.open_count() == 0


@pytest.mark.parametrize("who", [OTHER, ADMIN, APPROVER], ids=["another-user", "admin", "approver"])
def test_nobody_else_cancels_a_request(api, core, civitai, who):
    request = _create(api)

    response = api(who, "DELETE", f"/requests/{request['id']}")

    assert response.status_code == 403
    assert _error(response) == {
        "success": False,
        "error": "forbidden",
        "message": "Only the user who made a request can cancel it",
    }
    assert requests.get(request["id"]).status == "open"


def test_a_closed_request_cannot_be_cancelled(api, core, civitai):
    request = _create(api)
    _approve(api, request["id"]).raise_for_status()

    response = api(REQUESTER, "DELETE", f"/requests/{request['id']}")

    assert (response.status_code, _error(response)["error"]) == (409, "not_open")


def test_cancelling_an_unknown_request_is_not_found(api, core, civitai):
    assert api(REQUESTER, "DELETE", "/requests/nope").status_code == 404


@pytest.fixture
def real_checks(app, monkeypatch, core, civitai):
    state = SimpleNamespace(capabilities=frozenset())
    app.dependency_overrides[get_current_active_user] = lambda: user("user-1")
    monkeypatch.setattr(permissions, "resolve_grant", lambda who: replace(USER_GRANT, capabilities=state.capabilities))
    with TestClient(app) as client:
        yield client, state


def test_the_request_routes_check_their_capabilities(real_checks):
    client, state = real_checks

    state.capabilities = frozenset({"browse"})
    assert client.post(f"{BASE}/requests", json=_body()).status_code == 403
    response = client.get(f"{BASE}/requests")
    assert response.status_code == 403
    assert response.json()["detail"]["message"] == "This action needs one of these permissions: 'request', 'approve'"

    state.capabilities = frozenset({"browse", "request"})
    created = client.post(f"{BASE}/requests", json=_body())
    assert created.status_code == 200
    request_id = created.json()["data"]["request"]["id"]
    assert client.get(f"{BASE}/requests").status_code == 200
    for action in ("approve", "reject"):
        refused = client.post(f"{BASE}/requests/{request_id}/{action}", json={})
        assert (refused.status_code, refused.json()["detail"]["message"]) == (403, "This action needs the 'approve' permission")

    state.capabilities = frozenset()
    assert client.delete(f"{BASE}/requests/{request_id}").status_code == 200


def test_the_request_routes_need_a_signed_in_user(app, core):
    with TestClient(app) as anonymous:
        assert anonymous.post(f"{BASE}/requests", json=_body()).status_code in (401, 403)
        assert anonymous.get(f"{BASE}/requests").status_code in (401, 403)
        assert anonymous.post(f"{BASE}/requests/x/approve", json={}).status_code in (401, 403)
        assert anonymous.post(f"{BASE}/requests/x/reject", json={}).status_code in (401, 403)
        assert anonymous.delete(f"{BASE}/requests/x").status_code in (401, 403)


async def test_an_approved_request_is_done_once_the_watcher_has_the_model_ready(core, civitai, monkeypatch):
    from backend.watcher import Watcher

    request = requests.create(
        user_id="user-1", source="civitai", source_model_id="122359", source_version_id="135867",
        file_id="1001", snapshot={},
    )
    requests.set_status(request.id, "approved", decided_by="admin-1", only_from=("open",))
    prepared = await downloads.prepare_download(
        permissions.principal_for(core.users["user-1"]),
        downloads.DownloadRequest.model_validate(_body()),
        purpose="request",
        on_behalf=True,
    )
    answer = await downloads.queue_prepared(prepared, request_id=request.id, created_by="admin-1")
    requests.update(request.id, sha256=answer["pending"]["sha256"], assign_to=prepared.assign_to)
    core.downloads["dl-1"]["status"] = "completed"
    core.models_by_sha[SHA] = model("model-1")

    await Watcher().run_pass()

    assert pending.get(SHA).status == "enriched"
    assert requests.get(request.id).status == "done"
    assert core.calls_to("assign_to_user") == [("model-1", "user-1")]
    assert [n["user_id"] for n in core.notifications if n["notification_type"] == "model-browser.download_done"] == ["user-1"]
