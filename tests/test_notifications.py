from types import SimpleNamespace

import pytest

from backend import core_bridge, notifications

REQUEST = SimpleNamespace(
    id="req-1",
    user_id="user-1",
    source="civitai",
    source_model_id="122359",
    source_version_id="135867",
    file_id="1001",
    sha256="a" * 64,
    note=None,
    decision_note=None,
    snapshot={"name": "Detail Tweaker XL", "file_name": "add-detail-xl.safetensors"},
)


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(core_bridge, "notify", lambda **kwargs: calls.append(kwargs))
    return calls


def _request(**changes):
    values = dict(vars(REQUEST))
    values.update(changes)
    return SimpleNamespace(**values)


def test_the_types_are_the_five_plugin_types():
    assert [item.key for item in notifications.TYPES] == [
        "model-browser.request_received",
        "model-browser.request_approved",
        "model-browser.request_rejected",
        "model-browser.download_done",
        "model-browser.download_failed",
    ]
    assert all(item.category == "plugin" and item.label and item.description for item in notifications.TYPES)
    assert len({item.label for item in notifications.TYPES}) == 5


def test_send_never_raises(monkeypatch, caplog):
    def broken(**kwargs):
        raise RuntimeError("notifications are down")

    monkeypatch.setattr(core_bridge, "notify", broken)

    assert notifications.send("user-1", notifications.DOWNLOAD_DONE, title="t", message="m") is False
    assert "Could not send the model-browser.download_done notification to user-1" in caplog.text


def test_a_new_request_goes_to_every_approver_but_the_requester(sent):
    count = notifications.request_received(["admin-1", "user-1", "approver-1", "admin-1", ""], REQUEST, "user-1")

    assert count == 2
    assert [call["user_id"] for call in sent] == ["admin-1", "approver-1"]
    first = sent[0]
    assert (first["notification_type"], first["level"], first["title"]) == (
        "model-browser.request_received", "info", "New model request",
    )
    assert first["message"] == "user-1 asked for Detail Tweaker XL (add-detail-xl.safetensors)."
    assert first["metadata"] == {
        "request_id": "req-1",
        "source": "civitai",
        "source_model_id": "122359",
        "source_version_id": "135867",
        "file_id": "1001",
        "link": "/plugins/model-browser-c?tab=requests",
        "link_label": "Open requests",
    }


def test_a_long_note_is_quoted_on_one_line_and_cut(sent):
    note = "For my\nworkflow " + "x" * 400

    notifications.request_received(["admin-1"], _request(note=note), "user-1")

    message = sent[0]["message"]
    assert '. Note: "For my workflow xxx' in message
    assert message.endswith('..."') and "\n" not in message
    assert len(message) < 320


@pytest.mark.parametrize(
    "flags, outcome",
    [
        ({}, "it is being downloaded for you"),
        ({"already_installed": True}, "already installed and is being added to your models"),
        ({"already_queued": True}, "already being downloaded; you will be told when it is ready"),
    ],
)
def test_an_approval_tells_the_requester_what_happens_next(sent, flags, outcome):
    assert notifications.request_approved(_request(decision_note="Approved for the team"), "admin-1", **flags)

    [call] = sent
    assert (call["user_id"], call["notification_type"], call["level"]) == (
        "user-1", "model-browser.request_approved", "success",
    )
    assert call["message"].startswith("admin-1 approved your request for Detail Tweaker XL (add-detail-xl.safetensors): ")
    assert outcome in call["message"]
    assert call["message"].endswith(' Note: "Approved for the team"')
    assert call["metadata"]["sha256"] == "a" * 64 and call["metadata"]["request_id"] == "req-1"


def test_a_rejection_tells_the_requester_with_the_note(sent):
    notifications.request_rejected(_request(decision_note="Too large for the server"), "admin-1")

    [call] = sent
    assert (call["user_id"], call["notification_type"], call["level"], call["title"]) == (
        "user-1", "model-browser.request_rejected", "warning", "Model request declined",
    )
    assert call["message"] == (
        'admin-1 declined your request for Detail Tweaker XL (add-detail-xl.safetensors). Note: "Too large for the server"'
    )


def test_a_rejection_without_a_note_says_so_plainly(sent):
    notifications.request_rejected(_request(snapshot={"file_name": "x.safetensors"}), "admin-1")

    assert sent[0]["message"] == "admin-1 declined your request for x.safetensors."


def test_download_notifications_keep_their_titles_and_messages(sent):
    metadata = {"model_id": "model-1"}

    notifications.download_done("user-1", "Detail Tweaker XL (x.safetensors)", "it is now in your models", metadata)
    notifications.download_failed("user-2", "x.safetensors", "HTTP 401", metadata)
    notifications.download_failed("user-2", "x.safetensors", None, metadata)

    assert [(call["notification_type"], call["title"], call["level"], call["message"]) for call in sent] == [
        ("model-browser.download_done", "Model downloaded", "success", "Detail Tweaker XL (x.safetensors): it is now in your models."),
        ("model-browser.download_failed", "Model download failed", "error", "x.safetensors: HTTP 401"),
        ("model-browser.download_failed", "Model download failed", "error", "x.safetensors: the download failed"),
    ]
    assert all(call["metadata"] == metadata for call in sent)


def test_the_request_label_falls_back_to_what_the_snapshot_knows():
    assert notifications.request_label({"name": "A", "file_name": "a.safetensors"}) == "A (a.safetensors)"
    assert notifications.request_label({"name": "a.safetensors", "file_name": "a.safetensors"}) == "a.safetensors"
    assert notifications.request_label({"name": "A"}) == "A"
    assert notifications.request_label({}) == "a model"
