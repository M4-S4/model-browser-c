from types import SimpleNamespace

import pytest

from backend import core_bridge, limit_handlers, limits
from backend.limits import DOWNLOAD_BYTES, DOWNLOADS, QuotaExceeded, check_quota

from .download_fakes import GIB, FakeCore, limited, unlimited


@pytest.fixture
def core(monkeypatch):
    return FakeCore().install(monkeypatch)


def _refused(user_id="user-1", size_bytes=GIB, **kwargs):
    with pytest.raises(QuotaExceeded) as raised:
        check_quota(user_id, size_bytes, **kwargs)
    return raised.value


def test_a_user_without_a_plan_limit_is_never_refused(core):
    check_quota("user-1", 50 * GIB)

    assert core.calls_to("limit_status") == [("user-1", DOWNLOADS), ("user-1", DOWNLOAD_BYTES)]


def test_an_exempt_admin_is_never_refused(core):
    core.limits[("admin-1", DOWNLOADS)] = {**unlimited(DOWNLOADS), "limit": 0, "exempt": True}
    core.limits[("admin-1", DOWNLOAD_BYTES)] = {**unlimited(DOWNLOAD_BYTES), "limit": 0, "exempt": True}

    check_quota("admin-1", GIB)


def test_downloads_within_the_count_pass_and_the_last_one_is_refused(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 3, 2)
    check_quota("user-1", GIB)

    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 3, 3)
    error = _refused()

    assert (error.kind, error.used, error.limit, error.incoming) == (DOWNLOADS, 3, 3, 1)
    assert error.message == (
        "You have reached your model downloads per month limit: 3 of 3 used. "
        "It resets on Nov 1 at 00:00 UTC. Ask your admin for more."
    )
    assert error.payload() == {
        "kind": DOWNLOADS,
        "label": "Model downloads per month",
        "used": 3,
        "limit": 3,
        "incoming": 1,
        "resets_at": "2026-11-01T00:00:00+00:00",
    }


def test_a_limit_of_zero_refuses_every_download(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 0, 0)

    assert _refused().kind == DOWNLOADS


def test_a_file_that_fits_the_volume_passes_and_one_that_does_not_is_refused(core):
    core.limits[("user-1", DOWNLOAD_BYTES)] = limited(DOWNLOAD_BYTES, 5 * GIB, 3 * GIB)
    check_quota("user-1", 2 * GIB)

    error = _refused(size_bytes=2 * GIB + 1)

    assert (error.kind, error.used, error.limit, error.incoming) == (DOWNLOAD_BYTES, 3 * GIB, 5 * GIB, 2 * GIB + 1)
    assert error.message == (
        "This file (2 GB) would go over your downloaded model bytes per month limit: you have used 3 GB of 5 GB. "
        "It resets on Nov 1 at 00:00 UTC. Ask your admin for more."
    )


def test_a_file_of_unknown_size_is_refused_only_once_the_volume_is_used_up(core):
    core.limits[("user-1", DOWNLOAD_BYTES)] = limited(DOWNLOAD_BYTES, 5 * GIB, 4.9 * GIB)
    check_quota("user-1", None)

    core.limits[("user-1", DOWNLOAD_BYTES)] = limited(DOWNLOAD_BYTES, 5 * GIB, 5 * GIB)
    error = _refused(size_bytes=None)

    assert error.incoming is None
    assert error.message.startswith("You have reached your downloaded model bytes per month limit: 5 GB of 5 GB used.")


def test_the_count_is_checked_before_the_volume(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 1, 1)
    core.limits[("user-1", DOWNLOAD_BYTES)] = limited(DOWNLOAD_BYTES, GIB, GIB)

    assert _refused().kind == DOWNLOADS


def test_for_an_approver_the_message_speaks_of_the_requester(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 1, 1)
    assert _refused(requester_name="mb-test-blocked").message == (
        "mb-test-blocked has reached their model downloads per month limit: 1 of 1 used. It resets on Nov 1 at 00:00 UTC."
    )

    core.limits[("user-1", DOWNLOADS)] = unlimited(DOWNLOADS)
    core.limits[("user-1", DOWNLOAD_BYTES)] = limited(DOWNLOAD_BYTES, GIB, 512 * 1024 ** 2)
    assert _refused(size_bytes=GIB, requester_name="mb-test-blocked").message == (
        "This file (1 GB) would go over the downloaded model bytes per month limit of mb-test-blocked: "
        "they have used 512 MB of 1 GB. It resets on Nov 1 at 00:00 UTC."
    )


def test_a_status_that_cannot_be_read_does_not_block(core, monkeypatch, caplog):
    def broken(user_id, kind):
        raise RuntimeError("plans are down")

    monkeypatch.setattr(core_bridge, "limit_status", broken)

    check_quota("user-1", GIB)

    assert "not enforcing it" in caplog.text


def test_broken_plans_read_as_unlimited_through_the_real_limit_status(monkeypatch, caplog):
    def no_container(user_id, kind):
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(core_bridge, "api_limits", SimpleNamespace(limit_status=no_container))

    check_quota("user-1", 100 * GIB)

    assert "treating it as unlimited" in caplog.text


def test_the_reset_is_left_out_when_unknown(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 1, 1, resets_at=None, contact_line="")

    assert _refused().message == "You have reached your model downloads per month limit: 1 of 1 used."


def test_the_reset_is_told_in_utc_whatever_the_offset_it_comes_with(core):
    core.limits[("user-1", DOWNLOADS)] = limited(DOWNLOADS, 1, 1, resets_at="2026-11-01T00:00:00+01:00", contact_line="")

    assert _refused().message == (
        "You have reached your model downloads per month limit: 1 of 1 used. It resets on Oct 31 at 23:00 UTC."
    )


@pytest.mark.parametrize(
    "kind, value, text",
    [
        (DOWNLOADS, 3, "3"),
        (DOWNLOADS, 2.5, "2.5"),
        (DOWNLOAD_BYTES, 228_000_000, "217.4 MB"),
        (DOWNLOAD_BYTES, 5 * GIB, "5 GB"),
        (DOWNLOAD_BYTES, 0, "0 B"),
        (DOWNLOAD_BYTES, 3 * 1024 ** 4, "3 TB"),
    ],
)
def test_amounts_are_shown_as_counts_or_sizes(kind, value, text):
    assert limits.format_amount(kind, value) == text


def test_record_counts_one_download_and_its_size(core):
    limits.record("user-1", 228_000_000, "model-browser:ref-1")

    assert core.limit_events == [
        ("user-1", DOWNLOADS, 1, "model-browser:ref-1"),
        ("user-1", DOWNLOAD_BYTES, 228_000_000, "model-browser:ref-1"),
    ]


@pytest.mark.parametrize("size", [None, 0])
def test_record_counts_the_download_only_when_the_size_is_unknown(core, size):
    limits.record("user-1", size, "model-browser:ref-1")

    assert core.limit_events == [("user-1", DOWNLOADS, 1, "model-browser:ref-1")]


def test_refund_cancels_the_events_of_a_ref_and_ignores_none(core):
    limits.record("user-1", 10, "model-browser:ref-1")

    assert limits.refund("model-browser:ref-1") == 2
    assert limits.refund(None) == 0
    assert core.refunds == ["model-browser:ref-1"]


def test_record_and_refund_never_raise(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("AppContainer not initialized yet")

    monkeypatch.setattr(limits, "LimitEvents", SimpleNamespace(record=broken, refund=broken))

    limits.record("user-1", 10, "model-browser:ref-1")
    assert limits.refund("model-browser:ref-1") == 0
    assert "Could not record a model-browser.downloads event" in caplog.text
    assert "Could not refund" in caplog.text


def test_ref_ids_are_unique_and_name_the_plugin():
    first, second = limits.new_ref_id(), limits.new_ref_id()

    assert first != second
    assert first.startswith("model-browser-c:") and len(first) == len("model-browser-c:") + 26


def test_the_core_is_never_asked_to_apply_the_kinds():
    assert limit_handlers.never_applies(SimpleNamespace(point="submit", user_id="user-1")) is False


def test_the_usage_is_the_ledger_total_of_the_window():
    context = SimpleNamespace(events=SimpleNamespace(total=lambda: 3))

    assert limit_handlers.ledger_total(context) == 3.0
