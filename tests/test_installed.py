import asyncio
import logging
import threading

import pytest

from backend import core_bridge
from backend.installed import installed_map, mark_installed
from backend.schema import SearchItem

from .download_fakes import OTHER_SHA, SHA, FakeCore, admin, model, principal

THIRD_SHA = "c" * 64


@pytest.fixture
def core(monkeypatch):
    return FakeCore().install(monkeypatch)


def _item(item_id, sha256=None):
    return SearchItem(
        source="civitai",
        id=item_id,
        name=f"Model {item_id}",
        type="LORA",
        source_url=f"https://civitai.com/models/{item_id}",
        primary_sha256=sha256,
    )


def _core_moved(*args):
    raise RuntimeError("The AppContainer no longer provides model_repository.get_by_sha256")


async def test_an_admin_sees_every_indexed_file_whose_file_is_present(core):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha[THIRD_SHA] = model("model-3", THIRD_SHA, is_available=False)

    result = await installed_map([SHA, OTHER_SHA, THIRD_SHA], admin())

    assert result == {SHA: True, OTHER_SHA: False, THIRD_SHA: False}
    assert "visible_model_ids" not in core.names()


async def test_a_user_only_counts_the_models_they_can_see(core):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha[OTHER_SHA] = model("model-2", OTHER_SHA)
    core.visible["user-1"] = ["model-1", "model-9"]

    result = await installed_map([SHA, OTHER_SHA], principal())

    assert result == {SHA: True, OTHER_SHA: False}
    assert core.calls_to("visible_model_ids") == [("user-1",)]


async def test_the_visible_models_are_those_of_the_caller_not_of_someone_else(core):
    core.models_by_sha[SHA] = model("model-1")
    core.visible["user-1"] = ["model-1"]
    core.visible["user-2"] = []

    assert await installed_map([SHA], principal("user-1")) == {SHA: True}
    assert await installed_map([SHA], principal("user-2")) == {SHA: False}


async def test_a_visible_model_whose_file_has_gone_does_not_count(core):
    core.models_by_sha[SHA] = model("model-1", is_available=False)
    core.visible["user-1"] = ["model-1"]

    assert await installed_map([SHA], principal()) == {SHA: False}


async def test_a_model_without_the_availability_flag_counts_as_present(core):
    indexed = model("model-1")
    del indexed["is_available"]
    core.models_by_sha[SHA] = indexed

    assert await installed_map([SHA], admin()) == {SHA: True}


@pytest.mark.parametrize("who", [admin(), principal()], ids=["admin", "user"])
async def test_a_hash_the_core_does_not_know_is_not_installed(core, who):
    result = await installed_map([SHA], who)

    assert result == {SHA: False}
    assert core.calls_to("find_model_by_sha256") == [(SHA,)]


async def test_hashes_are_normalised_and_looked_up_once_each(core):
    core.models_by_sha[SHA] = model("model-1")

    result = await installed_map([SHA.upper(), " " + SHA + " ", SHA, "", "   "], admin())

    assert result == {SHA: True}
    assert core.calls == [("find_model_by_sha256", SHA)]


async def test_no_hash_asks_the_core_nothing(core):
    assert await installed_map([], principal()) == {}
    assert await installed_map(["", " "], principal()) == {}
    assert await installed_map(iter(()), admin()) == {}
    assert core.calls == []


async def test_the_core_is_read_in_one_worker_thread_hop_never_on_the_event_loop(core, monkeypatch):
    core.models_by_sha[SHA] = model("model-1")
    core.visible["user-1"] = ["model-1"]
    hops = []
    threads = set()
    real_to_thread = asyncio.to_thread
    real_find = core.find_model_by_sha256

    async def counting_to_thread(function, /, *args, **kwargs):
        hops.append(function)
        return await real_to_thread(function, *args, **kwargs)

    def find_in_thread(sha256):
        threads.add(threading.get_ident())
        return real_find(sha256)

    monkeypatch.setattr(asyncio, "to_thread", counting_to_thread)
    monkeypatch.setattr(core_bridge, "find_model_by_sha256", find_in_thread)

    await installed_map([SHA, OTHER_SHA, THIRD_SHA], principal())

    assert len(hops) == 1
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert core.names() == ["visible_model_ids", "find_model_by_sha256", "find_model_by_sha256", "find_model_by_sha256"]


@pytest.mark.parametrize("failing", ["find_model_by_sha256", "visible_model_ids"])
async def test_installed_map_raises_what_the_core_raises(core, monkeypatch, failing):
    monkeypatch.setattr(core_bridge, failing, _core_moved)

    with pytest.raises(RuntimeError, match="no longer provides"):
        await installed_map([SHA], principal())


async def test_items_are_marked_installed_not_installed_or_left_unknown(core):
    core.models_by_sha[SHA] = model("model-1")
    items = [_item("1", SHA), _item("2", OTHER_SHA), _item("3")]

    await mark_installed(items, admin())

    assert [item.installed for item in items] == [True, False, None]
    assert [item.primary_sha256 for item in items] == [SHA, OTHER_SHA, None]


async def test_a_user_sees_installed_only_for_what_is_visible_to_them_with_one_lookup_for_the_page(core):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha[OTHER_SHA] = model("model-2", OTHER_SHA)
    core.visible["user-1"] = ["model-1"]
    items = [_item("1", SHA), _item("2", OTHER_SHA), _item("3", THIRD_SHA), _item("4", SHA), _item("5")]

    await mark_installed(items, principal())

    assert [item.installed for item in items] == [True, False, False, True, None]
    assert core.names().count("visible_model_ids") == 1
    assert core.calls_to("find_model_by_sha256") == [(SHA,), (OTHER_SHA,), (THIRD_SHA,)]


async def test_a_mixed_case_hash_on_an_item_still_finds_its_model(core):
    core.models_by_sha[SHA] = model("model-1")
    items = [_item("1", SHA.upper())]

    await mark_installed(items, admin())

    assert items[0].installed is True


async def test_a_page_without_hashes_asks_the_core_nothing(core):
    items = [_item("org/one"), _item("org/two")]

    await mark_installed(items, principal())
    await mark_installed([], principal())

    assert [item.installed for item in items] == [None, None]
    assert core.calls == []


@pytest.mark.parametrize("failing", ["find_model_by_sha256", "visible_model_ids"])
async def test_a_core_error_leaves_every_item_unmarked_and_is_logged_at_debug(core, monkeypatch, caplog, failing):
    core.models_by_sha[SHA] = model("model-1")
    monkeypatch.setattr(core_bridge, failing, _core_moved)
    items = [_item("1", SHA), _item("2", OTHER_SHA), _item("3")]

    with caplog.at_level(logging.DEBUG, logger="plugins.model-browser"):
        await mark_installed(items, principal())

    assert [item.installed for item in items] == [None, None, None]
    [record] = [found for found in caplog.records if found.name == "plugins.model-browser"]
    assert record.levelno == logging.DEBUG
    assert "Could not tell which results are installed" in record.getMessage()
    assert record.exc_info is not None and "no longer provides" in str(record.exc_info[1])


async def test_a_core_error_is_silent_above_debug(core, monkeypatch, caplog):
    monkeypatch.setattr(core_bridge, "find_model_by_sha256", _core_moved)
    items = [_item("1", SHA)]

    with caplog.at_level(logging.INFO, logger="plugins.model-browser"):
        await mark_installed(items, principal())

    assert items[0].installed is None
    assert caplog.records == []


@pytest.mark.parametrize("who", [admin(), principal(), principal("user-2")], ids=["admin", "user", "other-user"])
async def test_the_badge_answers_exactly_what_the_installed_map_answers(core, who):
    core.models_by_sha[SHA] = model("model-1")
    core.models_by_sha[OTHER_SHA] = model("model-2", OTHER_SHA)
    core.models_by_sha[THIRD_SHA] = model("model-3", THIRD_SHA, is_available=False)
    core.visible["user-1"] = ["model-1", "model-3"]
    core.visible["user-2"] = ["model-2"]
    hashes = [SHA, OTHER_SHA, THIRD_SHA, "d" * 64]
    items = [_item(str(number), sha256) for number, sha256 in enumerate(hashes)]

    await mark_installed(items, who)

    assert {item.primary_sha256: item.installed for item in items} == await installed_map(hashes, who)
