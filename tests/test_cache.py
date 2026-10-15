import json

import pytest

from backend import nsfw_filter
from backend.cache import Cache, cache_key, format_timestamp
from backend.sources import SearchQuery
from backend.tables import create_tables

from .civitai_fakes import FakeTransport, body, context, fake_settings, image_urls, make_source

T0 = 1_791_720_000.0


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def db(plugin_db, monkeypatch):
    create_tables()
    fake_settings(monkeypatch, cache_ttl_minutes=10)
    return plugin_db


@pytest.fixture
def clock():
    return Clock()


def _rows(db):
    with db.get_cursor() as cursor:
        cursor.execute("SELECT key, source, payload, fetched_at FROM model_browser_cache ORDER BY fetched_at")
        return [dict(row) for row in cursor.fetchall()]


REQUEST = {"url": "https://civitai.com/api/v1/models", "params": [["limit", "44"], ["nsfw", "true"]]}


def test_the_key_is_the_md5_of_the_source_and_the_canonical_request():
    import hashlib

    expected = hashlib.md5(
        ('civitai{"params":[["limit","44"],["nsfw","true"]],"url":"https://civitai.com/api/v1/models"}').encode()
    ).hexdigest()

    assert cache_key("civitai", REQUEST) == expected


def test_key_order_does_not_matter_but_every_parameter_does():
    reordered = {"params": [["limit", "44"], ["nsfw", "true"]], "url": "https://civitai.com/api/v1/models"}
    safe = {"url": "https://civitai.com/api/v1/models", "params": [["limit", "44"], ["nsfw", "false"]]}

    assert cache_key("civitai", reordered) == cache_key("civitai", REQUEST)
    assert cache_key("civitai", safe) != cache_key("civitai", REQUEST)
    assert cache_key("huggingface", REQUEST) != cache_key("civitai", REQUEST)


def test_miss_then_hit(db, clock):
    cache = Cache(clock=clock)

    assert cache.get("civitai", REQUEST) is None
    cache.set("civitai", REQUEST, {"items": [1, 2]})

    assert cache.get("civitai", REQUEST) == {"items": [1, 2]}
    assert _rows(db)[0]["source"] == "civitai"
    assert json.loads(_rows(db)[0]["payload"]) == {"items": [1, 2]}
    assert _rows(db)[0]["fetched_at"] == "2026-10-11 12:00:00.000000"


def test_an_entry_expires_after_the_ttl_setting(db, clock, monkeypatch):
    cache = Cache(clock=clock)
    cache.set("civitai", REQUEST, {"n": 1})

    clock.now = T0 + 600
    assert cache.get("civitai", REQUEST) == {"n": 1}
    clock.now = T0 + 601
    assert cache.get("civitai", REQUEST) is None

    fake_settings(monkeypatch, cache_ttl_minutes=60)
    assert cache.get("civitai", REQUEST) == {"n": 1}


def test_an_explicit_ttl_overrides_the_setting(db, clock):
    cache = Cache(clock=clock)
    cache.set("civitai", REQUEST, {"enums": True}, ttl_seconds=86400)

    clock.now = T0 + 3600
    assert cache.get("civitai", REQUEST) is None
    assert cache.get("civitai", REQUEST, ttl_seconds=86400) == {"enums": True}
    clock.now = T0 + 86401
    assert cache.get("civitai", REQUEST, ttl_seconds=86400) is None


def test_set_replaces_the_entry_and_its_time(db, clock):
    cache = Cache(clock=clock)
    cache.set("civitai", REQUEST, {"v": 1})
    clock.now = T0 + 300
    cache.set("civitai", REQUEST, {"v": 2})

    clock.now = T0 + 800
    assert cache.get("civitai", REQUEST) == {"v": 2}
    assert len(_rows(db)) == 1


def test_an_unreadable_payload_is_a_miss(db, clock):
    cache = Cache(clock=clock)
    with db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO model_browser_cache (key, source, payload, fetched_at) VALUES (?, ?, ?, ?)",
            (cache_key("civitai", REQUEST), "civitai", "{not json", format_timestamp(T0)),
        )

    assert cache.get("civitai", REQUEST) is None


def _request(n):
    return {"url": "https://civitai.com/api/v1/models", "params": [["cursor", str(n)]]}


def test_purge_drops_expired_rows_and_keeps_long_lived_ones(db, clock):
    cache = Cache(clock=clock, purge_every=0)
    cache.set("civitai", _request(1), "old")
    cache.set("civitai", _request(2), "enums", ttl_seconds=86400)
    clock.now = T0 + 700
    cache.set("civitai", _request(3), "fresh")

    assert cache.purge() == 1
    assert [json.loads(row["payload"]) for row in _rows(db)] == ["enums", "fresh"]

    clock.now = T0 + 86401
    cache.purge()
    assert _rows(db) == []


def test_purge_keeps_only_the_newest_rows_beyond_the_cap(db, clock):
    cache = Cache(clock=clock, max_rows=3, purge_every=0)
    for n in range(6):
        clock.now = T0 + n
        cache.set("civitai", _request(n), n)

    assert cache.purge() == 3
    assert [json.loads(row["payload"]) for row in _rows(db)] == [3, 4, 5]


def test_writes_purge_lazily_on_the_first_write_then_every_n(db, clock):
    cache = Cache(clock=clock, purge_every=3)
    calls = []
    real_purge = cache.purge
    cache.purge = lambda: calls.append(cache._writes) or real_purge()

    for n in range(7):
        cache.set("civitai", _request(n), n)

    assert calls == [1, 4, 7]


def test_a_lazy_purge_removes_expired_rows_while_writing(db, clock):
    cache = Cache(clock=clock, purge_every=2)
    cache.set("civitai", _request(1), "a")
    clock.now = T0 + 700
    cache.set("civitai", _request(2), "b")
    cache.set("civitai", _request(3), "c")

    assert [json.loads(row["payload"]) for row in _rows(db)] == ["b", "c"]


def test_a_failing_purge_does_not_fail_the_write(db, clock):
    cache = Cache(clock=clock, purge_every=1)

    def broken():
        raise RuntimeError("disk full")

    cache.purge = broken
    cache.set("civitai", REQUEST, {"ok": True})

    assert cache.get("civitai", REQUEST) == {"ok": True}


async def test_one_raw_entry_serves_two_users_without_leaks(db, clock):
    raw = body("models_flux_nsfwlevel")
    nsfw_image = raw["items"][0]["modelVersions"][0]["images"][0]
    nsfw_image["nsfwLevel"] = 16
    transport = FakeTransport().add("/models", (200, {}, raw))
    source = make_source(transport, cache=Cache(clock=clock))
    query = SearchQuery(limit=2, safe=False)

    shown = nsfw_filter.filter_page(await source.search(query, max_level=2), context("allowed", "show"))
    blurred = nsfw_filter.filter_page(await source.search(query, max_level=2), context("blur", "blur"))

    assert len(transport.calls) == 1, "the second user is served from the cache"
    assert [row["source"] for row in _rows(db)] == ["civitai"]
    assert json.loads(_rows(db)[0]["payload"]) == raw, "the entry stays raw"
    assert shown.items[0].thumbnail.url == nsfw_image["url"]
    assert blurred.items[0].thumbnail.url is None
    assert blurred.items[0].thumbnail.blurhash == nsfw_image["hash"]
    assert nsfw_image["url"] not in image_urls(blurred.model_dump())
    assert blurred.items[1].thumbnail.url is not None


async def test_blocked_and_hiding_users_share_the_safe_entry(db, clock):
    transport = FakeTransport().add("/models", "models_lora_page1")
    source = make_source(transport, cache=Cache(clock=clock))
    blocked = context("blocked", "hide")
    hiding = context("allowed", "hide")
    query = SearchQuery(limit=2, safe=True)

    first = nsfw_filter.filter_page(await source.search(query, max_level=2), blocked)
    second = nsfw_filter.filter_page(await source.search(query, max_level=2), hiding)

    assert len(transport.calls) == 1
    assert transport.calls_to("/models")[0]["nsfw"] == "false"
    assert [item.id for item in first.items] == [item.id for item in second.items] == ["58390"]
