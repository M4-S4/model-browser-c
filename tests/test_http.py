import asyncio
import json
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import aiohttp
import pytest

from backend import PLUGIN_VERSION
from backend import http as http_module
from backend.http import (
    USER_AGENT,
    SourceAnswerTooLarge,
    SourceError,
    SourceHttp,
    SourceInvalidRequest,
    SourceNotFound,
    SourceRateLimited,
    SourceUnavailable,
    decode_text,
    normalise_params,
    parse_retry_after,
    upstream_message,
)

from .civitai_fakes import load

URL = "https://civitai.com/api/v1/models"


def _answer(status=200, body=b"{}", headers=None):
    async def fetch(url, params):
        return status, headers or {}, body

    return fetch


def _fixture_answer(name):
    import json

    fixture = load(name)

    async def fetch(url, params):
        return fixture["_status"], fixture["_headers"], json.dumps(fixture["body"]).encode()

    return fetch


def _raising(exc):
    async def fetch(url, params):
        raise exc

    return fetch


def _client(fetch, **kwargs):
    kwargs.setdefault("limits", {"civitai": (0.0, 4)})
    return SourceHttp(fetch=fetch, **kwargs)


async def test_a_2xx_answer_is_decoded_json():
    client = _client(_fixture_answer("models_ids"))

    data = await client.get_json("civitai", URL, {"ids": "122359"})

    assert data["items"][0]["id"] == 122359


async def test_the_transport_gets_the_url_and_the_parameters_as_pairs():
    seen = []

    async def fetch(url, params):
        seen.append((url, params))
        return 200, {}, b"[]"

    await _client(fetch).get_json("civitai", URL, [("types", "LORA"), ("types", "Checkpoint"), ("nsfw", False)])

    assert seen == [(URL, [("types", "LORA"), ("types", "Checkpoint"), ("nsfw", "false")])]


def test_normalise_params():
    assert normalise_params(None) == []
    assert normalise_params({"limit": 44, "nsfw": True, "cursor": None, "types": ["LORA", "VAE"]}) == [
        ("limit", "44"),
        ("nsfw", "true"),
        ("types", "LORA"),
        ("types", "VAE"),
    ]


def test_the_user_agent_names_the_plugin_and_its_version():
    assert USER_AGENT == f"PotionUI-ModelBrowser/{PLUGIN_VERSION} (+https://github.com/PotionUI/PotionUI)"


async def test_429_is_rate_limited_with_the_retry_after_seconds():
    client = _client(_answer(429, b'{"error": "Too many requests"}', {"Retry-After": "17"}))

    with pytest.raises(SourceRateLimited) as caught:
        await client.get_json("civitai", URL)

    assert caught.value.retry_after == 17.0
    assert caught.value.status == 429


async def test_429_reads_a_lowercase_header_and_an_http_date():
    when = datetime.now(timezone.utc) + timedelta(seconds=120)
    client = _client(_answer(429, b"", {"retry-after": format_datetime(when, usegmt=True)}))

    with pytest.raises(SourceRateLimited) as caught:
        await client.get_json("civitai", URL)

    assert 100 < caught.value.retry_after <= 120


async def test_429_without_retry_after_has_none():
    with pytest.raises(SourceRateLimited) as caught:
        await _client(_answer(429)).get_json("civitai", URL)

    assert caught.value.retry_after is None


def test_parse_retry_after():
    in_a_minute = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)

    assert parse_retry_after("30") == 30.0
    assert parse_retry_after(" 2.5 ") == 2.5
    assert parse_retry_after("-4") == 0.0
    assert 50 < parse_retry_after(in_a_minute) <= 60
    assert parse_retry_after("Sun, 11 Oct 2020 11:00:00 GMT") == 0.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None


@pytest.mark.parametrize("status", [500, 502, 503, 504, 520])
async def test_5xx_is_unavailable(status):
    with pytest.raises(SourceUnavailable) as caught:
        await _client(_answer(status, b"<html>Bad gateway</html>")).get_json("civitai", URL)

    assert caught.value.status == status
    assert str(status) in caught.value.message


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        aiohttp.ServerTimeoutError("read timeout"),
        aiohttp.ClientConnectionError("connection reset"),
        aiohttp.ClientPayloadError("truncated"),
        ConnectionRefusedError(111, "refused"),
        ssl.SSLError("handshake failure"),
    ],
)
async def test_transport_failures_are_unavailable(exc):
    with pytest.raises(SourceUnavailable) as caught:
        await _client(_raising(exc)).get_json("civitai", URL)

    assert caught.value.status is None


async def test_a_timeout_says_so():
    with pytest.raises(SourceUnavailable, match="did not answer within 30 s"):
        await _client(_raising(asyncio.TimeoutError())).get_json("civitai", URL)


async def test_a_certificate_failure_says_so():
    with pytest.raises(SourceUnavailable, match="TLS certificate"):
        await _client(_raising(ssl.SSLCertVerificationError("certificate verify failed"))).get_json("civitai", URL)


async def test_400_carries_the_upstream_zod_message():
    with pytest.raises(SourceInvalidRequest) as caught:
        await _client(_fixture_answer("error_400_limit")).get_json("civitai", URL, {"limit": 1000})

    assert caught.value.status == 400
    assert caught.value.message == "CivitAI refused the request: limit: Too big: expected number to be <=100"


@pytest.mark.parametrize("status", [400, 401, 403, 410, 422])
async def test_other_4xx_are_invalid_requests(status):
    with pytest.raises(SourceInvalidRequest) as caught:
        await _client(_answer(status, b"<html>no</html>")).get_json("civitai", URL)

    assert caught.value.message == f"CivitAI refused the request (HTTP {status})"


async def test_404_is_not_found_with_the_upstream_message():
    with pytest.raises(SourceNotFound) as caught:
        await _client(_fixture_answer("model_version_by_hash")).get_json("civitai", URL)

    assert caught.value.status == 404
    assert caught.value.message == "Model not found"


async def test_an_answer_that_is_not_json_is_unavailable():
    with pytest.raises(SourceUnavailable, match="not JSON"):
        await _client(_answer(200, b"<html>challenge</html>")).get_json("civitai", URL)


async def test_an_answer_over_the_size_cap_is_unavailable():
    with pytest.raises(SourceUnavailable, match="larger than"):
        await _client(_answer(200, b"[" + b"1," * 100 + b"1]"), max_body_bytes=64).get_json("civitai", URL)


async def test_a_source_error_from_the_transport_passes_through():
    with pytest.raises(SourceNotFound):
        await _client(_raising(SourceNotFound("gone", 404))).get_json("civitai", URL)


def test_every_error_is_a_source_error():
    for error in (SourceInvalidRequest, SourceNotFound, SourceRateLimited, SourceUnavailable):
        assert issubclass(error, SourceError)


def test_upstream_message_shapes():
    assert upstream_message(b'{"error": "Model not found"}') == "Model not found"
    assert upstream_message(b'{"message": "Bad   input\\n here"}') == "Bad input here"
    assert upstream_message(b'{"error": {"name": "Other", "message": "plain"}}') == "plain"
    assert upstream_message(b"<html>") is None
    assert upstream_message(b'{"error": "' + b"x" * 300 + b'"}') == "x" * 300
    assert upstream_message(b'{"error": "' + b"x" * 400 + b'"}') == "x" * 299 + "…"


async def test_sequential_requests_are_spaced_by_the_source_interval():
    now = [100.0]
    sleeps = []
    starts = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    async def fetch(url, params):
        starts.append(now[0])
        return 200, {}, b"{}"

    client = SourceHttp(fetch=fetch, limits={"civitai": (0.25, 4)}, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        await client.get_json("civitai", URL)

    assert starts == [100.0, 100.25, 100.5]
    assert sleeps == [0.25, 0.25]


async def test_a_request_after_a_quiet_spell_does_not_wait():
    now = [100.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = SourceHttp(fetch=_answer(), limits={"civitai": (0.25, 4)}, clock=lambda: now[0], sleep=sleep)
    await client.get_json("civitai", URL)
    now[0] += 5
    await client.get_json("civitai", URL)

    assert sleeps == []


async def test_concurrent_requests_start_at_least_the_interval_apart():
    starts = []

    async def fetch(url, params):
        starts.append(time.monotonic())
        await asyncio.sleep(0.01)
        return 200, {}, b"{}"

    client = SourceHttp(fetch=fetch, limits={"civitai": (0.05, 4)})
    await asyncio.gather(*(client.get_json("civitai", URL) for _ in range(4)))

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert len(starts) == 4
    assert all(gap >= 0.045 for gap in gaps), gaps


async def test_requests_in_flight_are_capped_per_source():
    in_flight = 0
    peak = 0

    async def fetch(url, params):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return 200, {}, b"{}"

    client = SourceHttp(fetch=fetch, limits={"civitai": (0.0, 2)})
    await asyncio.gather(*(client.get_json("civitai", URL) for _ in range(6)))

    assert peak == 2


async def test_each_source_has_its_own_limiter():
    now = [0.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = SourceHttp(
        fetch=_answer(), limits={"civitai": (1.0, 1), "huggingface": (1.0, 1)}, clock=lambda: now[0], sleep=sleep
    )
    await client.get_json("civitai", URL)
    await client.get_json("huggingface", "https://huggingface.co/api/models")

    assert sleeps == []


def test_civitai_waits_a_quarter_second_between_requests_by_default():
    assert http_module.DEFAULT_LIMITS["civitai"][0] == 0.25


async def test_a_failed_request_still_releases_its_slot():
    client = SourceHttp(fetch=_raising(asyncio.TimeoutError()), limits={"civitai": (0.0, 1)})
    for _ in range(3):
        with pytest.raises(SourceUnavailable):
            await asyncio.wait_for(client.get_json("civitai", URL), timeout=1)


def test_the_limiter_follows_a_new_event_loop():
    client = _client(_answer())

    asyncio.run(client.get_json("civitai", URL))
    asyncio.run(client.get_json("civitai", URL))


async def test_the_session_is_created_lazily_with_the_user_agent_and_closed():
    client = SourceHttp()
    assert client._session is None

    session = client._get_session()

    assert session.headers["User-Agent"] == USER_AGENT
    assert session.timeout.total == 30
    assert client._get_session() is session
    await client.close()
    assert session.closed


async def test_close_soon_closes_the_session_on_its_loop():
    client = SourceHttp()
    session = client._get_session()

    thread = threading.Thread(target=client.close_soon)
    thread.start()
    thread.join()
    for _ in range(20):
        if session.closed:
            break
        await asyncio.sleep(0.01)

    assert session.closed


def test_shutdown_without_a_session_does_nothing(monkeypatch):
    monkeypatch.setattr(http_module, "http", SourceHttp())

    http_module.shutdown()


HF_URL = "https://huggingface.co/api/models"
README_URL = "https://huggingface.co/latent-consistency/lcm-lora-sdxl/raw/main/README.md"


def _hf_client(fetch, **kwargs):
    kwargs.setdefault("limits", {"huggingface": (0.0, 4)})
    return SourceHttp(fetch=fetch, **kwargs)


async def test_get_gives_the_status_the_lower_case_headers_and_the_body():
    link = '<https://huggingface.co/api/models?cursor=abc>; rel="next"'
    client = _hf_client(_answer(200, b"[]", {"Link": link, "Content-Type": "application/json"}))

    status, headers, body = await client.get("huggingface", HF_URL, {"limit": 3})

    assert (status, body) == (200, b"[]")
    assert headers == {"link": link, "content-type": "application/json"}


async def test_get_json_with_headers_gives_the_link_header():
    from .huggingface_fakes import body as hf_body
    from .huggingface_fakes import load as hf_load

    fixture = hf_load("models_search_lora")
    client = _hf_client(_answer(200, json.dumps(fixture["body"]).encode(), fixture["_headers"]))

    data, headers = await client.get_json_with_headers("huggingface", HF_URL)

    assert data == hf_body("models_search_lora")
    assert headers["link"] == fixture["_headers"]["Link"]


async def test_get_text_decodes_the_readme():
    from .huggingface_fakes import load as hf_load

    fixture = hf_load("readme_lcm_lora_sdxl")
    client = _hf_client(_answer(200, fixture["body"].encode("utf-8"), fixture["_headers"]))

    text = await client.get_text("huggingface", README_URL)

    assert text == fixture["body"]
    assert "🤗" in text


@pytest.mark.parametrize(
    "headers, payload, expected",
    [
        ({"Content-Type": "text/plain; charset=utf-8"}, "é".encode("utf-8"), "é"),
        ({"Content-Type": "text/plain; charset=ISO-8859-1"}, "é".encode("latin-1"), "é"),
        ({"content-type": 'text/plain; charset="latin-1"'}, "é".encode("latin-1"), "é"),
        ({"Content-Type": "text/plain; charset=nope"}, "é".encode("utf-8"), "é"),
        ({"Content-Type": "text/plain; charset=base64"}, "é".encode("utf-8"), "é"),
        ({}, "\N{ZERO WIDTH NO-BREAK SPACE}hello".encode("utf-8"), "hello"),
        ({}, b"bad \xff byte", "bad \N{REPLACEMENT CHARACTER} byte"),
        ({}, b"", ""),
    ],
)
def test_decode_text(headers, payload, expected):
    assert decode_text(headers, payload) == expected


async def test_a_gated_readme_is_a_401_invalid_request():
    from .huggingface_fakes import load as hf_load

    fixture = hf_load("readme_flux_dev_gated")
    client = _hf_client(_answer(fixture["_status"], fixture["body"].encode(), fixture["_headers"]))

    with pytest.raises(SourceInvalidRequest) as caught:
        await client.get_text("huggingface", README_URL)

    assert caught.value.status == 401
    assert caught.value.message == "Hugging Face refused the request (HTTP 401)"


async def test_an_unknown_repo_is_a_401_with_hugging_face_s_message():
    from .huggingface_fakes import load as hf_load

    fixture = hf_load("error_401_unknown_repo")
    client = _hf_client(_answer(401, json.dumps(fixture["body"]).encode(), fixture["_headers"]))

    with pytest.raises(SourceInvalidRequest) as caught:
        await client.get_json_with_headers("huggingface", f"{HF_URL}/no-such-org/no-such-repo")

    assert caught.value.message == "Hugging Face refused the request: Invalid username or password."


async def test_max_bytes_refuses_a_larger_answer_for_one_call():
    client = _hf_client(_answer(200, b"x" * 101))

    with pytest.raises(SourceAnswerTooLarge, match="larger than 100 bytes") as caught:
        await client.get_text("huggingface", README_URL, max_bytes=100)
    assert await client.get_text("huggingface", README_URL, max_bytes=101) == "x" * 101
    assert await client.get_text("huggingface", README_URL) == "x" * 101

    assert isinstance(caught.value, SourceUnavailable)
    assert caught.value.status == 200


async def test_max_bytes_never_raises_the_client_cap():
    client = _hf_client(_answer(200, b"x" * 65), max_body_bytes=64)

    with pytest.raises(SourceAnswerTooLarge, match="larger than 64 bytes"):
        await client.get("huggingface", README_URL, max_bytes=10_000)


async def test_the_size_check_comes_before_the_status():
    client = _hf_client(_answer(503, b"x" * 65), max_body_bytes=64)

    with pytest.raises(SourceAnswerTooLarge):
        await client.get_json("huggingface", HF_URL)


async def test_get_raises_the_same_errors_as_get_json():
    for status, error in ((404, SourceNotFound), (429, SourceRateLimited), (400, SourceInvalidRequest), (502, SourceUnavailable)):
        client = _hf_client(_answer(status, b'{"error": "x"}'))
        with pytest.raises(error):
            await client.get("huggingface", HF_URL)
        with pytest.raises(error):
            await client.get_text("huggingface", README_URL)


def test_hugging_face_waits_a_fifth_of_a_second_between_requests_with_four_in_flight():
    assert http_module.DEFAULT_LIMITS["huggingface"] == (0.2, 4)


async def test_hugging_face_requests_are_spaced_by_its_interval():
    now = [10.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = SourceHttp(fetch=_answer(), clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        await client.get("huggingface", HF_URL)

    assert sleeps == [pytest.approx(0.2), pytest.approx(0.2)]


CIVARCHIVE_URL = "https://civarchive.com/_next/data/build/models/1.json"
NOT_FOUND_PAGE = b"<!DOCTYPE html><title>Error 500: Model not found:1</title>"


async def test_accept_statuses_hands_those_answers_back_to_read():
    client = SourceHttp(fetch=_answer(500, NOT_FOUND_PAGE, {"Content-Type": "text/html; charset=utf-8"}), limits={"civarchive": (0.0, 1)})

    status, headers, body = await client.get("civarchive", CIVARCHIVE_URL, accept_statuses=(500,))

    assert (status, headers["content-type"], body) == (500, "text/html; charset=utf-8", NOT_FOUND_PAGE)
    with pytest.raises(SourceUnavailable):
        await client.get("civarchive", CIVARCHIVE_URL)


async def test_accept_statuses_leaves_the_other_errors_and_the_size_cap_alone():
    not_found = SourceHttp(fetch=_answer(404, b"<html>404</html>"), limits={"civarchive": (0.0, 1)})
    too_large = SourceHttp(fetch=_answer(500, b"x" * 2048), limits={"civarchive": (0.0, 1)}, max_body_bytes=1024)

    with pytest.raises(SourceNotFound):
        await not_found.get("civarchive", CIVARCHIVE_URL, accept_statuses=(500,))
    with pytest.raises(SourceAnswerTooLarge):
        await too_large.get("civarchive", CIVARCHIVE_URL, accept_statuses=(500,))


def test_civarchive_waits_a_second_between_requests_one_at_a_time():
    assert http_module.DEFAULT_LIMITS["civarchive"] == (1.0, 1)
    assert http_module.source_label("civarchive") == "CivArchive"


async def test_civarchive_requests_never_overlap():
    in_flight = 0
    peak = 0
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds

    async def fetch(url, params):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        return 200, {}, b"{}"

    client = SourceHttp(fetch=fetch, clock=lambda: now[0], sleep=sleep)
    await asyncio.gather(*(client.get_json("civarchive", CIVARCHIVE_URL) for _ in range(4)))

    assert peak == 1
    assert now[0] == pytest.approx(3.0)
