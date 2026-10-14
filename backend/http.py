import asyncio
import email.utils
import json
import logging
import math
import ssl
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiohttp

from . import PLUGIN_VERSION

logger = logging.getLogger("plugins.model-browser")

USER_AGENT = f"PotionUI-ModelBrowser/{PLUGIN_VERSION} (+https://github.com/PotionUI/PotionUI)"
TIMEOUT_SECONDS = 30.0
MAX_BODY_BYTES = 32 * 1024 * 1024

DEFAULT_LIMITS: Dict[str, Tuple[float, int]] = {
    "civitai": (0.25, 4),
    "huggingface": (0.2, 4),
    "openmodeldb": (0.5, 2),
    "civarchive": (1.0, 1),
}
_FALLBACK_LIMIT = (0.5, 2)

SOURCE_LABELS = {
    "civitai": "CivitAI",
    "huggingface": "Hugging Face",
    "openmodeldb": "OpenModelDB",
    "civarchive": "CivArchive",
}

Params = List[Tuple[str, str]]
ParamsInput = Union[None, Mapping[str, Any], Sequence[Tuple[str, Any]]]
Fetch = Callable[[str, Params], Awaitable[Tuple[int, Mapping[str, str], bytes]]]


class SourceError(Exception):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class SourceInvalidRequest(SourceError):
    pass


class SourceNotFound(SourceError):
    pass


class SourceRateLimited(SourceError):
    def __init__(self, message: str, retry_after: Optional[float] = None, status: Optional[int] = 429) -> None:
        super().__init__(message, status)
        self.retry_after = retry_after


class SourceUnavailable(SourceError):
    pass


class SourceAnswerTooLarge(SourceUnavailable):
    pass


def source_label(source_id: str) -> str:
    return SOURCE_LABELS.get(source_id, source_id)


def parse_retry_after(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def upstream_message(body: bytes, limit: int = 300) -> Optional[str]:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    message: Optional[str] = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, str):
            message = error
        elif isinstance(error, dict):
            message = _zod_message(error.get("message")) or (
                error.get("message") if isinstance(error.get("message"), str) else None
            )
        if not message and isinstance(data.get("message"), str):
            message = data["message"]
    if not message:
        return None
    message = " ".join(message.split())
    return message if len(message) <= limit else message[: limit - 1].rstrip() + "…"


def _zod_message(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    try:
        issues = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(issues, list):
        return None
    parts = []
    for issue in issues:
        if not isinstance(issue, dict) or not isinstance(issue.get("message"), str):
            continue
        path = ".".join(str(part) for part in issue.get("path") or [])
        parts.append(f"{path}: {issue['message']}" if path else issue["message"])
    return "; ".join(parts) or None


def normalise_params(params: ParamsInput) -> Params:
    if not params:
        return []
    items = params.items() if isinstance(params, Mapping) else params
    pairs: Params = []
    for name, value in items:
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is None:
                continue
            if isinstance(item, bool):
                item = "true" if item else "false"
            pairs.append((str(name), str(item)))
    return pairs


def decode_json(source_id: str, status: int, body: bytes) -> Any:
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        raise SourceUnavailable(
            f"{source_label(source_id)} sent an answer that is not JSON (HTTP {status})", status
        ) from None


def decode_text(headers: Mapping[str, str], body: bytes) -> str:
    charset = "utf-8"
    content_type = next((str(value) for name, value in (headers or {}).items() if str(name).lower() == "content-type"), "")
    for part in content_type.split(";")[1:]:
        name, _, value = part.strip().partition("=")
        if name.strip().lower() == "charset" and value.strip().strip('"'):
            charset = value.strip().strip('"')
    try:
        text = (body or b"").decode(charset, errors="replace")
    except LookupError:
        text = (body or b"").decode("utf-8", errors="replace")
    return text[1:] if text.startswith("\N{ZERO WIDTH NO-BREAK SPACE}") else text


class _Limiter:
    def __init__(self, min_interval: float, concurrency: int, clock: Callable[[], float], sleep) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._clock = clock
        self._sleep = sleep
        self._next_start: Optional[float] = None

    async def __aenter__(self) -> "_Limiter":
        await self._semaphore.acquire()
        try:
            now = self._clock()
            start = now if self._next_start is None else max(now, self._next_start)
            self._next_start = start + self.min_interval
            if start > now:
                await self._sleep(start - now)
        except BaseException:
            self._semaphore.release()
            raise
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._semaphore.release()


class SourceHttp:
    def __init__(
        self,
        fetch: Optional[Fetch] = None,
        *,
        limits: Optional[Mapping[str, Tuple[float, int]]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_body_bytes: int = MAX_BODY_BYTES,
    ) -> None:
        self._fetch = fetch
        self._limits = dict(DEFAULT_LIMITS)
        self._limits.update(limits or {})
        self._clock = clock
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._max_body_bytes = max_body_bytes
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._limiters: Dict[str, _Limiter] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None
        self._closing: Set["asyncio.Future[Any]"] = set()

    async def get(
        self,
        source_id: str,
        url: str,
        params: ParamsInput = None,
        *,
        max_bytes: Optional[int] = None,
        accept_statuses: Collection[int] = (),
    ) -> Tuple[int, Dict[str, str], bytes]:
        pairs = normalise_params(params)
        label = source_label(source_id)
        cap = self._max_body_bytes if max_bytes is None else max(0, min(int(max_bytes), self._max_body_bytes))
        started = time.monotonic()
        async with self._limiter(source_id):
            try:
                if self._fetch is not None:
                    status, headers, body = await self._fetch(url, pairs)
                else:
                    status, headers, body = await self._aiohttp_fetch(url, pairs, max_bytes=cap)
            except SourceError:
                raise
            except asyncio.TimeoutError:
                raise SourceUnavailable(f"{label} did not answer within {self._timeout_seconds:g} s") from None
            except (aiohttp.ClientConnectorCertificateError, ssl.SSLCertVerificationError) as exc:
                raise SourceUnavailable(
                    f"Could not verify {label}'s TLS certificate; check the system certificate store "
                    f"or SSL_CERT_FILE ({exc})"
                ) from None
            except (aiohttp.ClientError, ssl.SSLError, OSError) as exc:
                raise SourceUnavailable(f"Could not reach {label}: {exc.__class__.__name__}: {exc}") from None
        headers = {str(name).lower(): str(value) for name, value in (headers or {}).items()}
        body = body or b""
        logger.debug(
            "[MODEL-BROWSER] GET %s %s -> %s (%d bytes, %.0f ms)",
            url, pairs, status, len(body), (time.monotonic() - started) * 1000,
        )
        if len(body) > cap:
            raise SourceAnswerTooLarge(f"{label} sent an answer larger than {cap} bytes", status)
        if status not in accept_statuses:
            self._raise_for_status(source_id, status, headers, body)
        return status, headers, body

    async def get_json(self, source_id: str, url: str, params: ParamsInput = None) -> Any:
        status, _, body = await self.get(source_id, url, params)
        return decode_json(source_id, status, body)

    async def get_json_with_headers(
        self, source_id: str, url: str, params: ParamsInput = None
    ) -> Tuple[Any, Dict[str, str]]:
        status, headers, body = await self.get(source_id, url, params)
        return decode_json(source_id, status, body), headers

    async def get_text(
        self, source_id: str, url: str, params: ParamsInput = None, *, max_bytes: Optional[int] = None
    ) -> str:
        _, headers, body = await self.get(source_id, url, params, max_bytes=max_bytes)
        return decode_text(headers, body)

    @staticmethod
    def _raise_for_status(source_id: str, status: int, headers: Mapping[str, str], body: bytes) -> None:
        if 200 <= status < 300:
            return
        label = source_label(source_id)
        message = upstream_message(body)
        if status == 404:
            raise SourceNotFound(message or f"Not found on {label}", status)
        if status == 429:
            retry_after = parse_retry_after(headers.get("retry-after"))
            logger.warning("[MODEL-BROWSER] %s rate limit reached (retry after %s s)", label, retry_after)
            raise SourceRateLimited(f"{label} rate limit reached", retry_after, status)
        if 400 <= status < 500:
            raise SourceInvalidRequest(
                f"{label} refused the request: {message}" if message else f"{label} refused the request (HTTP {status})",
                status,
            )
        logger.warning("[MODEL-BROWSER] %s answered HTTP %s", label, status)
        raise SourceUnavailable(f"{label} answered HTTP {status}", status)

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            if self._session is not None and self._session_loop is not loop:
                self.close_soon()
            self._loop = loop
            self._limiters = {}
        return loop

    def _limiter(self, source_id: str) -> _Limiter:
        self._bind_loop()
        limiter = self._limiters.get(source_id)
        if limiter is None:
            interval, concurrency = self._limits.get(source_id, _FALLBACK_LIMIT)
            limiter = _Limiter(interval, concurrency, self._clock, self._sleep or asyncio.sleep)
            self._limiters[source_id] = limiter
        return limiter

    def _get_session(self) -> aiohttp.ClientSession:
        loop = self._bind_loop()
        if self._session is None or self._session.closed:
            from src.plugin_api.providers import aiohttp_connector

            self._session = aiohttp.ClientSession(
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                connector=aiohttp_connector(),
                timeout=aiohttp.ClientTimeout(total=self._timeout_seconds),
            )
            self._session_loop = loop
        return self._session

    async def _aiohttp_fetch(
        self, url: str, params: Params, max_bytes: Optional[int] = None
    ) -> Tuple[int, Mapping[str, str], bytes]:
        cap = self._max_body_bytes if max_bytes is None else max_bytes
        session = self._get_session()
        async with session.get(url, params=params) as response:
            too_large = SourceAnswerTooLarge(
                f"The answer from {response.url.host} is larger than {cap} bytes", response.status
            )
            length = response.content_length
            if length is not None and length > cap:
                raise too_large
            chunks: List[bytes] = []
            size = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > cap:
                    raise too_large
                chunks.append(chunk)
            return response.status, response.headers, b"".join(chunks)

    async def close(self) -> None:
        session, self._session = self._session, None
        self._session_loop = None
        if session is not None and not session.closed:
            await session.close()

    def close_soon(self) -> None:
        session, loop = self._session, self._session_loop
        self._session = None
        self._session_loop = None
        if session is None or session.closed or loop is None or loop.is_closed():
            return

        def _close() -> None:
            task = loop.create_task(session.close())
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)

        try:
            loop.call_soon_threadsafe(_close)
        except RuntimeError:
            pass


http = SourceHttp()


def shutdown() -> None:
    http.close_soon()
