import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from .civitai_fakes import ABSENT, MemoryCache

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "civarchive"
SITE = "https://civarchive.com"
DATA_PREFIX = "/_next/data/"
CAPTURED_BUILD_ID = "ga7UfXN2zTSyfNyGSR_kL"
JSON_HEADERS = {"Content-Type": "application/json", "Cache-Control": "private, no-cache, no-store, max-age=0, must-revalidate"}
HTML_HEADERS = {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "private, no-cache, no-store, max-age=0, must-revalidate"}

SHA_78291 = "e47d1de297b8ec749aab188969172e79898028e8ad5e9333e87d7d975087269a"
SHA_UNKNOWN = "5f1c3a9e7b2d4c6a8e0f1b3d5c7e9a1b2c4d6e8f0a1b3c5d7e9f1a2b3c4d5e6f"

DEFAULT_SEARCH = {"platform": "civitai", "kind": "version", "rating": "all", "sort": "top", "period": "all", "page": "1"}
NO_FILTERS = {name: ABSENT for name in ("q", "tags", "username", "type", "base_model", "platform_status")}

Response = Union[str, BaseException, Tuple[int, Dict[str, str], Any], Callable[[str, List[Tuple[str, str]]], Any]]

STATUSES = {"model_unknown_500": 500, "model_78291_unknown_version_500": 500, "build_id_404": 404}


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def props(name: str) -> Dict[str, Any]:
    return load(name)["pageProps"]


def model(name: str) -> Dict[str, Any]:
    return props(name)["model"]


def results(name: str) -> List[Dict[str, Any]]:
    return props(name)["data"]["results"]


def html(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def data_body(page_props: Dict[str, Any]) -> Dict[str, Any]:
    return {"pageProps": page_props, "__N_SSP": True}


def search_body(entries: List[Dict[str, Any]], total_hits: Optional[int] = 1000, **extra: Any) -> Dict[str, Any]:
    page_props = props("search_civitai_top")
    page_props["data"] = {"results": entries} if total_hits is None else {"results": entries, "total_hits": total_hits}
    page_props.update(extra)
    return data_body(page_props)


def full_page(entries: List[Dict[str, Any]], size: int = 50, first_model: int = 9_000_000) -> List[Dict[str, Any]]:
    page = []
    for index in range(size):
        entry = copy.deepcopy(entries[index % len(entries)])
        entry["model_id"] = str(first_model + index)
        entry["version_id"] = str(first_model + 500_000 + index)
        page.append(entry)
    return page


class CivArchiveTransport:
    def __init__(self, build_id: Optional[str] = CAPTURED_BUILD_ID) -> None:
        self.build_id = build_id
        self.calls: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.build_ids: List[str] = []
        self._routes: List[Tuple[str, Dict[str, Any], Response]] = []
        self.home: Response = "index"
        self._defaults()

    def add(self, path: str, response: Response, **params: Any) -> "CivArchiveTransport":
        self._routes.append((path, params, response))
        return self

    def _defaults(self) -> None:
        search = dict(DEFAULT_SEARCH, **NO_FILTERS)
        self.add("search.json", "search_civitai_top", **search)
        self.add("search.json", "search_civitai_page2", **dict(search, page="2"))
        self.add("search.json", "search_safe", **dict(search, rating="safe"))
        self.add(
            "search.json",
            "search_civitai_lora_sd15_newest",
            **dict(DEFAULT_SEARCH, sort="newest", type="LORA", base_model="SD 1.5", q=ABSENT, tags=ABSENT, username=ABSENT),
        )
        self.add("search.json", "search_tag", **dict(DEFAULT_SEARCH, tags="anime", q=ABSENT, username=ABSENT, type=ABSENT))
        self.add("search.json", "search_user", **dict(DEFAULT_SEARCH, username="TangBohu", q=ABSENT, tags=ABSENT, type=ABSENT))
        self.add("search.json", "search_empty", **dict(DEFAULT_SEARCH, q="zzqxjv-no-such-model-0"))
        self.add("search.json", "search_filters", platform="civitai", kind=ABSENT, page=ABSENT)
        self.add("models/78291.json", "model_78291_redirect", platform="civitai", modelVersionId=ABSENT)
        for model_id, version_id in ((78291, 83095), (122359, 135867), (298188, 334947), (195519, 225222), (48139, 52753), (1879, 2026)):
            self.add(f"models/{model_id}.json", f"model_{model_id}_v{version_id}", platform="civitai", modelVersionId=version_id)
        self.add("models/78291.json", "model_78291_v1", platform="civitai", modelVersionId=1)
        self.add("models/78291.json", "model_78291_unknown_version_500", platform="civitai", modelVersionId=999999999)
        self.add("models/999999999.json", "model_unknown_500")
        self.add(f"sha256/{SHA_78291}.json", "sha256_e47d")
        self.add(f"sha256/{SHA_UNKNOWN}.json", "sha256_unknown")

    def clear(self, path: str) -> "CivArchiveTransport":
        self._routes = [route for route in self._routes if route[0] != path]
        return self

    def redeploy(self, build_id: str) -> None:
        self.build_id = build_id

    @property
    def home_fetches(self) -> int:
        return sum(1 for path, _ in self.calls if path == "/")

    def data_calls(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        return [_as_dict(params) for called, params in self.calls if called != "/" and (path is None or called == path)]

    def paths(self) -> List[str]:
        return [path for path, _ in self.calls]

    async def __call__(self, url: str, params):
        parts = urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}" == SITE, url
        assert not parts.query, url
        params = [(str(name), str(value)) for name, value in params]
        if parts.path == "/":
            self.calls.append(("/", params))
            await asyncio.sleep(0)
            return self._answer("/", params, self.home)
        assert parts.path.startswith(DATA_PREFIX), url
        build_id, _, path = parts.path[len(DATA_PREFIX):].partition("/")
        self.calls.append((path, params))
        self.build_ids.append(build_id)
        await asyncio.sleep(0)
        if build_id != self.build_id:
            return 404, dict(HTML_HEADERS), html("build_id_404").encode("utf-8")
        best = None
        for route_path, required, response in self._routes:
            if route_path != path or not _matches(required, params):
                continue
            if best is None or len(required) >= len(best[0]):
                best = (required, response)
        if best is None:
            return 404, dict(HTML_HEADERS), html("build_id_404").encode("utf-8")
        return self._answer(path, params, best[1])

    def _answer(self, path: str, params: List[Tuple[str, str]], response: Response):
        if callable(response) and not isinstance(response, BaseException):
            response = response(path, params)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return self._fixture(response)
        status, headers, payload = response
        if isinstance(payload, (dict, list)):
            return status, dict(headers or JSON_HEADERS), json.dumps(payload).encode("utf-8")
        data = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        return status, dict(headers or HTML_HEADERS), data

    def _fixture(self, name: str):
        status = STATUSES.get(name, 200)
        if (FIXTURES / f"{name}.json").exists():
            return status, dict(JSON_HEADERS), (FIXTURES / f"{name}.json").read_bytes()
        page = html(name)
        if name == "index":
            if self.build_id is None:
                page = page.replace(f'"buildId":"{CAPTURED_BUILD_ID}",', "")
            else:
                page = page.replace(f'"buildId":"{CAPTURED_BUILD_ID}"', f'"buildId":"{self.build_id}"')
        return status, dict(HTML_HEADERS), page.encode("utf-8")


def _matches(required: Dict[str, Any], params: List[Tuple[str, str]]) -> bool:
    names = {name for name, _ in params}
    for name, value in required.items():
        if value is ABSENT:
            if name in names:
                return False
        elif (name, str(value)) not in params:
            return False
    return True


def _as_dict(params: List[Tuple[str, str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, value in params:
        if name in result:
            existing = result[name]
            result[name] = existing + [value] if isinstance(existing, list) else [existing, value]
        else:
            result[name] = value
    return result


def make_source(transport: CivArchiveTransport, cache: Any = None, clock: Optional[Callable[[], float]] = None):
    from backend.http import SourceHttp
    from backend.sources.civarchive import CivArchiveSource

    client = SourceHttp(fetch=transport, limits={"civarchive": (0.0, 4)})
    store = cache if cache is not None else MemoryCache()
    if clock is None:
        return CivArchiveSource(http_client=client, cache_store=store)
    return CivArchiveSource(http_client=client, cache_store=store, clock=clock)
