import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlsplit

from .civitai_fakes import ABSENT, MemoryCache

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "huggingface"
SITE = "https://huggingface.co"

REPOS = {
    "lcm_lora_sdxl": "latent-consistency/lcm-lora-sdxl",
    "flux_dev_gated": "black-forest-labs/FLUX.1-dev",
    "sd35_large": "stabilityai/stable-diffusion-3.5-large",
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "real_esrgan": "ai-forever/Real-ESRGAN",
}

Response = Union[str, Tuple[int, Dict[str, str], Any], Callable[[str, List[Tuple[str, str]]], Any]]


def load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def body(name: str) -> Any:
    return copy.deepcopy(load(name)["body"])


def item(name: str, repo: str) -> Dict[str, Any]:
    for raw in body(name):
        if raw["id"] == repo:
            return raw
    raise KeyError(repo)


def link_header(name: str) -> Optional[str]:
    return load(name)["_headers"].get("Link")


def cursor_of(name: str) -> str:
    link = link_header(name)
    url = link[link.index("<") + 1:link.index(">")]
    return unquote(url.split("cursor=", 1)[1])


def encode(headers: Dict[str, str], payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    content_type = next((value for key, value in (headers or {}).items() if key.lower() == "content-type"), "")
    if isinstance(payload, str) and not content_type.startswith("application/json"):
        return payload.encode("utf-8")
    return json.dumps(payload).encode("utf-8")


class HuggingFaceTransport:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, List[Tuple[str, str]]]] = []
        self._routes: List[Tuple[str, Dict[str, Any], Response]] = []

    def add(self, path: str, response: Response, **params: Any) -> "HuggingFaceTransport":
        self._routes.append((path, params, response))
        return self

    def add_repo(self, key: str, *, readme: Optional[Response] = None) -> str:
        repo = REPOS[key]
        self.add(f"/api/models/{repo}", f"model_{key}")
        self.add(f"/api/models/{repo}/tree/main", f"tree_{key}")
        if readme is None and (FIXTURES / f"readme_{key}.json").exists():
            readme = f"readme_{key}"
        if readme is not None:
            self.add(f"/{repo}/raw/main/README.md", readme)
        return repo

    def paths(self) -> List[str]:
        return [path for path, _ in self.calls]

    def calls_to(self, path: str) -> List[Dict[str, Any]]:
        return [_as_dict(params) for called, params in self.calls if called == path]

    async def __call__(self, url: str, params):
        parts = urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}" == SITE, url
        assert not parts.query, url
        path = parts.path
        params = [(str(name), str(value)) for name, value in params]
        self.calls.append((path, params))
        best = None
        for route_path, required, response in self._routes:
            if route_path != path or not _matches(required, params):
                continue
            if best is None or len(required) >= len(best[0]):
                best = (required, response)
        if best is None:
            return 404, {"Content-Type": "application/json"}, b'{"error": "Repository not found"}'
        response = best[1]
        if callable(response):
            response = response(path, params)
        if isinstance(response, str):
            fixture = load(response)
            headers = dict(fixture["_headers"])
            return fixture["_status"], headers, encode(headers, fixture["body"])
        status, headers, payload = response
        headers = dict(headers or {})
        if isinstance(payload, (dict, list)):
            headers.setdefault("Content-Type", "application/json; charset=utf-8")
        return status, headers, encode(headers, payload)


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


def make_source(transport: HuggingFaceTransport, cache: Any = None):
    from backend.http import SourceHttp
    from backend.sources.huggingface import HuggingFaceSource

    client = SourceHttp(fetch=transport, limits={"huggingface": (0.0, 4)})
    return HuggingFaceSource(http_client=client, cache_store=cache if cache is not None else MemoryCache())


OPENMODELDB_SITE = "https://openmodeldb.info"
CIVARCHIVE_SITE = "https://civarchive.com"


def combined_fetch(civitai, huggingface, openmodeldb=None, civarchive=None):
    async def fetch(url: str, params):
        if url.startswith(SITE + "/"):
            return await huggingface(url, params)
        if url.startswith(OPENMODELDB_SITE + "/"):
            if openmodeldb is None:
                return 503, {"Content-Type": "text/plain"}, b"OpenModelDB is not faked in this test"
            return await openmodeldb(url, params)
        if url.startswith(CIVARCHIVE_SITE + "/"):
            if civarchive is None:
                return 503, {"Content-Type": "text/plain"}, b"CivArchive is not faked in this test"
            return await civarchive(url, params)
        return await civitai(url, params)

    return fetch
