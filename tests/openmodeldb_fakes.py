import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from .civitai_fakes import MemoryCache

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openmodeldb"
SITE = "https://openmodeldb.info"
MODELS = "/api/v1/models.json"
TAGS = "/api/v1/tags.json"
ARCHITECTURES = "/api/v1/architectures.json"
HEADERS = {"Content-Type": "application/json", "Cache-Control": "max-age=600"}

ORACLE = "16x-ESRGAN"
GITHUB_RELEASE = "1x-Archiver-AntiLines"
GITHUB_RAW = "4x-Nomos8kSC"
RAW_GITHUB = "2x-LiveActionV1-SPAN"
HUGGING_FACE = "4x-Remacri"
BACKBLAZE = "1x-DeJpeg-Fatality-PlusULTRA"
SMALL_ESRGAN = "1x-Debandurh-FS-Ultra-lite"
GOOGLE_DRIVE_ONLY = "3x-OmniSR-DF2K"
MEGA_ONLY = "4x-UltraSharp"
MIXED = "4x-UltraSharpV2"
ONNX_FIRST = "4x-Nomos8kSCHAT-S"
ONNX_ONLY = "4x-SPANkendata"
ONNX_AND_DRIVE = "4x-Nomos8kSCSRFormer"
THREE_FORMATS = "1x-DeNoise-realplksr-otf"
ZIP_ONLY = "2x-AnimeJaNai-Standard-v1-SuperUltraCompact"
PTH_TAR = "4x-Swift-SRGAN"
HTTP_ONLY = "4x-AnalogFrames-1-0"
NO_LICENSE = "1x-KDM003-scans"
VIDEO = "2x-cainREALTIME"
TWO_AUTHORS = "2x-LD-Anime-Compact"
ONE_NAME_LIST = "8x-classicalSR-DIV2K-s64w8-SwinIR-M"
MISSPELT_KEYS = "1x-BW-Denoise"
STANDALONE_IMAGES = "1x-BroadcastToStudio-SPAN"
PAIR_WITHOUT_SIZES = "2x-LiveActionV1-SPAN"
INLINE_HTML = "1x-Book-Compact"
GRAYSCALE = "1x-MangaJPEGLQ"

Response = Union[str, BaseException, Tuple[int, Dict[str, str], Any], Callable[[str, List[Tuple[str, str]]], Any]]


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def models() -> Dict[str, Any]:
    return load("models")


def model(model_id: str) -> Dict[str, Any]:
    return copy.deepcopy(models()[model_id])


def catalog(models_body: Optional[Dict[str, Any]] = None, *, tags: bool = True, architectures: bool = True):
    from backend.sources.openmodeldb import build_catalog

    return build_catalog(
        models() if models_body is None else models_body,
        load("tags") if tags else None,
        load("architectures") if architectures else None,
        fetched_at=0.0,
    )


class OpenModelDBTransport:
    def __init__(self) -> None:
        self.calls: List[str] = []
        self.responses: Dict[str, Response] = {MODELS: "models", TAGS: "tags", ARCHITECTURES: "architectures"}

    def count(self, path: str) -> int:
        return self.calls.count(path)

    async def __call__(self, url: str, params):
        parts = urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}" == SITE, url
        assert not parts.query and not list(params), url
        self.calls.append(parts.path)
        response = self.responses.get(parts.path)
        if response is None:
            return 404, {"Content-Type": "text/html"}, b"<!doctype html><title>Not found</title>"
        if callable(response):
            response = response(parts.path, [(str(name), str(value)) for name, value in params])
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return 200, dict(HEADERS), (FIXTURES / f"{response}.json").read_bytes()
        status, headers, payload = response
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return status, dict(headers or {}), data


def make_source(transport: OpenModelDBTransport, cache: Any = None, clock: Optional[Callable[[], float]] = None):
    from backend.http import SourceHttp
    from backend.sources.openmodeldb import OpenModelDBSource

    client = SourceHttp(fetch=transport, limits={"openmodeldb": (0.0, 2)})
    store = cache if cache is not None else MemoryCache()
    if clock is None:
        return OpenModelDBSource(http_client=client, cache_store=store)
    return OpenModelDBSource(http_client=client, cache_store=store, clock=clock)
