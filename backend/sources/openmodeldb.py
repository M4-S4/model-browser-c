import asyncio
import hashlib
import html
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from .. import cache as cache_module
from .. import http as http_module
from ..http import SourceError, SourceInvalidRequest, SourceNotFound, SourceRateLimited, SourceUnavailable
from ..schema import (
    File,
    FilterOption,
    Image,
    License,
    ModelDetail,
    SearchItem,
    SearchPage,
    SourceInfo,
    Stats,
    Version,
)
from ..security import is_allowed_download_url, sanitize_html
from .base import SearchQuery
from .huggingface import MARKDOWN_EXTRAS

logger = logging.getLogger("plugins.model-browser")

SOURCE_ID = "openmodeldb"
LABEL = "OpenModelDB"
SITE = "https://openmodeldb.info"
API_BASE = SITE + "/api/v1"
MODELS_URL = API_BASE + "/models.json"
TAGS_URL = API_BASE + "/tags.json"
ARCHITECTURES_URL = API_BASE + "/architectures.json"

CATALOG_TTL_SECONDS = 6 * 3600
RETRY_SECONDS = 10 * 60
FAILURE_HOLD_SECONDS = 30

MB_TYPE = "upscaler"
IMAGE_LEVEL = 1
VERSION_NAME = "Release"
MAX_IMAGES = 12
DEAD_IMAGE_HOSTS: Tuple[str, ...] = ("imgsli.com", "github.com", "private-user-images.githubusercontent.com")
MAX_MARKDOWN_CHARS = 32 * 1024

SORTS: Tuple[Tuple[str, str], ...] = (
    ("downloadable", "Downloadable first"),
    ("newest", "Newest"),
    ("oldest", "Oldest"),
    ("name", "Name"),
)
DEFAULT_SORT = "downloadable"
_SORT_KEYS = frozenset(key for key, _ in SORTS)

SEARCH_HELP = "Text, #tag (anime, photo, restoration, …), @author, or an openmodeldb.info model link"

_FORMATS = {
    "safetensors": "SafeTensor",
    "pth": "PickleTensor",
    "pt": "PickleTensor",
    "ckpt": "PickleTensor",
    "bin": "PickleTensor",
}
_PLATFORMS = {"pytorch": "PyTorch"}
_PYTORCH = "pytorch"
_ONNX = "onnx"
_LOADABLE_ARCHITECTURES = frozenset({"esrgan", "esrgan+"})

_FILE_HOSTS = frozenset({"objects.githubusercontent.com", "raw.githubusercontent.com"})
_FILE_HOST_SUFFIXES = (".oraclecloud.com", ".backblazeb2.com")

_HOST_LABELS: Tuple[Tuple[str, str], ...] = (
    ("drive.google.com", "Google Drive"),
    ("docs.google.com", "Google Drive"),
    ("mega.nz", "Mega"),
    ("mega.co.nz", "Mega"),
    ("1drv.ms", "OneDrive"),
    ("onedrive.live.com", "OneDrive"),
    ("mediafire.com", "MediaFire"),
    ("pcloud.link", "pCloud"),
    ("pcloud.com", "pCloud"),
    ("drive.proton.me", "Proton Drive"),
    ("icedrive.net", "Icedrive"),
    ("1fichier.com", "1fichier"),
    ("dropbox.com", "Dropbox"),
    ("zippyshare.com", "Zippyshare"),
    ("catbox.moe", "Catbox"),
    ("lbry.tv", "LBRY"),
    ("github.com", "GitHub"),
    ("githubusercontent.com", "GitHub"),
    ("huggingface.co", "Hugging Face"),
    ("oraclecloud.com", "Oracle Cloud"),
    ("backblazeb2.com", "Backblaze B2"),
)

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_CURSOR = re.compile(r"[0-9]{1,9}")
_SCALE = re.compile(r"([0-9]{1,3})x?", re.I)
_EXTENSION = re.compile(r"[a-z0-9]{1,16}")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_ODD_URL_CHARACTERS = frozenset("\\\"'<>`")


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _positive(value: Any) -> Optional[int]:
    number = _int(value)
    return number if number is not None and number > 0 else None


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _strings(values: Any) -> List[str]:
    return list(dict.fromkeys(value.strip() for value in _list(values) if isinstance(value, str) and value.strip()))


def _sha256(value: Any) -> Optional[str]:
    return value.lower() if isinstance(value, str) and _SHA256.fullmatch(value) else None


def is_model_id(value: Any) -> bool:
    return isinstance(value, str) and ".." not in value and bool(_MODEL_ID.fullmatch(value))


def model_page_url(model_id: str) -> str:
    return f"{SITE}/models/{model_id}"


def authors(raw: Mapping[str, Any]) -> List[str]:
    value = raw.get("author")
    return _strings(value) if isinstance(value, list) else ([value.strip()] if _text(value) else [])


def scale_of(raw: Mapping[str, Any]) -> Optional[int]:
    return _positive(raw.get("scale"))


def date_of(raw: Mapping[str, Any]) -> Optional[str]:
    value = _text(raw.get("date"))
    return value if value is not None and _DATE.fullmatch(value) else None


def resources(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [resource for resource in _list(raw.get("resources")) if isinstance(resource, dict)]


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def is_direct_url(url: Any) -> bool:
    if not isinstance(url, str) or not is_allowed_download_url(SOURCE_ID, url):
        return False
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host == "huggingface.co":
        return "/resolve/" in parts.path
    if host == "github.com":
        segments = parts.path.split("/")[1:]
        if len(segments) >= 6 and segments[2:4] == ["releases", "download"]:
            return all(segments[:2]) and bool(segments[-1])
        if len(segments) >= 5 and segments[2] == "raw":
            return all(segments[:2]) and bool(segments[-1])
        return False
    return host in _FILE_HOSTS or host.endswith(_FILE_HOST_SUFFIXES)


def _extension(kind: Any) -> str:
    kind = kind.strip().lower() if isinstance(kind, str) else ""
    return f".{kind}" if _EXTENSION.fullmatch(kind) else ""


def direct_url(resource: Mapping[str, Any]) -> Optional[str]:
    extension = _extension(resource.get("type"))
    for url in _strings(resource.get("urls")):
        if not is_direct_url(url):
            continue
        name = url_file_name(url)
        if extension and not (name and name.lower().endswith(extension)):
            continue
        return url
    return None


def is_onnx(resource: Mapping[str, Any]) -> bool:
    return _ONNX in ((_text(resource.get("platform")) or "").lower(), (_text(resource.get("type")) or "").lower())


def file_url(resource: Mapping[str, Any]) -> Optional[str]:
    return None if is_onnx(resource) else direct_url(resource)


def host_label(host: str) -> str:
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    for domain, label in _HOST_LABELS:
        if host == domain or host.endswith("." + domain):
            return label
    return host


def other_downloads(raw: Mapping[str, Any]) -> List[str]:
    labels: Dict[str, None] = {}
    for resource in resources(raw):
        urls = _strings(resource.get("urls"))
        if not urls or file_url(resource) is not None:
            continue
        if is_onnx(resource):
            labels.setdefault("ONNX", None)
            continue
        for url in urls:
            host = _host(url)
            if host:
                labels.setdefault(host_label(host), None)
    return list(labels)


def url_file_name(url: str) -> Optional[str]:
    try:
        segment = urlsplit(url).path.rsplit("/", 1)[-1]
    except ValueError:
        return None
    name = unquote(segment).strip()
    if not name or name in (".", "..") or any(char in "/\\:" for char in name):
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in name):
        return None
    return name


def file_name(model_id: str, url: str, kind: Optional[str]) -> str:
    extension = _extension(kind)
    name = url_file_name(url)
    if name and (not extension or name.lower().endswith(extension)):
        return name
    return f"{model_id}{extension}"


def model_files(model_id: str, raw: Mapping[str, Any]) -> List[File]:
    found: List[Tuple[Dict[str, Any], str, str]] = []
    seen = set()
    for index, resource in enumerate(resources(raw)):
        url = file_url(resource)
        if url is None:
            continue
        sha256 = _sha256(resource.get("sha256"))
        file_id = sha256 or str(index)
        if file_id in seen:
            continue
        seen.add(file_id)
        found.append((resource, url, file_id))
    platforms = [(_text(resource.get("platform")) or "").lower() for resource, _, _ in found]
    primary = next((index for index, platform in enumerate(platforms) if platform == _PYTORCH), 0)
    files: List[File] = []
    for index, (resource, url, file_id) in enumerate(found):
        kind = _text(resource.get("type"))
        platform = _text(resource.get("platform"))
        size = _int(resource.get("size"))
        files.append(
            File(
                id=file_id,
                name=file_name(model_id, url, kind),
                size_bytes=size if size is not None and size >= 0 else None,
                sha256=_sha256(resource.get("sha256")),
                format=_FORMATS.get(kind.lower()) if kind else None,
                precision=None,
                file_type=_PLATFORMS.get(platform.lower(), platform) if platform else None,
                primary=index == primary,
                download_url=url,
                auth_provider_id=None,
            )
        )
    return files[primary:primary + 1] + files[:primary] + files[primary + 1:]


def https_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or any(
        char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F or char in _ODD_URL_CHARACTERS for char in value
    ):
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() != "https" or not parts.hostname or port not in (None, 443) or "@" in parts.netloc:
        return None
    return value


def site_url(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip().startswith("/") and not value.strip().startswith("//"):
        return https_url(SITE + value.strip())
    return https_url(value)


def preview_url(value: Any) -> Optional[str]:
    url = site_url(value)
    if url is None:
        return None
    host = _host(url)
    if any(host == dead or host.endswith("." + dead) for dead in DEAD_IMAGE_HOSTS):
        return None
    return url


def _image(url: str, width: Optional[int] = None, height: Optional[int] = None) -> Image:
    return Image(
        url=url,
        nsfw=False,
        nsfw_level=IMAGE_LEVEL,
        width=width,
        height=height,
        kind="image",
        id=hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:16],
    )


def _gallery(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [image for image in _list(raw.get("images")) if isinstance(image, dict)]


def _gallery_url(image: Mapping[str, Any]) -> Optional[str]:
    if image.get("type") == "paired":
        return preview_url(image.get("SR"))
    return preview_url(image.get("url")) or preview_url(image.get("SR"))


def model_thumbnail(raw: Mapping[str, Any]) -> Optional[Image]:
    own = raw.get("thumbnail")
    if not isinstance(own, dict):
        return None
    if own.get("type") == "paired":
        url = preview_url(own.get("SR"))
        if url:
            sizes = own.get("SRSize") if isinstance(own.get("SRSize"), dict) else {}
            return _image(url, _positive(sizes.get("width")), _positive(sizes.get("height")))
    url = preview_url(own.get("url")) or preview_url(own.get("SR"))
    return _image(url) if url else None


def version_images(raw: Mapping[str, Any]) -> List[Image]:
    images: List[Image] = []
    seen = set()
    own = model_thumbnail(raw)
    if own is not None:
        images.append(own)
        seen.add(own.url)
    for entry in _gallery(raw):
        if len(images) >= MAX_IMAGES:
            break
        url = _gallery_url(entry)
        if url is None or url in seen:
            continue
        seen.add(url)
        images.append(_image(url))
    return images


def thumbnail(raw: Mapping[str, Any]) -> Optional[Image]:
    own = model_thumbnail(raw)
    if own is not None:
        return own
    for entry in _gallery(raw):
        small = preview_url(entry.get("thumbnail"))
        if small:
            return _image(small)
    return None


_SPDX_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}")
NO_LICENSE = "No license stated"


def license_from(raw: Mapping[str, Any]) -> License:
    value = _text(raw.get("license"))
    if value is None:
        return License(summary=NO_LICENSE)
    url = f"https://spdx.org/licenses/{value}.html" if _SPDX_ID.fullmatch(value) else None
    return License(summary=value, url=url)


@dataclass(frozen=True)
class Entry:
    id: str
    name_key: str
    date: str
    authors: FrozenSet[str]
    tags: FrozenSet[str]
    architecture: Optional[str]
    scale: Optional[int]
    downloadable: bool
    text: str


@dataclass(frozen=True)
class Catalog:
    models: Mapping[str, Mapping[str, Any]]
    by_sha256: Mapping[str, Tuple[str, Mapping[str, Any]]]
    tags: Mapping[str, str]
    architectures: Mapping[str, Mapping[str, Optional[str]]]
    fetched_at: float
    orders: Mapping[str, Tuple[Entry, ...]]
    folded_ids: Mapping[str, str]

    def tag_name(self, key: str) -> str:
        return self.tags.get(key, key)

    def architecture_label(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        known = self.architectures.get(key)
        return (known.get("name") if known else None) or key

    def architecture_input(self, key: Optional[str]) -> Optional[str]:
        known = self.architectures.get(key) if key else None
        return known.get("input") if known else None

    def find(self, model_id: str) -> Optional[str]:
        if model_id in self.models:
            return model_id
        return self.folded_ids.get(model_id.casefold())

    def architecture_options(self) -> List[Tuple[str, str]]:
        keys = {entry.architecture for entry in self.orders["name"] if entry.architecture}
        options = [(key, self.architecture_label(key) or key) for key in keys]
        return sorted(options, key=lambda option: (option[1].casefold(), option[0]))

    def architecture_keys(self) -> List[str]:
        keys = {entry.architecture for entry in self.orders["name"] if entry.architecture}
        return sorted(keys | set(self.architectures))

    def scales(self) -> List[int]:
        return sorted({entry.scale for entry in self.orders["name"] if entry.scale})


def _parse_tags(body: Any) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    if isinstance(body, dict):
        for key, value in body.items():
            if isinstance(key, str) and key.strip():
                tags[key] = (_text(value.get("name")) if isinstance(value, dict) else None) or key
    return tags


def _parse_architectures(body: Any) -> Dict[str, Dict[str, Optional[str]]]:
    architectures: Dict[str, Dict[str, Optional[str]]] = {}
    if isinstance(body, dict):
        for key, value in body.items():
            if isinstance(key, str) and key.strip() and isinstance(value, dict):
                architectures[key] = {"name": _text(value.get("name")) or key, "input": _text(value.get("input"))}
    return architectures


def _entry(model_id: str, raw: Mapping[str, Any], tags: Mapping[str, str], architectures: Mapping[str, Any]) -> Entry:
    name = _text(raw.get("name")) or model_id
    names = authors(raw)
    keys = _strings(raw.get("tags"))
    labels = [tags.get(key, key) for key in keys]
    architecture = _text(raw.get("architecture"))
    known = architectures.get(architecture) if architecture else None
    description = raw.get("description") if isinstance(raw.get("description"), str) else ""
    words = [model_id, name, *names, description, *keys, *labels]
    if architecture:
        words += [architecture, (known or {}).get("name") or ""]
    return Entry(
        id=model_id,
        name_key=name.casefold(),
        date=date_of(raw) or "",
        authors=frozenset(author.casefold() for author in names),
        tags=frozenset(tag.casefold() for tag in keys + labels),
        architecture=architecture,
        scale=scale_of(raw),
        downloadable=any(file_url(resource) is not None for resource in resources(raw)),
        text="\n".join(words).casefold(),
    )


def build_catalog(
    models_body: Any, tags_body: Any = None, architectures_body: Any = None, fetched_at: float = 0.0
) -> Catalog:
    if not isinstance(models_body, dict):
        raise SourceUnavailable("OpenModelDB sent a catalogue this browser cannot read")
    tags = _parse_tags(tags_body)
    architectures = _parse_architectures(architectures_body)
    models: Dict[str, Mapping[str, Any]] = {}
    skipped = 0
    for model_id, raw in models_body.items():
        if is_model_id(model_id) and isinstance(raw, dict):
            models[model_id] = raw
        else:
            skipped += 1
    if skipped:
        logger.warning("[MODEL-BROWSER] OpenModelDB: left out %d catalogue entries this browser cannot read", skipped)

    by_sha256: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
    for model_id, raw in models.items():
        for resource in resources(raw):
            sha256 = _sha256(resource.get("sha256"))
            if sha256 is None:
                continue
            known = by_sha256.get(sha256)
            if known is None or (file_url(known[1]) is None and file_url(resource) is not None):
                by_sha256[sha256] = (model_id, resource)

    entries = [_entry(model_id, raw, tags, architectures) for model_id, raw in models.items()]
    by_name = sorted(entries, key=lambda entry: (entry.name_key, entry.id))
    newest = sorted(by_name, key=lambda entry: entry.date, reverse=True)
    orders = {
        "downloadable": tuple(sorted(newest, key=lambda entry: not entry.downloadable)),
        "newest": tuple(newest),
        "oldest": tuple(sorted(by_name, key=lambda entry: (entry.date == "", entry.date))),
        "name": tuple(by_name),
    }
    folded: Dict[str, str] = {}
    for model_id in models:
        folded.setdefault(model_id.casefold(), model_id)
    return Catalog(
        models=models,
        by_sha256=by_sha256,
        tags=tags,
        architectures=architectures,
        fetched_at=float(fetched_at),
        orders=orders,
        folded_ids=folded,
    )


_RENDER_MEMO_SIZE = 64
_render_memo: "OrderedDict[str, str]" = OrderedDict()
_render_lock = threading.Lock()


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{html.escape(part.strip())}</p>" for part in _PARAGRAPH_BREAK.split(text) if part.strip())


def _render(text: str) -> str:
    if len(text) > MAX_MARKDOWN_CHARS:
        return _paragraphs(text)
    try:
        import markdown2
    except ImportError:
        return _paragraphs(text)
    try:
        return str(markdown2.markdown(text, extras=dict(MARKDOWN_EXTRAS)))
    except Exception:  # noqa: BLE001
        logger.warning("[MODEL-BROWSER] OpenModelDB: could not convert a description; showing it as text", exc_info=True)
        return _paragraphs(text)


def render_markdown(text: str) -> str:
    key = hashlib.md5(text.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()
    with _render_lock:
        hit = _render_memo.get(key)
        if hit is not None:
            _render_memo.move_to_end(key)
            return hit
    markup = _render(text)
    with _render_lock:
        _render_memo[key] = markup
        while len(_render_memo) > _RENDER_MEMO_SIZE:
            _render_memo.popitem(last=False)
    return markup


def facts_line(raw: Mapping[str, Any], catalog: Catalog) -> Optional[str]:
    facts: List[str] = []
    architecture = _text(raw.get("architecture"))
    label = catalog.architecture_label(architecture)
    if label:
        facts.append(f"Architecture: {label}")
    scale = scale_of(raw)
    if scale:
        facts.append(f"Scale: {scale}x")
    kind = catalog.architecture_input(architecture)
    if kind:
        facts.append(f"Input: {kind}")
    channels_in, channels_out = _positive(raw.get("inputChannels")), _positive(raw.get("outputChannels"))
    if channels_in and channels_out:
        facts.append(f"Channels: {channels_in} → {channels_out}")
    dataset = _text(raw.get("dataset"))
    if dataset:
        facts.append(f"Dataset: {' '.join(dataset.split())}")
    iterations, epochs = _positive(raw.get("trainingIterations")), _positive(raw.get("trainingEpochs"))
    training = [f"{iterations:,} iterations" if iterations else "", f"{epochs:,} epochs" if epochs else ""]
    if any(training):
        facts.append("Training: " + ", ".join(part for part in training if part))
    return " · ".join(facts) or None


def compatibility_note(raw: Mapping[str, Any], catalog: Catalog, files: Sequence[File]) -> Optional[str]:
    if not files:
        return None
    architecture = _text(raw.get("architecture"))
    primary = next((file for file in files if file.primary), files[0])
    if (architecture or "").lower() in _LOADABLE_ARCHITECTURES and primary.name.lower().endswith(".pth"):
        return None
    label = catalog.architecture_label(architecture)
    return (
        "Note: PotionUI's built-in upscaler currently loads ESRGAN (.pth) models only; "
        f"this {label + ' ' if label else ''}model will download and index but may not run in PotionUI."
    )


def description_html(
    model_id: str,
    raw: Mapping[str, Any],
    catalog: Catalog,
    *,
    notes: bool = True,
    files: Optional[Sequence[File]] = None,
) -> Optional[str]:
    parts: List[str] = []
    facts = facts_line(raw, catalog)
    if facts:
        parts.append(f"<p>{html.escape(facts)}</p>")
    text = _text(raw.get("description"))
    if text:
        parts.append(render_markdown(text))
    if notes:
        others = other_downloads(raw)
        if others:
            page = html.escape(model_page_url(model_id), quote=True)
            parts.append(
                f'<p>Other downloads on the <a href="{page}">OpenModelDB page</a>: {html.escape(", ".join(others))}</p>'
            )
        note = compatibility_note(raw, catalog, model_files(model_id, raw) if files is None else files)
        if note:
            parts.append(f"<p>{html.escape(note)}</p>")
    return sanitize_html("".join(parts)) or None


def base_model_of(raw: Mapping[str, Any], catalog: Catalog) -> Optional[str]:
    label = catalog.architecture_label(_text(raw.get("architecture")))
    scale = scale_of(raw)
    return " ".join(part for part in (label, f"{scale}x" if scale else None) if part) or None


def tag_names(raw: Mapping[str, Any], catalog: Catalog) -> List[str]:
    return list(dict.fromkeys(catalog.tag_name(key) for key in _strings(raw.get("tags"))))


def normalize_item(model_id: str, raw: Mapping[str, Any], catalog: Catalog) -> SearchItem:
    files = model_files(model_id, raw)
    primary = next((file for file in files if file.primary), None)
    base_model = base_model_of(raw, catalog)
    return SearchItem(
        source=SOURCE_ID,
        id=model_id,
        name=_text(raw.get("name")) or model_id,
        type=catalog.architecture_label(_text(raw.get("architecture"))) or "Upscaler",
        mb_type=MB_TYPE if files else None,
        base_model=base_model,
        base_models=[base_model] if base_model else [],
        creator=", ".join(authors(raw)) or None,
        nsfw=False,
        poi=False,
        early_access=False,
        gated=False,
        thumbnail=thumbnail(raw),
        stats=Stats(),
        source_url=model_page_url(model_id),
        primary_sha256=primary.sha256 if primary is not None else None,
    )


def normalize_detail(model_id: str, raw: Mapping[str, Any], catalog: Catalog) -> ModelDetail:
    files = model_files(model_id, raw)
    version = Version(
        id=model_id,
        name=VERSION_NAME,
        base_model=base_model_of(raw, catalog),
        trigger_words=[],
        created_at=date_of(raw),
        early_access=False,
        nsfw=False,
        images=version_images(raw),
        files=files,
    )
    return ModelDetail(
        source=SOURCE_ID,
        id=model_id,
        name=_text(raw.get("name")) or model_id,
        type=catalog.architecture_label(_text(raw.get("architecture"))) or "Upscaler",
        mb_type=MB_TYPE if files else None,
        creator=", ".join(authors(raw)) or None,
        description_html=description_html(model_id, raw, catalog, files=files),
        tags=tag_names(raw, catalog),
        license=license_from(raw),
        source_url=model_page_url(model_id),
        nsfw=False,
        poi=False,
        gated=False,
        stats=Stats(),
        versions=[version],
    )


@dataclass(frozen=True)
class ParsedQuery:
    kind: str = "all"
    value: str = ""
    words: Tuple[str, ...] = ()


def parse_model_url(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text or any(char.isspace() for char in text):
        return None
    candidate = text if "://" in text else f"https://{text}"
    try:
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    if host.startswith("www."):
        host = host[4:]
    if host != "openmodeldb.info":
        return None
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if len(segments) < 2 or segments[0] != "models" or not is_model_id(segments[1]):
        return None
    return segments[1]


def parse_query(q: Optional[str]) -> ParsedQuery:
    text = (q or "").strip()
    if not text:
        return ParsedQuery()
    model_id = parse_model_url(text)
    if model_id is not None:
        return ParsedQuery(kind="model", value=model_id)
    if text.startswith("@"):
        author = " ".join(text[1:].split())
        return ParsedQuery(kind="author", value=author) if author else ParsedQuery()
    if text.startswith("#"):
        tag = " ".join(text[1:].split())
        return ParsedQuery(kind="tag", value=tag) if tag else ParsedQuery()
    return ParsedQuery(kind="text", value=text, words=tuple(text.casefold().split()))


def parse_cursor(cursor: Optional[str]) -> int:
    if cursor is None or cursor == "":
        return 0
    if not _CURSOR.fullmatch(cursor):
        raise SourceInvalidRequest("Invalid OpenModelDB cursor")
    return int(cursor)


def _clean(values: Iterable[Any]) -> List[str]:
    return sorted({str(value).strip() for value in values or () if str(value).strip()})


def architecture_filter(types: Iterable[str], catalog: Catalog) -> Optional[FrozenSet[str]]:
    chosen = _clean(types)
    if not chosen:
        return None
    known = {key.casefold(): key for key in catalog.architecture_keys()}
    keys = set()
    for value in chosen:
        key = known.get(value.casefold())
        if key is None:
            raise SourceInvalidRequest(
                f"Unknown OpenModelDB architecture {value[:40]!r}; use one of: {', '.join(sorted(known.values()))}"
            )
        keys.add(key)
    return frozenset(keys)


def scale_filter(base_models: Iterable[str]) -> Optional[FrozenSet[int]]:
    chosen = _clean(base_models)
    if not chosen:
        return None
    scales = set()
    for value in chosen:
        match = _SCALE.fullmatch(value)
        if match is None or int(match.group(1)) <= 0:
            raise SourceInvalidRequest(f"Unknown OpenModelDB scale {value[:40]!r}; use a number such as 4")
        scales.add(int(match.group(1)))
    return frozenset(scales)


def matches(
    entry: Entry,
    parsed: ParsedQuery,
    architectures: Optional[FrozenSet[str]] = None,
    scales: Optional[FrozenSet[int]] = None,
) -> bool:
    if architectures is not None and entry.architecture not in architectures:
        return False
    if scales is not None and entry.scale not in scales:
        return False
    if parsed.kind == "author":
        return parsed.value.casefold() in entry.authors
    if parsed.kind == "tag":
        return parsed.value.casefold() in entry.tags
    if parsed.kind == "text":
        return all(word in entry.text for word in parsed.words)
    return True


def _again(error: SourceError) -> SourceError:
    if isinstance(error, SourceRateLimited):
        return SourceRateLimited(error.message, error.retry_after, error.status)
    return type(error)(error.message, error.status)


def _fresh(fetched_at: Any, now: float) -> bool:
    if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)) or not math.isfinite(fetched_at):
        return False
    return 0 <= now - fetched_at < CATALOG_TTL_SECONDS


class OpenModelDBSource:
    id = SOURCE_ID
    label = LABEL

    def __init__(self, http_client: Any = None, cache_store: Any = None, clock: Callable[[], float] = time.time) -> None:
        self._http_client = http_client
        self._cache_store = cache_store
        self._clock = clock
        self._snapshot: Optional[Catalog] = None
        self._failure: Optional[Tuple[float, SourceError]] = None
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _client(self) -> Any:
        return self._http_client if self._http_client is not None else http_module.http

    def _store(self) -> Any:
        return self._cache_store if self._cache_store is not None else cache_module.cache

    @property
    def snapshot(self) -> Optional[Catalog]:
        return self._snapshot

    def _loop_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def catalog(self) -> Catalog:
        snapshot = self._snapshot
        if snapshot is not None and _fresh(snapshot.fetched_at, self._clock()):
            return snapshot
        async with self._loop_lock():
            snapshot = self._snapshot
            now = self._clock()
            if snapshot is not None and _fresh(snapshot.fetched_at, now):
                return snapshot
            if self._failure is not None:
                failed_at, error = self._failure
                waited = now - failed_at
                if snapshot is not None and 0 <= waited < RETRY_SECONDS:
                    return snapshot
                if snapshot is None and 0 <= waited < FAILURE_HOLD_SECONDS:
                    raise _again(error)
            try:
                loaded = await self._load(now)
            except SourceError as exc:
                self._failure = (self._clock(), _again(exc))
                if snapshot is None:
                    raise
                logger.warning(
                    "[MODEL-BROWSER] OpenModelDB: could not refresh the catalogue (%s); "
                    "still using the one from %.1f h ago",
                    exc.message,
                    max(0.0, now - snapshot.fetched_at) / 3600,
                )
                return snapshot
            self._snapshot = loaded
            self._failure = None
            logger.info("[MODEL-BROWSER] OpenModelDB catalogue ready: %d models", len(loaded.models))
            return loaded

    async def _load(self, now: float) -> Catalog:
        models, tags, architectures = await asyncio.gather(
            self._read(MODELS_URL, now),
            self._read(TAGS_URL, now),
            self._read(ARCHITECTURES_URL, now),
            return_exceptions=True,
        )
        if isinstance(models, BaseException):
            raise models
        optional: List[Any] = []
        for name, result in (("tags.json", tags), ("architectures.json", architectures)):
            if isinstance(result, BaseException):
                if not isinstance(result, SourceError):
                    raise result
                logger.warning("[MODEL-BROWSER] OpenModelDB: no %s (%s); showing the keys as labels", name, result.message)
                optional.append(None)
            else:
                optional.append(result[0])
        body, fetched_at = models
        return await asyncio.to_thread(build_catalog, body, optional[0], optional[1], fetched_at)

    async def _read(self, url: str, now: float) -> Tuple[Any, float]:
        request = {"url": url, "params": []}
        cached = await asyncio.to_thread(self._cache_get, request)
        if isinstance(cached, dict) and "body" in cached and _fresh(cached.get("fetched_at"), now):
            return cached["body"], float(cached["fetched_at"])
        body = await self._client().get_json(SOURCE_ID, url)
        fetched_at = self._clock()
        await asyncio.to_thread(self._cache_set, request, {"body": body, "fetched_at": fetched_at})
        return body, fetched_at

    def _cache_get(self, request: Dict[str, Any]) -> Any:
        try:
            return self._store().get(SOURCE_ID, request, ttl_seconds=CATALOG_TTL_SECONDS)
        except Exception:  # noqa: BLE001
            logger.warning("[MODEL-BROWSER] Could not read the cache; asking OpenModelDB", exc_info=True)
            return None

    def _cache_set(self, request: Dict[str, Any], payload: Any) -> None:
        try:
            self._store().set(SOURCE_ID, request, payload, ttl_seconds=CATALOG_TTL_SECONDS)
        except Exception:  # noqa: BLE001
            logger.warning("[MODEL-BROWSER] Could not write the cache", exc_info=True)

    async def find_sha256(self, sha256: str) -> Optional[Tuple[str, Mapping[str, Any], Mapping[str, Any], Catalog]]:
        catalog = await self.catalog()
        found = catalog.by_sha256.get(str(sha256).strip().lower())
        if found is None:
            return None
        model_id, resource = found
        return model_id, catalog.models[model_id], resource, catalog

    async def info(self) -> SourceInfo:
        types: List[FilterOption] = []
        scales: List[FilterOption] = []
        try:
            catalog = await self.catalog()
        except SourceError as exc:
            logger.warning("[MODEL-BROWSER] OpenModelDB filter lists unavailable: %s", exc.message)
        else:
            types = [FilterOption(value=key, label=label) for key, label in catalog.architecture_options()]
            scales = [FilterOption(value=str(scale), label=f"{scale}x") for scale in catalog.scales()]
        return SourceInfo(
            id=SOURCE_ID,
            label=LABEL,
            types=types,
            base_models=scales,
            sorts=[FilterOption(value=key, label=label) for key, label in SORTS],
            periods=[],
            search_help=SEARCH_HELP,
            types_label="Architecture",
            base_models_label="Scale",
            base_model_label="Architecture",
        )

    async def search(self, query: SearchQuery, *, max_level: Optional[int] = None) -> SearchPage:
        sort = query.sort or DEFAULT_SORT
        if sort not in _SORT_KEYS:
            raise SourceInvalidRequest(
                f"Unknown sort {str(sort)[:40]!r}; use one of: {', '.join(key for key, _ in SORTS)}"
            )
        offset = parse_cursor(query.cursor)
        parsed = parse_query(query.q)
        catalog = await self.catalog()
        if parsed.kind == "model":
            model_id = catalog.find(parsed.value)
            items = self._normalise([model_id], catalog) if model_id is not None and offset == 0 else []
            return SearchPage(source=SOURCE_ID, items=items, next_cursor=None)
        architectures = architecture_filter(query.types, catalog)
        scales = scale_filter(query.base_models)
        found = [entry.id for entry in catalog.orders[sort] if matches(entry, parsed, architectures, scales)]
        end = offset + query.limit
        return SearchPage(
            source=SOURCE_ID,
            items=self._normalise(found[offset:end], catalog),
            next_cursor=str(end) if end < len(found) else None,
        )

    async def detail(self, model_id: str, *, max_level: Optional[int] = None) -> ModelDetail:
        text = str(model_id).strip()
        wanted = parse_model_url(text) or text
        if not is_model_id(wanted):
            raise SourceNotFound("OpenModelDB model ids look like 4x-UltraSharp")
        catalog = await self.catalog()
        found = catalog.find(wanted)
        if found is None:
            raise SourceNotFound(f"No OpenModelDB model {wanted!r}", 404)
        try:
            return await asyncio.to_thread(normalize_detail, found, catalog.models[found], catalog)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise SourceUnavailable(
                f"OpenModelDB sent a model this browser cannot read ({exc.__class__.__name__})"
            ) from None

    @staticmethod
    def _normalise(model_ids: Sequence[str], catalog: Catalog) -> List[SearchItem]:
        items: List[SearchItem] = []
        for model_id in model_ids:
            try:
                items.append(normalize_item(model_id, catalog.models[model_id], catalog))
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "[MODEL-BROWSER] Skipping an OpenModelDB model this browser cannot read (%r): %s", model_id, exc
                )
        return items
