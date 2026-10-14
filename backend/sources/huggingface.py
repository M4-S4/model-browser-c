import asyncio
import hashlib
import html
import logging
import math
import posixpath
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from pydantic import ValidationError

from .. import cache as cache_module
from .. import http as http_module
from ..http import SourceAnswerTooLarge, SourceError, SourceInvalidRequest, SourceNotFound, SourceUnavailable
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
from ..security import sanitize_html
from ..type_mapping import WEIGHT_EXTENSIONS, huggingface_mb_type
from .base import SearchQuery

logger = logging.getLogger("plugins.model-browser")

SOURCE_ID = "huggingface"
LABEL = "Hugging Face"
SITE = "https://huggingface.co"
API_BASE = SITE + "/api"
REVISION = "main"

NSFW_TAG = "not-for-all-audiences"
MAX_README_BYTES = 64 * 1024
MAX_IMAGES = 6
MAX_TREE_PAGES = 5
SAFE_IMAGE_LEVEL = 1
NSFW_IMAGE_LEVEL = 8

SORTS: Tuple[Tuple[str, str, str], ...] = (
    ("trending", "trendingScore", "Trending"),
    ("downloads", "downloads", "Most downloaded"),
    ("likes", "likes", "Most liked"),
    ("updated", "lastModified", "Recently updated"),
    ("newest", "createdAt", "Newest"),
)
DEFAULT_SORT = "trending"
_SORT_VALUES = {key: value for key, value, _ in SORTS}

PIPELINE_TYPES: Tuple[Tuple[str, str], ...] = (
    ("text-to-image", "Text to image"),
    ("image-to-image", "Image to image"),
    ("text-to-video", "Text to video"),
    ("image-to-video", "Image to video"),
    ("text-to-audio", "Text to audio"),
    ("text-to-3d", "Text to 3D"),
    ("image-to-3d", "Image to 3D"),
)
TAG_TYPES: Tuple[Tuple[str, str], ...] = (
    ("lora", "LoRA"),
    ("controlnet", "ControlNet"),
    ("gguf", "GGUF"),
    ("safetensors", "Safetensors"),
    ("diffusers", "Diffusers"),
)
_PIPELINES = frozenset(value for value, _ in PIPELINE_TYPES)
_TAG_FILTERS = frozenset(value for value, _ in TAG_TYPES)
_KIND_TAGS = ("lora", "controlnet", "vae", "gguf")
_HIDDEN_TAG_PREFIXES = ("license:", "base_model:", "region:")

SEARCH_HELP = "org/repo, @author, #tag, free text, or a pasted huggingface.co link"

_FORMATS = {
    ".safetensors": "SafeTensor",
    ".sft": "SafeTensor",
    ".gguf": "GGUF",
    ".ckpt": "PickleTensor",
    ".pt": "PickleTensor",
    ".pth": "PickleTensor",
    ".bin": "PickleTensor",
    ".onnx": "ONNX",
}
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif")
_ODD_URL_CHARACTERS = frozenset("\\\"'<>`")
_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt")
_HF_HOSTS = frozenset({"huggingface.co", "hf.co"})
_RESERVED_SEGMENTS = frozenset({
    "api", "avatars", "blog", "changelog", "chat", "collections", "datasets", "docs", "enterprise",
    "front", "inference", "inference-endpoints", "jobs", "join", "learn", "login", "logout", "models",
    "new", "new-dataset", "new-space", "notifications", "oauth", "organizations", "papers", "posts",
    "pricing", "privacy", "search", "settings", "spaces", "storage", "support", "tasks",
    "terms-of-service", "welcome",
})

_NAME = r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?"
_REPO_ID = re.compile(rf"(?:{_NAME}/)?{_NAME}")
_NAMESPACED_REPO_ID = re.compile(rf"{_NAME}/{_NAME}")
_ACCOUNT = re.compile(_NAME)
_BASE_MODEL = re.compile(r"[A-Za-z0-9_.\-/:]{1,200}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_LINK_ENTRY = re.compile(r"<([^>]*)>((?:\s*;\s*[^;,]*)*)")
_LINK_REL = re.compile(r'rel\s*=\s*"?([^";,]*)"?', re.I)
_FRONTMATTER = re.compile(
    r"\A(?:\N{ZERO WIDTH NO-BREAK SPACE})?---[ \t]*\r?\n(?:.*?\r?\n)?(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]\n]{0,500}\]\(\s*<?([^\s)>]+)")
_BLOB_PATH = re.compile(r"^/([^/]+/[^/]+)/blob/(.+)$")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_FLOAT_PRECISION = re.compile(
    r"(?<![a-z0-9])(fp32|fp16|bf16|fp8(?:_?e4m3fn|_?e5m2)?|nvfp4|mxfp4|fp4|nf4|int8|int4|f32|f16)(?![a-z0-9])"
)
_GGUF_QUANT = re.compile(r"(?<![a-z0-9])(i?q[1-8](?:_(?:k|[0-9]|xxs|xs|s|m|l|nl))*|bf16|f16|f32)(?![a-z0-9])")
_PRECISION_ALIASES = {"f32": "fp32", "f16": "fp16"}
_TEXT_ENCODER_TOKENS = frozenset({"clip", "t5", "t5xxl", "umt5"})


@dataclass(frozen=True)
class ParsedQuery:
    kind: str = "all"
    value: str = ""


def is_repo_id(value: Any, *, namespaced: bool = False) -> bool:
    if not isinstance(value, str) or ".." in value:
        return False
    return bool((_NAMESPACED_REPO_ID if namespaced else _REPO_ID).fullmatch(value))


def parse_hf_url(text: str) -> Optional[ParsedQuery]:
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
    if host not in _HF_HOSTS:
        return None
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if not segments or segments[0].lower() in _RESERVED_SEGMENTS:
        return None
    if len(segments) >= 2:
        repo = f"{segments[0]}/{segments[1]}"
        return ParsedQuery(kind="repo", value=repo) if is_repo_id(repo, namespaced=True) else None
    return ParsedQuery(kind="author", value=segments[0]) if _ACCOUNT.fullmatch(segments[0]) else None


def parse_query(q: Optional[str]) -> ParsedQuery:
    text = (q or "").strip()
    if not text:
        return ParsedQuery()
    link = parse_hf_url(text)
    if link is not None:
        return link
    if text.startswith("@"):
        author = text[1:].strip()
        return ParsedQuery(kind="author", value=author) if author else ParsedQuery()
    if text.startswith("#"):
        tag = text[1:].strip()
        return ParsedQuery(kind="tag", value=tag) if tag else ParsedQuery()
    if is_repo_id(text, namespaced=True):
        return ParsedQuery(kind="repo", value=text)
    return ParsedQuery(kind="text", value=text)


def next_cursor_from_link(link: Optional[str]) -> Optional[str]:
    if not isinstance(link, str) or not link:
        return None
    for match in _LINK_ENTRY.finditer(link):
        rels = " ".join(_LINK_REL.findall(match.group(2))).lower().split()
        if "next" not in rels:
            continue
        for pair in urlsplit(match.group(1).strip()).query.split("&"):
            name, _, value = pair.partition("=")
            if unquote(name) == "cursor" and value:
                return unquote(value)
        return None
    return None


def _clean_list(values: Iterable[Any]) -> List[str]:
    return sorted({str(value).strip() for value in values or () if str(value).strip()})


def _sort_value(sort: Optional[str]) -> str:
    key = sort or DEFAULT_SORT
    try:
        return _SORT_VALUES[key]
    except KeyError:
        raise SourceInvalidRequest(f"Unknown sort {str(key)[:40]!r}; use one of: {', '.join(_SORT_VALUES)}") from None


def type_params(types: Iterable[str]) -> List[Tuple[str, str]]:
    chosen = _clean_list(str(value).lower() for value in types or ())
    for value in chosen:
        if value not in _PIPELINES and value not in _TAG_FILTERS:
            known = ", ".join(value for value, _ in PIPELINE_TYPES + TAG_TYPES)
            raise SourceInvalidRequest(f"Unknown Hugging Face type {value[:40]!r}; use one of: {known}")
    pipelines = [value for value in chosen if value in _PIPELINES]
    if len(pipelines) > 1:
        raise SourceInvalidRequest(f"Hugging Face filters on one task at a time; pick one of: {', '.join(pipelines)}")
    params = [("pipeline_tag", pipelines[0])] if pipelines else []
    return params + [("filter", value) for value in chosen if value in _TAG_FILTERS]


def base_model_params(base_models: Iterable[str]) -> List[Tuple[str, str]]:
    params = []
    for name in _clean_list(base_models):
        if name.lower().startswith("base_model:"):
            name = name[len("base_model:"):]
        if not _BASE_MODEL.fullmatch(name):
            raise SourceInvalidRequest(f"Invalid Hugging Face base model {name[:40]!r}")
        params.append(("filter", f"base_model:{name}"))
    return params


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip()))


def _card(raw: Dict[str, Any]) -> Dict[str, Any]:
    card = raw.get("cardData")
    return card if isinstance(card, dict) else {}


def _repo_id(raw: Dict[str, Any]) -> str:
    value = raw.get("id") if isinstance(raw.get("id"), str) else raw.get("modelId")
    if not isinstance(value, str) or not is_repo_id(value.strip()):
        raise ValueError("model without a valid repo id")
    return value.strip()


def is_gated(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "none")
    return False


def is_nsfw(tags: Iterable[Any]) -> bool:
    return any(isinstance(tag, str) and tag.strip().lower() == NSFW_TAG for tag in tags)


def base_models_from_tags(tags: Iterable[Any]) -> List[str]:
    names: Dict[str, None] = {}
    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith("base_model:"):
            name = tag.rsplit(":", 1)[-1].strip()
            if name:
                names.setdefault(name, None)
    return list(names)


def display_tags(tags: Iterable[Any]) -> List[str]:
    return [tag for tag in tags if isinstance(tag, str) and not tag.lower().startswith(_HIDDEN_TAG_PREFIXES)]


def card_base_model(card: Dict[str, Any]) -> Optional[str]:
    value = card.get("base_model")
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item.strip()), None)
    return _text(value)


def trigger_words(card: Dict[str, Any]) -> List[str]:
    value = card.get("instance_prompt")
    values = value if isinstance(value, list) else [value]
    return list(dict.fromkeys(item.strip() for item in values if isinstance(item, str) and item.strip()))


def source_type(tags: Sequence[str], pipeline_tag: Optional[str], library_name: Optional[str]) -> str:
    if pipeline_tag:
        return pipeline_tag
    folded = {tag.lower() for tag in tags}
    for kind in _KIND_TAGS:
        if kind in folded:
            return kind
    return library_name or "model"


def _sibling_names(raw: Dict[str, Any]) -> List[str]:
    siblings = raw.get("siblings")
    if not isinstance(siblings, list):
        return []
    return [item["rfilename"] for item in siblings if isinstance(item, dict) and isinstance(item.get("rfilename"), str)]


def _creator(raw: Dict[str, Any], repo: str) -> Optional[str]:
    return _text(raw.get("author")) or (repo.split("/", 1)[0] if "/" in repo else None)


def _stats(raw: Dict[str, Any]) -> Stats:
    return Stats(downloads=_int(raw.get("downloads")), likes=_int(raw.get("likes")))


def model_page_url(repo: str) -> str:
    return f"{SITE}/{repo}"


def _quote_path(path: str) -> str:
    return "/".join(quote(part, safe="") for part in path.split("/"))


def resolve_url(repo: str, path: str) -> str:
    return f"{SITE}/{repo}/resolve/{REVISION}/{_quote_path(path)}"


def blob_url(repo: str, path: str) -> str:
    return f"{SITE}/{repo}/blob/{REVISION}/{_quote_path(path)}"


def normalize_item(raw: Dict[str, Any]) -> SearchItem:
    repo = _repo_id(raw)
    tags = _strings(raw.get("tags"))
    pipeline = _text(raw.get("pipeline_tag"))
    card_base = card_base_model(_card(raw))
    base_models = base_models_from_tags(tags) or ([card_base] if card_base else [])
    return SearchItem(
        source=SOURCE_ID,
        id=repo,
        name=repo,
        type=source_type(tags, pipeline, _text(raw.get("library_name"))),
        mb_type=huggingface_mb_type(tags, pipeline, _sibling_names(raw), repo_id=repo),
        base_model=card_base or (base_models[0] if base_models else None),
        base_models=base_models,
        creator=_creator(raw, repo),
        nsfw=is_nsfw(tags),
        gated=is_gated(raw.get("gated")),
        early_access=False,
        thumbnail=None,
        stats=_stats(raw),
        tags=display_tags(tags),
        updated_at=_text(raw.get("lastModified")),
        source_url=model_page_url(repo),
    )


def _entry_path(entry: Dict[str, Any]) -> Optional[str]:
    path = entry.get("path")
    if not isinstance(path, str) or not path or "\\" in path:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in path):
        return None
    if any(segment in ("", ".", "..") for segment in path.split("/")):
        return None
    return path


def _entry_size(entry: Dict[str, Any]) -> Optional[int]:
    lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
    size = _int(lfs.get("size"))
    if size is None:
        size = _int(entry.get("size"))
    return size if size is not None and size >= 0 else None


def tree_files(entries: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for entry in entries or ():
        if isinstance(entry, dict) and entry.get("type") == "file":
            path = _entry_path(entry)
            if path is not None:
                files.setdefault(path, entry)
    return files


def weight_extension(path: str) -> Optional[str]:
    lowered = path.lower()
    return next((extension for extension in WEIGHT_EXTENSIONS if lowered.endswith(extension)), None)


def file_precision(path: str) -> Optional[str]:
    name = posixpath.basename(path).lower()
    extension = weight_extension(name)
    stem = name[: -len(extension)] if extension else name
    if extension == ".gguf":
        match = _GGUF_QUANT.search(stem)
        if match:
            return _PRECISION_ALIASES.get(match.group(1), match.group(1))
    match = _FLOAT_PRECISION.search(stem)
    if not match:
        return None
    value = match.group(1)
    return "fp8" if value.startswith("fp8") else _PRECISION_ALIASES.get(value, value)


def file_type_from_path(path: str) -> str:
    lowered = path.lower()
    folders = lowered.split("/")[:-1]
    name = posixpath.basename(lowered)
    extension = weight_extension(name)
    stem = name[: -len(extension)] if extension else name
    tokens = set(_TOKEN_SPLIT.split(stem))
    if any(folder.startswith("vae") for folder in folders) or "vae" in tokens or stem == "ae":
        return "VAE"
    if any(folder.startswith("text_encoder") for folder in folders) or tokens & _TEXT_ENCODER_TOKENS:
        return "Text Encoder"
    if any(folder == "unet" or folder.startswith("transformer") for folder in folders):
        return "UNet"
    if "lora" in tokens or "loras" in tokens or any(folder in ("lora", "loras") for folder in folders):
        return "LoRA"
    return "Model"


def weight_files(repo: str, entries: Iterable[Any]) -> List[File]:
    candidates: List[Tuple[int, str, Dict[str, Any], str]] = []
    for index, (path, entry) in enumerate(tree_files(entries).items()):
        extension = weight_extension(path)
        if extension is not None:
            candidates.append((index, path, entry, extension))
    primary_path: Optional[str] = None
    best: Optional[Tuple[int, int]] = None
    for index, path, entry, _ in candidates:
        if "/" in path:
            continue
        size = _entry_size(entry)
        rank = (size if size is not None else -1, -index)
        if best is None or rank > best:
            best, primary_path = rank, path
    files = []
    for _, path, entry, extension in candidates:
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        oid = lfs.get("oid")
        files.append(
            File(
                id=path,
                name=path,
                size_bytes=_entry_size(entry),
                sha256=oid.lower() if isinstance(oid, str) and _SHA256.fullmatch(oid) else None,
                format=_FORMATS.get(extension),
                precision=file_precision(path),
                file_type=file_type_from_path(path),
                primary=path == primary_path,
                download_url=resolve_url(repo, path),
                auth_provider_id=SOURCE_ID,
            )
        )
    files.sort(key=lambda file: (not file.primary, "/" in file.id, file.id.lower()))
    return files


def _http_link(value: str) -> Optional[str]:
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    try:
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host:
        return None
    return value


def license_url(repo: str, link: Any, file_paths: Iterable[str] = ()) -> Optional[str]:
    link = _text(link)
    if link is not None:
        if _URL_SCHEME.match(link):
            return _http_link(link)
        return _http_link(urljoin(f"{SITE}/{repo}/blob/{REVISION}/", link))
    root = {path.lower(): path for path in file_paths if "/" not in path}
    for name in _LICENSE_FILES:
        if name.lower() in root:
            return blob_url(repo, root[name.lower()])
    return None


def license_from(raw: Dict[str, Any], repo: str, file_paths: Iterable[str] = ()) -> Optional[License]:
    card = _card(raw)
    license_id = next(
        (tag[len("license:"):].strip() for tag in _strings(raw.get("tags")) if tag.lower().startswith("license:")),
        None,
    )
    if not license_id:
        value = card.get("license")
        if isinstance(value, list):
            value = next((item for item in value if isinstance(item, str) and item.strip()), None)
        license_id = _text(value)
    if not license_id:
        return None
    summary = license_id
    if license_id.lower() == "other":
        summary = _text(card.get("license_name")) or license_id
    return License(summary=summary, url=license_url(repo, card.get("license_link"), file_paths))


MARKDOWN_EXTRAS: Dict[str, Any] = {
    "fenced-code-blocks": None,
    "highlightjs-lang": None,
    "tables": None,
    "strike": None,
    "cuddled-lists": None,
    "middle-word-em": False,
}
_RENDER_MEMO_SIZE = 16
_render_memo: "OrderedDict[str, Tuple[str, Tuple[str, ...]]]" = OrderedDict()
_render_lock = threading.Lock()


def strip_frontmatter(text: str) -> str:
    text = text or ""
    match = _FRONTMATTER.match(text)
    return text[match.end():] if match else text


class _ImageCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.srcs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            src = next((value for name, value in attrs if name == "src" and value), None)
            if src:
                self.srcs.append(src)


def _img_srcs(markup: str) -> List[str]:
    collector = _ImageCollector()
    collector.feed(markup)
    collector.close()
    return collector.srcs


def _plain_render(text: str) -> Tuple[str, List[str]]:
    paragraphs = [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]
    markup = "".join(f"<p>{html.escape(part)}</p>" for part in paragraphs)
    return markup, [match.group(1) for match in _MARKDOWN_IMAGE.finditer(text)] + _img_srcs(text)


def _render(text: str) -> Tuple[str, List[str]]:
    try:
        import markdown2
    except ImportError:
        return _plain_render(text)
    try:
        rendered = str(markdown2.markdown(text, extras=dict(MARKDOWN_EXTRAS)))
    except Exception:  # noqa: BLE001
        logger.warning("[MODEL-BROWSER] Hugging Face: could not convert a README; showing it as text", exc_info=True)
        return _plain_render(text)
    return sanitize_html(rendered), _img_srcs(rendered)


def render_markdown(text: str) -> Tuple[str, List[str]]:
    key = hashlib.md5(text.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()
    with _render_lock:
        hit = _render_memo.get(key)
        if hit is not None:
            _render_memo.move_to_end(key)
            return hit[0], list(hit[1])
    markup, srcs = _render(text)
    with _render_lock:
        _render_memo[key] = (markup, tuple(srcs))
        while len(_render_memo) > _RENDER_MEMO_SIZE:
            _render_memo.popitem(last=False)
    return markup, list(srcs)


def readme_image_url(repo: str, src: Any, known_paths: Optional[Set[str]] = None) -> Optional[str]:
    if not isinstance(src, str):
        return None
    src = src.strip()
    if not src or any(ord(char) < 0x20 or ord(char) == 0x7F or char in _ODD_URL_CHARACTERS for char in src):
        return None
    if src.startswith("//"):
        src = "https:" + src
    try:
        parts = urlsplit(src)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if parts.scheme or parts.netloc or src.startswith("/"):
        if any(char.isspace() for char in src):
            return None
        if parts.scheme or parts.netloc:
            if parts.scheme.lower() not in ("http", "https") or port is not None or "@" in parts.netloc:
                return None
            if host != "huggingface.co" and not host.endswith(".huggingface.co"):
                return None
        else:
            host = "huggingface.co"
        path = parts.path
        if host == "huggingface.co":
            path = _BLOB_PATH.sub(r"/\1/resolve/\2", path)
        if not path.lower().endswith(_IMAGE_EXTENSIONS):
            return None
        return urlunsplit(("https", host, path, parts.query, ""))
    path = posixpath.normpath(unquote(parts.path))
    if path in (".", "..") or path.startswith(("../", "/")) or not path.lower().endswith(_IMAGE_EXTENSIONS):
        return None
    if known_paths is not None and path not in known_paths:
        return None
    return resolve_url(repo, path)


def readme_images(repo: str, srcs: Iterable[Any], nsfw: bool, known_paths: Optional[Set[str]] = None) -> List[Image]:
    images: List[Image] = []
    seen: Set[str] = set()
    for src in srcs:
        url = readme_image_url(repo, src, known_paths)
        if url is None or url in seen:
            continue
        seen.add(url)
        images.append(
            Image(
                url=url,
                nsfw=nsfw,
                nsfw_level=NSFW_IMAGE_LEVEL if nsfw else SAFE_IMAGE_LEVEL,
                id=hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:16],
            )
        )
        if len(images) >= MAX_IMAGES:
            break
    return images


def normalize_detail(
    raw: Dict[str, Any],
    tree: Optional[Iterable[Any]] = None,
    readme: Optional[str] = None,
    *,
    tree_complete: bool = True,
) -> ModelDetail:
    repo = _repo_id(raw)
    tags = _strings(raw.get("tags"))
    card = _card(raw)
    pipeline = _text(raw.get("pipeline_tag"))
    files_by_path = tree_files(tree or ())
    nsfw = is_nsfw(tags)
    description_html: Optional[str] = None
    images: List[Image] = []
    body = strip_frontmatter(readme) if isinstance(readme, str) else ""
    if body.strip():
        markup, srcs = render_markdown(body)
        description_html = markup or None
        known_paths = set(files_by_path) if tree is not None and tree_complete else None
        images = readme_images(repo, srcs, nsfw, known_paths)
    base_models = base_models_from_tags(tags)
    version = Version(
        id=REVISION,
        name=REVISION,
        base_model=card_base_model(card) or (base_models[0] if base_models else None),
        trigger_words=trigger_words(card),
        created_at=_text(raw.get("createdAt")),
        early_access=False,
        nsfw=nsfw,
        images=images,
        files=weight_files(repo, tree or ()),
    )
    return ModelDetail(
        source=SOURCE_ID,
        id=repo,
        name=repo,
        type=source_type(tags, pipeline, _text(raw.get("library_name"))),
        mb_type=huggingface_mb_type(tags, pipeline, list(files_by_path) or _sibling_names(raw), repo_id=repo),
        creator=_creator(raw, repo),
        description_html=description_html,
        tags=display_tags(tags),
        license=license_from(raw, repo, files_by_path),
        source_url=model_page_url(repo),
        nsfw=nsfw,
        poi=False,
        gated=is_gated(raw.get("gated")),
        stats=_stats(raw),
        versions=[version],
    )


class HuggingFaceSource:
    id = SOURCE_ID
    label = LABEL

    def __init__(self, http_client: Any = None, cache_store: Any = None) -> None:
        self._http_client = http_client
        self._cache_store = cache_store

    def _client(self) -> Any:
        return self._http_client if self._http_client is not None else http_module.http

    def _store(self) -> Any:
        return self._cache_store if self._cache_store is not None else cache_module.cache

    def _cache_get(self, request: Dict[str, Any]) -> Any:
        try:
            return self._store().get(SOURCE_ID, request)
        except Exception:  # noqa: BLE001
            logger.warning("[MODEL-BROWSER] Could not read the cache; asking Hugging Face", exc_info=True)
            return None

    def _cache_set(self, request: Dict[str, Any], payload: Any) -> None:
        try:
            self._store().set(SOURCE_ID, request, payload)
        except Exception:  # noqa: BLE001
            logger.warning("[MODEL-BROWSER] Could not write the cache", exc_info=True)

    async def _get_json(
        self, url: str, params: Sequence[Tuple[str, Any]] = (), *, repo: Optional[str] = None
    ) -> Tuple[Any, Optional[str]]:
        pairs = sorted(http_module.normalise_params(list(params)))
        request = {"url": url, "params": [list(pair) for pair in pairs]}
        cached = self._cache_get(request)
        if isinstance(cached, dict) and "body" in cached:
            link = cached.get("link")
            return cached["body"], link if isinstance(link, str) else None
        try:
            body, headers = await self._client().get_json_with_headers(SOURCE_ID, url, pairs)
        except SourceInvalidRequest as exc:
            if repo is not None and exc.status == 401:
                raise SourceNotFound(
                    f"No public Hugging Face model {repo!r}: it does not exist, or it is private", 401
                ) from None
            raise
        link = headers.get("link")
        self._cache_set(request, {"body": body, "link": link})
        return body, link

    async def _model(self, repo: str) -> Dict[str, Any]:
        body, _ = await self._get_json(f"{API_BASE}/models/{repo}", repo=repo)
        if not isinstance(body, dict) or not isinstance(body.get("id") or body.get("modelId"), str):
            raise SourceUnavailable("Hugging Face sent an unexpected answer for a model")
        return body

    async def _tree(self, repo: str) -> Tuple[List[Dict[str, Any]], bool]:
        url = f"{API_BASE}/models/{repo}/tree/{REVISION}"
        entries: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(MAX_TREE_PAGES):
            params: List[Tuple[str, Any]] = [("recursive", "1")]
            if cursor:
                params.append(("cursor", cursor))
            body, link = await self._get_json(url, params, repo=repo)
            if not isinstance(body, list):
                raise SourceUnavailable("Hugging Face sent an unexpected file list")
            entries.extend(entry for entry in body if isinstance(entry, dict))
            next_cursor = next_cursor_from_link(link)
            if not next_cursor or next_cursor == cursor:
                return entries, True
            cursor = next_cursor
        logger.info(
            "[MODEL-BROWSER] Hugging Face: %s lists more than %d pages of files; showing the first ones",
            repo, MAX_TREE_PAGES,
        )
        return entries, False

    async def _readme(self, repo: str, files: Dict[str, Dict[str, Any]], tree_complete: bool) -> Optional[str]:
        entry = files.get("README.md")
        if entry is None and tree_complete:
            return None
        if entry is not None and (_entry_size(entry) or 0) > MAX_README_BYTES:
            return None
        url = f"{SITE}/{repo}/raw/{REVISION}/README.md"
        request = {"url": url, "params": [], "as": "text"}
        cached = self._cache_get(request)
        if isinstance(cached, dict) and "text" in cached:
            return cached["text"] if isinstance(cached["text"], str) else None
        text: Optional[str]
        try:
            text = await self._client().get_text(SOURCE_ID, url, max_bytes=MAX_README_BYTES)
        except (SourceAnswerTooLarge, SourceNotFound):
            text = None
        except SourceInvalidRequest as exc:
            if exc.status not in (401, 403):
                logger.warning("[MODEL-BROWSER] Hugging Face: no README for %s: %s", repo, exc.message)
                return None
            text = None
        except SourceError as exc:
            logger.warning("[MODEL-BROWSER] Hugging Face: no README for %s: %s", repo, exc.message)
            return None
        self._cache_set(request, {"text": text})
        return text

    async def info(self) -> SourceInfo:
        return SourceInfo(
            id=SOURCE_ID,
            label=LABEL,
            enabled=True,
            types=[FilterOption(value=value, label=label) for value, label in PIPELINE_TYPES + TAG_TYPES],
            base_models=[],
            sorts=[FilterOption(value=key, label=label) for key, _, label in SORTS],
            periods=[],
            search_help=SEARCH_HELP,
        )

    async def search(self, query: SearchQuery, *, max_level: Optional[int] = None) -> SearchPage:
        parsed = parse_query(query.q)
        sort = _sort_value(query.sort)
        filters = type_params(query.types) + base_model_params(query.base_models)
        if parsed.kind == "repo":
            if not query.cursor:
                item = await self._exact_item(parsed.value)
                if item is not None:
                    items = [] if query.safe and item.nsfw else [item]
                    return SearchPage(source=SOURCE_ID, items=items, next_cursor=None)
            parsed = ParsedQuery(kind="text", value=parsed.value)
        params: List[Tuple[str, Any]] = [
            ("limit", query.limit),
            ("full", "true"),
            ("sort", sort),
            ("direction", "-1"),
        ]
        if parsed.kind == "text":
            params.append(("search", parsed.value))
        elif parsed.kind == "author":
            params.append(("author", parsed.value))
        elif parsed.kind == "tag":
            params.append(("filter", parsed.value))
        params += filters
        if query.cursor:
            params.append(("cursor", query.cursor))
        body, link = await self._get_json(f"{API_BASE}/models", params)
        if not isinstance(body, list):
            raise SourceUnavailable("Hugging Face sent an unexpected search answer")
        items = self._normalise_items(body)
        if query.safe:
            items = [item for item in items if not item.nsfw]
        return SearchPage(source=SOURCE_ID, items=items, next_cursor=next_cursor_from_link(link))

    async def detail(self, model_id: str, *, max_level: Optional[int] = None) -> ModelDetail:
        repo = str(model_id).strip()
        if not is_repo_id(repo):
            raise SourceNotFound("Hugging Face model ids look like org/repo")
        raw, tree = await asyncio.gather(self._model(repo), self._tree(repo), return_exceptions=True)
        if isinstance(raw, BaseException):
            raise raw
        if raw.get("disabled") is True:
            raise SourceNotFound(f"{repo} is disabled on Hugging Face")
        entries: List[Dict[str, Any]] = []
        complete = False
        if isinstance(tree, BaseException):
            if not (isinstance(tree, SourceNotFound) and tree.status == 404):
                raise tree
            logger.info("[MODEL-BROWSER] Hugging Face: %s has no %s revision to list", repo, REVISION)
        else:
            entries, complete = tree
        readme = await self._readme(repo, tree_files(entries), complete)
        try:
            return await asyncio.to_thread(normalize_detail, raw, entries, readme, tree_complete=complete)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise SourceUnavailable(
                f"Hugging Face sent a model this browser cannot read ({exc.__class__.__name__})"
            ) from None

    async def _exact_item(self, repo: str) -> Optional[SearchItem]:
        try:
            raw = await self._model(repo)
        except SourceNotFound:
            return None
        if raw.get("disabled") is True:
            return None
        items = self._normalise_items([raw])
        return items[0] if items else None

    @staticmethod
    def _normalise_items(raw_items: Sequence[Any]) -> List[SearchItem]:
        items: List[SearchItem] = []
        seen = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            try:
                item = normalize_item(raw)
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "[MODEL-BROWSER] Skipping a Hugging Face model this browser cannot read (id %r): %s",
                    raw.get("id"), exc,
                )
                continue
            if item.id not in seen:
                seen.add(item.id)
                items.append(item)
        return items
