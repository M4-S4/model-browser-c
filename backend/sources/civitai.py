import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlsplit, urlunsplit

from pydantic import ValidationError

from .. import cache as cache_module
from .. import http as http_module
from .. import settings
from ..http import SourceError, SourceInvalidRequest, SourceNotFound, SourceUnavailable
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
from ..type_mapping import civitai_mb_type
from .base import SearchQuery

logger = logging.getLogger("plugins.model-browser")

SOURCE_ID = "civitai"
LABEL = "CivitAI"
SITE_SFW = "https://civitai.com"
SITE_FULL = "https://civitai.red"

OVERFETCH = 20
UPSTREAM_MAX_LIMIT = 100
IDS_CHUNK = 100
MAX_IDS = 1000
MAX_REFETCHES = 10
ENUMS_TTL_SECONDS = 24 * 3600
IDS_CURSOR_PREFIX = "ids:"

SORTS: Tuple[Tuple[str, str, str], ...] = (
    ("highest_rated", "Highest Rated", "Highest rated"),
    ("most_downloaded", "Most Downloaded", "Most downloaded"),
    ("most_liked", "Most Liked", "Most liked"),
    ("most_discussed", "Most Discussed", "Most discussed"),
    ("most_collected", "Most Collected", "Most collected"),
    ("most_images", "Most Images", "Most images"),
    ("newest", "Newest", "Newest"),
    ("oldest", "Oldest", "Oldest"),
)
PERIODS: Tuple[Tuple[str, str, str], ...] = (
    ("all_time", "AllTime", "All time"),
    ("year", "Year", "Past year"),
    ("month", "Month", "Past month"),
    ("week", "Week", "Past week"),
    ("day", "Day", "Past day"),
)
_SORT_VALUES = {key: value for key, value, _ in SORTS}
_PERIOD_VALUES = {key: value for key, value, _ in PERIODS}

REMOVED_MODES = frozenset({"Archived", "TakenDown"})
_OPEN_AVAILABILITY = frozenset({"Public", "Unsearchable"})

SEARCH_HELP = "Text, #tag, @creator, $#id1,id2 or a civitai.com / civitai.red model link"

_TYPE_LABELS = {
    "LORA": "LoRA",
    "LoCon": "LyCORIS (LoCon)",
    "DoRA": "DoRA",
    "TextualInversion": "Embedding",
    "Controlnet": "ControlNet",
    "VAE": "VAE",
    "UNet": "UNet",
    "CLIP": "CLIP",
    "CLIPVision": "CLIP Vision",
    "LLM": "LLM",
}

_COMMERCIAL_USES: Tuple[Tuple[str, str], ...] = (
    ("Image", "selling generated images"),
    ("RentCivit", "generation on CivitAI"),
    ("Rent", "generation services"),
    ("Sell", "selling the model"),
    ("SellMerge", "selling merges"),
)
_COMMERCIAL_LABELS = dict(_COMMERCIAL_USES)

_DIGITS = re.compile(r"[0-9]{1,12}")
_MODEL_PATH = re.compile(r"^/models/([0-9]{1,12})(?:/[^/]*)?/?$")
_ID_SEPARATORS = re.compile(r"[\s,;]+")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_IMAGE_ID_IN_URL = re.compile(r"/([0-9]{1,15})\.[A-Za-z0-9]{2,5}$")


@dataclass(frozen=True)
class ParsedQuery:
    kind: str = "all"
    value: str = ""
    ids: Tuple[int, ...] = ()
    version_id: Optional[int] = None


def parse_model_url(text: str) -> Optional[Tuple[int, Optional[int]]]:
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
    if host not in ("civitai.com", "civitai.red"):
        return None
    match = _MODEL_PATH.match(parts.path)
    if not match:
        return None
    version = (parse_qs(parts.query).get("modelVersionId") or [""])[0]
    return int(match.group(1)), int(version) if _DIGITS.fullmatch(version) else None


def parse_ids(text: str) -> Tuple[int, ...]:
    ids: Dict[int, None] = {}
    for token in _ID_SEPARATORS.split(text or ""):
        if _DIGITS.fullmatch(token) and int(token) > 0:
            ids.setdefault(int(token), None)
    return tuple(ids)


def parse_query(q: Optional[str]) -> ParsedQuery:
    text = (q or "").strip()
    if not text:
        return ParsedQuery()
    link = parse_model_url(text)
    if link is not None:
        return ParsedQuery(kind="ids", ids=(link[0],), version_id=link[1])
    if text.startswith("$#"):
        ids = parse_ids(text[2:])
        if not ids:
            raise SourceInvalidRequest("Write CivitAI model ids after $#, separated by commas: $#4201,122359")
        if len(ids) > MAX_IDS:
            raise SourceInvalidRequest(f"Search for at most {MAX_IDS} CivitAI model ids at once")
        return ParsedQuery(kind="ids", ids=ids)
    if text.startswith("#"):
        tag = text[1:].strip()
        return ParsedQuery(kind="tag", value=tag) if tag else ParsedQuery()
    if text.startswith("@"):
        username = text[1:].strip()
        return ParsedQuery(kind="username", value=username) if username else ParsedQuery()
    return ParsedQuery(kind="text", value=text)


def ids_chunk_index(cursor: Optional[str]) -> int:
    if cursor and cursor.startswith(IDS_CURSOR_PREFIX):
        rest = cursor[len(IDS_CURSOR_PREFIX):]
        if _DIGITS.fullmatch(rest):
            return int(rest)
    return 0


def _choice(value: Optional[str], choices: Dict[str, str], what: str) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        return choices[value]
    except KeyError:
        raise SourceInvalidRequest(
            f"Unknown {what} {str(value)[:40]!r}; use one of: {', '.join(choices)}"
        ) from None


def _clean_list(values: Iterable[str]) -> List[str]:
    return sorted({str(value).strip() for value in values or () if str(value).strip()})


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _id(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    if isinstance(value, str) and _DIGITS.fullmatch(value.strip()):
        return int(value.strip()) or None
    return None


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _bool_or_none(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _parse_date(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def safe_mask(max_level: int) -> int:
    return (1 << max(1, int(max_level)).bit_length()) - 1


def level_is_safe(level: Any, max_level: int) -> bool:
    level = _int(level)
    if level is None or level <= 0:
        return False
    return level & ~safe_mask(max_level) == 0


def mask_exceeds(mask: Any, max_level: int) -> bool:
    mask = _int(mask)
    if mask is None or mask <= 0:
        return False
    return bool(mask & ~safe_mask(max_level))


def model_page_url(model_id: str, nsfw: bool) -> str:
    return f"{SITE_FULL if nsfw else SITE_SFW}/models/{model_id}"


def type_label(model_type: str) -> str:
    if model_type in _TYPE_LABELS:
        return _TYPE_LABELS[model_type]
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", model_type) or [model_type]
    return " ".join([words[0]] + [word if word.isupper() else word.lower() for word in words[1:]])


def _image_id(raw: Dict[str, Any], url: Optional[str]) -> Optional[str]:
    image_id = _id(raw.get("id"))
    if image_id is not None:
        return str(image_id)
    if url:
        match = _IMAGE_ID_IN_URL.search(urlsplit(url).path)
        if match:
            return match.group(1)
    return None


def normalize_image(raw: Any, max_level: int) -> Optional[Image]:
    if not isinstance(raw, dict):
        return None
    url = raw.get("url")
    url = url if isinstance(url, str) and url.startswith("https://") else None
    blurhash = _text(raw.get("hash"))
    if url is None and blurhash is None:
        return None
    level = _int(raw.get("nsfwLevel"))
    return Image(
        url=url,
        blurhash=blurhash,
        nsfw=not level_is_safe(level, max_level),
        nsfw_level=level,
        width=_int(raw.get("width")),
        height=_int(raw.get("height")),
        kind="video" if raw.get("type") == "video" else "image",
        poi=_is_true(raw.get("poi")),
        id=_image_id(raw, url),
    )


def _images(raw_version: Dict[str, Any], max_level: int) -> List[Image]:
    images = raw_version.get("images")
    if not isinstance(images, list):
        return []
    return [image for image in (normalize_image(item, max_level) for item in images) if image is not None]


def pick_thumbnail(images: Sequence[Image]) -> Optional[Image]:
    if not images:
        return None
    best = min(
        range(len(images)),
        key=lambda index: (
            images[index].nsfw,
            images[index].url is None,
            images[index].poi,
            images[index].kind != "image",
            index,
        ),
    )
    return images[best]


def _file_sha256(raw: Any) -> Optional[str]:
    hashes = raw.get("hashes") if isinstance(raw, dict) else None
    sha256 = hashes.get("SHA256") if isinstance(hashes, dict) else None
    return sha256.lower() if isinstance(sha256, str) and _SHA256.fullmatch(sha256) else None


def primary_file_sha256(raw_version: Optional[Dict[str, Any]]) -> Optional[str]:
    if raw_version is None:
        return None
    files = [raw for raw in _list(raw_version.get("files")) if isinstance(raw, dict)]
    for raw in [raw for raw in files if _is_true(raw.get("primary"))] + files:
        sha256 = _file_sha256(raw)
        if sha256:
            return sha256
    return None


_RED_HOSTS = frozenset({"civitai.red", "www.civitai.red"})


def com_download_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return url
    if (parts.hostname or "").lower() not in _RED_HOSTS or port is not None or "@" in parts.netloc:
        return url
    return urlunsplit((parts.scheme, "civitai.com", parts.path, parts.query, parts.fragment))


def normalize_file(raw: Any) -> Optional[File]:
    if not isinstance(raw, dict):
        return None
    file_id = _id(raw.get("id"))
    url = _text(raw.get("downloadUrl"))
    if file_id is None or url is None:
        return None
    size_kb = _float(raw.get("sizeKB"))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return File(
        id=str(file_id),
        name=_text(raw.get("name")) or f"file-{file_id}",
        size_bytes=round(size_kb * 1024) if size_kb is not None and size_kb >= 0 else None,
        sha256=_file_sha256(raw),
        format=_text(metadata.get("format")),
        precision=_text(metadata.get("fp")),
        file_type=_text(raw.get("type")),
        primary=_is_true(raw.get("primary")),
        download_url=com_download_url(url),
        auth_provider_id=SOURCE_ID,
    )


def _trigger_words(raw_version: Dict[str, Any]) -> List[str]:
    words: Dict[str, None] = {}
    for word in _list(raw_version.get("trainedWords")):
        if isinstance(word, str):
            cleaned = word.strip().strip(",").strip()
            if cleaned:
                words.setdefault(cleaned, None)
    return list(words)


def _version_date(raw_version: Dict[str, Any]) -> Optional[str]:
    return (
        _text(raw_version.get("publishedAt"))
        or _text(raw_version.get("createdAt"))
        or _text(raw_version.get("updatedAt"))
    )


def version_early_access(raw_version: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    availability = raw_version.get("availability")
    if isinstance(availability, str) and availability not in _OPEN_AVAILABILITY:
        return True
    if raw_version.get("paidAccess") or _is_true(raw_version.get("earlyAccess")):
        return True
    now = now or datetime.now(timezone.utc)
    for key in ("earlyAccessEndsAt", "earlyAccessDeadline"):
        ends = _parse_date(raw_version.get(key))
        if ends is not None and ends > now:
            return True
    return False


def model_early_access(raw: Dict[str, Any], first_version: Optional[Dict[str, Any]]) -> bool:
    availability = raw.get("availability")
    if isinstance(availability, str) and availability not in _OPEN_AVAILABILITY:
        return True
    if _is_true(raw.get("earlyAccess")):
        return True
    return first_version is not None and version_early_access(first_version)


def _raw_versions(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    versions = raw.get("modelVersions")
    if not isinstance(versions, list):
        return []
    return [version for version in versions if isinstance(version, dict)]


def normalize_version(raw: Dict[str, Any], max_level: int, model_flagged: bool = False) -> Optional[Version]:
    version_id = _id(raw.get("id"))
    if version_id is None:
        return None
    images = _images(raw, max_level)
    files = [file for file in (normalize_file(item) for item in _list(raw.get("files"))) if file is not None]
    nsfw = model_flagged or mask_exceeds(raw.get("nsfwLevel"), max_level) or not any(not image.nsfw for image in images)
    return Version(
        id=str(version_id),
        name=_text(raw.get("name")) or str(version_id),
        base_model=_text(raw.get("baseModel")),
        trigger_words=_trigger_words(raw),
        created_at=_version_date(raw),
        early_access=version_early_access(raw),
        nsfw=nsfw,
        images=images,
        files=files,
    )


def model_is_nsfw(raw: Dict[str, Any], images: Sequence[Image], max_level: int) -> bool:
    return (
        _is_true(raw.get("nsfw"))
        or mask_exceeds(raw.get("nsfwLevel"), max_level)
        or not any(not image.nsfw for image in images)
    )


def license_summary(
    commercial_use: Optional[List[str]],
    no_credit: Optional[bool],
    derivatives: Optional[bool],
    different_license: Optional[bool],
) -> str:
    parts = []
    if no_credit is True:
        parts.append("No credit required")
    elif no_credit is False:
        parts.append("Credit required")
    if commercial_use is not None:
        if commercial_use:
            labels = [_COMMERCIAL_LABELS.get(use, use) for use in commercial_use]
            parts.append("Commercial use: " + ", ".join(labels))
        else:
            parts.append("No commercial use")
    if derivatives is True:
        if different_license is True:
            parts.append("Merges allowed, under any license")
        elif different_license is False:
            parts.append("Merges allowed, with the same permissions")
        else:
            parts.append("Merges allowed")
    elif derivatives is False:
        parts.append("No merges")
    return "; ".join(parts) or "No license information"


def _commercial_uses(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        return None
    values = [item for item in dict.fromkeys(values) if item and item != "None"]
    known = [use for use, _ in _COMMERCIAL_USES if use in values]
    return known + [item for item in values if item not in _COMMERCIAL_LABELS]


def license_from(raw: Dict[str, Any]) -> Optional[License]:
    flags = ("allowCommercialUse", "allowNoCredit", "allowDerivatives", "allowDifferentLicense")
    if not any(flag in raw for flag in flags):
        return None
    commercial = _commercial_uses(raw.get("allowCommercialUse"))
    no_credit = _bool_or_none(raw.get("allowNoCredit"))
    derivatives = _bool_or_none(raw.get("allowDerivatives"))
    different = _bool_or_none(raw.get("allowDifferentLicense"))
    return License(
        summary=license_summary(commercial, no_credit, derivatives, different),
        commercial_use=commercial,
        no_credit=no_credit,
        derivatives=derivatives,
        different_license=different,
    )


def _stats(raw: Dict[str, Any]) -> Stats:
    stats = raw.get("stats") if isinstance(raw.get("stats"), dict) else {}
    return Stats(
        downloads=_int(stats.get("downloadCount")),
        likes=_int(stats.get("thumbsUpCount")),
        rating=_float(stats.get("rating")),
    )


def _tags(raw: Dict[str, Any]) -> List[str]:
    tags: Dict[str, None] = {}
    for tag in _list(raw.get("tags")):
        name = tag.get("name") if isinstance(tag, dict) else tag
        if isinstance(name, str) and name.strip():
            tags.setdefault(name.strip(), None)
    return list(tags)


def _creator(raw: Dict[str, Any]) -> Optional[str]:
    creator = raw.get("creator")
    return _text(creator.get("username")) if isinstance(creator, dict) else None


def _base_models(raw: Dict[str, Any], versions: Sequence[Dict[str, Any]]) -> List[str]:
    names = _strings(raw.get("baseModels"))
    if not names:
        names = [name for name in (_text(version.get("baseModel")) for version in versions) if name]
    return list(dict.fromkeys(names))


def _model_id(raw: Dict[str, Any]) -> str:
    model_id = _id(raw.get("id"))
    if model_id is None:
        raise ValueError("model without an id")
    return str(model_id)


def normalize_item(raw: Dict[str, Any], max_level: int) -> SearchItem:
    model_id = _model_id(raw)
    versions = _raw_versions(raw)
    images = [image for version in versions for image in _images(version, max_level)]
    nsfw = model_is_nsfw(raw, images, max_level)
    first = versions[0] if versions else None
    model_type = _text(raw.get("type")) or "Other"
    return SearchItem(
        source=SOURCE_ID,
        id=model_id,
        name=_text(raw.get("name")) or f"Model {model_id}",
        type=model_type,
        mb_type=civitai_mb_type(model_type),
        base_model=_text(first.get("baseModel")) if first else None,
        base_models=_base_models(raw, versions),
        creator=_creator(raw),
        nsfw=nsfw,
        poi=_is_true(raw.get("poi")),
        early_access=model_early_access(raw, first),
        thumbnail=pick_thumbnail(images),
        stats=_stats(raw),
        tags=_tags(raw),
        updated_at=_version_date(first) if first else None,
        source_url=model_page_url(model_id, nsfw),
        primary_sha256=primary_file_sha256(first),
    )


def normalize_detail(raw: Dict[str, Any], max_level: int) -> ModelDetail:
    model_id = _model_id(raw)
    flagged = _is_true(raw.get("nsfw"))
    versions = [
        version
        for version in (normalize_version(item, max_level, flagged) for item in _raw_versions(raw))
        if version is not None
    ]
    images = [image for version in versions for image in version.images]
    nsfw = model_is_nsfw(raw, images, max_level)
    model_type = _text(raw.get("type")) or "Other"
    description = raw.get("description")
    return ModelDetail(
        source=SOURCE_ID,
        id=model_id,
        name=_text(raw.get("name")) or f"Model {model_id}",
        type=model_type,
        mb_type=civitai_mb_type(model_type),
        creator=_creator(raw),
        description_html=(sanitize_html(description) or None) if isinstance(description, str) else None,
        tags=_tags(raw),
        license=license_from(raw),
        source_url=model_page_url(model_id, nsfw),
        nsfw=nsfw,
        poi=_is_true(raw.get("poi")),
        stats=_stats(raw),
        versions=versions,
    )


def is_removed(raw: Dict[str, Any]) -> bool:
    return raw.get("mode") in REMOVED_MODES


class CivitaiSource:
    id = SOURCE_ID
    label = LABEL

    def __init__(self, http_client: Any = None, cache_store: Any = None) -> None:
        self._http_client = http_client
        self._cache_store = cache_store

    def _client(self) -> Any:
        return self._http_client if self._http_client is not None else http_module.http

    def _store(self) -> Any:
        return self._cache_store if self._cache_store is not None else cache_module.cache

    @staticmethod
    def api_base() -> str:
        return str(settings.get_setting("civitai_api_base")).rstrip("/")

    @staticmethod
    def _max_level(max_level: Optional[int]) -> int:
        return int(settings.get_setting("nsfw_max_level") if max_level is None else max_level)

    async def _get(self, path: str, params: Sequence[Tuple[str, Any]] = (), *, ttl_seconds: Optional[float] = None) -> Any:
        url = self.api_base() + path
        pairs = sorted(http_module.normalise_params(list(params)))
        request = {"url": url, "params": [list(pair) for pair in pairs]}
        store = self._store()
        try:
            cached = store.get(SOURCE_ID, request, ttl_seconds=ttl_seconds)
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not read the cache; asking CivitAI", exc_info=True)
            cached = None
        if cached is not None:
            return cached
        body = await self._client().get_json(SOURCE_ID, url, pairs)
        try:
            store.set(SOURCE_ID, request, body, ttl_seconds=ttl_seconds)
        except Exception:
            logger.warning("[MODEL-BROWSER] Could not write the cache", exc_info=True)
        return body

    async def _enums(self) -> Dict[str, Any]:
        body = await self._get("/enums", ttl_seconds=ENUMS_TTL_SECONDS)
        if not isinstance(body, dict):
            raise SourceUnavailable("CivitAI sent unexpected filter lists")
        return body

    async def _model(self, model_id: int) -> Dict[str, Any]:
        body = await self._get(f"/models/{int(model_id)}")
        if not isinstance(body, dict) or _id(body.get("id")) is None:
            raise SourceUnavailable("CivitAI sent an unexpected answer for a model")
        return body

    async def info(self) -> SourceInfo:
        types: List[FilterOption] = []
        base_models: List[FilterOption] = []
        try:
            enums = await self._enums()
        except SourceError as exc:
            logger.warning("[MODEL-BROWSER] CivitAI filter lists unavailable: %s", exc.message)
        else:
            types = [FilterOption(value=value, label=type_label(value)) for value in _strings(enums.get("ModelType"))]
            base_names = _strings(enums.get("ActiveBaseModel")) or _strings(enums.get("BaseModel"))
            base_models = [FilterOption(value=value, label=value) for value in base_names]
        return SourceInfo(
            id=SOURCE_ID,
            label=LABEL,
            enabled=True,
            types=types,
            base_models=base_models,
            sorts=[FilterOption(value=key, label=label) for key, _, label in SORTS],
            periods=[FilterOption(value=key, label=label) for key, _, label in PERIODS],
            search_help=SEARCH_HELP,
        )

    async def search(self, query: SearchQuery, *, max_level: Optional[int] = None) -> SearchPage:
        level = self._max_level(max_level)
        parsed = parse_query(query.q)
        sort = _choice(query.sort, _SORT_VALUES, "sort")
        period = _choice(query.period, _PERIOD_VALUES, "period")
        if parsed.kind == "ids":
            raw_items, next_cursor = await self._search_ids(parsed, query, sort)
        else:
            raw_items, next_cursor = await self._search_list(parsed, query, sort, period)
        raw_items = await self._complete_versions(raw_items)
        return SearchPage(source=SOURCE_ID, items=self._normalise_items(raw_items, level), next_cursor=next_cursor)

    async def detail(self, model_id: str, *, max_level: Optional[int] = None) -> ModelDetail:
        level = self._max_level(max_level)
        text = str(model_id).strip()
        if not _DIGITS.fullmatch(text) or int(text) <= 0:
            raise SourceNotFound("CivitAI model ids are numbers")
        raw = await self._model(int(text))
        mode = raw.get("mode")
        if mode == "TakenDown":
            raise SourceNotFound("This model was taken down on CivitAI")
        if mode == "Archived":
            raise SourceNotFound("This model was archived on CivitAI: its files are gone")
        try:
            return normalize_detail(raw, level)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise SourceUnavailable(f"CivitAI sent a model this browser cannot read ({exc.__class__.__name__})") from None

    async def _search_list(
        self, parsed: ParsedQuery, query: SearchQuery, sort: Optional[str], period: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        params: List[Tuple[str, Any]] = [
            ("limit", min(query.limit + OVERFETCH, UPSTREAM_MAX_LIMIT)),
            ("nsfw", not query.safe),
        ]
        if parsed.kind == "text":
            params.append(("query", parsed.value))
        elif parsed.kind == "tag":
            params.append(("tag", parsed.value))
        elif parsed.kind == "username":
            params.append(("username", parsed.value))
        params += [("types", value) for value in _clean_list(query.types)]
        params += [("baseModels", value) for value in await self._partial_base_models(query.base_models)]
        if sort:
            params.append(("sort", sort))
        if period:
            params.append(("period", period))
        if query.cursor:
            params.append(("cursor", query.cursor))
        body = await self._get("/models", params)
        if not isinstance(body, dict):
            raise SourceUnavailable("CivitAI sent an unexpected search answer")
        items = [item for item in _list(body.get("items")) if isinstance(item, dict)]
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        next_cursor = metadata.get("nextCursor")
        return items, (str(next_cursor) if next_cursor not in (None, "") else None)

    async def _search_ids(
        self, parsed: ParsedQuery, query: SearchQuery, sort: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        index = ids_chunk_index(query.cursor)
        chunk = parsed.ids[index * IDS_CHUNK:(index + 1) * IDS_CHUNK]
        if not chunk:
            return [], None
        params: List[Tuple[str, Any]] = [
            ("ids", ",".join(str(model_id) for model_id in chunk)),
            ("limit", UPSTREAM_MAX_LIMIT),
            ("nsfw", not query.safe),
            ("period", "AllTime"),
        ]
        if sort:
            params.append(("sort", sort))
        body = await self._get("/models", params)
        if not isinstance(body, dict):
            raise SourceUnavailable("CivitAI sent an unexpected search answer")
        returned: Dict[int, Dict[str, Any]] = {}
        for item in _list(body.get("items")):
            if isinstance(item, dict) and _id(item.get("id")) is not None:
                returned.setdefault(_id(item.get("id")), item)
        missing = [model_id for model_id in chunk if model_id not in returned]
        if missing:
            returned.update(await self._models_by_id(missing))
        more = (index + 1) * IDS_CHUNK < len(parsed.ids)
        ordered = [returned[model_id] for model_id in chunk if model_id in returned]
        return ordered, (f"{IDS_CURSOR_PREFIX}{index + 1}" if more else None)

    async def _models_by_id(self, model_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        found: Dict[int, Dict[str, Any]] = {}
        for model_id in list(model_ids)[:MAX_REFETCHES]:
            try:
                found[model_id] = await self._model(model_id)
            except SourceNotFound:
                continue
            except SourceError as exc:
                logger.warning("[MODEL-BROWSER] CivitAI: could not fetch model %s by id: %s", model_id, exc.message)
                break
        return found

    async def _complete_versions(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        empty = [_id(item.get("id")) for item in raw_items if not _raw_versions(item) and _id(item.get("id"))]
        if not empty:
            return raw_items
        found = await self._models_by_id(empty)
        completed = []
        for item in raw_items:
            replacement = found.get(_id(item.get("id"))) if not _raw_versions(item) else None
            completed.append(replacement if replacement is not None and _raw_versions(replacement) else item)
        return completed

    async def _partial_base_models(self, selected: Iterable[str]) -> List[str]:
        chosen = _clean_list(selected)
        if not chosen:
            return []
        try:
            enums = await self._enums()
        except SourceError:
            return chosen
        active = set(_strings(enums.get("ActiveBaseModel")))
        if active and active.issubset(chosen):
            return []
        return chosen

    @staticmethod
    def _normalise_items(raw_items: Sequence[Dict[str, Any]], max_level: int) -> List[SearchItem]:
        items: List[SearchItem] = []
        seen = set()
        for raw in raw_items:
            if is_removed(raw):
                continue
            try:
                item = normalize_item(raw, max_level)
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                logger.warning("[MODEL-BROWSER] Skipping a CivitAI model this browser cannot read (id %r): %s", raw.get("id"), exc)
                continue
            if item.id not in seen:
                seen.add(item.id)
                items.append(item)
        return items
