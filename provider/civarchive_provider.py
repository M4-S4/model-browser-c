import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from src.plugin_api import (
    MarketplaceProviderBase,
    ProviderCapability,
    ProviderConnectionError,
    ProviderMetadata,
    ProviderModelInfo,
)

from .pending_provider import DESCRIPTION_LIMIT, preview_policy, preview_urls

if __name__.count(".") >= 2:
    from ..backend import PLUGIN_VERSION, sources
    from ..backend import settings as plugin_settings
    from ..backend.http import SourceError, SourceNotFound
    from ..backend.security import html_to_text
    from ..backend.sources import civarchive
    from ..backend.type_mapping import civitai_mb_type
else:
    from backend import PLUGIN_VERSION, sources
    from backend import settings as plugin_settings
    from backend.http import SourceError, SourceNotFound
    from backend.security import html_to_text
    from backend.sources import civarchive
    from backend.type_mapping import civitai_mb_type

logger = logging.getLogger("plugins.model-browser")

PROVIDER_ID = "civarchive"
SETTING = "enable_civarchive"
MAX_MEDIA_URLS = 6

_SHA256 = re.compile(r"[0-9a-f]{64}")


class CivArchiveProvider(MarketplaceProviderBase):
    def __init__(self, source: Any = None) -> None:
        self._source = source

    @staticmethod
    def enabled() -> bool:
        try:
            return bool(plugin_settings.get_setting(SETTING))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[MODEL-BROWSER] Could not read %s: %s", SETTING, exc)
            return False

    def _archive(self) -> Any:
        if self._source is not None:
            return self._source
        try:
            return sources.get_source(civarchive.SOURCE_ID)
        except sources.UnknownSource:
            raise ProviderConnectionError("The CivArchive source of the Model Browser is not available") from None

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            id=PROVIDER_ID,
            name="CivArchive",
            description=(
                "Model details from civarchive.com, an unofficial community index of CivitAI, matched by "
                "file hash: description, tags, trigger words, safe previews and a link to the model's page "
                "(deleted CivitAI models included). Answers only while the Model Browser's CivArchive "
                "source is enabled."
            ),
            website=civarchive.SITE,
            capabilities=[
                ProviderCapability.HASH_LOOKUP,
                ProviderCapability.MODEL_INFO,
                ProviderCapability.MEDIA_DOWNLOAD,
            ],
            version=PLUGIN_VERSION,
        )

    async def initialize(self, settings: Dict[str, Any]) -> bool:
        return True

    async def shutdown(self) -> None:
        return None

    async def test_connection(self) -> bool:
        if not self.enabled():
            return False
        try:
            await self._archive().build_id()
        except Exception as exc:  # noqa: BLE001
            logger.info("[MODEL-BROWSER] CivArchive connection test failed: %s", exc)
            return False
        return True

    def get_settings_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def get_download_headers(self) -> Dict[str, str]:
        return {}

    def matches_download_url(self, url: str) -> bool:
        return False

    async def get_model_by_hash(self, sha256: str) -> Optional[ProviderModelInfo]:
        key = sha256.strip().lower() if isinstance(sha256, str) else ""
        if not _SHA256.fullmatch(key) or not self.enabled():
            return None
        try:
            page = await self._archive().hash_page(key)
        except SourceNotFound:
            return None
        except SourceError as exc:
            logger.warning("[MODEL-BROWSER] CivArchive unavailable for a hash lookup: %s", exc.message)
            raise ProviderConnectionError(f"Could not read CivArchive: {exc.message}") from exc
        raw = pick_model(page)
        if raw is None:
            return None
        max_level, hide_poi = preview_policy()
        try:
            return build_model_info(key, raw, page, max_level=max_level, hide_poi=hide_poi)
        except Exception:
            logger.exception("[MODEL-BROWSER] Could not build the CivArchive details of %s", key[:12])
            return None

    def get_model_page_url(
        self, provider_model_id: Optional[str], provider_version_id: Optional[str] = None
    ) -> Optional[str]:
        return model_page_url(provider_model_id, provider_version_id)


def pick_model(page: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    listed = page.get("models") if isinstance(page.get("models"), list) else []
    models = [model for model in listed if isinstance(model, dict) and isinstance(model.get("version"), dict)]
    on_civitai = [model for model in models if model.get("platform") == civarchive.PLATFORM]
    return (on_civitai or models or [None])[0]


def build_model_info(
    sha256: str, raw: Mapping[str, Any], page: Mapping[str, Any], *, max_level: int, hide_poi: bool
) -> ProviderModelInfo:
    detail = civarchive.normalize_detail(raw, max_level)
    version = detail.versions[0]
    raw_version = raw["version"]
    raw_files = raw_version.get("files") if isinstance(raw_version.get("files"), list) else []
    raw_file = next(
        (item for item in raw_files if isinstance(item, dict) and str(item.get("sha256") or "").lower() == sha256), None
    )
    file = next((item for item in version.files if item.sha256 == sha256), None)
    if file is not None:
        download_url: Optional[str] = file.download_url
    else:
        found = civarchive.download_for({"mirrors": page.get("files") or []}, served_by_civitai=False)
        download_url = found.url if found is not None else None
    file_nsfw = raw_file is not None and civarchive.level_nsfw(
        raw_file.get("nsfw_level"), raw_file.get("is_nsfw"), max_level
    )
    return ProviderModelInfo(
        provider_id=PROVIDER_ID,
        provider_model_id=f"{PROVIDER_ID}:{detail.id}",
        provider_version_id=version.id,
        name=detail.name,
        description=html_to_text(detail.description_html, limit=DESCRIPTION_LIMIT) or None,
        tags=list(detail.tags),
        trigger_words=list(version.trigger_words),
        nsfw=detail.nsfw or version.nsfw or file_nsfw,
        download_url=download_url,
        media_urls=preview_urls(version.images, max_level=max_level, hide_poi=hide_poi)[:MAX_MEDIA_URLS],
        model_type=civitai_mb_type(detail.type),
        base_model=version.base_model,
        created_at=_parse_datetime(version.created_at),
        extra_data={
            "source": civarchive.SOURCE_ID,
            "source_url": detail.source_url,
            "mb_base_model": version.base_model,
        },
    )


def model_page_url(provider_model_id: Any, provider_version_id: Any = None) -> Optional[str]:
    if not isinstance(provider_model_id, str):
        return None
    source, separator, model_id = provider_model_id.partition(":")
    if not separator or source != PROVIDER_ID:
        return None
    return civarchive.page_url_for(model_id, provider_version_id)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
