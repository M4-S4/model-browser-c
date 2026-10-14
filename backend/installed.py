import asyncio
import logging
from typing import Dict, Iterable, List, Sequence

from . import core_bridge
from .permissions import Principal
from .schema import SearchItem

logger = logging.getLogger("plugins.model-browser")


def _normal(sha256: str) -> str:
    return sha256.strip().lower()


def _lookup(sha256s: List[str], user_id: str, admin: bool) -> Dict[str, bool]:
    visible = None if admin else set(core_bridge.visible_model_ids(user_id))
    result: Dict[str, bool] = {}
    for sha256 in sha256s:
        model = core_bridge.find_model_by_sha256(sha256)
        result[sha256] = (
            model is not None
            and model.get("is_available", True) is not False
            and (visible is None or model.get("id") in visible)
        )
    return result


async def installed_map(sha256s: Iterable[str], principal: Principal) -> Dict[str, bool]:
    hashes = list(dict.fromkeys(_normal(value) for value in sha256s if value and value.strip()))
    if not hashes:
        return {}
    return await asyncio.to_thread(_lookup, hashes, principal.user.id, principal.is_admin)


async def mark_installed(items: Sequence[SearchItem], principal: Principal) -> None:
    hashes = [item.primary_sha256 for item in items if item.primary_sha256]
    if not hashes:
        return
    try:
        found = await installed_map(hashes, principal)
    except Exception:
        logger.debug("[MODEL-BROWSER] Could not tell which results are installed: leaving them unmarked", exc_info=True)
        return
    for item in items:
        if item.primary_sha256:
            item.installed = found.get(_normal(item.primary_sha256))
