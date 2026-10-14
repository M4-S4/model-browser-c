from fastapi import APIRouter, Depends

from src.plugin_api import User, get_current_active_user

from . import PLUGIN_ID, PLUGIN_VERSION
from .routes_downloads import router as downloads_router
from .routes_permissions import router as permissions_router
from .routes_requests import router as requests_router
from .routes_sources import router as sources_router

router = APIRouter(prefix="/api/plugins/model-browser-c", tags=["model-browser"])


@router.get("/health")
async def health(current_user: User = Depends(get_current_active_user)):
    return {
        "success": True,
        "data": {"status": "ok", "plugin": PLUGIN_ID, "version": PLUGIN_VERSION},
    }


router.include_router(permissions_router)
router.include_router(downloads_router)
router.include_router(requests_router)
router.include_router(sources_router)
