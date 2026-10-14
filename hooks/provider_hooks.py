import importlib
import logging

from src.plugin_api import HookContext

logger = logging.getLogger("plugins.model-browser")

PROVIDERS = (
    ("pending_provider", "PendingProvider"),
    ("openmodeldb_provider", "OpenModelDBProvider"),
    ("civarchive_provider", "CivArchiveProvider"),
)


def _provider_module(name: str):
    if __name__.count(".") >= 2:
        return importlib.import_module(f"..provider.{name}", __package__)
    return importlib.import_module(f"provider.{name}")


def register_providers(context: HookContext) -> HookContext:
    for module_name, class_name in PROVIDERS:
        try:
            module = _provider_module(module_name)
            providers = context.data.get("providers")
            if providers is None:
                providers = context.data["providers"] = {}
            providers[module.PROVIDER_ID] = getattr(module, class_name)
            logger.info("[MODEL-BROWSER] Registered the %s marketplace provider", module.PROVIDER_ID)
        except Exception:
            logger.exception("[MODEL-BROWSER] Could not register the %s marketplace provider", class_name)
    return context
