import logging

from src.plugin_api import HookContext

if __name__.count(".") >= 2:
    from ..backend import core_bridge, notifications
else:
    from backend import core_bridge, notifications

logger = logging.getLogger("plugins.model-browser")


def register_types(context: HookContext) -> HookContext:
    try:
        registry = context.data.get("registry") if isinstance(context.data, dict) else None
        if registry is None:
            logger.warning("[MODEL-BROWSER] notification_type.register carried no registry: no notification type declared")
            return context
        added = core_bridge.register_notification_types(registry, notifications.TYPES)
        logger.info("[MODEL-BROWSER] Notification types declared: %s", ", ".join(added) or "none new")
    except Exception:
        logger.exception("[MODEL-BROWSER] Could not declare the plugin's notification types")
    return context
