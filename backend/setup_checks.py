import logging

from src.plugin_api import SetupCheck, SetupCheckContext, SetupCheckResult

logger = logging.getLogger("plugins.model-browser")


class _ProviderReady(SetupCheck):
    provider_id = ""
    plugin_id = ""
    name = ""
    credential = ""

    def evaluate(self, context: SetupCheckContext) -> SetupCheckResult:
        action_label = f"Open the {self.name} provider"
        action_href = f"/admin?tab=plugins&id={self.plugin_id}"
        try:
            from src.plugin_api import get_provider_registry

            ready = bool(get_provider_registry().is_provider_initialized(self.provider_id))
        except Exception as exc:
            logger.warning("[MODEL-BROWSER] Setup check: could not read the provider registry: %s", exc)
            return SetupCheckResult(
                done=False,
                description=f"Could not check the {self.name} provider; see the server log.",
                action_label=action_label,
                action_href=action_href,
            )
        if ready:
            return SetupCheckResult(done=True, description=f"The {self.name} provider is enabled.")
        return SetupCheckResult(
            done=False,
            description=(
                f"Enable and configure the {self.name} provider plugin ({self.credential}) so downloads "
                "that need a login work; browsing works without it."
            ),
            action_label=action_label,
            action_href=action_href,
        )


class CivitaiProviderReady(_ProviderReady):
    provider_id = "civitai"
    plugin_id = "civitai-provider"
    name = "CivitAI"
    credential = "API key"


class HuggingFaceProviderReady(_ProviderReady):
    provider_id = "huggingface"
    plugin_id = "huggingface-provider"
    name = "Hugging Face"
    credential = "token"
