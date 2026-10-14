from types import SimpleNamespace

import pytest

import src.plugin_api
from backend.setup_checks import CivitaiProviderReady, HuggingFaceProviderReady
from src.plugin_api import SetupCheck, SetupCheckContext, SetupCheckResult

CONTEXT = SetupCheckContext(plugin_id="model-browser", settings={"nsfw_ceiling": "inherit"})


class _Ids(set):
    asked: list


@pytest.fixture
def ready(monkeypatch):
    ids = _Ids()
    ids.asked = []

    class Registry:
        def is_provider_initialized(self, provider_id):
            ids.asked.append(provider_id)
            return provider_id in ids

    monkeypatch.setattr(src.plugin_api, "get_provider_registry", lambda: Registry())
    return ids


def test_the_checks_are_setup_checks():
    assert issubclass(CivitaiProviderReady, SetupCheck)
    assert issubclass(HuggingFaceProviderReady, SetupCheck)


def test_done_when_the_civitai_provider_is_initialised(ready):
    ready.add("civitai")

    assert CivitaiProviderReady().evaluate(CONTEXT) == SetupCheckResult(
        done=True, description="The CivitAI provider is enabled."
    )
    assert ready.asked == ["civitai"]


def test_done_when_the_hugging_face_provider_is_initialised(ready):
    ready.add("huggingface")

    assert HuggingFaceProviderReady().evaluate(CONTEXT) == SetupCheckResult(
        done=True, description="The Hugging Face provider is enabled."
    )
    assert ready.asked == ["huggingface"]


def test_civitai_not_initialised_points_at_its_provider_plugin(ready):
    ready.add("huggingface")

    assert CivitaiProviderReady().evaluate(CONTEXT) == SetupCheckResult(
        done=False,
        description=(
            "Enable and configure the CivitAI provider plugin (API key) so downloads that need a login work; "
            "browsing works without it."
        ),
        action_label="Open the CivitAI provider",
        action_href="/admin?tab=plugins&id=civitai-provider",
    )


def test_hugging_face_not_initialised_points_at_its_provider_plugin(ready):
    ready.add("civitai")

    assert HuggingFaceProviderReady().evaluate(CONTEXT) == SetupCheckResult(
        done=False,
        description=(
            "Enable and configure the Hugging Face provider plugin (token) so downloads that need a login work; "
            "browsing works without it."
        ),
        action_label="Open the Hugging Face provider",
        action_href="/admin?tab=plugins&id=huggingface-provider",
    )


@pytest.mark.parametrize("check, name, plugin_id", [
    (CivitaiProviderReady, "CivitAI", "civitai-provider"),
    (HuggingFaceProviderReady, "Hugging Face", "huggingface-provider"),
])
@pytest.mark.parametrize("where", ["accessor", "lookup"])
def test_a_registry_error_is_a_step_still_to_do(monkeypatch, caplog, check, name, plugin_id, where):
    def fail(*args):
        raise RuntimeError("registry exploded")

    registry = SimpleNamespace(is_provider_initialized=fail)
    monkeypatch.setattr(src.plugin_api, "get_provider_registry", fail if where == "accessor" else lambda: registry)

    result = check().evaluate(CONTEXT)

    assert result == SetupCheckResult(
        done=False,
        description=f"Could not check the {name} provider; see the server log.",
        action_label=f"Open the {name} provider",
        action_href=f"/admin?tab=plugins&id={plugin_id}",
    )
    assert "registry exploded" in caplog.text


@pytest.mark.parametrize("answer", [None, 0, ""])
def test_a_falsy_registry_answer_is_not_done(monkeypatch, answer):
    registry = SimpleNamespace(is_provider_initialized=lambda provider_id: answer)
    monkeypatch.setattr(src.plugin_api, "get_provider_registry", lambda: registry)

    assert CivitaiProviderReady().evaluate(CONTEXT).done is False


def test_the_plugin_settings_do_not_matter(ready):
    ready.add("civitai")
    empty = SetupCheckContext(plugin_id="model-browser")

    assert CivitaiProviderReady().evaluate(empty).done is True
