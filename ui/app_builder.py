import gradio as gr

from config.settings import Settings
from config.task_registry import TaskRegistry
from core.hub_client import HubClient
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from ui.hub_browser_tab import create_hub_browser_tab
from ui.text_generation_tab import create_text_generation_tab
from ui.image_generation_tab import create_image_generation_tab
from ui.classification_tab import create_classification_tab
from ui.speech_tab import create_speech_tab
from ui.object_detection_tab import create_object_detection_tab
from ui.general_runner_tab import create_general_runner_tab
from ui.settings_tab import create_settings_tab


def build_app() -> gr.Blocks:
    registry = TaskRegistry()
    hub_client = HubClient()
    cache_manager = CacheManager()
    model_manager = ModelManager(registry)

    with gr.Blocks(title=Settings.APP_TITLE) as app:
        gr.Markdown(f"# {Settings.APP_TITLE}")
        gr.Markdown(
            "Browse, download, and run AI models from Hugging Face Hub locally."
        )

        create_hub_browser_tab(hub_client, cache_manager, model_manager)
        dropdowns = create_text_generation_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_image_generation_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_classification_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_speech_tab(model_manager, cache_manager)
        dropdowns += create_object_detection_tab(model_manager, cache_manager)
        dropdowns += create_general_runner_tab(model_manager, cache_manager, hub_client)
        create_settings_tab(model_manager, cache_manager)

        # Populate all model dropdowns when the page first loads
        def refresh_all_dropdowns():
            cached = cache_manager.get_cached_model_ids()
            return [gr.update(choices=cached) for _ in dropdowns]

        app.load(fn=refresh_all_dropdowns, outputs=dropdowns)

    return app
