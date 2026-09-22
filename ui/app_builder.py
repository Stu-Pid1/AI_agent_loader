import gradio as gr

from config.settings import Settings
from config.task_registry import TaskRegistry
from core.hub_client import HubClient
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from ui.hub_browser_tab import create_hub_browser_tab
from ui.text_generation_tab import create_text_generation_tab
from ui.image_generation_tab import create_image_generation_tab
from ui.video_generation_tab import create_video_generation_tab
from ui.face_swap_tab import create_face_swap_tab
from ui.classification_tab import create_classification_tab
from ui.speech_tab import create_speech_tab
from ui.voice_swap_tab import create_voice_swap_tab
from ui.orpheus_tts_tab import create_orpheus_tts_tab
from ui.object_detection_tab import create_object_detection_tab
from ui.general_runner_tab import create_general_runner_tab
from ui.settings_tab import create_settings_tab
from ui.download_queue_tab import create_download_queue_tab
from ui.agent_orchestrator_tab import create_agent_orchestrator_tab
from ui.components import build_compatible_dropdown_choices


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

        download_queue_state = gr.State([])

        create_hub_browser_tab(hub_client, cache_manager, model_manager, download_queue_state)
        create_download_queue_tab(download_queue_state, cache_manager, hub_client)
        dropdowns = create_text_generation_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_image_generation_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_video_generation_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_face_swap_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_classification_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_speech_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_voice_swap_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_orpheus_tts_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_object_detection_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_general_runner_tab(model_manager, cache_manager, hub_client)
        dropdowns += create_agent_orchestrator_tab(model_manager, cache_manager, hub_client)
        create_settings_tab(model_manager, cache_manager)

        # Populate all model dropdowns when the page first loads
        def refresh_all_dropdowns():
            cached = cache_manager.get_cached_model_ids()
            allowed_groups = [
                {"text-generation", "text2text-generation", "image-text-to-text", "video-text-to-text", "audio-text-to-text", "document-question-answering", "visual-question-answering"},
                {"text-to-image"},
                {"text-to-video"},
                {"face-swap", "head-swap", "body-swap"},
                {"text-classification"},
                {"image-classification"},
                {"token-classification", "named-entity-recognition"},
                {"automatic-speech-recognition"},
                {"text-to-speech"},
                {"instant-voice-cloning"},
                {"orpheus"},
                {"object-detection"},
                {"summarization"},
                {"translation"},
                {"text-generation", "text2text-generation", "image-text-to-text", "video-text-to-text", "audio-text-to-text", "document-question-answering", "visual-question-answering"},
                set(["text-generation", "summarization", "translation", "text-classification", "image-classification", "token-classification", "automatic-speech-recognition", "text-to-speech", "object-detection", "text-to-image", "text-to-video"]),
                set(["text-generation", "summarization", "translation", "text-classification", "image-classification", "token-classification", "automatic-speech-recognition", "text-to-speech", "object-detection", "text-to-image", "text-to-video"]),
                set(["text-generation", "summarization", "translation", "text-classification", "image-classification", "token-classification", "automatic-speech-recognition", "text-to-speech", "object-detection", "text-to-image", "text-to-video"]),
            ]
            return [
                gr.update(choices=build_compatible_dropdown_choices(cached, allowed_group, cache_manager))
                for allowed_group in allowed_groups
            ]

        app.load(fn=refresh_all_dropdowns, outputs=dropdowns)

    return app
