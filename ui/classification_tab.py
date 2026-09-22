import gradio as gr
import json
import logging

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from ui.components import build_compatible_dropdown_choices, is_model_compatible, update_model_status

logger = logging.getLogger("ai_agent_loader.ui.classification")


def create_classification_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Classification", id="classification"):
        gr.Markdown("## Text & Image Classification")

        with gr.Tabs():
            # --- Text Classification ---
            with gr.Tab("Text Classification"):
                with gr.Row():
                    text_model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    text_refresh_btn = gr.Button("Refresh", scale=1)
                    text_load_btn = gr.Button("Load", variant="primary", scale=1)
                    text_unload_btn = gr.Button("Unload", variant="stop", scale=1)

                text_status = gr.Markdown("**Status:** No model loaded")
                text_input = gr.Textbox(
                    label="Text to Classify",
                    placeholder="Enter text here...",
                    lines=4,
                )
                text_classify_btn = gr.Button("Classify", variant="primary")
                text_output = gr.JSON(label="Results")
                text_meta = gr.Markdown("")

            # --- Image Classification ---
            with gr.Tab("Image Classification"):
                with gr.Row():
                    img_model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    img_refresh_btn = gr.Button("Refresh", scale=1)
                    img_load_btn = gr.Button("Load", variant="primary", scale=1)
                    img_unload_btn = gr.Button("Unload", variant="stop", scale=1)

                img_status = gr.Markdown("**Status:** No model loaded")
                img_input = gr.Image(label="Image to Classify", type="filepath")
                img_classify_btn = gr.Button("Classify", variant="primary")
                img_output = gr.JSON(label="Results")
                img_meta = gr.Markdown("")

            # --- NER ---
            with gr.Tab("Named Entity Recognition"):
                with gr.Row():
                    ner_model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    ner_refresh_btn = gr.Button("Refresh", scale=1)
                    ner_load_btn = gr.Button("Load", variant="primary", scale=1)
                    ner_unload_btn = gr.Button("Unload", variant="stop", scale=1)

                ner_status = gr.Markdown("**Status:** No model loaded")
                ner_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="Apple Inc. was founded by Steve Jobs in Cupertino, California.",
                    lines=4,
                )
                ner_btn = gr.Button("Extract Entities", variant="primary")
                ner_output = gr.JSON(label="Entities")
                ner_meta = gr.Markdown("")

        # --- Shared Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            text_choices = build_compatible_dropdown_choices(cached, {"text-classification"}, cache_manager)
            img_choices = build_compatible_dropdown_choices(cached, {"image-classification"}, cache_manager)
            ner_choices = build_compatible_dropdown_choices(cached, {"token-classification", "named-entity-recognition"}, cache_manager)
            return (
                gr.update(choices=text_choices, value=None),
                gr.update(choices=img_choices, value=None),
                gr.update(choices=ner_choices, value=None),
            )

        def load_text_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"text-classification"}, cache_manager):
                return "**Status:** This model is not compatible with the Text Classification tab (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "text-classification")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def load_img_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"image-classification"}, cache_manager):
                return "**Status:** This model is not compatible with the Image Classification tab (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "image-classification")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def load_ner_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"token-classification", "named-entity-recognition"}, cache_manager):
                return "**Status:** This model is not compatible with the Named Entity Recognition tab (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "token-classification")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def classify_text(text, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not text.strip():
                return None, "Enter text."
            result = model_manager.run_inference(text.strip())
            if result.success:
                meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                return result.output, meta
            return None, f"Error: {result.error}"

        def classify_image(image_path, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not image_path:
                return None, "Upload an image."
            result = model_manager.run_inference(image_path)
            if result.success:
                meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                return result.output, meta
            return None, f"Error: {result.error}"

        def extract_entities(text, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not text.strip():
                return None, "Enter text."
            result = model_manager.run_inference(text.strip())
            if result.success:
                meta = (
                    f"**Duration:** {result.metadata.get('duration_seconds', '?')}s | "
                    f"**Entities found:** {result.metadata.get('num_entities', 0)}"
                )
                return result.output, meta
            return None, f"Error: {result.error}"

        # --- Wire Events ---
        text_refresh_btn.click(fn=refresh_models, outputs=[text_model_selector, img_model_selector, ner_model_selector])
        text_load_btn.click(fn=load_text_model, inputs=[text_model_selector], outputs=[text_status], concurrency_id="model_ops")
        text_unload_btn.click(fn=unload, outputs=[text_status], concurrency_id="model_ops")
        text_classify_btn.click(fn=classify_text, inputs=[text_input], outputs=[text_output, text_meta], concurrency_id="model_ops")

        img_refresh_btn.click(fn=refresh_models, outputs=[text_model_selector, img_model_selector, ner_model_selector])
        img_load_btn.click(fn=load_img_model, inputs=[img_model_selector], outputs=[img_status], concurrency_id="model_ops")
        img_unload_btn.click(fn=unload, outputs=[img_status], concurrency_id="model_ops")
        img_classify_btn.click(fn=classify_image, inputs=[img_input], outputs=[img_output, img_meta], concurrency_id="model_ops")

        ner_refresh_btn.click(fn=refresh_models, outputs=[text_model_selector, img_model_selector, ner_model_selector])
        ner_load_btn.click(fn=load_ner_model, inputs=[ner_model_selector], outputs=[ner_status], concurrency_id="model_ops")
        ner_unload_btn.click(fn=unload, outputs=[ner_status], concurrency_id="model_ops")
        ner_btn.click(fn=extract_entities, inputs=[ner_input], outputs=[ner_output, ner_meta], concurrency_id="model_ops")

        return [text_model_selector, img_model_selector, ner_model_selector]
