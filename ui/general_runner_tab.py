import gradio as gr
import logging

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from ui.components import (
    build_compatible_dropdown_choices,
    is_model_compatible,
    slider_with_manual_override,
    update_model_status,
)

logger = logging.getLogger("ai_agent_loader.ui.general_runner")


def create_general_runner_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Summarize / Translate", id="general_runner"):
        gr.Markdown("## Summarization & Translation")

        with gr.Tabs():
            # --- Summarization ---
            with gr.Tab("Summarization"):
                with gr.Row():
                    sum_model = gr.Dropdown(
                        label="Select Model", choices=[], interactive=True, scale=3
                    )
                    sum_refresh = gr.Button("Refresh", scale=1)
                    sum_load = gr.Button("Load", variant="primary", scale=1)
                    sum_unload = gr.Button("Unload", variant="stop", scale=1)

                sum_status = gr.Markdown("**Status:** No model loaded")
                sum_input = gr.Textbox(
                    label="Text to Summarize", lines=8,
                    placeholder="Paste a long article or text here...",
                )
                with gr.Row():
                    sum_max_len = slider_with_manual_override(
                        "Max Length", minimum=30, maximum=500, value=150, step=10
                    )
                    sum_min_len = slider_with_manual_override(
                        "Min Length", minimum=10, maximum=200, value=30, step=10
                    )
                sum_btn = gr.Button("Summarize", variant="primary")
                sum_output = gr.Textbox(label="Summary", lines=4, interactive=False)
                sum_meta = gr.Markdown("")

            # --- Translation ---
            with gr.Tab("Translation"):
                with gr.Row():
                    tr_model = gr.Dropdown(
                        label="Select Model", choices=[], interactive=True, scale=3
                    )
                    tr_refresh = gr.Button("Refresh", scale=1)
                    tr_load = gr.Button("Load", variant="primary", scale=1)
                    tr_unload = gr.Button("Unload", variant="stop", scale=1)

                tr_status = gr.Markdown("**Status:** No model loaded")
                tr_input = gr.Textbox(
                    label="Text to Translate", lines=4,
                    placeholder="Enter text to translate...",
                )
                tr_btn = gr.Button("Translate", variant="primary")
                tr_output = gr.Textbox(label="Translation", lines=4, interactive=False)
                tr_meta = gr.Markdown("")

        # --- Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            sum_choices = build_compatible_dropdown_choices(cached, {"summarization"}, cache_manager)
            tr_choices = build_compatible_dropdown_choices(cached, {"translation"}, cache_manager)
            return gr.update(choices=sum_choices, value=None), gr.update(choices=tr_choices, value=None)

        def load_sum_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"summarization"}, cache_manager):
                return "**Status:** This model is not compatible with the Summarization tab (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "summarization")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def load_tr_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"translation"}, cache_manager):
                return "**Status:** This model is not compatible with the Translation tab (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "translation")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def summarize(text, max_len, min_len, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return "No model loaded.", ""
            if not text.strip():
                return "Enter text.", ""
            result = model_manager.run_inference(
                text.strip(), max_length=int(max_len), min_length=int(min_len)
            )
            if result.success:
                return result.output, f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
            return f"Error: {result.error}", ""

        def translate(text, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return "No model loaded.", ""
            if not text.strip():
                return "Enter text.", ""
            result = model_manager.run_inference(text.strip())
            if result.success:
                return result.output, f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
            return f"Error: {result.error}", ""

        # --- Wire ---
        sum_refresh.click(fn=refresh_models, outputs=[sum_model, tr_model])
        sum_load.click(fn=load_sum_model, inputs=[sum_model], outputs=[sum_status], concurrency_id="model_ops")
        sum_unload.click(fn=unload, outputs=[sum_status], concurrency_id="model_ops")
        sum_btn.click(
            fn=summarize, inputs=[sum_input, sum_max_len, sum_min_len],
            outputs=[sum_output, sum_meta],
            concurrency_id="model_ops",
        )

        tr_refresh.click(fn=refresh_models, outputs=[sum_model, tr_model])
        tr_load.click(fn=load_tr_model, inputs=[tr_model], outputs=[tr_status], concurrency_id="model_ops")
        tr_unload.click(fn=unload, outputs=[tr_status], concurrency_id="model_ops")
        tr_btn.click(fn=translate, inputs=[tr_input], outputs=[tr_output, tr_meta], concurrency_id="model_ops")

        return [sum_model, tr_model]
