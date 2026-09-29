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

logger = logging.getLogger("ai_agent_loader.ui.image_generation")


def create_image_generation_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Image Generation", id="image_generation"):
        gr.Markdown("## Image Generation")

        # --- Model Controls ---
        with gr.Row():
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=[],
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        # --- Input ---
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful sunset over mountains, digital art, 4k...",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted...",
                    lines=2,
                )
            with gr.Column(scale=1):
                width = slider_with_manual_override(
                    "Width", minimum=256, maximum=2048, value=1024, step=64
                )
                height = slider_with_manual_override(
                    "Height", minimum=256, maximum=2048, value=1024, step=64
                )
                steps = slider_with_manual_override(
                    "Steps", minimum=1, maximum=100, value=30, step=1
                )
                cfg_scale = slider_with_manual_override(
                    "CFG Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5
                )
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                num_images = slider_with_manual_override(
                    "Number of Images", minimum=1, maximum=4, value=1, step=1
                )

        generate_btn = gr.Button("Generate", variant="primary")

        # --- Output ---
        output_gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
        output_meta = gr.Markdown("")

        # --- Event Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(
                cached,
                {"text-to-image"},
                cache_manager,
            )
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"text-to-image"}, cache_manager):
                return (
                    "**Status:** This model is not compatible with the Image Generation tab "
                    "(marked with ❌). Use a model whose pipeline tag is text-to-image."
                )
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "text-to-image")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def generate(prompt_text, neg_prompt, w, h, num_steps, cfg, s, n, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return [], "No model loaded."
            if not prompt_text.strip():
                return [], "Enter a prompt."

            inputs = {
                "prompt": prompt_text.strip(),
                "negative_prompt": neg_prompt.strip() if neg_prompt else "",
            }
            result = model_manager.run_inference(
                inputs,
                width=int(w),
                height=int(h),
                num_inference_steps=int(num_steps),
                guidance_scale=cfg,
                seed=int(s),
                num_images=int(n),
            )

            if result.success:
                meta = (
                    f"**Duration:** {result.metadata.get('duration_seconds', '?')}s | "
                    f"**Seed:** {result.metadata.get('seed', '?')}"
                )
                return result.output, meta
            return [], f"Error: {result.error}"

        # --- Wire Events ---
        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        generate_btn.click(
            fn=generate,
            inputs=[prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images],
            outputs=[output_gallery, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
