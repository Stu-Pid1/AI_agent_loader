import gradio as gr
import logging

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from ui.components import build_compatible_dropdown_choices, is_model_compatible, update_model_status

logger = logging.getLogger("ai_agent_loader.ui.video_generation")


def create_video_generation_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Video Generation", id="video_generation"):
        gr.Markdown("## Video Generation")

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

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A cinematic drone shot of neon-lit city streets at night, ultra detailed, smooth motion...",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, distorted, low quality, flicker...",
                    lines=2,
                )
            with gr.Column(scale=1):
                width = gr.Slider(
                    label="Width", minimum=256, maximum=1024, value=512, step=64
                )
                height = gr.Slider(
                    label="Height", minimum=256, maximum=1024, value=512, step=64
                )
                num_frames = gr.Slider(
                    label="Frames", minimum=8, maximum=64, value=16, step=1
                )
                steps = gr.Slider(
                    label="Inference Steps", minimum=1, maximum=100, value=25, step=1
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5
                )
                fps = gr.Slider(
                    label="FPS", minimum=4, maximum=24, value=8, step=1
                )
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

        generate_btn = gr.Button("Generate Video", variant="primary")

        output_video = gr.Video(label="Generated Video", autoplay=True, height=480)
        output_meta = gr.Markdown("")

        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, {"text-to-video"}, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"text-to-video"}, cache_manager):
                return (
                    "**Status:** This model is not compatible with the Video Generation tab "
                    "(marked with ❌). Use a text-to-video model instead."
                )
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "text-to-video")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def generate(
            prompt_text,
            neg_prompt,
            w,
            h,
            frames,
            num_steps,
            cfg,
            fps_value,
            s,
            progress: gr.Progress = gr.Progress(track_tqdm=True),
        ):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not prompt_text.strip():
                return None, "Enter a prompt."

            inputs = {
                "prompt": prompt_text.strip(),
                "negative_prompt": neg_prompt.strip() if neg_prompt else "",
            }
            result = model_manager.run_inference(
                inputs,
                width=int(w),
                height=int(h),
                num_frames=int(frames),
                num_inference_steps=int(num_steps),
                guidance_scale=float(cfg),
                fps=int(fps_value),
                seed=int(s),
            )

            if result.success:
                meta = (
                    f"**Duration:** {result.metadata.get('duration_seconds', '?')}s | "
                    f"**Frames:** {result.metadata.get('frames', '?')} | "
                    f"**FPS:** {result.metadata.get('fps', '?')}"
                )
                return result.output, meta
            return None, f"Error: {result.error}"

        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        generate_btn.click(
            fn=generate,
            inputs=[prompt, negative_prompt, width, height, num_frames, steps, cfg_scale, fps, seed],
            outputs=[output_video, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
