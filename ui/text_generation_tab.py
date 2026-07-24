import gradio as gr
import logging

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from ui.components import update_model_status

logger = logging.getLogger("ai_agent_loader.ui.text_generation")


def create_text_generation_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Text Generation", id="text_generation"):
        gr.Markdown("## Text Generation")

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

        # --- Input / Parameters ---
        with gr.Row():
            with gr.Column(scale=3):
                system_prompt = gr.Textbox(
                    label="System Prompt (optional)",
                    placeholder="You are a helpful assistant...",
                    lines=2,
                )
                user_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
            with gr.Column(scale=1):
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=32,
                    maximum=4096,
                    value=512,
                    step=32,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.05,
                )
                top_p = gr.Slider(
                    label="Top-p",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                )
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=200,
                    value=50,
                    step=1,
                )
                rep_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                )

        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            clear_btn = gr.Button("Clear")

        # --- Output ---
        output_text = gr.Textbox(
            label="Output",
            lines=12,
            interactive=False,
        )
        output_meta = gr.Markdown("")

        # --- Event Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            return gr.update(choices=cached, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            try:
                progress(0, desc=f"Loading {model_id}...")
                # Detect pipeline tag
                try:
                    detail = hub_client.get_model_detail(model_id)
                    tag = detail.pipeline_tag or "text-generation"
                except Exception:
                    tag = "text-generation"

                model_manager.load_model(model_id, tag)
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def generate(
            system, prompt, max_tok, temp, tp, tk, rep_pen,
            progress: gr.Progress = gr.Progress(track_tqdm=True),
        ):
            if not model_manager.active_model_id:
                return "No model loaded. Load a model first.", ""

            if not prompt.strip():
                return "Enter a prompt.", ""

            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system.strip()})
            messages.append({"role": "user", "content": prompt.strip()})

            result = model_manager.run_inference(
                messages,
                max_new_tokens=int(max_tok),
                temperature=temp,
                top_p=tp,
                top_k=int(tk),
                repetition_penalty=rep_pen,
            )

            if result.success:
                meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                return result.output, meta
            else:
                return f"Error: {result.error}", ""

        def clear_all():
            return "", "", ""

        # --- Wire Events ---
        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        generate_btn.click(
            fn=generate,
            inputs=[
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                rep_penalty,
            ],
            outputs=[output_text, output_meta],
            concurrency_id="model_ops",
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[user_prompt, output_text, output_meta],
        )

        return [model_selector]
