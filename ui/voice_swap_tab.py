import logging

import gradio as gr

from core.cache_manager import CacheManager
from core.hub_client import HubClient
from core.model_manager import ModelManager
from ui.components import build_compatible_dropdown_choices, is_model_compatible, update_model_status

logger = logging.getLogger("ai_agent_loader.ui.voice_swap")


def create_voice_swap_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Voice Swap", id="voice_swap"):
        gr.Markdown("## Voice Swap")
        gr.Markdown(
            "Convert an existing recording to sound like a different speaker, using a few "
            "reference clips of the target voice — the original words, timing, and delivery "
            "are kept, only the voice's tone/timbre changes.\n\n"
            "This is symmetric: to go the other way (put someone else's words in *your* "
            "voice, or vice versa), just swap which clip you put in **Source Audio** and "
            "which you put in **Target Voice Reference**."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Voice-Swap Model", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            with gr.Column():
                source_audio = gr.Audio(
                    label="Source Audio (the recording to convert — its words/timing are kept)",
                    type="filepath",
                )
            with gr.Column():
                target_refs = gr.File(
                    label="Target Voice Reference Clip(s) (a few short samples of the voice to apply)",
                    file_count="multiple",
                    file_types=["audio"],
                )

        tau = gr.Slider(
            label="Conversion Strength (tau)",
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.05,
        )

        convert_btn = gr.Button("Convert Voice", variant="primary")
        output_audio = gr.Audio(label="Converted Audio")
        output_meta = gr.Markdown("")

        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, {"instant-voice-cloning"}, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"instant-voice-cloning"}, cache_manager):
                if is_model_compatible(model_id, {"orpheus"}, cache_manager):
                    return (
                        "**Status:** This is an Orpheus-style voice-cloning **TTS** model, not an "
                        "audio-to-audio voice-swap model — it can only generate new speech from "
                        "typed text, not convert an existing recording. Use the **Voice Clone TTS** "
                        "tab instead."
                    )
                return "**Status:** This model is not a recognized voice-swap model (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "voice-swap")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model():
            model_manager.unload_current()
            return update_model_status(model_manager)

        def convert(src_path, ref_files, tau_value, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not src_path:
                return None, "Upload the source audio first."
            if not ref_files:
                return None, "Upload at least one target voice reference clip."

            ref_paths = [f.name if hasattr(f, "name") else f for f in ref_files]

            progress(0.3, desc="Converting voice...")
            result = model_manager.run_inference(
                {"source_audio": src_path, "target_audio": ref_paths},
                tau=float(tau_value),
            )
            if result.success:
                meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                return result.output, meta
            return None, f"Error: {result.error}"

        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        convert_btn.click(
            fn=convert,
            inputs=[source_audio, target_refs, tau],
            outputs=[output_audio, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
