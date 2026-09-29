import logging

import gradio as gr

from core.cache_manager import CacheManager
from core.hub_client import HubClient
from core.model_manager import ModelManager
from ui.components import (
    build_compatible_dropdown_choices,
    is_model_compatible,
    slider_with_manual_override,
    update_model_status,
)

logger = logging.getLogger("ai_agent_loader.ui.orpheus_tts")


def create_orpheus_tts_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Voice Clone TTS", id="orpheus_tts"):
        gr.Markdown("## Voice Clone TTS")
        gr.Markdown(
            "Type new text and have it spoken in a cloned voice, using a short reference "
            "clip — for Orpheus-style discrete-audio-token models (e.g. Svara-TTS). "
            "These models also require the `hubertsiuzdak/snac_24khz` audio codec to be "
            "downloaded via the Hub Browser tab.\n\n"
            "This is the reverse direction of Voice Swap: there you convert an *existing* "
            "recording's voice, here you generate *brand-new* speech from typed text in a "
            "cloned voice."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Voice-Clone TTS Model", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter the text you want spoken in the cloned voice...",
                    lines=4,
                )
                reference_transcript = gr.Textbox(
                    label="Reference Transcript (optional)",
                    placeholder="What is actually said in the reference clip, if known — improves cloning quality.",
                    lines=2,
                )
            with gr.Column(scale=1):
                reference_audio = gr.Audio(
                    label="Reference Voice Clip (a short sample of the voice to clone)",
                    type="filepath",
                )

        with gr.Row():
            max_new_tokens = slider_with_manual_override("Max New Tokens", minimum=100, maximum=1048576, value=1200, step=100)
            temperature = slider_with_manual_override("Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.05)
            top_p = slider_with_manual_override("Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)

        generate_btn = gr.Button("Generate Speech", variant="primary")
        output_audio = gr.Audio(label="Generated Speech")
        output_meta = gr.Markdown("")

        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, {"orpheus"}, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"orpheus"}, cache_manager):
                if is_model_compatible(model_id, {"instant-voice-cloning"}, cache_manager):
                    return (
                        "**Status:** This is an audio-to-audio voice-swap model, not an Orpheus-style "
                        "TTS model — it converts existing recordings rather than generating new "
                        "speech from text. Use the **Voice Swap** tab instead."
                    )
                return "**Status:** This model is not a recognized Orpheus-style voice-clone model (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "orpheus-tts")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model():
            model_manager.unload_current()
            return update_model_status(model_manager)

        def generate(text, ref_audio, ref_transcript, max_tok, temp, tp, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not text or not text.strip():
                return None, "Enter some text to speak."
            if not ref_audio:
                return None, "Upload a reference voice clip first."

            progress(0.3, desc="Generating speech...")
            result = model_manager.run_inference(
                {
                    "text": text.strip(),
                    "reference_audio": ref_audio,
                    "reference_transcript": (ref_transcript or "").strip() or None,
                },
                max_new_tokens=int(max_tok),
                temperature=float(temp),
                top_p=float(tp),
            )
            if result.success:
                meta = (
                    f"**Duration:** {result.metadata.get('duration_seconds', '?')}s | "
                    f"**Audio length:** {result.metadata.get('audio_seconds', '?')}s"
                )
                return result.output, meta
            return None, f"Error: {result.error}"

        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        generate_btn.click(
            fn=generate,
            inputs=[text_input, reference_audio, reference_transcript, max_new_tokens, temperature, top_p],
            outputs=[output_audio, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
