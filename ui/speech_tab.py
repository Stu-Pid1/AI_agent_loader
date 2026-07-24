import gradio as gr
import numpy as np
import logging

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from ui.components import update_model_status

logger = logging.getLogger("ai_agent_loader.ui.speech")


def create_speech_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
):
    with gr.Tab("Speech", id="speech"):
        gr.Markdown("## Speech Processing")

        with gr.Tabs():
            # --- Speech to Text ---
            with gr.Tab("Speech to Text"):
                with gr.Row():
                    stt_model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    stt_refresh_btn = gr.Button("Refresh", scale=1)
                    stt_load_btn = gr.Button("Load", variant="primary", scale=1)
                    stt_unload_btn = gr.Button("Unload", variant="stop", scale=1)

                stt_status = gr.Markdown("**Status:** No model loaded")

                with gr.Row():
                    audio_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        scale=2,
                    )
                    audio_mic = gr.Audio(
                        label="Record from Microphone",
                        sources=["microphone"],
                        type="filepath",
                        scale=2,
                    )

                stt_btn = gr.Button("Transcribe", variant="primary")
                stt_output = gr.Textbox(label="Transcription", lines=6, interactive=False)
                stt_meta = gr.Markdown("")

            # --- Text to Speech ---
            with gr.Tab("Text to Speech"):
                with gr.Row():
                    tts_model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    tts_refresh_btn = gr.Button("Refresh", scale=1)
                    tts_load_btn = gr.Button("Load", variant="primary", scale=1)
                    tts_unload_btn = gr.Button("Unload", variant="stop", scale=1)

                tts_status = gr.Markdown("**Status:** No model loaded")

                tts_input = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter text to convert to speech...",
                    lines=4,
                )
                tts_btn = gr.Button("Synthesize", variant="primary")
                tts_output = gr.Audio(label="Generated Audio")
                tts_meta = gr.Markdown("")

        # --- Event Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            return gr.update(choices=cached, value=None)

        def load_stt(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "automatic-speech-recognition")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def load_tts(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "text-to-speech")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def transcribe(audio_file, mic_file, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return "No model loaded.", ""

            audio = audio_file or mic_file
            if not audio:
                return "Upload or record audio first.", ""

            result = model_manager.run_inference(audio)
            if result.success:
                meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                return result.output, meta
            return f"Error: {result.error}", ""

        def synthesize(text, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not text.strip():
                return None, "Enter text."

            result = model_manager.run_inference(text.strip())
            if result.success:
                audio_data = result.output
                sr = audio_data.get("sampling_rate", 16000)
                waveform = audio_data.get("audio")
                if waveform is not None:
                    if isinstance(waveform, np.ndarray):
                        audio_tuple = (sr, waveform)
                    else:
                        audio_tuple = (sr, np.array(waveform))
                    meta = f"**Duration:** {result.metadata.get('duration_seconds', '?')}s"
                    return audio_tuple, meta
                return None, "No audio generated."
            return None, f"Error: {result.error}"

        # --- Wire Events ---
        stt_refresh_btn.click(fn=refresh_models, outputs=[stt_model_selector])
        stt_load_btn.click(fn=load_stt, inputs=[stt_model_selector], outputs=[stt_status], concurrency_id="model_ops")
        stt_unload_btn.click(fn=unload, outputs=[stt_status], concurrency_id="model_ops")
        stt_btn.click(
            fn=transcribe,
            inputs=[audio_upload, audio_mic],
            outputs=[stt_output, stt_meta],
            concurrency_id="model_ops",
        )

        tts_refresh_btn.click(fn=refresh_models, outputs=[tts_model_selector])
        tts_load_btn.click(fn=load_tts, inputs=[tts_model_selector], outputs=[tts_status], concurrency_id="model_ops")
        tts_unload_btn.click(fn=unload, outputs=[tts_status], concurrency_id="model_ops")
        tts_btn.click(
            fn=synthesize,
            inputs=[tts_input],
            outputs=[tts_output, tts_meta],
            concurrency_id="model_ops",
        )

        return [stt_model_selector, tts_model_selector]
