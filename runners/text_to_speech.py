import time
import logging
from typing import Any, Dict, Optional

import torch
from transformers import pipeline as hf_pipeline

from config.settings import Settings
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_to_speech")

# Used only for optional voice-sample conditioning: extracts a speaker
# embedding from a reference clip for architectures that accept one (e.g.
# SpeechT5's `speaker_embeddings` forward param). Models that don't support
# per-utterance voice conditioning just ignore it — see run() below.
SPEAKER_ENCODER_MODEL_ID = "speechbrain/spkrec-xvect-voxceleb"


class TextToSpeechRunner(BaseRunner):

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._speaker_encoder = None

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text to Speech",
            supported_pipeline_tags=["text-to-speech", "text-to-audio"],
            supported_libraries=["transformers"],
            description="Convert text to speech audio using TTS models. Optionally clones a "
            "reference voice sample for architectures that support it (e.g. SpeechT5).",
            input_description="Text to speak, plus an optional reference voice clip",
            output_description="Audio waveform",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            pipeline_kwargs = {
                "model": model_id,
            }
            # TTS models are small enough to always fit on a single GPU, and
            # device_map="auto" has been observed to shard components like
            # SpeechT5's decoder/vocoder across multiple GPUs in a way that
            # breaks with cross-device tensor errors — pin to one GPU instead.
            if self._device == "cuda":
                pipeline_kwargs["device"] = 0
            else:
                pipeline_kwargs["device"] = -1

            self._pipeline = hf_pipeline(
                "text-to-speech",
                **pipeline_kwargs,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded text-to-speech model: {model_id}")

        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(f"Insufficient VRAM to load {model_id}.")
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def _get_speaker_encoder(self):
        if self._speaker_encoder is not None:
            return self._speaker_encoder

        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        local_dir = cache_manager.resolve_local_model_path(SPEAKER_ENCODER_MODEL_ID)
        if not local_dir:
            raise ModelLoadError(
                f"Voice-sample cloning needs '{SPEAKER_ENCODER_MODEL_ID}' to extract a speaker "
                f"embedding from your reference clip, and it's not present in the local cache "
                f"at {Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser tab first."
            )

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        # This model's hyperparams.yaml hardcodes `pretrained_path:
        # speechbrain/spkrec-xvect-voxceleb` and builds every checkpoint path
        # from it, ignoring the `source` we pass here — so without this
        # override it always tries to fetch from the Hub regardless of what
        # local directory we point at. Overriding the variable directly (a
        # standard HyperPyYAML mechanism) makes it resolve against our local
        # snapshot instead. local_strategy=COPY avoids symlink creation,
        # which requires elevated privileges on Windows.
        self._speaker_encoder = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            overrides={"pretrained_path": local_dir},
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cuda:0" if self._device == "cuda" else "cpu"},
        )
        return self._speaker_encoder

    def _extract_speaker_embedding(self, reference_audio: str):
        import librosa

        encoder = self._get_speaker_encoder()
        audio, _ = librosa.load(reference_audio, sr=16000, mono=True)
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            embedding = encoder.encode_batch(signal)
        return embedding.reshape(1, -1)

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            reference_audio: Optional[str] = kwargs.get("reference_audio")
            forward_params = {}
            voice_conditioning_note = None

            if reference_audio:
                try:
                    embedding = self._extract_speaker_embedding(reference_audio)
                    # With a multi-GPU device_map, the pipeline's input device
                    # may not match wherever the speaker encoder ran — move
                    # the embedding to match, or generation fails with a
                    # cross-device tensor error.
                    pipeline_device = getattr(self._pipeline, "device", None)
                    if pipeline_device is not None:
                        embedding = embedding.to(pipeline_device)
                    forward_params["speaker_embeddings"] = embedding
                except ModelLoadError:
                    raise
                except Exception as e:
                    voice_conditioning_note = f"Could not process the reference clip: {e}"

            start_time = time.time()
            voice_conditioning_used = False
            try:
                if forward_params:
                    result = self._pipeline(inputs, forward_params=forward_params)
                    voice_conditioning_used = True
                else:
                    result = self._pipeline(inputs)
            except TypeError:
                if not forward_params:
                    raise
                # This model's architecture doesn't accept speaker_embeddings —
                # only some (e.g. SpeechT5) support per-utterance voice
                # conditioning. Fall back to the model's default voice.
                logger.warning("Model does not support voice-sample conditioning; ignoring reference audio.")
                voice_conditioning_note = (
                    "This model doesn't support voice-sample conditioning (only some "
                    "architectures like SpeechT5 do) — generated with its default voice instead."
                )
                result = self._pipeline(inputs)
            elapsed = time.time() - start_time

            audio = result.get("audio", None) if isinstance(result, dict) else None
            sampling_rate = (
                result.get("sampling_rate", 16000) if isinstance(result, dict) else 16000
            )

            self._status = RunnerStatus.READY
            metadata = {
                "duration_seconds": round(elapsed, 2),
                "model_id": self._model_id,
                "sampling_rate": sampling_rate,
                "voice_conditioning_used": voice_conditioning_used,
            }
            if voice_conditioning_note:
                metadata["voice_conditioning_note"] = voice_conditioning_note

            return RunResult(
                success=True,
                output={"audio": audio, "sampling_rate": sampling_rate},
                metadata=metadata,
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def unload(self) -> None:
        self._speaker_encoder = None
        super().unload()

    def get_default_parameters(self) -> Dict[str, Any]:
        return {}

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            tag = getattr(info, "pipeline_tag", None)
            return tag in ("text-to-speech", "text-to-audio")
        except Exception:
            return False
