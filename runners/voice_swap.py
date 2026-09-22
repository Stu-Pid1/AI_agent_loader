import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from config.settings import Settings
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.voice_swap")


def _stub_wavmark_if_missing() -> None:
    """openvoice_cli's ToneColorConverter unconditionally imports wavmark and
    invokes its watermarking model inside convert() unless disabled — but its
    own constructor forwards **kwargs (including "enable_watermark") straight
    to the base class, which rejects unknown kwargs, so it can't actually be
    disabled through the public API. We don't need audio watermarking here,
    so install a harmless stand-in only if the real package isn't already
    installed, letting convert() run without an extra dependency or a
    watermark-model download.
    """
    try:
        import wavmark  # noqa: F401

        return
    except ImportError:
        pass

    import sys
    import types

    class _NoOpWatermarker:
        def to(self, *_args, **_kwargs):
            return self

        def encode(self, signal, _message):
            return signal.clone()

    stub = types.ModuleType("wavmark")
    stub.load_model = lambda: _NoOpWatermarker()
    sys.modules["wavmark"] = stub


class VoiceSwapRunner(BaseRunner):
    """Audio-to-audio voice conversion using OpenVoice's tone-color converter.

    Unlike text-to-speech, this takes an existing recording and re-renders it
    in a different speaker's voice/timbre, keeping the original words, timing,
    and prosody — a "face swap" for voices. It's symmetric: swapping which
    clip is the source and which is the target reverses the direction.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._converter = None

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Voice Swap",
            supported_pipeline_tags=["voice-swap"],
            supported_libraries=["openvoice"],
            description="Convert an existing recording to sound like a different speaker, "
            "using a few reference clips of the target voice.",
            input_description="Source audio (the recording to convert) + target voice reference clip(s)",
            output_description="Source audio re-rendered in the target voice",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            from core.cache_manager import CacheManager

            cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
            local_dir = cache_manager.resolve_local_model_path(model_id)
            if not local_dir:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser tab first."
                )

            config_path = Path(local_dir) / "converter" / "config.json"
            checkpoint_path = Path(local_dir) / "converter" / "checkpoint.pth"
            if not config_path.exists() or not checkpoint_path.exists():
                raise ModelLoadError(
                    f"'{model_id}' is cached but is missing converter/config.json or "
                    f"converter/checkpoint.pth — the download may be incomplete."
                )

            _stub_wavmark_if_missing()
            from openvoice_cli.api import ToneColorConverter

            device = self._device if self._device == "cuda" else "cpu"
            self._converter = ToneColorConverter(str(config_path), device=device)
            self._converter.load_ckpt(str(checkpoint_path))

            self._pipeline = self._converter
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded voice-swap converter: {model_id}")

        except ModelLoadError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded() or self._converter is None:
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            source_audio = inputs["source_audio"]
            target_audio = inputs["target_audio"]
            if isinstance(target_audio, str):
                target_audio = [target_audio]

            params = {**self.get_default_parameters(), **kwargs}

            start_time = time.time()
            src_se = self._converter.extract_se(source_audio)
            tgt_se = self._converter.extract_se(target_audio)

            out_path = str(Path(tempfile.gettempdir()) / f"voiceswap_{uuid.uuid4().hex}.wav")
            self._converter.convert(
                audio_src_path=source_audio,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=out_path,
                tau=float(params["tau"]),
            )
            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=out_path,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "tau": float(params["tau"]),
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def unload(self) -> None:
        self._converter = None
        super().unload()

    def get_default_parameters(self) -> Dict[str, Any]:
        return {"tau": 0.3}

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            tags = {str(t).lower() for t in (getattr(info, "tags", None) or [])}
            return "instant-voice-cloning" in tags
        except Exception:
            return False
