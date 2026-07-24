import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_to_speech")


class TextToSpeechRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text to Speech",
            supported_pipeline_tags=["text-to-speech", "text-to-audio"],
            supported_libraries=["transformers"],
            description="Convert text to speech audio using TTS models.",
            input_description="Text to speak",
            output_description="Audio waveform",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            device = 0 if self._device == "cuda" else -1
            self._pipeline = hf_pipeline(
                "text-to-speech",
                model=model_id,
                device=device,
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

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            start_time = time.time()
            result = self._pipeline(inputs)
            elapsed = time.time() - start_time

            audio = result.get("audio", None) if isinstance(result, dict) else None
            sampling_rate = (
                result.get("sampling_rate", 16000) if isinstance(result, dict) else 16000
            )

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output={"audio": audio, "sampling_rate": sampling_rate},
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "sampling_rate": sampling_rate,
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

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
