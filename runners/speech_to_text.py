import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from config.settings import Settings
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.speech_to_text")


class SpeechToTextRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Speech to Text",
            supported_pipeline_tags=["automatic-speech-recognition"],
            supported_libraries=["transformers"],
            description="Transcribe audio to text using models like Whisper.",
            input_description="Audio file (WAV, MP3, etc.) or microphone recording",
            output_description="Transcribed text",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device_map="auto",
                torch_dtype=dtype,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded speech-to-text model: {model_id}")

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
            params = {**self.get_default_parameters(), **kwargs}
            start_time = time.time()

            pipe_kwargs = {}
            if params.get("language"):
                pipe_kwargs["generate_kwargs"] = {"language": params["language"]}
            if params.get("return_timestamps"):
                pipe_kwargs["return_timestamps"] = True

            result = self._pipeline(inputs, **pipe_kwargs)
            elapsed = time.time() - start_time

            text = result.get("text", "") if isinstance(result, dict) else str(result)

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=text.strip(),
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "chunks": result.get("chunks") if isinstance(result, dict) else None,
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "language": None,
            "return_timestamps": False,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "pipeline_tag", None) == "automatic-speech-recognition"
        except Exception:
            return False
