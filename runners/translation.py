import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.translation")


class TranslationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Translation",
            supported_pipeline_tags=["translation"],
            supported_libraries=["transformers"],
            description="Translate text between languages.",
            input_description="Text to translate",
            output_description="Translated text",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            device = 0 if self._device == "cuda" else -1
            self._pipeline = hf_pipeline(
                "translation",
                model=model_id,
                device=device,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded translation model: {model_id}")

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

            pipe_kwargs = {
                "max_length": int(params["max_length"]),
            }
            if params.get("src_lang"):
                pipe_kwargs["src_lang"] = params["src_lang"]
            if params.get("tgt_lang"):
                pipe_kwargs["tgt_lang"] = params["tgt_lang"]

            result = self._pipeline(inputs, **pipe_kwargs)
            elapsed = time.time() - start_time

            translation = result[0]["translation_text"] if result else ""

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=translation,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "max_length": 512,
            "src_lang": None,
            "tgt_lang": None,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "pipeline_tag", None) == "translation"
        except Exception:
            return False
