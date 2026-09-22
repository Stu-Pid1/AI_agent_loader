import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from config.settings import Settings
from core.device import DeviceManager
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.summarization")


class SummarizationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Summarization",
            supported_pipeline_tags=["summarization"],
            supported_libraries=["transformers"],
            description="Summarize long text into shorter form.",
            input_description="Text to summarize",
            output_description="Summarized text",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            pipeline_kwargs = {
                "model": model_id,
            }
            if self._device == "cuda" and DeviceManager.use_distributed_device_map():
                pipeline_kwargs["device_map"] = "auto"
            elif self._device == "cuda":
                pipeline_kwargs["device"] = 0
            else:
                pipeline_kwargs["device"] = -1

            self._pipeline = hf_pipeline(
                "summarization",
                **pipeline_kwargs,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded summarization model: {model_id}")

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

            result = self._pipeline(
                inputs,
                max_length=int(params["max_length"]),
                min_length=int(params["min_length"]),
                do_sample=False,
            )
            elapsed = time.time() - start_time

            summary = result[0]["summary_text"] if result else ""

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=summary,
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
            "max_length": 150,
            "min_length": 30,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "pipeline_tag", None) == "summarization"
        except Exception:
            return False
