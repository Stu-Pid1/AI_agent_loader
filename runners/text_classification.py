import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from config.settings import Settings
from core.device import DeviceManager
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_classification")


class TextClassificationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text Classification",
            supported_pipeline_tags=[
                "text-classification",
                "sentiment-analysis",
                "zero-shot-classification",
            ],
            supported_libraries=["transformers"],
            description="Classify text into categories: sentiment analysis, "
            "topic classification, zero-shot classification.",
            input_description="Text to classify",
            output_description="Labels with confidence scores",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            task = kwargs.get("task", "text-classification")
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
                task,
                **pipeline_kwargs,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded text classification model: {model_id}")

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

            pipe_kwargs = {}
            if kwargs.get("candidate_labels"):
                pipe_kwargs["candidate_labels"] = kwargs["candidate_labels"]
            if kwargs.get("top_k"):
                pipe_kwargs["top_k"] = kwargs["top_k"]

            result = self._pipeline(inputs, **pipe_kwargs)
            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=result,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {"top_k": 5}

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            tag = getattr(info, "pipeline_tag", None)
            return tag in (
                "text-classification",
                "sentiment-analysis",
                "zero-shot-classification",
            )
        except Exception:
            return False
