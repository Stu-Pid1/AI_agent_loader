import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.object_detection")


class ObjectDetectionRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Object Detection",
            supported_pipeline_tags=["object-detection"],
            supported_libraries=["transformers"],
            description="Detect objects in images with bounding boxes and labels.",
            input_description="Image file",
            output_description="Detected objects with bounding boxes, labels, and scores",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            device = 0 if self._device == "cuda" else -1
            self._pipeline = hf_pipeline(
                "object-detection",
                model=model_id,
                device=device,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded object detection model: {model_id}")

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
            threshold = kwargs.get("threshold", 0.5)
            result = self._pipeline(inputs, threshold=threshold)
            elapsed = time.time() - start_time

            detections = []
            for det in result:
                detections.append({
                    "label": det["label"],
                    "score": round(float(det["score"]), 4),
                    "box": det["box"],
                })

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=detections,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "num_detections": len(detections),
                    "model_id": self._model_id,
                },
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {"threshold": 0.5}

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "pipeline_tag", None) == "object-detection"
        except Exception:
            return False
