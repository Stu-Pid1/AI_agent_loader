import time
import logging
from typing import Any, Dict

import torch
from transformers import pipeline as hf_pipeline

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.token_classification")


class TokenClassificationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Token Classification",
            supported_pipeline_tags=["token-classification", "ner"],
            supported_libraries=["transformers"],
            description="Named Entity Recognition (NER), POS tagging, and other "
            "token-level classification tasks.",
            input_description="Text to analyze",
            output_description="Entities with labels, scores, and positions",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            device = 0 if self._device == "cuda" else -1
            self._pipeline = hf_pipeline(
                "token-classification",
                model=model_id,
                device=device,
                aggregation_strategy="simple",
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded token classification model: {model_id}")

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

            # Convert to serializable format
            entities = []
            for ent in result:
                entities.append({
                    "entity": ent.get("entity_group", ent.get("entity", "")),
                    "word": ent.get("word", ""),
                    "score": round(float(ent.get("score", 0)), 4),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                })

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=entities,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "num_entities": len(entities),
                    "model_id": self._model_id,
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
            return tag in ("token-classification", "ner")
        except Exception:
            return False
