import time
import logging
from typing import Any, Dict, Optional

import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError, InferenceError

logger = logging.getLogger("ai_agent_loader.runners.text_generation")


class TextGenerationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text Generation",
            supported_pipeline_tags=["text-generation", "text2text-generation"],
            supported_libraries=["transformers"],
            description="Generate text using large language models (LLMs). "
            "Supports chat and completion modes.",
            input_description="Text prompt or chat messages",
            output_description="Generated text continuation or response",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._pipeline = hf_pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded text generation model: {model_id}")

        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(
                f"Insufficient VRAM to load {model_id}. "
                f"Try a smaller or quantized model."
            )
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded. Load a model first.")

        self._status = RunnerStatus.RUNNING
        try:
            params = {**self.get_default_parameters(), **kwargs}

            messages = None
            prompt = inputs

            # Support chat format: list of {"role": ..., "content": ...}
            if isinstance(inputs, list) and all(
                isinstance(m, dict) and "role" in m for m in inputs
            ):
                messages = inputs
                prompt = None

            start_time = time.time()

            if messages:
                result = self._pipeline(
                    messages,
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    repetition_penalty=params["repetition_penalty"],
                    do_sample=params["temperature"] > 0,
                )
            else:
                result = self._pipeline(
                    prompt,
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    repetition_penalty=params["repetition_penalty"],
                    do_sample=params["temperature"] > 0,
                )

            elapsed = time.time() - start_time

            generated_text = result[0]["generated_text"]
            if isinstance(generated_text, list):
                # Chat format: extract last assistant message
                generated_text = generated_text[-1].get("content", str(generated_text[-1]))
            elif isinstance(generated_text, str) and prompt:
                # Strip the input prompt from the output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=generated_text.strip(),
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                },
            )

        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.READY
            torch.cuda.empty_cache()
            return RunResult(
                success=False,
                output=None,
                error="Out of VRAM during inference. Try shorter input or fewer tokens.",
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            tag = getattr(info, "pipeline_tag", None)
            return tag in ("text-generation", "text2text-generation")
        except Exception:
            return False
