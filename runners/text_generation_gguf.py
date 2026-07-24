import time
import logging
import glob
from typing import Any, Dict, List, Optional
from pathlib import Path

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_generation_gguf")


class GGUFTextGenerationRunner(BaseRunner):

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._llm = None

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text Generation (GGUF)",
            supported_pipeline_tags=["text-generation"],
            supported_libraries=["llama-cpp-python"],
            description="Run quantized GGUF models efficiently using llama-cpp-python. "
            "Great for running large models on limited VRAM.",
            input_description="Text prompt or chat messages",
            output_description="Generated text",
        )

    @staticmethod
    def find_gguf_file(model_id: str) -> Optional[str]:
        try:
            from huggingface_hub import model_info, hf_hub_download

            info = model_info(model_id)
            gguf_files = [
                s.rfilename
                for s in (info.siblings or [])
                if s.rfilename.endswith(".gguf")
            ]
            if not gguf_files:
                return None

            # Prefer Q4_K_M quantization, then Q5, then the smallest file
            for preferred in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "q4_k_m", "q5_k_m"]:
                for f in gguf_files:
                    if preferred in f:
                        return f
            return gguf_files[0]
        except Exception:
            return None

    @staticmethod
    def has_gguf_files(model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            return any(
                s.rfilename.endswith(".gguf") for s in (info.siblings or [])
            )
        except Exception:
            return False

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            from config.settings import Settings

            gguf_file = kwargs.get("gguf_file") or self.find_gguf_file(model_id)
            if not gguf_file:
                raise ModelLoadError(
                    f"No GGUF file found in {model_id}. "
                    f"This model may not have quantized versions."
                )

            logger.info(f"Downloading GGUF file: {model_id}/{gguf_file}")
            model_path = hf_hub_download(
                model_id,
                filename=gguf_file,
                token=Settings.HF_TOKEN,
            )

            n_gpu_layers = -1 if self._device == "cuda" else 0
            n_ctx = kwargs.get("n_ctx", 4096)

            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded GGUF model: {model_id} ({gguf_file})")

        except Exception as e:
            self._status = RunnerStatus.ERROR
            self._llm = None
            if "CUDA" in str(e) or "memory" in str(e).lower():
                raise VRAMError(f"Insufficient VRAM to load {model_id}: {e}")
            raise ModelLoadError(f"Failed to load GGUF model {model_id}: {e}")

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded() or self._llm is None:
            raise ModelNotLoadedError("No GGUF model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            params = {**self.get_default_parameters(), **kwargs}
            start_time = time.time()

            # Chat format
            if isinstance(inputs, list) and all(
                isinstance(m, dict) and "role" in m for m in inputs
            ):
                response = self._llm.create_chat_completion(
                    messages=inputs,
                    max_tokens=int(params["max_new_tokens"]),
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=int(params["top_k"]),
                    repeat_penalty=params["repetition_penalty"],
                )
                text = response["choices"][0]["message"]["content"]
            else:
                # Completion format
                prompt = inputs if isinstance(inputs, str) else str(inputs)
                response = self._llm(
                    prompt,
                    max_tokens=int(params["max_new_tokens"]),
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=int(params["top_k"]),
                    repeat_penalty=params["repetition_penalty"],
                )
                text = response["choices"][0]["text"]

            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=text.strip(),
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "backend": "llama-cpp-python",
                },
            )

        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        super().unload()

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

    def validate_model(self, model_id: str) -> bool:
        return self.has_gguf_files(model_id)
