import os
import sys
import time
import logging
import glob
from typing import Any, Dict, List, Optional
from pathlib import Path

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_generation_gguf")


def _ensure_cuda_dll_path() -> None:
    """Make sure llama-cpp-python's CUDA backend can actually find the CUDA runtime DLLs.

    llama-cpp-python loads its native libraries with ctypes winmode=RTLD_GLOBAL
    (which is 0 on Windows), which disables the "safe" os.add_dll_directory()
    search path and falls back to the classic search order — PATH included,
    os.add_dll_directory() additions ignored. CUDA 13+ also moved its runtime
    DLLs (cudart64_*.dll, cublas64_*.dll, ...) from <toolkit>/bin into
    <toolkit>/bin/x64, which a typical CUDA install does not add to PATH.
    Without this, ggml-cuda.dll fails to load and everything silently falls
    back to CPU.
    """
    if sys.platform != "win32":
        return
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for sub in (os.path.join("bin", "x64"), "bin"):
        candidate = os.path.join(cuda_path, sub)
        if os.path.isdir(candidate) and candidate not in path_entries:
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            path_entries.insert(0, candidate)


class GGUFTextGenerationRunner(BaseRunner):

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._llm = None
        self._n_ctx = None

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
    def _local_gguf_files(model_id: str) -> List[str]:
        """GGUF filenames already present in the local cache — never the Hub."""
        from config.settings import Settings
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        return [f for f in cache_manager.list_local_files(model_id) if f.endswith(".gguf")]

    @classmethod
    def find_gguf_file(cls, model_id: str) -> Optional[str]:
        gguf_files = cls._local_gguf_files(model_id)
        if not gguf_files:
            return None

        # Prefer Q4_K_M quantization, then Q5, then the smallest file
        for preferred in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "q4_k_m", "q5_k_m"]:
            for f in gguf_files:
                if preferred in f:
                    return f
        return gguf_files[0]

    @classmethod
    def has_gguf_files(cls, model_id: str) -> bool:
        return bool(cls._local_gguf_files(model_id))

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            _ensure_cuda_dll_path()
            from llama_cpp import Llama
            from config.settings import Settings
            from core.cache_manager import CacheManager

            cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
            local_dir = cache_manager.resolve_local_model_path(model_id)
            if not local_dir:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser tab first."
                )

            gguf_file = kwargs.get("gguf_file") or self.find_gguf_file(model_id)
            if not gguf_file:
                raise ModelLoadError(
                    f"No GGUF file found in the local cache for {model_id}. "
                    f"This model may not have quantized versions, or the download is incomplete."
                )

            model_path = str(Path(local_dir) / gguf_file)
            if not Path(model_path).exists():
                raise ModelLoadError(
                    f"'{gguf_file}' was not found in the local cache for {model_id}."
                )

            logger.info(f"Loading local GGUF file: {model_path}")

            n_gpu_layers = -1 if self._device == "cuda" else 0
            n_ctx = int(kwargs.get("n_ctx") or 4096)

            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
            self._model_id = model_id
            self._n_ctx = n_ctx
            self._status = RunnerStatus.READY
            logger.info(f"Loaded GGUF model: {model_id} ({gguf_file}), n_ctx={n_ctx}")

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

            is_chat = isinstance(inputs, list) and all(
                isinstance(m, dict) and "role" in m for m in inputs
            )
            prompt_text = (
                "\n".join(str(m.get("content", "")) for m in inputs) if is_chat else str(inputs)
            )
            requested_tokens = int(params["max_new_tokens"])
            max_tokens, clamped = self._fit_to_context(prompt_text, requested_tokens)

            # Chat format
            if is_chat:
                response = self._llm.create_chat_completion(
                    messages=inputs,
                    max_tokens=max_tokens,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=int(params["top_k"]),
                    repeat_penalty=params["repetition_penalty"],
                )
                text = response["choices"][0]["message"]["content"]
            else:
                # Completion format
                response = self._llm(
                    prompt_text,
                    max_tokens=max_tokens,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=int(params["top_k"]),
                    repeat_penalty=params["repetition_penalty"],
                )
                text = response["choices"][0]["text"]

            elapsed = time.time() - start_time

            metadata = {
                "duration_seconds": round(elapsed, 2),
                "model_id": self._model_id,
                "backend": "llama-cpp-python",
                "n_ctx": self._n_ctx,
                "max_tokens_used": max_tokens,
            }
            if clamped:
                metadata["clamped_note"] = (
                    f"Requested {requested_tokens} tokens, but only {max_tokens} fit in the "
                    f"{self._n_ctx}-token context window alongside the current prompt — "
                    f"increase Context Window (n_ctx) when loading the model for more room."
                )

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=text.strip(),
                metadata=metadata,
            )

        except Exception as e:
            self._status = RunnerStatus.READY
            msg = str(e)
            if "exceed context window" in msg.lower():
                msg += (
                    f" — the prompt itself is longer than the {self._n_ctx}-token context "
                    f"window. Increase Context Window (n_ctx) when loading the model, or "
                    f"shorten the conversation/system prompt."
                )
            return RunResult(success=False, output=None, error=msg)

    def _fit_to_context(self, prompt_text: str, requested_tokens: int) -> "tuple[int, bool]":
        """Clamps max_tokens so prompt + generation can't exceed n_ctx.

        Without this, requesting more tokens than actually fit (easy to do
        now that the Max Tokens slider has no artificial ceiling) crashes
        with a raw llama.cpp "exceed context window" error instead of just
        generating as much as will fit.
        """
        available = self._n_ctx or 4096
        try:
            prompt_tokens = len(self._llm.tokenize(prompt_text.encode("utf-8")))
        except Exception:
            prompt_tokens = 0

        safety_margin = 64
        max_allowed = max(16, available - prompt_tokens - safety_margin)
        max_tokens = min(requested_tokens, max_allowed)
        return max_tokens, max_tokens < requested_tokens

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._n_ctx = None
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
