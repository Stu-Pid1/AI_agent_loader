import threading
import logging
from typing import Any, Dict, Optional

from runners.base import BaseRunner, RunResult, RunnerStatus
from config.task_registry import TaskRegistry
from core.device import DeviceManager
from utils.errors import ModelLoadError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.model_manager")


class ModelManager:

    def __init__(self, registry: TaskRegistry, device: Optional[str] = None):
        self._registry = registry
        self._device = device or DeviceManager.get_optimal_device()
        self._active_runner: Optional[BaseRunner] = None
        self._active_model_id: Optional[str] = None
        self._active_pipeline_tag: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def active_runner(self) -> Optional[BaseRunner]:
        return self._active_runner

    @property
    def active_pipeline_tag(self) -> Optional[str]:
        return self._active_pipeline_tag

    def load_model(self, model_id: str, pipeline_tag: str, **kwargs) -> str:
        with self._lock:
            # Auto-detect GGUF models for text generation
            runner_class = None
            if pipeline_tag in ("text-generation", "text2text-generation"):
                try:
                    from runners.text_generation_gguf import GGUFTextGenerationRunner
                    if GGUFTextGenerationRunner.has_gguf_files(model_id):
                        runner_class = GGUFTextGenerationRunner
                        logger.info(f"GGUF files detected for {model_id}, using GGUF runner")
                except ImportError:
                    pass

            if runner_class is None:
                runner_class = self._registry.get_runner_class(pipeline_tag)

            if runner_class is None:
                raise ModelLoadError(
                    f"No runner registered for pipeline tag: {pipeline_tag}"
                )

            # Unload any currently loaded model
            if self._active_runner is not None:
                logger.info(f"Unloading current model: {self._active_model_id}")
                self._active_runner.unload()
                self._active_runner = None
                self._active_model_id = None
                self._active_pipeline_tag = None

            runner = runner_class(device=self._device)
            runner.load(model_id, task=pipeline_tag, **kwargs)

            self._active_runner = runner
            self._active_model_id = model_id
            self._active_pipeline_tag = pipeline_tag

            logger.info(f"Loaded model: {model_id} ({pipeline_tag})")
            return f"Loaded {model_id}"

    def unload_current(self) -> str:
        with self._lock:
            if self._active_runner is None:
                return "No model loaded."

            model_id = self._active_model_id
            self._active_runner.unload()
            self._active_runner = None
            self._active_model_id = None
            self._active_pipeline_tag = None
            logger.info(f"Unloaded model: {model_id}")
            return f"Unloaded {model_id}"

    def run_inference(self, inputs: Any, **kwargs) -> RunResult:
        with self._lock:
            if self._active_runner is None:
                raise ModelNotLoadedError("No model loaded.")
            return self._active_runner.run(inputs, **kwargs)

    def get_active_status(self) -> Dict[str, Any]:
        gpu_info = DeviceManager.get_gpu_info()
        return {
            "model_id": self._active_model_id,
            "pipeline_tag": self._active_pipeline_tag,
            "status": self._active_runner.status.value
            if self._active_runner
            else "unloaded",
            "device": self._device,
            "gpu_name": gpu_info.device_name,
            "vram_used_gb": gpu_info.vram_used_gb,
            "vram_total_gb": gpu_info.vram_total_gb,
            "vram_free_gb": gpu_info.vram_free_gb,
        }
