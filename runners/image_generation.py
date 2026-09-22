import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from config.settings import Settings
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from core.device import DeviceManager
from utils.errors import ModelLoadError, VRAMError

logger = logging.getLogger("ai_agent_loader.runners.image_generation")

# File patterns that identify different model formats
_SINGLE_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")
_PIPELINE_INDEX = "model_index.json"
_LORA_INDICATORS = ("adapter_config.json", "pytorch_lora_weights.safetensors")


def _is_lora_tagged(tags: set) -> bool:
    if "lora" in tags or "template:sd-lora" in tags:
        return True
    return any(t.startswith("base_model:adapter:") for t in tags)


def _detect_model_format(model_id: str):
    """
    Returns (fmt, info) where fmt is one of:
    'pipeline', 'single_file', 'lora', 'unknown'.
    Inspects only the local snapshot's files and local README/config metadata
    — a model that's already cached should never require a Hub lookup to
    classify it.
    """
    try:
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        filenames = set(cache_manager.list_local_files(model_id))
        info = cache_manager.get_local_model_metadata(model_id)
        tags = {str(t).lower() for t in (info.tags or [])}
        pipeline_tag = (info.pipeline_tag or "").lower()

        if pipeline_tag in {"text-to-video", "video-generation"}:
            raise ValueError(f"'{model_id}' is a video model, not an image-generation model.")
        if "text-to-video" in tags:
            raise ValueError(f"'{model_id}' is a video model, not an image-generation model.")

        if _PIPELINE_INDEX in filenames:
            return "pipeline", info
        if any(f in filenames for f in _LORA_INDICATORS):
            return "lora", info
        if any(f.endswith(_SINGLE_FILE_EXTS) for f in filenames):
            # Custom-named LoRA weights (e.g. from ai-toolkit/glif training)
            # won't match the filename indicators above, so fall back to tags.
            if _is_lora_tagged(tags):
                return "lora", info
            return "single_file", info
        return "unknown", info
    except ValueError:
        raise
    except Exception:
        return "unknown", None


def _get_lora_base_model(info) -> Optional[str]:
    """Best-effort resolution of the base model a LoRA repo was trained on."""
    if info is None:
        return None
    return getattr(info, "base_model", None)


def _find_lora_weight_file(filenames) -> Optional[str]:
    """Pick the right LoRA weight file when a repo publishes several."""
    for name in filenames:
        if name in _LORA_INDICATORS:
            return name

    candidates = [n for n in filenames if n.endswith(".safetensors") and "/" not in n]
    if not candidates:
        return None

    import re

    step_pattern = re.compile(r"_(\d+)\.safetensors$")
    finals = [n for n in candidates if not step_pattern.search(n)]
    if finals:
        return finals[0]

    return max(candidates, key=lambda n: int(step_pattern.search(n).group(1)))


def _find_single_file(model_id: str) -> Optional[str]:
    """Return the filename of the best single-file checkpoint in the local snapshot."""
    try:
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        filenames = cache_manager.list_local_files(model_id)

        # Prefer fp16 safetensors, then any safetensors, then ckpt
        for name in filenames:
            if name.endswith(".safetensors") and "fp16" in name.lower():
                return name
        for name in filenames:
            if name.endswith(".safetensors") and "/" not in name:
                return name
        for name in filenames:
            if name.endswith((".ckpt", ".safetensors")):
                return name
    except Exception:
        pass
    return None


def _resolve_local_model_path(model_id: str) -> Optional[str]:
    try:
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        return cache_manager.resolve_local_model_path(model_id)
    except Exception:
        return None


class ImageGenerationRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Image Generation",
            supported_pipeline_tags=["text-to-image"],
            supported_libraries=["diffusers"],
            description="Generate images from text prompts using diffusion models "
            "(Stable Diffusion, SDXL, Flux, etc.).",
            input_description="Text prompt and optional negative prompt",
            output_description="Generated image(s)",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            local_path = _resolve_local_model_path(model_id)
            if not local_path:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. This app only loads cached local models."
                )

            logger.info(f"Using local cache snapshot for {model_id}: {local_path}")
            model_ref = local_path

            fmt, info = _detect_model_format(model_ref)
            logger.info(f"Detected format for {model_id}: {fmt}")

            if fmt == "unknown":
                raise ModelLoadError(
                    f"'{model_id}' is not a valid cached image-generation model in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}."
                )

            dtype = torch.float16 if self._device == "cuda" else torch.float32

            if fmt == "lora":
                base_model = _get_lora_base_model(info)
                if not base_model:
                    raise ModelLoadError(
                        f"'{model_id}' is a LoRA adapter — it cannot run standalone, and no "
                        f"base model could be determined from its tags/card data. "
                        f"Search for the base model instead and load that."
                    )
                self._load_lora(model_ref, info, base_model, dtype, **kwargs)
                self._model_id = f"{model_id} (LoRA on {base_model})"
            elif fmt == "single_file":
                self._load_single_file(model_ref, dtype, **kwargs)
                self._model_id = model_id
            else:
                self._load_pipeline(model_ref, dtype, **kwargs)
                self._model_id = model_id

            self._status = RunnerStatus.READY
            logger.info(f"Loaded image generation model: {model_id} ({fmt})")

        except (ModelLoadError, VRAMError):
            self._status = RunnerStatus.ERROR
            self.unload()
            raise
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(
                f"Insufficient VRAM to load {model_id}. "
                f"Try a smaller model or lower resolution."
            )
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            msg = f"Failed to load {model_id}: {e}"
            error_text = str(e).lower()
            if any(kw in error_text for kw in ("gated", "401", "403", "access to model", "restricted")):
                msg += (
                    " — this model may be gated. Accept its license on huggingface.co "
                    "and make sure HF_TOKEN is set."
                )
            raise ModelLoadError(msg)

    def _load_pipeline(self, model_id: str, dtype, **kwargs) -> None:
        from diffusers import DiffusionPipeline

        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "local_files_only": True,
        }
        if DeviceManager.use_distributed_device_map():
            # diffusers pipelines only accept "balanced" (or "cuda"/"cpu"),
            # not "auto" — that's a transformers-only value.
            load_kwargs["device_map"] = "balanced"

        # Try fp16 variant first (smaller download, faster load)
        if dtype == torch.float16:
            try:
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    variant="fp16",
                    cache_dir=str(Settings.DEFAULT_CACHE_DIR),
                    **load_kwargs,
                )
                logger.info(f"Loaded fp16 variant of {model_id}")
            except Exception:
                logger.info(f"No fp16 variant found, loading default weights")
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    cache_dir=str(Settings.DEFAULT_CACHE_DIR),
                    **load_kwargs,
                )
        else:
            self._pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=str(Settings.DEFAULT_CACHE_DIR),
                **load_kwargs,
            )

        if self._device == "cuda" and not DeviceManager.use_distributed_device_map():
            vram = DeviceManager.get_vram_usage()
            free_gb = vram["free"] / (1024 ** 3)
            if free_gb < 4.0:
                # Low VRAM: offload to CPU between inference calls
                logger.warning(
                    f"Low VRAM ({free_gb:.1f} GB free), enabling CPU offload"
                )
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline = self._pipeline.to("cuda")

    def _load_lora(self, model_id: str, info, base_model: str, dtype, **kwargs) -> None:
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        filenames = cache_manager.list_local_files(model_id)
        weight_name = _find_lora_weight_file(filenames)
        if not weight_name:
            raise ModelLoadError(f"Could not find a LoRA weight file in the local cache at {model_id}.")

        base_local_path = _resolve_local_model_path(base_model)
        if not base_local_path:
            raise ModelLoadError(
                f"LoRA base model '{base_model}' is not present in the local cache at "
                f"{Settings.DEFAULT_CACHE_DIR}. Download it first, then retry this LoRA."
            )

        logger.info(f"Loading base model {base_model} for LoRA at {model_id}")
        self._load_pipeline(base_local_path, dtype, **kwargs)

        logger.info(f"Applying LoRA weights: {model_id}/{weight_name}")
        self._pipeline.load_lora_weights(model_id, weight_name=weight_name)

    def _load_single_file(self, model_id: str, dtype, **kwargs) -> None:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        filename = _find_single_file(model_id)
        if not filename:
            raise ModelLoadError(f"Could not find a checkpoint file in the local cache at {model_id}.")

        local_path = str(Path(model_id) / filename)
        logger.info(f"Loading local single-file checkpoint: {local_path}")

        # Try SDXL first (larger models), fall back to SD 1.5/2.x
        for cls in (StableDiffusionXLPipeline, StableDiffusionPipeline):
            try:
                self._pipeline = cls.from_single_file(
                    local_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                if self._device == "cuda":
                    vram = DeviceManager.get_vram_usage()
                    free_gb = vram["free"] / (1024 ** 3)
                    if free_gb < 4.0:
                        self._pipeline.enable_model_cpu_offload()
                    else:
                        self._pipeline = self._pipeline.to("cuda")
                logger.info(f"Loaded as {cls.__name__}")
                return
            except Exception as e:
                logger.debug(f"{cls.__name__} failed: {e}")

        raise ModelLoadError(
            f"Could not load {filename} as SD or SDXL. "
            f"The checkpoint format may not be supported."
        )

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            from utils.errors import ModelNotLoadedError
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            params = {**self.get_default_parameters(), **kwargs}
            prompt = inputs if isinstance(inputs, str) else inputs.get("prompt", "")
            negative_prompt = (
                inputs.get("negative_prompt", "") if isinstance(inputs, dict) else ""
            )

            seed = params.get("seed", -1)
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=self._device).manual_seed(seed)

            start_time = time.time()

            pipe_kwargs = {
                "prompt": prompt,
                "num_inference_steps": int(params["num_inference_steps"]),
                "guidance_scale": params["guidance_scale"],
                "width": int(params["width"]),
                "height": int(params["height"]),
                "num_images_per_prompt": int(params.get("num_images", 1)),
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
            if generator:
                pipe_kwargs["generator"] = generator

            result = self._pipeline(**pipe_kwargs)
            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=result.images,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "seed": seed,
                    "model_id": self._model_id,
                    "num_images": len(result.images),
                },
            )

        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.READY
            torch.cuda.empty_cache()
            return RunResult(
                success=False,
                output=None,
                error="Out of VRAM. Try smaller resolution or fewer steps.",
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": -1,
            "num_images": 1,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "pipeline_tag", None) == "text-to-image"
        except Exception:
            return False
