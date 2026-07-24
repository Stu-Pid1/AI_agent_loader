import time
import logging
from typing import Any, Dict, Optional

import torch

from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from core.device import DeviceManager
from utils.errors import ModelLoadError, VRAMError

logger = logging.getLogger("ai_agent_loader.runners.image_generation")

# File patterns that identify different model formats
_SINGLE_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")
_PIPELINE_INDEX = "model_index.json"
_LORA_INDICATORS = ("adapter_config.json", "pytorch_lora_weights.safetensors")


_BASE_MODEL_TAG_PREFIXES = ("adapter:", "finetune:", "quantized:", "merge:")


def _is_lora_tagged(tags: set) -> bool:
    if "lora" in tags or "template:sd-lora" in tags:
        return True
    return any(t.startswith("base_model:adapter:") for t in tags)


def _detect_model_format(model_id: str):
    """
    Returns (fmt, info) where fmt is one of:
    'pipeline', 'single_file', 'lora', 'unknown'.
    Inspects the file list and tags from HF Hub without downloading anything.
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        filenames = {s.rfilename for s in (info.siblings or [])}
        tags = set(getattr(info, "tags", None) or [])

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
    except Exception:
        return "unknown", None


def _get_lora_base_model(info) -> Optional[str]:
    """Best-effort resolution of the base model a LoRA repo was trained on."""
    if info is None:
        return None

    card_data = getattr(info, "card_data", None)
    if card_data is not None:
        base_model = getattr(card_data, "base_model", None) or (
            card_data.get("base_model") if hasattr(card_data, "get") else None
        )
        if isinstance(base_model, list) and base_model:
            base_model = base_model[0]
        if isinstance(base_model, str) and base_model:
            return base_model

    for tag in getattr(info, "tags", None) or []:
        if not tag.startswith("base_model:"):
            continue
        remainder = tag[len("base_model:"):]
        for prefix in _BASE_MODEL_TAG_PREFIXES:
            if remainder.startswith(prefix):
                remainder = remainder[len(prefix):]
                break
        if remainder:
            return remainder
    return None


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
    """Return the filename of the best single-file checkpoint in the repo."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        filenames = [s.rfilename for s in (info.siblings or [])]

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
            fmt, info = _detect_model_format(model_id)
            logger.info(f"Detected format for {model_id}: {fmt}")

            if fmt == "unknown":
                raise ModelLoadError(
                    f"'{model_id}' does not appear to be a runnable image generation model. "
                    f"It has no model_index.json and no recognised checkpoint files."
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
                self._load_lora(model_id, info, base_model, dtype, **kwargs)
                self._model_id = f"{model_id} (LoRA on {base_model})"
            elif fmt == "single_file":
                self._load_single_file(model_id, dtype, **kwargs)
                self._model_id = model_id
            else:
                self._load_pipeline(model_id, dtype, **kwargs)
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
        }

        # Try fp16 variant first (smaller download, faster load)
        if dtype == torch.float16:
            try:
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_id, variant="fp16", **load_kwargs
                )
                logger.info(f"Loaded fp16 variant of {model_id}")
            except Exception:
                logger.info(f"No fp16 variant found, loading default weights")
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_id, **load_kwargs
                )
        else:
            self._pipeline = DiffusionPipeline.from_pretrained(
                model_id, **load_kwargs
            )

        if self._device == "cuda":
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
        filenames = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
        weight_name = _find_lora_weight_file(filenames)
        if not weight_name:
            raise ModelLoadError(f"Could not find a LoRA weight file in {model_id}.")

        logger.info(f"Loading base model {base_model} for LoRA {model_id}")
        self._load_pipeline(base_model, dtype, **kwargs)

        logger.info(f"Applying LoRA weights: {model_id}/{weight_name}")
        self._pipeline.load_lora_weights(model_id, weight_name=weight_name)

    def _load_single_file(self, model_id: str, dtype, **kwargs) -> None:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download

        filename = _find_single_file(model_id)
        if not filename:
            raise ModelLoadError(f"Could not find a checkpoint file in {model_id}.")

        logger.info(f"Downloading single checkpoint file: {filename}")
        local_path = hf_hub_download(model_id, filename=filename)

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
