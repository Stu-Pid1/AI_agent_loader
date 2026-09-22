import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch

from config.settings import Settings
from core.device import DeviceManager
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, ModelNotLoadedError, VRAMError

logger = logging.getLogger("ai_agent_loader.runners.face_swap")

DEFAULT_PROMPT = (
    "face_swap: take the face from Picture 1 and swap it onto the person in Picture 2. "
    "the result should show the exact face and features from Picture 1 on the body, pose, "
    "and scene from Picture 2, with natural skin tone blending and matching lighting."
)


class FaceSwapRunner(BaseRunner):
    """Generic FLUX.2-Klein face/head-swap LoRA runner.

    Unlike the generic image-generation runner, this always takes two
    conditioning images (a face reference and a target scene) rather than a
    free-form single-image prompt. Different LoRA packs (ilkerzgi/face-swap,
    Alissonerdx/BFS-Best-Face-Swap, etc.) ship different weight filenames,
    recommended prompts, and even different conventions for which image goes
    first — none of that is hardcoded here. The LoRA weight file, prompt, and
    image order are all supplied by the caller (see runners/face_swap.py's
    `list_lora_weight_files` for discovering the first, and the UI for the
    other two).
    """

    BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

    @staticmethod
    def list_lora_weight_files(model_id: str) -> list:
        """All .safetensors files in a cached LoRA repo, for the UI to offer as choices.

        Face/head-swap LoRA packs commonly bundle variants trained for several
        different base models (FLUX.2-Klein 4B/9B, Qwen-Image-Edit, Krea 2,
        LTX-2, ...) in one repo. This runner only supports FLUX.2-Klein-9B as
        a base, so pick a filename that names "9b"/"klein" — but we don't
        hard-filter, since naming conventions vary between authors.
        """
        from core.cache_manager import CacheManager

        cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))
        return [f for f in cache_manager.list_local_files(model_id) if f.endswith(".safetensors")]

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Face Swap",
            supported_pipeline_tags=["face-swap"],
            supported_libraries=["diffusers"],
            description="Swap a reference face onto a person in an image using a FLUX.2-Klein face-swap LoRA.",
            input_description="Face reference image + target scene image",
            output_description="Face-swapped image",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            from core.cache_manager import CacheManager

            cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))

            lora_dir = cache_manager.resolve_local_model_path(model_id)
            if not lora_dir:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser tab first."
                )

            base_dir = cache_manager.resolve_local_model_path(self.BASE_MODEL_ID)
            base_incomplete = not base_dir or not (Path(base_dir) / "model_index.json").exists()
            if base_incomplete:
                raise ModelLoadError(
                    f"Base model '{self.BASE_MODEL_ID}' is not fully downloaded at "
                    f"{Settings.DEFAULT_CACHE_DIR} (missing model_index.json / the model weights). "
                    f"It's a gated model — this usually means the download only pulled the public "
                    f"README/LICENSE while the actual weights were skipped because no HF_TOKEN is "
                    f"set. To fix: accept the license at "
                    f"huggingface.co/{self.BASE_MODEL_ID}, create a read token at "
                    f"huggingface.co/settings/tokens, add `HF_TOKEN=hf_...` to a .env file in the "
                    f"project root, restart the app, then re-download the model from the Hub "
                    f"Browser tab."
                )

            variant = kwargs.get("lora_variant")
            if not variant:
                raise ModelLoadError(
                    f"No LoRA weight file selected for {model_id}. Pick one from the "
                    f"LoRA Variant dropdown."
                )
            weight_path = Path(lora_dir) / variant
            if not weight_path.exists():
                raise ModelLoadError(
                    f"LoRA weight file '{variant}' was not found in the local cache for {model_id}."
                )

            from diffusers import Flux2KleinPipeline

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            }
            if self._device == "cuda" and DeviceManager.use_distributed_device_map():
                # diffusers pipelines only accept "balanced" (or "cuda"/"cpu"),
                # not "auto" — that's a transformers-only value.
                load_kwargs["device_map"] = "balanced"

            self._pipeline = Flux2KleinPipeline.from_pretrained(base_dir, **load_kwargs)
            self._pipeline.load_lora_weights(lora_dir, weight_name=variant)

            if self._device == "cuda" and "device_map" not in load_kwargs:
                vram = DeviceManager.get_vram_usage()
                free_gb = vram["free"] / (1024 ** 3)
                if free_gb < 20.0:
                    self._pipeline.enable_model_cpu_offload()
                else:
                    self._pipeline = self._pipeline.to("cuda")

            self._model_id = model_id
            self._lora_variant = variant
            self._status = RunnerStatus.READY
            logger.info(f"Loaded face-swap LoRA: {model_id} ({variant}) on base {self.BASE_MODEL_ID}")

        except (ModelLoadError, VRAMError):
            self._status = RunnerStatus.ERROR
            self.unload()
            raise
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(
                f"Insufficient VRAM to load the face-swap pipeline. FLUX.2-Klein-9B is a large "
                f"model — free up VRAM or use a machine with more GPU memory."
            )
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            face_image = inputs["face_image"]
            scene_image = inputs["scene_image"]
            params = {**self.get_default_parameters(), **kwargs}

            prompt = params["prompt"] or DEFAULT_PROMPT
            images = [scene_image, face_image] if params["swap_image_order"] else [face_image, scene_image]

            start_time = time.time()
            result = self._pipeline(
                image=images,
                prompt=prompt,
                num_inference_steps=int(params["num_inference_steps"]),
                guidance_scale=float(params["guidance_scale"]),
            )
            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=result.images[0],
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "lora_variant": getattr(self, "_lora_variant", None),
                },
            )
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.READY
            torch.cuda.empty_cache()
            return RunResult(
                success=False,
                output=None,
                error="Out of VRAM during face swap. Try a smaller crop or fewer inference steps.",
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "num_inference_steps": 28,
            "guidance_scale": 1.0,
            "prompt": DEFAULT_PROMPT,
            "swap_image_order": False,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            tags = {str(t).lower() for t in (getattr(info, "tags", None) or [])}
            return bool(tags & {"face-swap", "head-swap", "body-swap"})
        except Exception:
            return False
