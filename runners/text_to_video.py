import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from config.settings import Settings
from core.device import DeviceManager
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, VRAMError, ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.runners.text_to_video")


class TextToVideoRunner(BaseRunner):

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Text to Video",
            supported_pipeline_tags=["text-to-video"],
            supported_libraries=["diffusers"],
            description="Generate short video clips from text prompts using diffusion-based video models such as MiniMax-style text-to-video models.",
            input_description="Text prompt and optional negative prompt",
            output_description="Generated video clip",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            from core.cache_manager import CacheManager

            local_path = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR)).resolve_local_model_path(model_id)
            if not local_path:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. This app only loads cached local video models."
                )

            logger.info(f"Using local cache snapshot for text-to-video model {model_id}: {local_path}")
            model_ref = local_path

            dtype = torch.float16 if self._device == "cuda" else torch.float32

            from diffusers import DiffusionPipeline

            load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": kwargs.get("trust_remote_code", False),
                "cache_dir": str(Settings.DEFAULT_CACHE_DIR),
                "local_files_only": True,
            }
            if self._device == "cuda" and DeviceManager.use_distributed_device_map():
                # diffusers pipelines only accept "balanced" (or "cuda"/"cpu"),
                # not "auto" — that's a transformers-only value.
                load_kwargs["device_map"] = "balanced"

            if dtype == torch.float16:
                try:
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        model_ref,
                        variant="fp16",
                        **load_kwargs,
                    )
                except Exception:
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        model_ref,
                        **load_kwargs,
                    )
            else:
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_ref,
                    **load_kwargs,
                )

            if self._device == "cuda" and not DeviceManager.use_distributed_device_map():
                vram = DeviceManager.get_vram_usage()
                free_gb = vram["free"] / (1024 ** 3)
                if free_gb < 6.0:
                    self._pipeline.enable_model_cpu_offload()
                else:
                    self._pipeline = self._pipeline.to("cuda")

            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded text-to-video model: {model_id}")

        except (ModelLoadError, VRAMError):
            self._status = RunnerStatus.ERROR
            self.unload()
            raise
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(f"Insufficient VRAM to load {model_id}. Try a smaller or lower-resolution video model.")
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            params = {**self.get_default_parameters(), **kwargs}
            prompt = inputs if isinstance(inputs, str) else inputs.get("prompt", "")
            negative_prompt = inputs.get("negative_prompt", "") if isinstance(inputs, dict) else ""

            seed = int(params.get("seed", -1))
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=self._device).manual_seed(seed)

            start_time = time.time()
            pipe_kwargs = {
                "prompt": prompt,
                "num_inference_steps": int(params["num_inference_steps"]),
                "guidance_scale": float(params["guidance_scale"]),
                "width": int(params["width"]),
                "height": int(params["height"]),
                "num_frames": int(params["num_frames"]),
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
            if generator is not None:
                pipe_kwargs["generator"] = generator

            result = self._pipeline(**pipe_kwargs)
            frames = self._extract_frames(result)
            video_path = self._frames_to_video(frames, fps=int(params.get("fps", 8)))
            elapsed = time.time() - start_time

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=video_path,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "frames": len(frames),
                    "fps": int(params.get("fps", 8)),
                    "seed": seed,
                },
            )
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.READY
            torch.cuda.empty_cache()
            return RunResult(
                success=False,
                output=None,
                error="Out of VRAM during video generation. Try a smaller resolution or fewer frames.",
            )
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def _extract_frames(self, result: Any) -> List[np.ndarray]:
        if hasattr(result, "frames"):
            frames = result.frames
        elif isinstance(result, dict):
            frames = result.get("frames") or result.get("video_frames")
        elif isinstance(result, (list, tuple)) and result and hasattr(result[0], "frames"):
            frames = result[0].frames
        else:
            frames = result

        if frames is None:
            raise ModelLoadError("The model did not return any video frames.")

        normalized: List[np.ndarray] = []
        for frame in frames:
            try:
                arr = np.asarray(frame)
            except Exception:
                continue
            if arr.size > 0:
                normalized.append(arr)

        if not normalized:
            raise ModelLoadError("No valid frames were generated for the video output.")
        return normalized

    def _frames_to_video(self, frames: List[np.ndarray], fps: int = 8) -> str:
        try:
            import imageio.v2 as imageio
        except Exception:
            # Fallback: return the first frame array if imageio is unavailable.
            return np.asarray(frames[0]) if frames else ""

        path = Path(tempfile.gettempdir()) / f"video_{uuid.uuid4().hex}.mp4"
        imageio.mimsave(str(path), [np.asarray(frame) for frame in frames], fps=fps)
        return str(path)

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "num_frames": 16,
            "fps": 8,
            "seed": -1,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            tag = getattr(info, "pipeline_tag", None)
            return tag in ("text-to-video", "video-generation")
        except Exception:
            return False
