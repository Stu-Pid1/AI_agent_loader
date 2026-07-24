from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("ai_agent_loader.device")


@dataclass
class GPUInfo:
    available: bool
    device_name: Optional[str] = None
    cuda_version: Optional[str] = None
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0


class DeviceManager:

    @staticmethod
    def get_gpu_info() -> GPUInfo:
        try:
            import torch

            if not torch.cuda.is_available():
                return GPUInfo(available=False)

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total = props.total_mem / (1024**3)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            free = total - reserved

            return GPUInfo(
                available=True,
                device_name=props.name,
                cuda_version=torch.version.cuda,
                vram_total_gb=round(total, 2),
                vram_used_gb=round(allocated, 2),
                vram_free_gb=round(free, 2),
            )
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
            return GPUInfo(available=False)

    @staticmethod
    def get_optimal_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def get_optimal_dtype():
        import torch

        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    @staticmethod
    def get_vram_usage() -> dict:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_mem
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)

            return {
                "allocated": allocated,
                "reserved": reserved,
                "free": total - reserved,
                "total": total,
            }
        except Exception:
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

    @staticmethod
    def estimate_model_fits(model_size_bytes: int) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_mem
            reserved = torch.cuda.memory_reserved(device)
            free = total - reserved
            # Leave 500MB headroom
            return model_size_bytes < (free - 500 * 1024 * 1024)
        except Exception:
            return False
