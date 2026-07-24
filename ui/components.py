import gradio as gr
from typing import List, Optional

from core.device import DeviceManager
from utils.formatting import format_bytes


def create_model_status_bar() -> gr.Markdown:
    return gr.Markdown(
        value="**Status:** No model loaded",
        elem_id="model_status_bar",
    )


def update_model_status(model_manager) -> str:
    status = model_manager.get_active_status()
    if status["model_id"]:
        vram_text = (
            f"{status['vram_used_gb']:.1f} / {status['vram_total_gb']:.1f} GB"
            if status["vram_total_gb"] > 0
            else "N/A"
        )
        return (
            f"**Model:** {status['model_id']} | "
            f"**Status:** {status['status']} | "
            f"**Device:** {status['device']} | "
            f"**VRAM:** {vram_text}"
        )
    return "**Status:** No model loaded"


def create_vram_display() -> gr.Markdown:
    gpu_info = DeviceManager.get_gpu_info()
    if gpu_info.available:
        pct = (
            (gpu_info.vram_used_gb / gpu_info.vram_total_gb * 100)
            if gpu_info.vram_total_gb > 0
            else 0
        )
        return gr.Markdown(
            f"**GPU:** {gpu_info.device_name} | "
            f"**VRAM:** {gpu_info.vram_used_gb:.1f} / {gpu_info.vram_total_gb:.1f} GB "
            f"({pct:.0f}% used)"
        )
    return gr.Markdown("**GPU:** Not available (CPU mode)")
