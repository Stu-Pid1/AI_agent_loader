import gradio as gr
from typing import Iterable, List, Optional, Sequence, Tuple

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


def slider_with_manual_override(
    label: str,
    minimum: float,
    maximum: float,
    value: float,
    step: Optional[float] = None,
    info: Optional[str] = None,
    scale: Optional[int] = None,
) -> gr.Number:
    """A slider for quick visual adjustment, paired with a plain number box
    that has no upper/lower bound.

    Gradio's Slider rejects (server-side, not just cosmetically) any typed
    value outside [minimum, maximum], so pairing it with an unbounded Number
    field is the only way to let manual entry exceed the slider's range.
    Use the *returned* Number component as the actual input to your event
    handlers — it always mirrors the slider while its value is in range, and
    holds the real value when the user manually enters something beyond it
    (the slider then just visually pins at its own limit).
    """
    with gr.Column(scale=scale or 1, min_width=160):
        slider = gr.Slider(label=label, minimum=minimum, maximum=maximum, value=value, step=step, info=info)
        number = gr.Number(value=value, show_label=False, container=False)

    def _slider_to_number(v):
        return v

    def _number_to_slider(v):
        if v is not None and minimum <= v <= maximum:
            return gr.update(value=v)
        return gr.update()

    # Use release(), not change(): change() fires a server round-trip for
    # every intermediate tick while dragging (potentially dozens per
    # second), which floods the queue and makes the UI appear to jump
    # around under the backlog. release() fires once, when the drag ends.
    slider.release(fn=_slider_to_number, inputs=[slider], outputs=[number])
    number.change(fn=_number_to_slider, inputs=[number], outputs=[slider])

    return number


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


def is_model_compatible(model_id: str, allowed_tags: Iterable[str], cache_manager) -> bool:
    """Checks compatibility using only metadata already on disk.

    Cached models never require a Hub lookup to classify — the app reads
    README front matter / config.json from the local snapshot instead.
    """
    if not model_id:
        return False

    allowed = {tag.lower() for tag in allowed_tags}
    try:
        detail = cache_manager.get_local_model_metadata(model_id)
        actual = (detail.pipeline_tag or "").lower()
        if actual in allowed:
            return True
        for tag in (detail.tags or []):
            if str(tag).lower() in allowed:
                return True
        return False
    except Exception:
        return False


def build_compatible_dropdown_choices(model_ids: Sequence[str], allowed_tags: Iterable[str], cache_manager) -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for model_id in model_ids:
        compatible = is_model_compatible(model_id, allowed_tags, cache_manager)
        label = model_id if compatible else f"{model_id} ❌"
        choices.append((label, model_id))
    return choices
