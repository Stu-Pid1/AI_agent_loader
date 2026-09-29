import gradio as gr
import logging
from PIL import Image, ImageDraw, ImageFont

from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from ui.components import (
    build_compatible_dropdown_choices,
    is_model_compatible,
    slider_with_manual_override,
    update_model_status,
)

logger = logging.getLogger("ai_agent_loader.ui.object_detection")


def draw_detections(image_path: str, detections: list) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#FFD700",
    ]

    for i, det in enumerate(detections):
        box = det["box"]
        color = colors[i % len(colors)]
        x0, y0 = box["xmin"], box["ymin"]
        x1, y1 = box["xmax"], box["ymax"]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        label = f"{det['label']} ({det['score']:.2f})"
        text_bbox = draw.textbbox((x0, y0), label)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x0, y0), label, fill="white")

    return img


def create_object_detection_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Object Detection", id="object_detection"):
        gr.Markdown("## Object Detection")

        with gr.Row():
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=[],
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="filepath", scale=1)
            output_image = gr.Image(label="Detections", scale=1)

        with gr.Row():
            threshold = slider_with_manual_override(
                "Confidence Threshold",
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
            )
            detect_btn = gr.Button("Detect Objects", variant="primary")

        output_json = gr.JSON(label="Detection Details")
        output_meta = gr.Markdown("")

        # --- Event Handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, {"object-detection"}, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, {"object-detection"}, cache_manager):
                return (
                    "**Status:** This model is not compatible with the Object Detection tab "
                    "(marked with ❌)."
                )
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "object-detection")
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            model_manager.unload_current()
            return update_model_status(model_manager)

        def detect(image_path, thresh, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_manager.active_model_id:
                return None, None, "No model loaded."
            if not image_path:
                return None, None, "Upload an image."

            result = model_manager.run_inference(image_path, threshold=thresh)
            if result.success:
                annotated = draw_detections(image_path, result.output)
                meta = (
                    f"**Duration:** {result.metadata.get('duration_seconds', '?')}s | "
                    f"**Objects found:** {result.metadata.get('num_detections', 0)}"
                )
                return annotated, result.output, meta
            return None, None, f"Error: {result.error}"

        # --- Wire Events ---
        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload, outputs=[status_bar], concurrency_id="model_ops")
        detect_btn.click(
            fn=detect,
            inputs=[input_image, threshold],
            outputs=[output_image, output_json, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
