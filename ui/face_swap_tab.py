import logging
import tempfile
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.cache_manager import CacheManager
from core.hub_client import HubClient
from core.model_manager import ModelManager
from runners.face_swap import DEFAULT_PROMPT, FaceSwapRunner
from ui.components import build_compatible_dropdown_choices, is_model_compatible, update_model_status

logger = logging.getLogger("ai_agent_loader.ui.face_swap")

_FACE_CASCADE = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    faces = sorted(faces, key=lambda f: f[0])  # stable left-to-right numbering
    return [tuple(int(v) for v in f) for f in faces]


def _draw_face_boxes(frame_bgr, faces):
    annotated = frame_bgr.copy()
    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(annotated, str(i), (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def _read_frame_at(video_path: str, time_seconds: float):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(max(0, int(time_seconds * fps)), max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _expand_box(x, y, w, h, frame_w, frame_h, margin=0.6):
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(frame_w, cx + half))
    y1 = int(min(frame_h, cy + half))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, frame_w, frame_h
    return x0, y0, x1, y1


def _feathered_paste(base_img: Image.Image, patch_img: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return base_img

    patch_resized = patch_img.resize((w, h), Image.LANCZOS)

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    pad = max(1, int(min(w, h) * 0.08))
    draw.ellipse([pad, pad, w - pad, h - pad], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(4, min(w, h) // 12)))

    result = base_img.copy()
    result.paste(patch_resized, (x0, y0), mask)
    return result


def create_face_swap_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Face Swap", id="face_swap"):
        gr.Markdown("## Video Face Swap")
        gr.Markdown(
            "Swap a reference face onto a selected person in a video, using a FLUX.2-Klein "
            "face/head-swap LoRA. Requires both the base model "
            f"`{FaceSwapRunner.BASE_MODEL_ID}` (**gated** — accept its license on huggingface.co "
            "and set `HF_TOKEN`) and a compatible LoRA (e.g. `ilkerzgi/face-swap` or "
            "`Alissonerdx/BFS-Best-Face-Swap`), already downloaded via the Hub Browser tab.\n\n"
            "Different LoRA packs use different prompts and expect the two images in a "
            "different order — check the **LoRA's own model card** for its recommended prompt "
            "and whether Picture 1 should be the face or the scene, and set those below. Some "
            "LoRA packs (like BFS) bundle variants trained for other base models "
            "(Qwen-Image-Edit, Krea 2, LTX-2, FLUX.2-Klein-**4B**) alongside ones for "
            f"FLUX.2-Klein-**9B** — only the 9B-trained file will work here; others are "
            "marked ❌ in the dropdown.\n\n"
            "⚠️ Each processed frame is a full diffusion edit (several seconds each), so this "
            "processes a short trimmed segment at a reduced frame rate — not the full video at "
            "its native frame rate. The original audio is not carried over, since the output "
            "frame rate no longer matches the source."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Face-Swap LoRA", choices=[], interactive=True, scale=3)
            lora_variant = gr.Dropdown(
                label="LoRA Weight File",
                choices=[],
                interactive=True,
                scale=2,
            )
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt (check the LoRA's model card for the recommended wording)",
                value=DEFAULT_PROMPT,
                lines=3,
                scale=3,
            )
            swap_order = gr.Checkbox(
                label="Invert image order (Picture 1 = scene/body, Picture 2 = face) — "
                "many LoRAs like BFS Head V3+ need this",
                value=False,
                scale=1,
            )

        gr.Markdown("### 1. Upload a video and pick the face to replace")
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Source Video")
                frame_time = gr.Slider(label="Preview Frame (seconds)", minimum=0, maximum=120, value=0, step=0.5)
                detect_btn = gr.Button("Detect Faces in Frame")
            with gr.Column(scale=1):
                face_preview = gr.Image(label="Detected Faces", interactive=False)
                face_choice = gr.Radio(label="Face to Replace", choices=[], interactive=True)

        faces_state = gr.State([])
        frame_size_state = gr.State((0, 0))

        gr.Markdown("### 2. Upload the replacement face")
        face_reference = gr.Image(label="New Face (reference photo)", type="pil")

        gr.Markdown("### 3. Choose what to process")
        with gr.Row():
            start_time = gr.Number(label="Start Time (s)", value=0, precision=1)
            duration = gr.Slider(label="Duration to Process (s)", minimum=0.5, maximum=15, value=3, step=0.5)
            frame_stride = gr.Slider(label="Process Every Nth Frame", minimum=1, maximum=15, value=5, step=1)
            output_fps = gr.Slider(label="Output FPS", minimum=1, maximum=24, value=6, step=1)

        with gr.Row():
            steps = gr.Slider(label="Inference Steps", minimum=4, maximum=50, value=28, step=1)
            guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=6.0, value=1.0, step=0.5)

        process_btn = gr.Button("Process Video", variant="primary")
        output_video = gr.Video(label="Face-Swapped Output")
        output_meta = gr.Markdown("")

        _COMPATIBLE_TAGS = {"face-swap", "head-swap", "body-swap"}

        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, _COMPATIBLE_TAGS, cache_manager)
            return gr.update(choices=choices, value=None), gr.update(choices=[], value=None)

        _OTHER_BASE_HINTS = ("4b", "qwen", "krea", "ltx", "wan22")

        def _variant_choice(filename: str):
            lower = filename.lower()
            if any(hint in lower for hint in _OTHER_BASE_HINTS) and "9b" not in lower:
                return (f"{filename}  (❌ needs a different base model)", filename)
            return (filename, filename)

        def refresh_variants(model_id):
            if not model_id:
                return gr.update(choices=[], value=None)
            files = FaceSwapRunner.list_lora_weight_files(model_id)
            if not files:
                return gr.update(choices=[], value=None)
            choices = [_variant_choice(f) for f in files]
            likely_ok = [f for f in files if "9b" in f.lower() or not any(h in f.lower() for h in _OTHER_BASE_HINTS)]
            default = likely_ok[0] if likely_ok else files[0]
            return gr.update(choices=choices, value=default)

        def load_model(model_id, variant, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No LoRA selected."
            if not is_model_compatible(model_id, _COMPATIBLE_TAGS, cache_manager):
                return "**Status:** This model is not a recognized face-swap LoRA (marked with ❌)."
            if not variant:
                return "**Status:** No LoRA weight file selected — pick one from the dropdown."
            try:
                progress(0, desc=f"Loading {model_id}...")
                model_manager.load_model(model_id, "face-swap", lora_variant=variant)
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model():
            model_manager.unload_current()
            return update_model_status(model_manager)

        def detect_faces(video_path, t):
            if not video_path:
                return None, gr.update(choices=[], value=None), [], (0, 0)
            frame = _read_frame_at(video_path, t)
            if frame is None:
                return None, gr.update(choices=[], value=None), [], (0, 0)
            faces = _detect_faces(frame)
            annotated = _draw_face_boxes(frame, faces)
            h, w = frame.shape[:2]
            if not faces:
                return annotated, gr.update(choices=[], value=None), [], (w, h)
            labels = [f"Face {i + 1}" for i in range(len(faces))]
            return annotated, gr.update(choices=labels, value=labels[0]), faces, (w, h)

        def process_video(
            video_path, face_ref_img, faces, frame_wh, face_label,
            start_s, dur_s, stride, out_fps, num_steps, cfg, prompt_text, invert_order,
            progress: gr.Progress = gr.Progress(),
        ):
            if not model_manager.active_model_id:
                return None, "No model loaded."
            if not video_path:
                return None, "Upload a video first."
            if face_ref_img is None:
                return None, "Upload a replacement face image first."
            if not faces or not face_label:
                return None, "Detect and select a face first."

            try:
                face_idx = int(str(face_label).split()[-1]) - 1
            except (ValueError, IndexError):
                face_idx = -1
            if face_idx < 0 or face_idx >= len(faces):
                return None, "Selected face is out of range — re-run detection."

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = max(0, int(start_s * fps))
            end_frame = min(total_frames, int((start_s + dur_s) * fps))
            if end_frame <= start_frame:
                cap.release()
                return None, "Selected time range is empty — adjust start time/duration."

            box = _expand_box(*faces[face_idx], frame_wh[0], frame_wh[1])
            frame_indices = list(range(start_frame, end_frame, max(1, int(stride))))

            import imageio.v2 as imageio

            out_path = str(Path(tempfile.gettempdir()) / f"faceswap_{uuid.uuid4().hex}.mp4")
            writer = imageio.get_writer(out_path, fps=int(out_fps), macro_block_size=None)

            processed = 0
            errors = 0
            try:
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        continue
                    progress((i + 1) / len(frame_indices), desc=f"Processing frame {i + 1}/{len(frame_indices)}")

                    full_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    crop = full_img.crop(box)

                    result = model_manager.run_inference(
                        {"face_image": face_ref_img, "scene_image": crop},
                        num_inference_steps=int(num_steps),
                        guidance_scale=float(cfg),
                        prompt=prompt_text,
                        swap_image_order=bool(invert_order),
                    )
                    if not result.success:
                        errors += 1
                        logger.warning(f"Face-swap frame {frame_idx} failed: {result.error}")
                        continue

                    blended = _feathered_paste(full_img, result.output, box)
                    writer.append_data(np.array(blended))
                    processed += 1
            finally:
                writer.close()
                cap.release()

            if processed == 0:
                return None, f"**Status:** No frames were processed successfully ({errors} error(s))."

            meta = (
                f"**Processed:** {processed} frame(s) | **Errors:** {errors} | "
                f"**Output FPS:** {int(out_fps)}"
            )
            return out_path, meta

        refresh_btn.click(fn=refresh_models, outputs=[model_selector, lora_variant])
        model_selector.change(fn=refresh_variants, inputs=[model_selector], outputs=[lora_variant])
        load_btn.click(
            fn=load_model, inputs=[model_selector, lora_variant], outputs=[status_bar], concurrency_id="model_ops"
        )
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        detect_btn.click(
            fn=detect_faces,
            inputs=[video_input, frame_time],
            outputs=[face_preview, face_choice, faces_state, frame_size_state],
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                video_input, face_reference, faces_state, frame_size_state, face_choice,
                start_time, duration, frame_stride, output_fps, steps, guidance,
                prompt_box, swap_order,
            ],
            outputs=[output_video, output_meta],
            concurrency_id="model_ops",
        )

        return [model_selector]
