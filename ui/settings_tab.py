import gradio as gr
import logging

from core.device import DeviceManager
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from config.settings import Settings
from utils.formatting import format_bytes

logger = logging.getLogger("ai_agent_loader.ui.settings")


def create_settings_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
):
    with gr.Tab("Settings", id="settings"):
        gr.Markdown("## Settings & System Info")

        # --- Download Directory ---
        with gr.Accordion("Download Directory", open=True):
            gr.Markdown(
                "Set where models are downloaded and stored. "
                "The folder will be created if it does not exist."
            )
            with gr.Row():
                dir_input = gr.Textbox(
                    label="Download Directory",
                    value=str(Settings.DEFAULT_CACHE_DIR),
                    placeholder=r"e.g. D:\Models or C:\Users\you\models",
                    scale=4,
                )
                save_dir_btn = gr.Button("Save", variant="primary", scale=1)
            dir_status = gr.Markdown("")

        # --- GPU Info ---
        with gr.Accordion("GPU Information", open=True):
            gpu_info_display = gr.Markdown("")
            refresh_gpu_btn = gr.Button("Refresh GPU Info")

        # --- Cache Management ---
        with gr.Accordion("Cache Management", open=True):
            cache_info_display = gr.Markdown("")
            cached_models_table = gr.Dataframe(
                headers=["Model ID", "Size", "Files", "Path"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                label="Cached Models",
            )
            with gr.Row():
                refresh_cache_btn = gr.Button("Refresh Cache Info")
                delete_model_id = gr.Textbox(
                    label="Model ID to Delete",
                    placeholder="e.g. meta-llama/Llama-3.1-8B",
                    scale=2,
                )
                delete_btn = gr.Button("Delete from Cache", variant="stop", scale=1)

        # --- App Info ---
        with gr.Accordion("Application Info", open=False):
            gr.Markdown(
                f"**{Settings.APP_TITLE}** v{Settings.APP_VERSION}\n\n"
                f"**HF Token:** {'Configured' if Settings.HF_TOKEN else 'Not set — gated models (Llama, etc.) will be inaccessible'}"
            )

        # --- Event Handlers ---
        def save_download_dir(path: str):
            path = path.strip()
            if not path:
                return "**Error:** Please enter a directory path.", str(Settings.DEFAULT_CACHE_DIR)
            try:
                new_path = Settings.set_download_dir(path)
                # Update the cache manager to scan the new directory
                cache_manager._cache_dir = new_path
                return (
                    f"**Saved.** Models will be downloaded to `{new_path}`",
                    str(new_path),
                )
            except Exception as e:
                return f"**Error:** {e}", str(Settings.DEFAULT_CACHE_DIR)

        def get_gpu_info():
            info = DeviceManager.get_gpu_info()
            if info.available:
                pct = (
                    (info.vram_used_gb / info.vram_total_gb * 100)
                    if info.vram_total_gb > 0
                    else 0
                )
                return (
                    f"**GPU:** {info.device_name}\n\n"
                    f"**CUDA Version:** {info.cuda_version}\n\n"
                    f"**VRAM Total:** {info.vram_total_gb:.1f} GB\n\n"
                    f"**VRAM Used:** {info.vram_used_gb:.1f} GB ({pct:.0f}%)\n\n"
                    f"**VRAM Free:** {info.vram_free_gb:.1f} GB"
                )
            return "**GPU:** Not available. Running in CPU mode."

        def get_cache_info():
            cached = cache_manager.get_cached_models()
            total_size = sum(m.size_bytes for m in cached)
            current_dir = str(cache_manager._cache_dir)

            table_data = [
                [m.model_id, format_bytes(m.size_bytes), str(m.num_files), m.local_path]
                for m in cached
            ]

            summary = (
                f"**Directory:** `{current_dir}`\n\n"
                f"**Total cached models:** {len(cached)}\n\n"
                f"**Total cache size:** {format_bytes(total_size)}"
            )
            return summary, table_data

        def delete_model(model_id):
            if not model_id.strip():
                gr.Warning("Enter a model ID to delete.")
                return get_cache_info()
            if model_manager.active_model_id == model_id.strip():
                model_manager.unload_current()
            success = cache_manager.delete_model_cache(model_id.strip())
            if success:
                gr.Info(f"Deleted {model_id} from cache.")
            else:
                gr.Warning(f"Could not find {model_id} in cache.")
            return get_cache_info()

        # --- Wire Events ---
        save_dir_btn.click(
            fn=save_download_dir,
            inputs=[dir_input],
            outputs=[dir_status, dir_input],
        )
        refresh_gpu_btn.click(fn=get_gpu_info, outputs=[gpu_info_display])
        refresh_cache_btn.click(
            fn=get_cache_info,
            outputs=[cache_info_display, cached_models_table],
        )
        delete_btn.click(
            fn=delete_model,
            inputs=[delete_model_id],
            outputs=[cache_info_display, cached_models_table],
        )
