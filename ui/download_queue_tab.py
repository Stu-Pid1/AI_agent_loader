import os
import time
from pathlib import Path

import gradio as gr

from config.settings import Settings


def create_download_queue_tab(download_queue_state, cache_manager, hub_client):
    with gr.Tab("Download Queue", id="download_queue"):
        gr.Markdown("## Download Queue")
        gr.Markdown("Track queued model downloads, cancel pending entries, and change the save directory.")

        with gr.Row():
            queue_dir_input = gr.Textbox(
                label="Save Folder",
                value=str(Settings.DEFAULT_CACHE_DIR),
                placeholder=r"e.g. G:\AI\models",
                scale=4,
            )
            queue_dir_btn = gr.Button("Update Save Folder", variant="primary", scale=1)
        queue_dir_status = gr.Textbox(label="Folder Status", interactive=False, value="")

        selected_queue_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                queue_table = gr.Dataframe(
                    headers=["Model ID", "Status", "Progress", "ETA", "Remaining"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Queued Downloads",
                )
            with gr.Column(scale=1):
                cancel_btn = gr.Button("Remove Selected", variant="stop")
                refresh_btn = gr.Button("Refresh Queue")

        downloaded_table = gr.Dataframe(
            headers=["Model Name", "Type", "Size", "Files", "Location"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Downloaded Models",
        )

        queue_status = gr.Markdown("")
        queue_progress_display = gr.HTML("<div style='min-height: 44px;'>No active downloads.</div>")
        queue_timer = gr.Timer(value=1)

        def format_bytes(value):
            if value is None or value <= 0:
                return "0 B"
            units = ["B", "KB", "MB", "GB", "TB"]
            size = float(value)
            idx = 0
            while size >= 1024 and idx < len(units) - 1:
                size /= 1024
                idx += 1
            return f"{size:.1f} {units[idx]}" if idx > 0 else f"{size:.0f} {units[idx]}"

        def format_eta(seconds):
            if seconds is None or seconds <= 0:
                return "--"
            total_seconds = max(0, int(seconds))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            if hours:
                return f"{hours}h {minutes}m"
            if minutes:
                return f"{minutes}m {secs}s"
            return f"{secs}s"

        def estimate_model_progress(item):
            model_id = item.get("model_id")
            total_bytes = item.get("total_bytes") or 0
            save_path = item.get("save_path") or str(Settings.DEFAULT_CACHE_DIR)
            status = item.get("status")
            if not model_id:
                return 0, 0, 0, "queued"

            if status == "completed":
                return 100, 0, 0, "completed"

            if status == "failed":
                return 0, 0, 0, "failed"

            if total_bytes <= 0:
                try:
                    detail = hub_client.get_model_detail(model_id)
                    total_bytes = detail.total_size_bytes or 0
                    item["total_bytes"] = total_bytes
                except Exception:
                    total_bytes = 0

            current_bytes = 0
            if save_path:
                try:
                    p = Path(save_path)
                    if p.exists():
                        for path in p.rglob('*'):
                            if path.is_file() and path.name not in {".DS_Store"}:
                                try:
                                    current_bytes += path.stat().st_size
                                except OSError:
                                    pass
                except Exception:
                    current_bytes = 0

            if total_bytes > 0 and current_bytes > 0:
                progress = max(0, min(100, (current_bytes / total_bytes) * 100))
                remaining = max(0, total_bytes - current_bytes)
                started_at = item.get("started_at")
                elapsed = max(1.0, time.time() - float(started_at)) if started_at else 1.0
                if current_bytes > 0 and elapsed > 0:
                    current_rate = current_bytes / elapsed
                    eta = remaining / current_rate if current_rate > 0 else 0
                else:
                    eta = 0
            else:
                progress = 0
                remaining = 0
                eta = 0

            return progress, eta, remaining, status

        def make_progress_bar(progress):
            value = max(0, min(100, float(progress)))
            filled = int(value / 5)
            bar = "█" * filled + "░" * (20 - filled)
            return f"<div style='display:flex; align-items:center; gap:10px;'><div style='width:140px; border:1px solid #d0d7de; border-radius:6px; overflow:hidden; background:#f6f8fa; height:12px;'><div style='width:{value:.1f}%; height:100%; background:linear-gradient(90deg, #3b82f6, #22c55e);'></div></div><span>{value:.0f}%</span></div>"

        def render_queue_display(queue_items):
            active = [item for item in (queue_items or []) if item.get("status") in {"queued", "downloading"}]
            if not active:
                return "<div style='min-height: 44px;'>No active downloads.</div>"

            parts = []
            for item in active[:3]:
                model_id = item.get("model_id", "Unknown")
                progress, eta, remaining, status = estimate_model_progress(item)
                progress_bar = make_progress_bar(progress)
                parts.append(
                    f"<div style='margin-bottom:10px;'><strong>{model_id}</strong><br>{progress_bar}<br><small>{status.title()} • ETA {format_eta(eta)} • Remaining {format_bytes(remaining)}</small></div>"
                )
            return "<div style='min-height: 44px;'>" + "".join(parts) + "</div>"

        def format_queue_rows(queue_items):
            rows = []
            for item in queue_items or []:
                progress, eta, remaining, status = estimate_model_progress(item)
                rows.append([
                    item.get("model_id", ""),
                    status.title() if status else "Queued",
                    f"{progress:.0f}%",
                    format_eta(eta),
                    format_bytes(remaining),
                ])
            return rows

        def on_queue_row_select(queue_items, evt: gr.SelectData):
            if not queue_items or evt.index[0] >= len(queue_items):
                return None
            return queue_items[evt.index[0]]

        def update_save_dir(path: str):
            path = (path or "").strip()
            if not path:
                return "**Error:** Please enter a directory path.", str(Settings.DEFAULT_CACHE_DIR)
            try:
                new_path = Settings.set_download_dir(path)
                cache_manager._cache_dir = new_path
                return f"**Saved.** Downloads will go to `{new_path}`", str(new_path)
            except Exception as e:
                return f"**Error:** {e}", str(Settings.DEFAULT_CACHE_DIR)

        def list_downloaded_models():
            rows = []
            for model in cache_manager.get_cached_models():
                # Read from the local snapshot only — an already-downloaded
                # model should never need a Hub call just to display its type.
                meta = cache_manager.get_local_model_metadata(model.model_id)
                model_type = meta.pipeline_tag or meta.library_name or "Unknown"
                rows.append([
                    model.model_id,
                    model_type,
                    format_bytes(model.size_bytes),
                    str(model.num_files),
                    model.local_path,
                ])
            return rows

        def refresh_queue(queue_items):
            rows = format_queue_rows(queue_items)
            display = render_queue_display(queue_items)
            downloaded_rows = list_downloaded_models()
            if rows:
                return rows, display, "", downloaded_rows
            return rows, display, "*No items in the queue.*", downloaded_rows

        def tick_queue(queue_items):
            # Runs every second while this tab exists — must stay local/cheap.
            # The Downloaded Models table (which needs a Hub lookup per model)
            # is intentionally excluded here and only rebuilt on explicit
            # user actions, otherwise it would hammer the HF API once per
            # cached model every second and blow through rate limits.
            rows = format_queue_rows(queue_items)
            display = render_queue_display(queue_items)
            status = "" if rows else "*No items in the queue.*"
            return rows, display, status

        def remove_selected(queue_items, selected_item):
            if not selected_item:
                return (
                    format_queue_rows(queue_items),
                    render_queue_display(queue_items),
                    "**Select a row from the queue to remove it.**",
                    list_downloaded_models(),
                    queue_items,
                )

            model_id = selected_item.get("model_id")
            if not model_id:
                return (
                    format_queue_rows(queue_items),
                    render_queue_display(queue_items),
                    "**Selected queue item is missing a model ID.**",
                    list_downloaded_models(),
                    queue_items,
                )

            filtered = [item for item in (queue_items or []) if item.get("model_id") != model_id]
            return (
                format_queue_rows(filtered),
                render_queue_display(filtered),
                f"**Removed** `{model_id}` from the queue.",
                list_downloaded_models(),
                filtered,
            )

        queue_dir_btn.click(
            fn=update_save_dir,
            inputs=[queue_dir_input],
            outputs=[queue_dir_status, queue_dir_input],
        )

        refresh_btn.click(
            fn=refresh_queue,
            inputs=[download_queue_state],
            outputs=[queue_table, queue_progress_display, queue_status, downloaded_table],
        )

        queue_table.select(
            fn=on_queue_row_select,
            inputs=[download_queue_state],
            outputs=[selected_queue_state],
        )

        cancel_btn.click(
            fn=remove_selected,
            inputs=[download_queue_state, selected_queue_state],
            outputs=[queue_table, queue_progress_display, queue_status, downloaded_table, download_queue_state],
        )

        queue_timer.tick(
            fn=tick_queue,
            inputs=[download_queue_state],
            outputs=[queue_table, queue_progress_display, queue_status],
        )
