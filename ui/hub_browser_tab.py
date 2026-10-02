import time

import gradio as gr
import pandas as pd
import logging
from pathlib import Path

from core.hub_client import HubClient
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from config.settings import Settings
from ui.components import slider_with_manual_override
from utils.formatting import format_bytes, format_number
from utils.gguf import ALL_VARIANTS_LABEL, group_gguf_variants, pick_default_variant

logger = logging.getLogger("ai_agent_loader.ui.hub_browser")


def create_hub_browser_tab(
    hub_client: HubClient,
    cache_manager: CacheManager,
    model_manager: ModelManager,
    download_queue_state,
):
    with gr.Tab("Hub Browser", id="hub_browser"):
        gr.Markdown("## Hugging Face Model Browser")
        gr.Markdown("Search, explore, and download models from Hugging Face Hub.")

        # --- Search Controls ---
        with gr.Row():
            search_query = gr.Textbox(
                label="Search",
                placeholder="e.g. llama, stable-diffusion, whisper...",
                scale=3,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            task_filter = gr.Dropdown(
                label="Task Type",
                choices=["All"] + Settings.SUPPORTED_PIPELINE_TAGS,
                value="All",
                scale=1,
            )
            library_filter = gr.Dropdown(
                label="Library",
                choices=["All"] + Settings.SUPPORTED_LIBRARIES,
                value="All",
                scale=1,
            )
            sort_by = gr.Dropdown(
                label="Sort By",
                choices=list(Settings.SEARCH_SORT_OPTIONS.keys()),
                value="Downloads",
                scale=1,
            )
            result_limit = slider_with_manual_override(
                "Results",
                minimum=5,
                maximum=Settings.MAX_SEARCH_LIMIT,
                value=Settings.DEFAULT_SEARCH_LIMIT,
                step=5,
                scale=1,
            )

        # --- State ---
        search_results_state = gr.State([])
        selected_model_state = gr.State(None)
        gguf_variants_state = gr.State([])

        # --- Results Layout ---
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                results_table = gr.Dataframe(
                    headers=["Model ID", "Author", "Task", "Downloads", "Likes", "Library", "Cached"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Search Results",
                    wrap=True,
                )
            with gr.Column(scale=1):
                quant_selector = gr.Dropdown(
                    label="Quantization to Download",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )
                quick_download_btn = gr.Button("Quick Download", variant="primary", interactive=False)
                quick_load_btn = gr.Button("Load & Run", variant="secondary", interactive=False)
                quick_save_dir = gr.Textbox(
                    label="Save Folder",
                    value=str(Settings.DEFAULT_CACHE_DIR),
                    placeholder=r"e.g. G:\AI",
                )
                quick_save_btn = gr.Button("Save Folder")
                quick_status = gr.Textbox(label="Status", interactive=False, value="")
                queue_preview = gr.JSON(label="Queued Downloads", value=[])

        # --- Model Detail Panel ---
        with gr.Accordion("Model Details", open=False) as detail_accordion:
            with gr.Row():
                with gr.Column(scale=2):
                    detail_header = gr.Markdown("*Select a model from the search results above.*")
                    detail_files = gr.Markdown("")
                with gr.Column(scale=1):
                    with gr.Row():
                        download_btn = gr.Button(
                            "Download Model",
                            variant="primary",
                            interactive=False,
                        )
                        load_btn = gr.Button(
                            "Load & Run",
                            variant="secondary",
                            interactive=False,
                        )
                    download_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="",
                    )

            with gr.Accordion("Model Card", open=False):
                model_card_display = gr.Markdown("*No model selected.*")

        def build_queue_rows(queue_items):
            return [
                {
                    "model_id": item.get("model_id", ""),
                    "status": item.get("status", "queued"),
                    "save_path": item.get("save_path", str(Settings.DEFAULT_CACHE_DIR)),
                }
                for item in (queue_items or [])
            ]

        def update_save_dir(path: str):
            path = (path or "").strip()
            if not path:
                return "**Error:** Please enter a directory path.", str(Settings.DEFAULT_CACHE_DIR), []
            try:
                new_path = Settings.set_download_dir(path)
                cache_manager._cache_dir = new_path
                return (
                    f"**Saved.** Models will be downloaded to `{new_path}`",
                    str(new_path),
                    build_queue_rows(download_queue_state.value),
                )
            except Exception as e:
                return f"**Error:** {e}", str(Settings.DEFAULT_CACHE_DIR), build_queue_rows(download_queue_state.value)

        # --- Event Handlers ---
        def do_search(query, task, library, sort, limit):
            try:
                sort_value = Settings.SEARCH_SORT_OPTIONS.get(sort, "downloads")
                results = hub_client.search_models(
                    query=query,
                    pipeline_tag=task if task != "All" else None,
                    library=library if library != "All" else None,
                    sort=sort_value,
                    limit=int(limit),
                )

                cached_ids = set(cache_manager.get_cached_model_ids())

                table_data = []
                for r in results:
                    table_data.append([
                        r.model_id,
                        r.author,
                        r.pipeline_tag or "—",
                        format_number(r.downloads),
                        format_number(r.likes),
                        r.library_name or "—",
                        "Yes" if r.model_id in cached_ids else "",
                    ])

                df = pd.DataFrame(
                    table_data,
                    columns=["Model ID", "Author", "Task", "Downloads", "Likes", "Library", "Cached"],
                )

                return df, results

            except Exception as e:
                logger.error(f"Search error: {e}")
                gr.Warning(f"Search failed: {e}")
                return pd.DataFrame(), []

        def _quant_dropdown_choices(variants):
            choices = [
                (f"{v.label} — {format_bytes(v.size_bytes)}", v.label) for v in variants
            ]
            total = sum(v.size_bytes for v in variants)
            choices.append((f"All quantizations — {format_bytes(total)} total", ALL_VARIANTS_LABEL))
            return choices

        def on_row_select(results, evt: gr.SelectData):
            if not results or evt.index[0] >= len(results):
                return (
                    "*Select a model from the search results above.*",
                    "",
                    "*No model selected.*",
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(choices=[], value=None, visible=False),
                    [],
                )

            selected = results[evt.index[0]]
            model_id = selected.model_id

            try:
                detail = hub_client.get_model_detail(model_id)
                is_cached = cache_manager.is_model_cached(model_id)
                variants = group_gguf_variants(detail.siblings)

                header = (
                    f"### {model_id}\n\n"
                    f"**Task:** {detail.pipeline_tag or 'Unknown'} | "
                    f"**Library:** {detail.library_name or 'Unknown'} | "
                    f"**Downloads:** {format_number(detail.downloads)} | "
                    f"**Likes:** {format_number(detail.likes)}"
                )

                if detail.gated:
                    header += "\n\n> **Gated model** — requires HF token and access approval."

                files_text = "**Files:**\n\n"
                for f in detail.siblings[:20]:
                    size = format_bytes(f.get("size")) if f.get("size") else "?"
                    files_text += f"- `{f['filename']}` — {size}\n"
                if len(detail.siblings) > 20:
                    files_text += f"\n*...and {len(detail.siblings) - 20} more files*\n"

                total = format_bytes(detail.total_size_bytes) if detail.total_size_bytes else "Unknown"
                files_text += f"\n**Total size:** {total}"

                if variants:
                    files_text += (
                        f"\n\n**{len(variants)} GGUF quantization(s) available** — pick one in "
                        f"'Quantization to Download' to only fetch that version."
                    )

                if is_cached:
                    cached_size = cache_manager.get_model_cache_size(model_id)
                    files_text += f"\n\n**Cached locally** ({format_bytes(cached_size)})"

                card = hub_client.get_model_card(model_id)
                status_text = "Already downloaded" if is_cached else "Ready to download"

                if variants:
                    quant_update = gr.update(
                        choices=_quant_dropdown_choices(variants),
                        value=pick_default_variant(variants),
                        visible=True,
                    )
                else:
                    quant_update = gr.update(choices=[], value=None, visible=False)

                return (
                    header,
                    files_text,
                    card,
                    model_id,
                    gr.update(interactive=True),
                    gr.update(interactive=is_cached),
                    status_text,
                    gr.update(interactive=True),
                    gr.update(interactive=is_cached),
                    quant_update,
                    variants,
                )

            except Exception as e:
                logger.error(f"Error loading details for {model_id}: {e}")
                return (
                    f"*Error loading details for {model_id}*",
                    "",
                    "*Error*",
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    f"Error: {e}",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(choices=[], value=None, visible=False),
                    [],
                )

        def do_download(
            model_id, queue_items, save_path, quant_choice, variants,
            progress: gr.Progress = gr.Progress(track_tqdm=True),
        ):
            if not model_id:
                return "No model selected.", "", gr.update(interactive=False), queue_items

            allow_patterns = None
            selected_size = 0
            if variants and quant_choice and quant_choice != ALL_VARIANTS_LABEL:
                match = next((v for v in variants if v.label == quant_choice), None)
                if match:
                    allow_patterns = list(match.files)
                    selected_size = match.size_bytes

            target_dir = (save_path or "").strip() or str(Settings.DEFAULT_CACHE_DIR)
            queue = list(queue_items or [])
            total_bytes = selected_size or (hub_client.get_model_detail(model_id).total_size_bytes or 0)
            item_found = False
            for item in queue:
                if item.get("model_id") == model_id:
                    item["status"] = "downloading"
                    item["save_path"] = target_dir
                    item["started_at"] = time.time()
                    item["total_bytes"] = total_bytes
                    item_found = True
                    break
            if not item_found:
                queue.append({
                    "model_id": model_id,
                    "status": "downloading",
                    "save_path": target_dir,
                    "started_at": time.time(),
                    "total_bytes": total_bytes,
                    "progress": 0,
                    "eta_seconds": 0,
                    "remaining_bytes": 0,
                })

            quant_note = f" ({quant_choice})" if allow_patterns else ""
            yield (
                f"Queued {model_id}{quant_note}...",
                f"Queued {model_id}{quant_note} to {target_dir}.",
                gr.update(interactive=False),
                queue,
            )

            try:
                progress(5, desc=f"Downloading {model_id}{quant_note}...")
                new_path = Settings.set_download_dir(target_dir)
                cache_manager._cache_dir = Path(new_path)
                hub_client.download_model(model_id, cache_dir=str(new_path), allow_patterns=allow_patterns)
                for item in queue:
                    if item.get("model_id") == model_id:
                        item["status"] = "completed"
                        item["save_path"] = str(new_path)
                        item["progress"] = 100
                        item["eta_seconds"] = 0
                        item["remaining_bytes"] = 0
                        break
                yield (
                    f"Downloaded {model_id}{quant_note} successfully!",
                    f"Downloaded {model_id}{quant_note} to {new_path}.",
                    gr.update(interactive=True),
                    queue,
                )
            except Exception as e:
                for item in queue:
                    if item.get("model_id") == model_id:
                        item["status"] = "failed"
                        item["progress"] = item.get("progress", 0)
                        item["eta_seconds"] = 0
                        item["remaining_bytes"] = item.get("remaining_bytes", 0)
                        break
                yield (
                    f"Download failed: {e}",
                    f"Download failed: {e}",
                    gr.update(interactive=False),
                    queue,
                )

        def do_load_and_run(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "No model selected.", ""

            try:
                progress(0, desc=f"Loading {model_id}...")
                detail = hub_client.get_model_detail(model_id)
                tag = detail.pipeline_tag
                if not tag:
                    return f"Cannot determine task type for {model_id}.", ""

                model_manager.load_model(model_id, tag)
                return f"Loaded {model_id} — switch to the appropriate runner tab.", ""
            except Exception as e:
                return f"Load failed: {e}", ""

        # --- Wire Events ---
        search_btn.click(
            fn=do_search,
            inputs=[search_query, task_filter, library_filter, sort_by, result_limit],
            outputs=[results_table, search_results_state],
        )

        search_query.submit(
            fn=do_search,
            inputs=[search_query, task_filter, library_filter, sort_by, result_limit],
            outputs=[results_table, search_results_state],
        )

        results_table.select(
            fn=on_row_select,
            inputs=[search_results_state],
            outputs=[
                detail_header,
                detail_files,
                model_card_display,
                selected_model_state,
                download_btn,
                load_btn,
                download_status,
                quick_download_btn,
                quick_load_btn,
                quant_selector,
                gguf_variants_state,
            ],
        )

        download_btn.click(
            fn=do_download,
            inputs=[selected_model_state, download_queue_state, quick_save_dir, quant_selector, gguf_variants_state],
            outputs=[download_status, quick_status, load_btn, download_queue_state],
            concurrency_id="model_ops",
        )

        quick_download_btn.click(
            fn=do_download,
            inputs=[selected_model_state, download_queue_state, quick_save_dir, quant_selector, gguf_variants_state],
            outputs=[quick_status, download_status, load_btn, download_queue_state],
            concurrency_id="model_ops",
        )

        quick_save_btn.click(
            fn=update_save_dir,
            inputs=[quick_save_dir],
            outputs=[quick_status, quick_save_dir, queue_preview],
        )

        load_btn.click(
            fn=do_load_and_run,
            inputs=[selected_model_state],
            outputs=[download_status, quick_status],
            concurrency_id="model_ops",
        )

        quick_load_btn.click(
            fn=do_load_and_run,
            inputs=[selected_model_state],
            outputs=[quick_status, download_status],
            concurrency_id="model_ops",
        )
