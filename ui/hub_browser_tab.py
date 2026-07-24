import gradio as gr
import pandas as pd
import logging
from typing import Optional

from core.hub_client import HubClient
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from config.settings import Settings
from utils.formatting import format_bytes, format_number

logger = logging.getLogger("ai_agent_loader.ui.hub_browser")


def create_hub_browser_tab(
    hub_client: HubClient,
    cache_manager: CacheManager,
    model_manager: ModelManager,
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
            result_limit = gr.Slider(
                label="Results",
                minimum=5,
                maximum=Settings.MAX_SEARCH_LIMIT,
                value=Settings.DEFAULT_SEARCH_LIMIT,
                step=5,
                scale=1,
            )

        # --- State ---
        search_results_state = gr.State([])
        selected_model_state = gr.State(None)

        # --- Results Table ---
        results_table = gr.Dataframe(
            headers=["Model ID", "Author", "Task", "Downloads", "Likes", "Library", "Cached"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="Search Results",
            wrap=True,
        )

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
                )

            selected = results[evt.index[0]]
            model_id = selected.model_id

            try:
                detail = hub_client.get_model_detail(model_id)
                is_cached = cache_manager.is_model_cached(model_id)

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

                if is_cached:
                    cached_size = cache_manager.get_model_cache_size(model_id)
                    files_text += f"\n\n**Cached locally** ({format_bytes(cached_size)})"

                card = hub_client.get_model_card(model_id)

                status_text = "Already downloaded" if is_cached else "Ready to download"

                return (
                    header,
                    files_text,
                    card,
                    model_id,
                    gr.update(interactive=True),
                    gr.update(interactive=is_cached),
                    status_text,
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
                )

        def do_download(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "No model selected.", gr.update(interactive=False)

            try:
                yield f"Downloading {model_id}...", gr.update(interactive=False)
                hub_client.download_model(model_id)
                yield (
                    f"Downloaded {model_id} successfully!",
                    gr.update(interactive=True),
                )
            except Exception as e:
                yield f"Download failed: {e}", gr.update(interactive=False)

        def do_load_and_run(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "No model selected."

            try:
                progress(0, desc=f"Loading {model_id}...")
                detail = hub_client.get_model_detail(model_id)
                tag = detail.pipeline_tag
                if not tag:
                    return f"Cannot determine task type for {model_id}."

                model_manager.load_model(model_id, tag)
                return f"Loaded {model_id} — switch to the appropriate runner tab."
            except Exception as e:
                return f"Load failed: {e}"

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
            ],
        )

        download_btn.click(
            fn=do_download,
            inputs=[selected_model_state],
            outputs=[download_status, load_btn],
            concurrency_id="model_ops",
        )

        load_btn.click(
            fn=do_load_and_run,
            inputs=[selected_model_state],
            outputs=[download_status],
            concurrency_id="model_ops",
        )
