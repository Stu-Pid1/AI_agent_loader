import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

from utils.logging_config import setup_logging

setup_logging()

import gradio as gr
import uvicorn
from fastapi import FastAPI

from ui.app_builder import build_app
from core.api_routes import create_api_router
from config.settings import Settings

if __name__ == "__main__":
    blocks, model_manager, cache_manager = build_app()
    blocks.queue(default_concurrency_limit=8)

    # Mount the LLM REST API (Ollama-style + OpenAI-compatible) alongside the
    # Gradio UI on the same port, rather than launching Gradio's own server
    # directly — this is the supported way to add custom routes to a Gradio app.
    fastapi_app = FastAPI(title=f"{Settings.APP_TITLE} API")
    fastapi_app.include_router(create_api_router(model_manager, cache_manager))
    app = gr.mount_gradio_app(fastapi_app, blocks, path="/", theme=gr.themes.Soft())

    uvicorn.run(app, host="0.0.0.0", port=Settings.SERVER_PORT)
