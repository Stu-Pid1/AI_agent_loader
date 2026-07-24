import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

from utils.logging_config import setup_logging

setup_logging()

import gradio as gr
from ui.app_builder import build_app
from config.settings import Settings

if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=8)
    app.launch(
        server_name="0.0.0.0",
        server_port=Settings.SERVER_PORT,
        share=False,
        theme=gr.themes.Soft(),
    )
