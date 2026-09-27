import gradio as gr

from config.settings import Settings
from core.model_manager import ModelManager


def create_api_info_tab(model_manager: ModelManager):
    with gr.Tab("API", id="api_info"):
        gr.Markdown("## LLM API")
        gr.Markdown(
            f"This app also exposes the currently loaded text-generation model over HTTP, "
            f"on the **same port** as this web UI (`{Settings.SERVER_PORT}`) — no separate "
            f"server to run. It operates on the same single active model as the Chat tab: "
            f"calling it with a different `model` swaps the loaded model, same as clicking "
            f"Load in the UI.\n\n"
            f"### Ollama-style\n"
            f"- `GET /api/tags` — list locally cached models\n"
            f"- `POST /api/generate` — `{{\"model\": \"...\", \"prompt\": \"...\"}}`\n"
            f"- `POST /api/chat` — `{{\"model\": \"...\", \"messages\": [{{\"role\": \"user\", \"content\": \"...\"}}]}}`\n\n"
            f"### OpenAI-compatible\n"
            f"- `GET /v1/models`\n"
            f"- `POST /v1/chat/completions` — standard OpenAI chat-completion request/response "
            f"shape, so existing OpenAI-SDK-based tools can point at this server.\n\n"
            f"### Other\n"
            f"- `GET /api/status` — currently loaded model, device, VRAM\n\n"
            f"**Note:** streaming (`\"stream\": true`) is not implemented yet — responses are "
            f"always returned as a single complete JSON object.\n\n"
            f"This API has no authentication, matching the rest of this app — it's reachable "
            f"by anything that can reach this machine on this port, same as the web UI itself."
        )

        gr.Markdown("### Try it")
        example = f"""```bash
curl http://localhost:{Settings.SERVER_PORT}/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "Say hello in five words."}}]}}'
```"""
        gr.Markdown(example)

        status_display = gr.Markdown("")
        refresh_btn = gr.Button("Refresh Status")

        def get_status():
            status = model_manager.get_active_status()
            if status["model_id"]:
                return (
                    f"**Currently reachable via the API:** `{status['model_id']}` "
                    f"({status['pipeline_tag']}, status: {status['status']})"
                )
            return "**Currently reachable via the API:** no model loaded — load one in the Chat tab, or pass `model` in your request."

        refresh_btn.click(fn=get_status, outputs=[status_display])
