import logging

import gradio as gr

from core.cache_manager import CacheManager
from core.conversation_store import Conversation, ConversationStore
from core.hub_client import HubClient
from core.model_manager import ModelManager
from ui.components import (
    build_compatible_dropdown_choices,
    is_model_compatible,
    slider_with_manual_override,
    update_model_status,
)

logger = logging.getLogger("ai_agent_loader.ui.chat")

_CHAT_TAGS = {
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
    "document-question-answering",
    "visual-question-answering",
}


def _conversation_choices(store: ConversationStore):
    convs = store.list_conversations()
    return [
        (f"{c['title']} ({c['num_messages']} msgs)", c["id"])
        for c in convs
    ]


def create_chat_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    store = ConversationStore()

    with gr.Tab("Chat", id="chat"):
        gr.Markdown("## Chat")
        gr.Markdown(
            "A multi-turn chat interface (like Ollama's chat UI) with full conversation "
            "memory — every message is sent along with the full history so far. Save a "
            "conversation to come back to it later, or start a new one at any time.\n\n"
            "This same loaded model is also reachable programmatically — see the **API** "
            "tab for the REST endpoints (Ollama-style and OpenAI-compatible)."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Select Model", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Conversations")
                conversation_list = gr.Dropdown(
                    label="Saved Conversations", choices=_conversation_choices(store), interactive=True
                )
                with gr.Row():
                    refresh_conv_btn = gr.Button("Refresh List", scale=1)
                    load_conv_btn = gr.Button("Load", scale=1)
                delete_conv_btn = gr.Button("Delete Selected", variant="stop")
                gr.Markdown("---")
                conversation_title = gr.Textbox(
                    label="Conversation Title",
                    placeholder="Leave blank to auto-title from the first message",
                )
                save_conv_btn = gr.Button("Save Conversation", variant="primary")
                new_conv_btn = gr.Button("New Conversation")
                conv_status = gr.Markdown("")

                with gr.Accordion("System Prompt", open=False):
                    system_prompt = gr.Textbox(
                        label="System Prompt (optional)",
                        placeholder="You are a helpful assistant...",
                        lines=3,
                    )

                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens = slider_with_manual_override("Max Tokens", minimum=32, maximum=1048576, value=512, step=32)
                    temperature = slider_with_manual_override("Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.05)
                    top_p = slider_with_manual_override("Top-p", minimum=0.0, maximum=1.0, value=0.9, step=0.05)
                    top_k = slider_with_manual_override("Top-k", minimum=1, maximum=200, value=50, step=1)
                    rep_penalty = slider_with_manual_override(
                        "Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05
                    )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=520)
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="Type a message and press Enter or click Send...",
                        lines=2,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                regenerate_btn = gr.Button("Regenerate Last Response")

        history_state = gr.State([])  # list[{"role": ..., "content": ...}]
        conversation_id_state = gr.State("")

        # --- Model handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, _CHAT_TAGS, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, _CHAT_TAGS, cache_manager):
                return "**Status:** This model is not compatible with Chat (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                tag = cache_manager.get_local_model_metadata(model_id).pipeline_tag or "text-generation"
                model_manager.load_model(model_id, tag)
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model():
            model_manager.unload_current()
            return update_model_status(model_manager)

        # --- Chat handlers ---
        def _run_turn(history, sys_prompt, max_tok, temp, tp, tk, rep_pen):
            messages = ([{"role": "system", "content": sys_prompt.strip()}] if sys_prompt and sys_prompt.strip() else [])
            messages += history
            result = model_manager.run_inference(
                messages,
                max_new_tokens=int(max_tok),
                temperature=float(temp),
                top_p=float(tp),
                top_k=int(tk),
                repetition_penalty=float(rep_pen),
            )
            if result.success:
                return result.output, None
            return None, result.error

        def send_message(user_msg, history, sys_prompt, max_tok, temp, tp, tk, rep_pen):
            if not model_manager.active_model_id:
                gr.Warning("No model loaded — load one above first.")
                return history, history, ""
            if not user_msg or not user_msg.strip():
                return history, history, ""

            history = list(history) + [{"role": "user", "content": user_msg.strip()}]
            reply, error = _run_turn(history, sys_prompt, max_tok, temp, tp, tk, rep_pen)
            if error:
                history = history + [{"role": "assistant", "content": f"⚠️ Error: {error}"}]
            else:
                history = history + [{"role": "assistant", "content": reply}]
            return history, history, ""

        def regenerate(history, sys_prompt, max_tok, temp, tp, tk, rep_pen):
            if not model_manager.active_model_id:
                gr.Warning("No model loaded — load one above first.")
                return history, history
            if not history:
                return history, history
            trimmed = list(history)
            if trimmed and trimmed[-1].get("role") == "assistant":
                trimmed = trimmed[:-1]
            if not trimmed:
                return history, history
            reply, error = _run_turn(trimmed, sys_prompt, max_tok, temp, tp, tk, rep_pen)
            if error:
                trimmed = trimmed + [{"role": "assistant", "content": f"⚠️ Error: {error}"}]
            else:
                trimmed = trimmed + [{"role": "assistant", "content": reply}]
            return trimmed, trimmed

        def new_conversation():
            return [], [], "", "", ""

        # --- Persistence handlers ---
        def save_conversation(history, title, conv_id, sys_prompt):
            if not history:
                return "**Status:** Nothing to save yet — send a message first.", conv_id, gr.update()
            conv = Conversation(
                id=conv_id or "",
                title=(title or "").strip(),
                model_id=model_manager.active_model_id,
                system_prompt=sys_prompt or "",
                messages=history,
            )
            new_id = store.save(conv)
            return (
                f"**Saved** as `{conv.title}`.",
                new_id,
                gr.update(choices=_conversation_choices(store), value=new_id),
            )

        def load_conversation(conv_id):
            if not conv_id:
                return [], [], "", "", conv_id, "**Status:** Select a conversation first."
            conv = store.load(conv_id)
            if conv is None:
                return [], [], "", "", conv_id, "**Status:** Could not find that conversation."
            return (
                conv.messages,
                conv.messages,
                conv.system_prompt,
                conv.title,
                conv.id,
                f"**Loaded** `{conv.title}` ({len(conv.messages)} messages). "
                + (f"Originally used model `{conv.model_id}`." if conv.model_id else ""),
            )

        def delete_conversation(conv_id):
            if not conv_id:
                return "**Status:** Select a conversation first.", gr.update()
            deleted = store.delete(conv_id)
            msg = "**Deleted.**" if deleted else "**Status:** Could not find that conversation."
            return msg, gr.update(choices=_conversation_choices(store), value=None)

        def refresh_conversation_list():
            return gr.update(choices=_conversation_choices(store))

        # --- Wire events ---
        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        gen_inputs = [history_state, system_prompt, max_tokens, temperature, top_p, top_k, rep_penalty]

        send_btn.click(
            fn=send_message,
            inputs=[user_input] + gen_inputs,
            outputs=[chatbot, history_state, user_input],
            concurrency_id="model_ops",
        )
        user_input.submit(
            fn=send_message,
            inputs=[user_input] + gen_inputs,
            outputs=[chatbot, history_state, user_input],
            concurrency_id="model_ops",
        )
        regenerate_btn.click(
            fn=regenerate,
            inputs=gen_inputs,
            outputs=[chatbot, history_state],
            concurrency_id="model_ops",
        )

        new_conv_btn.click(
            fn=new_conversation,
            outputs=[chatbot, history_state, conversation_id_state, conversation_title, conv_status],
        )

        save_conv_btn.click(
            fn=save_conversation,
            inputs=[history_state, conversation_title, conversation_id_state, system_prompt],
            outputs=[conv_status, conversation_id_state, conversation_list],
        )
        load_conv_btn.click(
            fn=load_conversation,
            inputs=[conversation_list],
            outputs=[chatbot, history_state, system_prompt, conversation_title, conversation_id_state, conv_status],
        )
        delete_conv_btn.click(
            fn=delete_conversation,
            inputs=[conversation_list],
            outputs=[conv_status, conversation_list],
        )
        refresh_conv_btn.click(fn=refresh_conversation_list, outputs=[conversation_list])

        return [model_selector]
