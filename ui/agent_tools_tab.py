import json
import logging
import shutil

import gradio as gr

from core import security_tools, tool_agent
from core.cache_manager import CacheManager
from core.hub_client import HubClient
from core.model_manager import ModelManager
from core.scope_manager import ScopeManager
from ui.components import (
    build_compatible_dropdown_choices,
    is_model_compatible,
    slider_with_manual_override,
    update_model_status,
)

logger = logging.getLogger("ai_agent_loader.ui.agent_tools")

_CHAT_TAGS = {
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
    "document-question-answering",
    "visual-question-answering",
}

_REQUIRED_BINARIES = ["nmap", "nikto", "nuclei"]


def _tool_availability_markdown() -> str:
    lines = ["**Tool availability:**"]
    for tool in security_tools.list_tools():
        if tool.requires_binary:
            found = shutil.which(tool.requires_binary) is not None
            lines.append(f"- `{tool.name}` — {'✅ found' if found else '❌ not installed'} (`{tool.requires_binary}`)")
        else:
            lines.append(f"- `{tool.name}` — ✅ built in")
    return "\n".join(lines)


def _scope_rows(scope: ScopeManager):
    return [[e.target, e.note, e.added_at] for e in scope.list_entries()]


def _format_proposal(step: tool_agent.AgentStep) -> str:
    args_json = json.dumps(step.tool_arguments, indent=2)
    lines = [f"### Proposed tool call: `{step.tool_name}`", "```json", args_json, "```"]
    if step.in_scope:
        lines.append("✅ Target is in scope — Approve to execute, or Deny.")
    else:
        lines.append(f"❌ **Blocked — cannot approve:** {step.scope_reason}")
    return "\n".join(lines)


def create_agent_tools_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    scope = ScopeManager()

    with gr.Tab("Security Agent", id="security_agent"):
        gr.Markdown("## Security Agent")
        gr.Markdown(
            "Turns the loaded LLM into a tool-using agent for security testing — recon, active "
            "scanning, and LLM-authored PoC execution — using OpenAI-compatible function-calling "
            "schemas under the hood (prompt-based, so it works with any loaded chat model, not "
            "just ones with a native tool-calling template).\n\n"
            "**Two safety rails, always on:** (1) the agent can only act on targets you've "
            "explicitly added to the scope list below — every proposed call is checked against "
            "it before you ever see an Approve button; (2) **every single tool call requires your "
            "explicit approval** — you see the exact tool, arguments, and (for the exploit runner) "
            "the full generated code before anything executes. Nothing runs unattended.\n\n"
            "Only add targets you're authorized to test. The exploit runner has no sandboxing "
            "beyond a timeout — it runs with this app's own privileges on this machine, so review "
            "generated code carefully before approving it."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Select Model", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)

        status_bar = gr.Markdown("**Status:** No model loaded")

        with gr.Row():
            n_ctx = slider_with_manual_override(
                "Context Window (n_ctx) — GGUF models only", minimum=512, maximum=32768, value=4096, step=512
            )

        with gr.Accordion("Scope (targets the agent may act on)", open=True):
            scope_table = gr.Dataframe(
                headers=["Target", "Note", "Added"],
                datatype=["str", "str", "str"],
                interactive=False,
                value=_scope_rows(scope),
                label="Current Scope",
            )
            with gr.Row():
                scope_target_input = gr.Textbox(
                    label="Target",
                    placeholder="example.com, 10.0.0.5, or 10.0.0.0/24",
                    scale=2,
                )
                scope_note_input = gr.Textbox(label="Note (optional)", placeholder="e.g. client engagement #123", scale=2)
                scope_add_btn = gr.Button("Add to Scope", variant="primary", scale=1)
                scope_remove_btn = gr.Button("Remove Selected", variant="stop", scale=1)
            selected_scope_target = gr.State("")
            scope_status = gr.Markdown("")

            gr.Markdown("---\n**Scan for live hosts** — ping-sweeps a scoped target/range directly, no LLM involved.")
            with gr.Row():
                discovery_target = gr.Dropdown(
                    label="Target",
                    choices=[e.target for e in scope.list_entries()],
                    interactive=True,
                    scale=3,
                )
                discovery_refresh_btn = gr.Button("Refresh List", scale=1)
                discovery_scan_btn = gr.Button("Scan for Live Hosts", variant="primary", scale=2)
            discovery_output = gr.Textbox(label="Discovery Results", interactive=False, lines=8)

        with gr.Accordion("Tool availability", open=False):
            tool_status = gr.Markdown(_tool_availability_markdown())
            tool_refresh_btn = gr.Button("Refresh Tool Availability")

        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row():
                max_steps = slider_with_manual_override("Max Agent Steps", minimum=1, maximum=50, value=15, step=1)
                max_tokens = slider_with_manual_override("Max Tokens / Step", minimum=64, maximum=8192, value=1024, step=64)
            with gr.Row():
                temperature = slider_with_manual_override("Temperature", minimum=0.0, maximum=2.0, value=0.3, step=0.05)
                top_p = slider_with_manual_override("Top-p", minimum=0.0, maximum=1.0, value=0.9, step=0.05)
                top_k = slider_with_manual_override("Top-k", minimum=1, maximum=200, value=50, step=1)
                rep_penalty = slider_with_manual_override("Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05)

        goal_input = gr.Textbox(
            label="Goal",
            placeholder="e.g. Enumerate subdomains for example.com and check which respond on port 443",
            lines=3,
        )
        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            reset_btn = gr.Button("New Session")

        agent_chat = gr.Chatbot(label="Agent", height=420)
        pending_display = gr.Markdown("*No pending tool call.*")
        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary", interactive=False)
            deny_btn = gr.Button("🚫 Deny", variant="stop", interactive=False)
        run_status = gr.Markdown("")

        history_state = gr.State([])   # raw messages sent to the LLM (no system prompt)
        pending_state = gr.State(None)  # dict describing the proposed tool call, or None
        step_state = gr.State(0)

        # --- Model handlers ---
        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            choices = build_compatible_dropdown_choices(cached, _CHAT_TAGS, cache_manager)
            return gr.update(choices=choices, value=None)

        def load_model(model_id, ctx_size, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not model_id:
                return "**Status:** No model selected."
            if not is_model_compatible(model_id, _CHAT_TAGS, cache_manager):
                return "**Status:** This model is not compatible with the Security Agent (marked with ❌)."
            try:
                progress(0, desc=f"Loading {model_id}...")
                tag = cache_manager.get_local_model_metadata(model_id).pipeline_tag or "text-generation"
                model_manager.load_model(model_id, tag, n_ctx=int(ctx_size))
                return update_model_status(model_manager)
            except Exception as e:
                return f"**Status:** Load failed — {e}"

        def unload_model():
            model_manager.unload_current()
            return update_model_status(model_manager)

        # --- Scope handlers ---
        def _discovery_choices():
            return gr.update(choices=[e.target for e in scope.list_entries()])

        def add_scope(target, note):
            if not target or not target.strip():
                return _scope_rows(scope), "**Status:** Enter a target first.", _discovery_choices()
            scope.add(target.strip(), (note or "").strip())
            return _scope_rows(scope), f"**Added** `{target.strip()}` to scope.", _discovery_choices()

        def select_scope_row(evt: gr.SelectData):
            try:
                return str(evt.row_value[0])
            except Exception:
                return ""

        def remove_scope(selected_target):
            if not selected_target:
                return (
                    _scope_rows(scope),
                    "**Status:** Click a row in the scope table first, then Remove Selected.",
                    _discovery_choices(),
                )
            removed = scope.remove(selected_target)
            msg = f"**Removed** `{selected_target}`." if removed else "**Status:** Could not find that entry."
            return _scope_rows(scope), msg, _discovery_choices()

        def do_host_discovery(target, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not target:
                return "Select a scope entry first."
            if not scope.is_in_scope(target):
                return f"'{target}' is not in scope — refresh the list or add it under Scope first."
            progress(0, desc=f"Pinging {target}...")
            result = security_tools.host_discovery(target)
            return result.output if result.success else f"Error: {result.error}"

        def refresh_tool_status():
            return _tool_availability_markdown()

        # --- Agent loop ---
        def _advance(history, chat_display, step_count, steps_cap, mt, temp, tp, tk, rp):
            if step_count >= int(steps_cap):
                chat_display = chat_display + [
                    {"role": "assistant", "content": "⚠️ Step limit reached — increase Max Agent Steps to continue."}
                ]
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "Step limit reached.", gr.update(interactive=False), gr.update(interactive=False),
                )

            if not model_manager.active_model_id:
                chat_display = chat_display + [{"role": "assistant", "content": "⚠️ No model loaded."}]
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "No model loaded.", gr.update(interactive=False), gr.update(interactive=False),
                )

            messages = [{"role": "system", "content": tool_agent.build_system_prompt()}] + history
            step = tool_agent.propose_next_step(
                model_manager, messages, scope,
                max_new_tokens=int(mt), temperature=float(temp), top_p=float(tp),
                top_k=int(tk), repetition_penalty=float(rp),
            )

            if step.kind == "error":
                chat_display = chat_display + [{"role": "assistant", "content": f"⚠️ {step.raw_text}"}]
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "Error — see conversation.", gr.update(interactive=False), gr.update(interactive=False),
                )

            if step.kind == "final_answer":
                history = history + [{"role": "assistant", "content": step.raw_text}]
                chat_display = chat_display + [{"role": "assistant", "content": step.raw_text}]
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "Done.", gr.update(interactive=False), gr.update(interactive=False),
                )

            # tool_proposal
            pending = {
                "name": step.tool_name,
                "arguments": step.tool_arguments,
                "raw_text": step.raw_text,
                "in_scope": step.in_scope,
                "scope_reason": step.scope_reason,
            }
            chat_display = chat_display + [
                {"role": "assistant", "content": f"🔧 Proposing `{step.tool_name}` — see the approval panel below."}
            ]
            return (
                chat_display, _format_proposal(step), history, pending, step_count,
                "Awaiting approval." if step.in_scope else "Blocked (out of scope) — cannot approve.",
                gr.update(interactive=bool(step.in_scope)), gr.update(interactive=True),
            )

        def start_agent(goal, chat_display, steps_cap, mt, temp, tp, tk, rp):
            if not goal or not goal.strip():
                return (
                    chat_display, "*No pending tool call.*", [], None, 0,
                    "Enter a goal first.", gr.update(interactive=False), gr.update(interactive=False),
                )
            history = [{"role": "user", "content": goal.strip()}]
            chat_display = list(chat_display or []) + [{"role": "user", "content": goal.strip()}]
            return _advance(history, chat_display, 0, steps_cap, mt, temp, tp, tk, rp)

        def approve_step(history, chat_display, pending, step_count, steps_cap, mt, temp, tp, tk, rp):
            if not pending:
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "Nothing to approve.", gr.update(interactive=False), gr.update(interactive=False),
                )
            if not pending.get("in_scope"):
                return (
                    chat_display, _format_proposal_from_dict(pending), history, pending, step_count,
                    "Blocked — out of scope.", gr.update(interactive=False), gr.update(interactive=True),
                )

            result = tool_agent.execute_tool(pending["name"], pending["arguments"])
            result_msg = tool_agent.format_tool_result_message(pending["name"], result)

            history = history + [
                {"role": "assistant", "content": pending["raw_text"]},
                {"role": "user", "content": result_msg},
            ]
            chat_display = chat_display + [
                {"role": "assistant", "content": f"✅ Approved `{pending['name']}`."},
                {"role": "assistant", "content": result_msg},
            ]
            return _advance(history, chat_display, step_count + 1, steps_cap, mt, temp, tp, tk, rp)

        def deny_step(history, chat_display, pending, step_count, steps_cap, mt, temp, tp, tk, rp):
            if not pending:
                return (
                    chat_display, "*No pending tool call.*", history, None, step_count,
                    "Nothing to deny.", gr.update(interactive=False), gr.update(interactive=False),
                )
            denial_msg = tool_agent.format_denial_message(pending["name"], pending.get("scope_reason") or "")
            history = history + [
                {"role": "assistant", "content": pending["raw_text"]},
                {"role": "user", "content": denial_msg},
            ]
            chat_display = chat_display + [{"role": "assistant", "content": f"🚫 Denied `{pending['name']}`."}]
            return _advance(history, chat_display, step_count + 1, steps_cap, mt, temp, tp, tk, rp)

        def _format_proposal_from_dict(pending: dict) -> str:
            args_json = json.dumps(pending["arguments"], indent=2)
            lines = [f"### Proposed tool call: `{pending['name']}`", "```json", args_json, "```"]
            if pending.get("in_scope"):
                lines.append("✅ Target is in scope — Approve to execute, or Deny.")
            else:
                lines.append(f"❌ **Blocked — cannot approve:** {pending.get('scope_reason')}")
            return "\n".join(lines)

        def new_session():
            return [], "*No pending tool call.*", [], None, 0, "", gr.update(interactive=False), gr.update(interactive=False)

        gen_params = [max_steps, max_tokens, temperature, top_p, top_k, rep_penalty]
        agent_outputs = [
            agent_chat, pending_display, history_state, pending_state, step_state,
            run_status, approve_btn, deny_btn,
        ]

        # --- Wire events ---
        refresh_btn.click(fn=refresh_models, outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector, n_ctx], outputs=[status_bar], concurrency_id="model_ops")
        unload_btn.click(fn=unload_model, outputs=[status_bar], concurrency_id="model_ops")

        scope_add_btn.click(
            fn=add_scope,
            inputs=[scope_target_input, scope_note_input],
            outputs=[scope_table, scope_status, discovery_target],
        )
        scope_table.select(fn=select_scope_row, outputs=[selected_scope_target])
        scope_remove_btn.click(
            fn=remove_scope, inputs=[selected_scope_target], outputs=[scope_table, scope_status, discovery_target]
        )
        tool_refresh_btn.click(fn=refresh_tool_status, outputs=[tool_status])

        discovery_refresh_btn.click(fn=_discovery_choices, outputs=[discovery_target])
        discovery_scan_btn.click(fn=do_host_discovery, inputs=[discovery_target], outputs=[discovery_output])

        start_btn.click(
            fn=start_agent,
            inputs=[goal_input, agent_chat, max_steps, max_tokens, temperature, top_p, top_k, rep_penalty],
            outputs=agent_outputs,
            concurrency_id="model_ops",
        )
        approve_btn.click(
            fn=approve_step,
            inputs=[history_state, agent_chat, pending_state, step_state] + gen_params,
            outputs=agent_outputs,
            concurrency_id="model_ops",
        )
        deny_btn.click(
            fn=deny_step,
            inputs=[history_state, agent_chat, pending_state, step_state] + gen_params,
            outputs=agent_outputs,
            concurrency_id="model_ops",
        )
        reset_btn.click(fn=new_session, outputs=agent_outputs)

        return [model_selector]
