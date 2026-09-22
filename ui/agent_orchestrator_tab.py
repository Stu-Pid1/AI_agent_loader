import json
import logging

import gradio as gr

from core.agent_orchestrator import AgentOrchestrator, SpecialistAgent
from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from core.hub_client import HubClient
from ui.components import build_compatible_dropdown_choices

logger = logging.getLogger("ai_agent_loader.ui.agent_orchestrator")


def _agent_task_options():
    return [
        "text-generation",
        "summarization",
        "translation",
        "text-classification",
        "image-classification",
        "token-classification",
        "automatic-speech-recognition",
        "text-to-speech",
        "object-detection",
        "text-to-image",
        "text-to-video",
    ]


def create_agent_orchestrator_tab(
    model_manager: ModelManager,
    cache_manager: CacheManager,
    hub_client: HubClient,
):
    with gr.Tab("Agent Orchestrator", id="agent_orchestrator"):
        gr.Markdown("## Multi-Agent Orchestrator")
        gr.Markdown(
            "Design a planner LLM plus a small set of specialist agents to break a high-level goal into executable steps."
        )

        with gr.Row():
            planner_model = gr.Dropdown(
                label="Planner Model (LLM)",
                choices=[],
                interactive=True,
                scale=3,
            )
            planner_refresh = gr.Button("Refresh", scale=1)

        task_goal = gr.Textbox(
            label="High-level Goal",
            placeholder="Summarize this report, translate the result, and prepare a short action plan...",
            lines=4,
        )

        with gr.Row():
            agent1_name = gr.Textbox(label="Agent 1 Name", value="Researcher")
            agent1_task = gr.Dropdown(label="Task Type", choices=_agent_task_options(), value="text-generation")
            agent1_model = gr.Dropdown(label="Agent 1 Model", choices=[], interactive=True)

        with gr.Row():
            agent2_name = gr.Textbox(label="Agent 2 Name", value="Summarizer")
            agent2_task = gr.Dropdown(label="Task Type", choices=_agent_task_options(), value="summarization")
            agent2_model = gr.Dropdown(label="Agent 2 Model", choices=[], interactive=True)

        with gr.Row():
            agent3_name = gr.Textbox(label="Agent 3 Name", value="Specialist")
            agent3_task = gr.Dropdown(label="Task Type", choices=_agent_task_options(), value="translation")
            agent3_model = gr.Dropdown(label="Agent 3 Model", choices=[], interactive=True)

        with gr.Row():
            generate_plan_btn = gr.Button("Generate Plan", variant="primary")
            run_plan_btn = gr.Button("Execute Plan", variant="primary")

        plan_output = gr.JSON(label="Planner Output")
        execution_output = gr.JSON(label="Execution Results")
        status_bar = gr.Markdown("**Status:** Idle")

        orchestrator = AgentOrchestrator(model_manager, cache_manager, hub_client)

        def refresh_models():
            cached = cache_manager.get_cached_model_ids()
            planner_choices = build_compatible_dropdown_choices(
                cached,
                {
                    "text-generation",
                    "text2text-generation",
                    "image-text-to-text",
                    "video-text-to-text",
                    "audio-text-to-text",
                    "document-question-answering",
                    "visual-question-answering",
                },
                cache_manager,
            )
            all_choices = build_compatible_dropdown_choices(cached, set(_agent_task_options()), cache_manager)
            return (
                gr.update(choices=planner_choices, value=None),
                gr.update(choices=all_choices, value=None),
                gr.update(choices=all_choices, value=None),
                gr.update(choices=all_choices, value=None),
            )

        def build_agents():
            return [
                SpecialistAgent(agent1_name.value or "Agent 1", agent1_task.value, agent1_model.value or "", ""),
                SpecialistAgent(agent2_name.value or "Agent 2", agent2_task.value, agent2_model.value or "", ""),
                SpecialistAgent(agent3_name.value or "Agent 3", agent3_task.value, agent3_model.value or "", ""),
            ]

        def generate_plan(goal, planner_id, a1_name, a1_task, a1_model_id, a2_name, a2_task, a2_model_id, a3_name, a3_task, a3_model_id):
            if not goal.strip():
                return {"error": "Provide the high-level goal first."}, "**Status:** Missing objective"
            agents = [
                SpecialistAgent(a1_name or "Agent 1", a1_task, a1_model_id or "", ""),
                SpecialistAgent(a2_name or "Agent 2", a2_task, a2_model_id or "", ""),
                SpecialistAgent(a3_name or "Agent 3", a3_task, a3_model_id or "", ""),
            ]
            plan_text = orchestrator.build_plan(goal, planner_id, agents)
            try:
                parsed = json.loads(plan_text)
                return parsed, "**Status:** Plan generated"
            except Exception:
                return {"raw_plan": plan_text}, "**Status:** Plan generated"

        def execute_plan(goal, planner_id, a1_name, a1_task, a1_model_id, a2_name, a2_task, a2_model_id, a3_name, a3_task, a3_model_id):
            agents = [
                SpecialistAgent(a1_name or "Agent 1", a1_task, a1_model_id or "", ""),
                SpecialistAgent(a2_name or "Agent 2", a2_task, a2_model_id or "", ""),
                SpecialistAgent(a3_name or "Agent 3", a3_task, a3_model_id or "", ""),
            ]
            result = orchestrator.execute_plan(goal, planner_id, agents)
            if not result["steps"]:
                return {"result": result, "warning": "No executable specialist agents selected."}, "**Status:** No action taken"
            return result, "**Status:** Execution finished"

        planner_refresh.click(fn=refresh_models, outputs=[planner_model, agent1_model, agent2_model, agent3_model])

        generate_plan_btn.click(
            fn=generate_plan,
            inputs=[task_goal, planner_model, agent1_name, agent1_task, agent1_model, agent2_name, agent2_task, agent2_model, agent3_name, agent3_task, agent3_model],
            outputs=[plan_output, status_bar],
            concurrency_id="model_ops",
        )

        run_plan_btn.click(
            fn=execute_plan,
            inputs=[task_goal, planner_model, agent1_name, agent1_task, agent1_model, agent2_name, agent2_task, agent2_model, agent3_name, agent3_task, agent3_model],
            outputs=[execution_output, status_bar],
            concurrency_id="model_ops",
        )

        return [planner_model, agent1_model, agent2_model, agent3_model]
