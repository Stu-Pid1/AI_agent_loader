import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("ai_agent_loader.agent_orchestrator")


@dataclass
class SpecialistAgent:
    name: str
    task: str
    model_id: str
    description: str = ""


class AgentOrchestrator:
    """Small orchestration layer for task-specific specialist models.

    This is intentionally pragmatic: it keeps the current single-active-model design,
    but allows a planner LLM to compose a set of specialist agents on top of the
    existing model registry and runners.
    """

    def __init__(self, model_manager, cache_manager=None, hub_client=None):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.hub_client = hub_client

    def validate_agent(self, agent: SpecialistAgent) -> str:
        if not agent.name.strip():
            return "Give each specialist agent a clear name."
        if not agent.model_id.strip():
            return f"{agent.name}: select a model first."
        if not agent.task.strip():
            return f"{agent.name}: choose the task type."

        allowed = {
            "text-generation",
            "text2text-generation",
            "image-text-to-text",
            "video-text-to-text",
            "audio-text-to-text",
            "document-question-answering",
            "visual-question-answering",
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
        }
        if agent.task not in allowed:
            return f"{agent.name}: unsupported task '{agent.task}'."

        if self.cache_manager is not None:
            try:
                from ui.components import is_model_compatible

                if not is_model_compatible(agent.model_id, {agent.task}, self.cache_manager):
                    return (
                        f"{agent.name}: '{agent.model_id}' is not compatible with "
                        f"the required task '{agent.task}'."
                    )
            except Exception:
                logger.debug("Compatibility validation skipped for %s", agent.model_id)

        return ""

    def build_plan(self, user_goal: str, planner_model_id: str, agents: List[SpecialistAgent]) -> str:
        if not user_goal.strip():
            return "Provide a clear objective before generating a plan."
        if not planner_model_id:
            return "Select a planner LLM before generating a plan."

        invalid = [self.validate_agent(a) for a in agents if a.model_id]
        invalid = [msg for msg in invalid if msg]
        if invalid:
            return "\n".join(invalid)

        prompt = self._make_planner_prompt(user_goal, agents)
        self.model_manager.load_model(planner_model_id, "text-generation")
        result = self.model_manager.run_inference(prompt)
        if not result.success:
            return f"Planner failed: {result.error}"

        output = result.output
        if isinstance(output, list):
            output = "\n".join(str(x) for x in output)
        return str(output)

    def execute_plan(self, user_goal: str, planner_model_id: str, agents: List[SpecialistAgent]) -> Dict[str, Any]:
        steps = []
        for idx, agent in enumerate(agents, start=1):
            if not agent.model_id:
                continue
            error = self.validate_agent(agent)
            if error:
                steps.append({
                    "step": idx,
                    "agent": agent.name,
                    "status": "blocked",
                    "reason": error,
                })
                continue

            try:
                self.model_manager.load_model(agent.model_id, agent.task)
                response = self.model_manager.run_inference(
                    self._build_agent_prompt(user_goal, agent),
                    max_new_tokens=512,
                )
                if response.success:
                    steps.append({
                        "step": idx,
                        "agent": agent.name,
                        "task": agent.task,
                        "model_id": agent.model_id,
                        "status": "completed",
                        "output": response.output,
                    })
                else:
                    steps.append({
                        "step": idx,
                        "agent": agent.name,
                        "task": agent.task,
                        "model_id": agent.model_id,
                        "status": "failed",
                        "reason": response.error,
                    })
            except Exception as exc:
                steps.append({
                    "step": idx,
                    "agent": agent.name,
                    "task": agent.task,
                    "model_id": agent.model_id,
                    "status": "error",
                    "reason": str(exc),
                })

        return {
            "goal": user_goal,
            "planner_model": planner_model_id,
            "agents": [asdict(agent) for agent in agents if agent.model_id],
            "steps": steps,
        }

    def _make_planner_prompt(self, user_goal: str, agents: List[SpecialistAgent]) -> str:
        agent_lines = []
        for i, agent in enumerate(agents, start=1):
            if not agent.model_id:
                continue
            agent_lines.append(f"{i}. {agent.name}: {agent.task} using {agent.model_id}")

        return (
            "You are the planner for a multi-agent system. Break the task into a small set of "
            "ordered steps using only the specialist agents listed below.\n"
            "Return valid JSON with a top-level array named 'plan' and each item containing: "
            "'step', 'agent_name', 'task', 'reason', 'prompt'.\n\n"
            f"Goal: {user_goal}\n\n"
            "Available specialist agents:\n"
            + ("\n".join(agent_lines) if agent_lines else "No specialist agents selected.")
            + "\n\n"
            "Rules:\n"
            "- Keep it realistic and task-specific.\n"
            "- Do not invent unavailable tools.\n"
            "- Prefer the most capable agent for each stage.\n"
            "- Keep the plan concise and executable."
        )

    def _build_agent_prompt(self, user_goal: str, agent: SpecialistAgent) -> str:
        return (
            f"You are acting as the specialist agent '{agent.name}' for the task '{agent.task}'.\n"
            f"Objective: {user_goal}\n\n"
            "Use the best available approach for this task, but stay focused on the assigned role.\n"
            "Return a concise, actionable result suitable for downstream orchestration."
        )
