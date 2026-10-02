"""Prompt-based tool-calling agent loop for the Security Agent tab.

Works with any loaded chat model (not just ones with native function-calling
templates): the system prompt documents the available tools in OpenAI
function-calling schema form and instructs the model to respond with a
fenced JSON block to invoke one. Every proposed call is scope-checked before
it's ever shown to the human, and nothing executes without that human
clicking Approve — see ui/agent_tools_tab.py for the approval loop this
drives.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core import security_tools
from core.scope_manager import ScopeManager

logger = logging.getLogger("ai_agent_loader.tool_agent")

MAX_STEPS = 20

_TOOL_CALL_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def build_system_prompt() -> str:
    lines = [spec.name + "(" + ", ".join(spec.parameters.get("properties", {}).keys()) + "): " + spec.description
             for spec in security_tools.list_tools()]
    tools_block = "\n".join("- " + line for line in lines)
    schemas_json = json.dumps(security_tools.openai_tool_schemas(), indent=2)

    return (
        "You are a security-testing assistant with access to tools for reconnaissance and "
        "authorized security testing, operated by a human who is responsible for authorization "
        "and scope. You may only act on targets the human has explicitly placed in scope — if "
        "asked to test something else, say so instead of proposing a tool call for it.\n\n"
        "Available tools:\n" + tools_block + "\n\n"
        "Full tool schemas (OpenAI function-calling format):\n```json\n" + schemas_json + "\n```\n\n"
        "To call a tool, respond with ONLY a single fenced JSON block in exactly this shape, "
        "and nothing else before or after it:\n"
        "```json\n"
        '{"tool_call": {"name": "<tool name>", "arguments": {"<param>": "<value>"}}}\n'
        "```\n\n"
        "A human reviews every proposed call before it runs. Your next message will contain "
        "either the real tool output or a note that the human denied the call — use that to "
        "decide your next step.\n\n"
        "When you have enough information, respond in plain text with your findings and "
        "recommendations. Do not wrap a final answer in a tool_call block."
    )


@dataclass
class AgentStep:
    kind: str  # "tool_proposal" | "final_answer" | "error"
    raw_text: str
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    in_scope: Optional[bool] = None
    scope_reason: Optional[str] = None


def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    candidates = _TOOL_CALL_FENCE_RE.findall(text)
    if not candidates:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates = [text[start:end + 1]]

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("tool_call"), dict):
            return data["tool_call"]
    return None


def propose_next_step(model_manager, messages: List[Dict[str, str]], scope: ScopeManager, **gen_kwargs) -> AgentStep:
    result = model_manager.run_inference(messages, **gen_kwargs)
    if not result.success:
        return AgentStep(kind="error", raw_text=f"LLM error: {result.error}")

    text = result.output or ""
    parsed = _extract_tool_call(text)
    if parsed is None:
        return AgentStep(kind="final_answer", raw_text=text)

    name = parsed.get("name")
    arguments = parsed.get("arguments") or {}
    if not isinstance(arguments, dict):
        return AgentStep(kind="error", raw_text=f"Model's tool_call.arguments was not an object.\n\nRaw:\n{text}")

    spec = security_tools.get_tool(str(name))
    if spec is None:
        return AgentStep(kind="error", raw_text=f"Model proposed an unknown tool '{name}'.\n\nRaw:\n{text}")

    in_scope = True
    scope_reason = None
    if spec.target_field:
        target_value = arguments.get(spec.target_field)
        if not target_value:
            in_scope = False
            scope_reason = f"Missing required target field '{spec.target_field}'."
        else:
            host = security_tools.extract_host(str(target_value))
            if not scope.is_in_scope(host):
                in_scope = False
                scope_reason = f"'{host}' is not in the configured scope list — add it first if this is authorized."

    if spec.requires_binary:
        import shutil
        if shutil.which(spec.requires_binary) is None:
            scope_reason = (scope_reason + " " if scope_reason else "") + f"Also: '{spec.requires_binary}' is not installed on this machine."

    return AgentStep(
        kind="tool_proposal",
        raw_text=text,
        tool_name=str(name),
        tool_arguments=arguments,
        in_scope=in_scope,
        scope_reason=scope_reason,
    )


def execute_tool(name: str, arguments: Dict[str, Any]) -> security_tools.ToolResult:
    spec = security_tools.get_tool(name)
    if spec is None:
        return security_tools.ToolResult(success=False, error=f"Unknown tool '{name}'")
    try:
        return spec.func(**arguments)
    except TypeError as e:
        return security_tools.ToolResult(success=False, error=f"Bad arguments for {name}: {e}")
    except Exception as e:
        logger.exception(f"Tool {name} raised")
        return security_tools.ToolResult(success=False, error=str(e))


def format_tool_result_message(name: str, result: security_tools.ToolResult) -> str:
    if result.success:
        return f"[Tool Result: {name}]\n{result.output}"
    return f"[Tool Result: {name}]\nFAILED: {result.error}"


def format_denial_message(name: str, reason: str = "") -> str:
    suffix = f" ({reason})" if reason else ""
    return f"[Tool Call Denied by human operator]{suffix}"
