import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.cache_manager import CacheManager
from core.model_manager import ModelManager
from utils.errors import ModelNotLoadedError

logger = logging.getLogger("ai_agent_loader.api")

CHAT_TAGS = {
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
    "document-question-answering",
    "visual-question-answering",
}

_OPTION_KEY_MAP = {
    "num_predict": "max_new_tokens",
    "max_tokens": "max_new_tokens",
    "max_new_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repeat_penalty": "repetition_penalty",
    "repetition_penalty": "repetition_penalty",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    system: Optional[str] = None
    stream: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)


class OllamaChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False


def _gen_kwargs_from_options(options: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {}
    for src, dst in _OPTION_KEY_MAP.items():
        if src in options and options[src] is not None:
            kwargs[dst] = options[src]
    return kwargs


def _ensure_chat_model_ready(model_manager: ModelManager, cache_manager: CacheManager, model_id: Optional[str]) -> None:
    """Loads the requested model on demand (Ollama-style) and verifies it's a
    text-generation-capable model — this API is for the LLM, not for
    whatever else might currently be loaded (an image model, etc.)."""
    if model_id and model_id != model_manager.active_model_id:
        tag = cache_manager.get_local_model_metadata(model_id).pipeline_tag or "text-generation"
        if tag not in CHAT_TAGS:
            tag = "text-generation"
        try:
            model_manager.load_model(model_id, tag)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load model '{model_id}': {e}")

    if not model_manager.active_model_id:
        raise HTTPException(
            status_code=400,
            detail="No model specified and none currently loaded. Pass a 'model' field "
            "naming a locally cached model, or load one first via the app's Chat tab.",
        )

    if model_manager.active_pipeline_tag not in CHAT_TAGS:
        raise HTTPException(
            status_code=400,
            detail=f"The currently loaded model ('{model_manager.active_model_id}') is a "
            f"{model_manager.active_pipeline_tag} model, not a text-generation model.",
        )


def create_api_router(model_manager: ModelManager, cache_manager: CacheManager) -> APIRouter:
    """REST API for the currently loaded LLM — an Ollama-style surface
    (/api/generate, /api/chat, /api/tags) plus an OpenAI-compatible
    /v1/chat/completions endpoint, so existing tools built against either
    convention can talk to whatever model is loaded in this app.

    Like the rest of this app, there's a single active model shared between
    the UI and this API (protected by ModelManager's own lock) — calling
    these with a different `model` than what's active will swap the loaded
    model, same as clicking Load in the UI would.
    """
    router = APIRouter()

    @router.get("/api/tags")
    def list_tags():
        models = cache_manager.get_cached_models()
        return {
            "models": [
                {
                    "name": m.model_id,
                    "model": m.model_id,
                    "size": m.size_bytes,
                    "modified_at": m.last_accessed,
                }
                for m in models
            ]
        }

    @router.get("/v1/models")
    def list_openai_models():
        models = cache_manager.get_cached_models()
        return {
            "object": "list",
            "data": [{"id": m.model_id, "object": "model", "owned_by": "local"} for m in models],
        }

    @router.get("/api/status")
    def status():
        return model_manager.get_active_status()

    @router.post("/api/generate")
    def ollama_generate(req: OllamaGenerateRequest):
        _ensure_chat_model_ready(model_manager, cache_manager, req.model)
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        try:
            result = model_manager.run_inference(messages, **_gen_kwargs_from_options(req.options))
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "model": model_manager.active_model_id,
            "created_at": time.time(),
            "response": result.output,
            "done": True,
        }

    @router.post("/api/chat")
    def ollama_chat(req: OllamaChatRequest):
        _ensure_chat_model_ready(model_manager, cache_manager, req.model)
        messages = [m.model_dump() for m in req.messages]

        try:
            result = model_manager.run_inference(messages, **_gen_kwargs_from_options(req.options))
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "model": model_manager.active_model_id,
            "created_at": time.time(),
            "message": {"role": "assistant", "content": result.output},
            "done": True,
        }

    @router.post("/v1/chat/completions")
    def openai_chat_completions(req: OpenAIChatRequest):
        _ensure_chat_model_ready(model_manager, cache_manager, req.model)
        messages = [m.model_dump() for m in req.messages]

        gen_kwargs = {}
        if req.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = req.max_tokens
        if req.temperature is not None:
            gen_kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            gen_kwargs["top_p"] = req.top_p

        try:
            result = model_manager.run_inference(messages, **gen_kwargs)
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_manager.active_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return router
