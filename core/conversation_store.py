import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ai_agent_loader.conversation_store")

CONVERSATIONS_DIR = Path(__file__).parent.parent / "conversations"

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass
class Conversation:
    id: str
    title: str
    model_id: Optional[str] = None
    system_prompt: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            title=data.get("title") or "Untitled",
            model_id=data.get("model_id"),
            system_prompt=data.get("system_prompt", ""),
            messages=list(data.get("messages") or []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class ConversationStore:
    """Persists chat conversations as one JSON file per conversation.

    Kept entirely separate from the model cache (Settings.DEFAULT_CACHE_DIR,
    which may live on a different drive dedicated to large model files) —
    conversations are small text data that belongs with the project itself.
    """

    def __init__(self, directory: Optional[Path] = None):
        self._dir = Path(directory) if directory else CONVERSATIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, conversation_id: str) -> Path:
        safe_id = _SAFE_ID_RE.sub("", conversation_id) or uuid.uuid4().hex
        return self._dir / f"{safe_id}.json"

    def list_conversations(self) -> List[Dict[str, Any]]:
        results = []
        for path in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not read conversation file {path}: {e}")
                continue
            results.append(
                {
                    "id": data.get("id", path.stem),
                    "title": data.get("title") or path.stem,
                    "model_id": data.get("model_id"),
                    "updated_at": data.get("updated_at", ""),
                    "num_messages": len(data.get("messages") or []),
                }
            )
        return results

    def load(self, conversation_id: str) -> Optional[Conversation]:
        if not conversation_id:
            return None
        path = self._path_for(conversation_id)
        if not path.exists():
            return None
        try:
            return Conversation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.error(f"Could not load conversation {conversation_id}: {e}")
            return None

    def save(self, conversation: Conversation) -> str:
        now = datetime.now(timezone.utc).isoformat()
        if not conversation.id:
            conversation.id = uuid.uuid4().hex
        if not conversation.created_at:
            conversation.created_at = now
        conversation.updated_at = now
        if not conversation.title.strip():
            conversation.title = self._auto_title(conversation.messages)

        path = self._path_for(conversation.id)
        path.write_text(json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return conversation.id

    def delete(self, conversation_id: str) -> bool:
        path = self._path_for(conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _auto_title(messages: List[Dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content", "").strip():
                text = msg["content"].strip().replace("\n", " ")
                return text[:60] + ("…" if len(text) > 60 else "")
        return "New Conversation"
