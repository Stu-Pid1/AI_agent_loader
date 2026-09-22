import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_USER_SETTINGS_FILE = Path(__file__).parent.parent / "user_settings.json"
_HF_DEFAULT = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


def _load_user_settings() -> dict:
    try:
        if _USER_SETTINGS_FILE.exists():
            return json.loads(_USER_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_user_settings(data: dict) -> None:
    _USER_SETTINGS_FILE.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


class Settings:
    APP_TITLE = "AI Agent Loader"
    APP_VERSION = "0.1.0"
    SERVER_PORT = 7860

    HF_TOKEN = os.getenv("HF_TOKEN", None)
    HF_HOME = os.getenv("HF_HOME", None)

    # Load persisted download dir, fall back to HF default
    _user = _load_user_settings()
    DEFAULT_CACHE_DIR: Path = Path(_user.get("download_dir", _HF_DEFAULT))

    @classmethod
    def apply_download_dir(cls, path: str) -> Path:
        """Persist the same cache path to all relevant HF environment variables."""
        new_path = Path(path).expanduser().resolve()
        new_path.mkdir(parents=True, exist_ok=True)
        cls.DEFAULT_CACHE_DIR = new_path

        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ):
            os.environ[key] = str(new_path)

        os.environ["HF_HOME"] = str(new_path)
        os.environ["HF_HUB_CACHE"] = str(new_path)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(new_path)
        os.environ["TRANSFORMERS_CACHE"] = str(new_path)
        return new_path

    SUPPORTED_PIPELINE_TAGS = [
        "text-generation",
        "text2text-generation",
        "image-text-to-text",
        "video-text-to-text",
        "audio-text-to-text",
        "document-question-answering",
        "visual-question-answering",
        "image-to-text",
        "text-to-image",
        "text-to-video",
        "text-classification",
        "sentiment-analysis",
        "zero-shot-classification",
        "token-classification",
        "ner",
        "automatic-speech-recognition",
        "text-to-speech",
        "text-to-audio",
        "audio-to-audio",
        "image-classification",
        "object-detection",
        "summarization",
        "translation",
    ]

    SUPPORTED_LIBRARIES = [
        "transformers",
        "diffusers",
        "llama-cpp-python",
    ]

    SEARCH_SORT_OPTIONS = {
        "Downloads": "downloads",
        "Likes": "likes",
        "Last Modified": "last_modified",
        "Trending": "trending_score",
        "Newest": "created_at",
    }

    DEFAULT_SEARCH_LIMIT = 20
    MAX_SEARCH_LIMIT = 100

    @classmethod
    def set_download_dir(cls, path: str) -> Path:
        """Persist a new download directory and apply it immediately."""
        new_path = cls.apply_download_dir(path)
        user = _load_user_settings()
        user["download_dir"] = str(new_path)
        _save_user_settings(user)
        return new_path


Settings.apply_download_dir(str(Settings.DEFAULT_CACHE_DIR))
