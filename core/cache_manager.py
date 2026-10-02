from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging
import re
import shutil

import yaml
from huggingface_hub import scan_cache_dir

from config.settings import Settings

logger = logging.getLogger("ai_agent_loader.cache_manager")

# Best-effort mapping from transformers `config.json` architecture class name
# suffixes to a pipeline_tag, used only when a cached model has no README
# front matter to read the tag from directly. Local models are matched by
# their config on disk — this app never calls the Hub to classify a model
# it has already downloaded.
_ARCHITECTURE_TASK_HINTS = (
    ("ForCausalLM", "text-generation"),
    ("ForConditionalGeneration", "text2text-generation"),
    ("ForSequenceClassification", "text-classification"),
    ("ForTokenClassification", "token-classification"),
    ("ForQuestionAnswering", "question-answering"),
    ("ForImageClassification", "image-classification"),
    ("ForObjectDetection", "object-detection"),
    ("ForSpeechSeq2Seq", "automatic-speech-recognition"),
    ("ForCTC", "automatic-speech-recognition"),
    ("ForTextToSpectrogram", "text-to-speech"),
    ("ForTextToWaveform", "text-to-speech"),
)


@dataclass
class LocalModelMetadata:
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    library_name: Optional[str] = None
    base_model: Optional[str] = None


@dataclass
class CachedModel:
    model_id: str
    size_bytes: int
    num_files: int
    last_accessed: Optional[str]
    local_path: str


class CacheManager:

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else Settings.DEFAULT_CACHE_DIR

    def get_cached_models(self) -> List[CachedModel]:
        if not self._cache_dir.exists():
            return []
        try:
            cache_info = scan_cache_dir(self._cache_dir)
            models = []
            for repo in cache_info.repos:
                if repo.repo_type != "model":
                    continue
                models.append(
                    CachedModel(
                        model_id=repo.repo_id,
                        size_bytes=repo.size_on_disk,
                        num_files=repo.nb_files,
                        last_accessed=str(repo.last_accessed)
                        if hasattr(repo, "last_accessed")
                        else None,
                        local_path=str(repo.repo_path),
                    )
                )
            return models

        except Exception as e:
            logger.warning(f"Could not scan cache: {e}")
            return []

    def is_model_cached(self, model_id: str) -> bool:
        cached = self.get_cached_models()
        return any(m.model_id == model_id for m in cached)

    def get_model_cache_size(self, model_id: str) -> Optional[int]:
        cached = self.get_cached_models()
        for m in cached:
            if m.model_id == model_id:
                return m.size_bytes
        return None

    def get_total_cache_size(self) -> int:
        return sum(m.size_bytes for m in self.get_cached_models())

    def get_cached_model_ids(self) -> List[str]:
        return [m.model_id for m in self.get_cached_models()]

    def resolve_local_model_path(self, model_id: str) -> Optional[str]:
        if not model_id:
            return None

        model_id = model_id.strip()
        if not model_id:
            return None

        candidate = Path(model_id)
        if candidate.exists():
            return str(candidate)

        cache_root = self._cache_dir
        if not cache_root.exists():
            return None

        normalized = model_id.rstrip("/")
        repo_dir_name = "models--" + normalized.replace("/", "--")

        exact = cache_root / repo_dir_name
        if exact.exists():
            snapshots = exact / "snapshots"
            if snapshots.exists():
                revision_dirs = sorted(
                    [p for p in snapshots.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if revision_dirs:
                    return str(revision_dirs[0])
            return str(exact)

        for repo_dir in sorted(cache_root.glob("models--*"), key=lambda p: p.name):
            display_name = repo_dir.name[len("models--") :].replace("--", "/")
            if display_name != normalized:
                continue
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                revision_dirs = sorted(
                    [p for p in snapshots.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if revision_dirs:
                    return str(revision_dirs[0])
            return str(repo_dir)

        return None

    def get_local_model_metadata(self, model_id: str) -> LocalModelMetadata:
        """Best-effort task/library classification read entirely from disk.

        Used to filter/label cached models in the UI without ever calling
        the Hub — a model that is already downloaded should never require
        network access just to figure out what task it's for.
        """
        local_path = self.resolve_local_model_path(model_id)
        if not local_path:
            return LocalModelMetadata()

        root = Path(local_path)

        pipeline_tag: Optional[str] = None
        tags: List[str] = []
        library_name: Optional[str] = None
        base_model: Optional[str] = None

        readme = root / "README.md"
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="ignore")
                match = re.match(r"^---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
                if match:
                    front_matter = yaml.safe_load(match.group(1)) or {}
                    if isinstance(front_matter, dict):
                        pipeline_tag = front_matter.get("pipeline_tag")
                        tags = list(front_matter.get("tags") or [])
                        library_name = front_matter.get("library_name")
                        raw_base_model = front_matter.get("base_model")
                        if isinstance(raw_base_model, list) and raw_base_model:
                            raw_base_model = raw_base_model[0]
                        if isinstance(raw_base_model, str) and raw_base_model:
                            base_model = raw_base_model
            except Exception as e:
                logger.debug(f"Could not parse README front matter for {model_id}: {e}")

        if not base_model:
            for tag in tags:
                if not str(tag).startswith("base_model:"):
                    continue
                remainder = str(tag)[len("base_model:"):]
                for prefix in ("adapter:", "finetune:", "quantized:", "merge:"):
                    if remainder.startswith(prefix):
                        remainder = remainder[len(prefix):]
                        break
                if remainder:
                    base_model = remainder
                    break

        if (root / "model_index.json").exists():
            library_name = library_name or "diffusers"
            if not pipeline_tag:
                try:
                    index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
                    class_name = str(index.get("_class_name", ""))
                    if "video" in class_name.lower():
                        pipeline_tag = "text-to-video"
                    elif class_name:
                        pipeline_tag = "text-to-image"
                except Exception as e:
                    logger.debug(f"Could not read model_index.json for {model_id}: {e}")

        config_file = root / "config.json"
        if not pipeline_tag and config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                library_name = library_name or "transformers"
                architectures = config.get("architectures") or []
                for arch in architectures:
                    for suffix, hint in _ARCHITECTURE_TASK_HINTS:
                        if arch.endswith(suffix):
                            pipeline_tag = hint
                            break
                    if pipeline_tag:
                        break
            except Exception as e:
                logger.debug(f"Could not read config.json for {model_id}: {e}")

        return LocalModelMetadata(
            pipeline_tag=pipeline_tag, tags=tags, library_name=library_name, base_model=base_model
        )

    def list_local_files(self, model_id: str) -> List[str]:
        """Relative POSIX paths of every file in a cached model's snapshot."""
        local_path = self.resolve_local_model_path(model_id)
        if not local_path:
            return []
        root = Path(local_path)
        try:
            return sorted(str(p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file())
        except Exception as e:
            logger.debug(f"Could not list local files for {model_id}: {e}")
            return []

    def list_local_file_sizes(self, model_id: str) -> List[Dict]:
        """Relative POSIX path + size (bytes) for every file in a cached
        model's snapshot — follows symlinks so sizes reflect the real blob."""
        local_path = self.resolve_local_model_path(model_id)
        if not local_path:
            return []
        root = Path(local_path)
        results = []
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                results.append({"filename": str(p.relative_to(root).as_posix()), "size": size})
            return sorted(results, key=lambda e: e["filename"])
        except Exception as e:
            logger.debug(f"Could not list local file sizes for {model_id}: {e}")
            return []

    def delete_model_files(self, model_id: str, filenames: List[str]) -> int:
        """Deletes specific files from a cached model's local snapshot,
        freeing the underlying blob too if the file is a HF-cache symlink.
        Returns the number of bytes freed."""
        local_path = self.resolve_local_model_path(model_id)
        if not local_path:
            return 0
        root = Path(local_path)
        freed = 0
        for rel in filenames:
            file_path = root / rel
            if not file_path.exists() and not file_path.is_symlink():
                continue
            try:
                is_link = file_path.is_symlink()
                blob_target = file_path.resolve() if is_link else None
                size = file_path.stat().st_size if not is_link else (
                    blob_target.stat().st_size if blob_target and blob_target.exists() else 0
                )
                file_path.unlink()
                if blob_target and blob_target.exists() and "blobs" in blob_target.parts:
                    try:
                        blob_target.unlink()
                    except OSError:
                        pass
                freed += size
                logger.info(f"Deleted {rel} for {model_id} ({size} bytes)")
            except OSError as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        return freed

    def get_cached_model_ids_by_tag(
        self, pipeline_tags: List[str]
    ) -> List[str]:
        # This requires checking model info for each cached model,
        # which is expensive. For now, return all cached models
        # and let the UI filter based on known model metadata.
        return self.get_cached_model_ids()

    def delete_model_cache(self, model_id: str) -> bool:
        try:
            cache_info = scan_cache_dir(self._cache_dir)
            revisions_to_delete = []
            for repo in cache_info.repos:
                if repo.repo_id == model_id and repo.repo_type == "model":
                    for revision in repo.revisions:
                        revisions_to_delete.append(revision.commit_hash)

            if revisions_to_delete:
                delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
                delete_strategy.execute()
                logger.info(f"Deleted cache for {model_id}")
                return True

            repo_dir_name = "models--" + model_id.replace("/", "--")
            repo_path = self._cache_dir / repo_dir_name
            if repo_path.exists():
                shutil.rmtree(repo_path)
                logger.info(f"Deleted partial cache folder for {model_id} at {repo_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting cache for {model_id}: {e}")
            try:
                repo_dir_name = "models--" + model_id.replace("/", "--")
                repo_path = self._cache_dir / repo_dir_name
                if repo_path.exists():
                    shutil.rmtree(repo_path)
                    logger.info(f"Deleted partial cache folder for {model_id} at {repo_path}")
                    return True
            except Exception:
                pass
            return False
