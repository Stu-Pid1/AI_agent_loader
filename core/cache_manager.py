from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import logging

from huggingface_hub import scan_cache_dir

from config.settings import Settings

logger = logging.getLogger("ai_agent_loader.cache_manager")


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

            if not revisions_to_delete:
                return False

            delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
            delete_strategy.execute()
            logger.info(f"Deleted cache for {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting cache for {model_id}: {e}")
            return False
