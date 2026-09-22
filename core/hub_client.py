from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from huggingface_hub import HfApi, snapshot_download, hf_hub_download

from config.settings import Settings

logger = logging.getLogger("ai_agent_loader.hub_client")


@dataclass
class ModelSearchResult:
    model_id: str
    author: str
    pipeline_tag: Optional[str]
    tags: List[str]
    downloads: int
    likes: int
    last_modified: str
    library_name: Optional[str]
    is_cached_locally: bool = False


@dataclass
class ModelDetail:
    model_id: str
    pipeline_tag: Optional[str]
    tags: List[str]
    downloads: int
    likes: int
    author: str
    siblings: List[Dict[str, Any]]
    total_size_bytes: Optional[int]
    library_name: Optional[str]
    card_data: Optional[Dict[str, Any]] = None
    safetensors: Optional[Dict[str, Any]] = None
    gated: Optional[Any] = None


class HubClient:

    def __init__(self, token: Optional[str] = None):
        self._api = HfApi(token=token or Settings.HF_TOKEN)
        self._detail_cache: Dict[str, ModelDetail] = {}

    def search_models(
        self,
        query: str = "",
        author: str = "",
        pipeline_tag: Optional[str] = None,
        library: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 20,
    ) -> List[ModelSearchResult]:
        try:
            kwargs: Dict[str, Any] = {
                "sort": sort,
                "limit": limit,
            }
            if query:
                kwargs["search"] = query
            if author:
                kwargs["author"] = author
            if pipeline_tag and pipeline_tag != "All":
                kwargs["pipeline_tag"] = pipeline_tag
            # library is no longer a direct param — pass as a filter tag
            if library and library != "All":
                kwargs["filter"] = f"library:{library}"

            models = list(self._api.list_models(**kwargs))

            results = []
            for m in models:
                author_name = m.id.split("/")[0] if "/" in m.id else ""
                results.append(
                    ModelSearchResult(
                        model_id=m.id,
                        author=author_name,
                        pipeline_tag=getattr(m, "pipeline_tag", None),
                        tags=list(getattr(m, "tags", []) or []),
                        downloads=getattr(m, "downloads", 0) or 0,
                        likes=getattr(m, "likes", 0) or 0,
                        last_modified=str(getattr(m, "last_modified", "")),
                        library_name=getattr(m, "library_name", None),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error searching models: {e}")
            raise

    def get_model_detail(self, model_id: str, use_cache: bool = True) -> ModelDetail:
        if use_cache and model_id in self._detail_cache:
            return self._detail_cache[model_id]
        try:
            info = self._api.model_info(model_id, files_metadata=True)

            siblings = []
            total_size = 0
            for s in info.siblings or []:
                file_info = {
                    "filename": s.rfilename,
                    "size": getattr(s, "size", None),
                }
                siblings.append(file_info)
                if file_info["size"]:
                    total_size += file_info["size"]

            author = model_id.split("/")[0] if "/" in model_id else ""

            detail = ModelDetail(
                model_id=info.id,
                pipeline_tag=getattr(info, "pipeline_tag", None),
                tags=list(getattr(info, "tags", []) or []),
                downloads=getattr(info, "downloads", 0) or 0,
                likes=getattr(info, "likes", 0) or 0,
                author=author,
                siblings=siblings,
                total_size_bytes=total_size if total_size > 0 else None,
                library_name=getattr(info, "library_name", None),
                card_data=getattr(info, "card_data", None),
                safetensors=getattr(info, "safetensors", None),
                gated=getattr(info, "gated", None),
            )
            self._detail_cache[model_id] = detail
            return detail

        except Exception as e:
            logger.error(f"Error getting model detail for {model_id}: {e}")
            raise

    def download_model(
        self,
        model_id: str,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        try:
            path = snapshot_download(
                model_id,
                token=Settings.HF_TOKEN,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
            )
            logger.info(f"Downloaded {model_id} to {path}")
            return path

        except Exception as e:
            logger.error(f"Error downloading {model_id}: {e}")
            raise

    def download_single_file(
        self,
        model_id: str,
        filename: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        try:
            path = hf_hub_download(
                model_id,
                filename=filename,
                token=Settings.HF_TOKEN,
                cache_dir=cache_dir,
            )
            return path
        except Exception as e:
            logger.error(f"Error downloading {model_id}/{filename}: {e}")
            raise

    def get_model_card(self, model_id: str) -> str:
        try:
            info = self._api.model_info(model_id)
            card = getattr(info, "card_data", None)
            if card and hasattr(card, "text"):
                return card.text

            # Fall back to downloading README
            try:
                path = hf_hub_download(
                    model_id,
                    filename="README.md",
                    token=Settings.HF_TOKEN,
                )
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return "*No model card available.*"

        except Exception as e:
            logger.warning(f"Could not fetch model card for {model_id}: {e}")
            return "*No model card available.*"
