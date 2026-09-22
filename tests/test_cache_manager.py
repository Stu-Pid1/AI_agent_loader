import shutil
import tempfile
import unittest
from pathlib import Path

from config.settings import Settings
from config.task_registry import TaskRegistry
from core.cache_manager import CacheManager


class CacheManagerDeleteModelCacheTests(unittest.TestCase):
    def test_delete_model_cache_removes_partial_and_complete_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_id = "meta-llama/Llama-3.2-1B"
            repo_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
            repo_dir.mkdir(parents=True)
            (repo_dir / "snapshots").mkdir(parents=True)
            (repo_dir / "snapshots" / "partial").write_text("partial model file", encoding="utf-8")

            manager = CacheManager(cache_dir=str(cache_dir))
            deleted = manager.delete_model_cache(model_id)

            self.assertTrue(deleted)
            self.assertFalse(repo_dir.exists())


class TaskRegistryMultimodalTagsTests(unittest.TestCase):
    def test_any_to_any_and_gemma_multimodal_tags_are_registered(self):
        registry = TaskRegistry()
        required_tags = {
            "image-text-to-text",
            "video-text-to-text",
            "audio-text-to-text",
            "document-question-answering",
            "visual-question-answering",
        }

        for tag in required_tags:
            self.assertIn(tag, Settings.SUPPORTED_PIPELINE_TAGS)
            self.assertTrue(registry.is_supported(tag), f"missing registry entry: {tag}")

    def test_text_to_video_tag_is_registered(self):
        registry = TaskRegistry()
        self.assertIn("text-to-video", Settings.SUPPORTED_PIPELINE_TAGS)
        self.assertTrue(registry.is_supported("text-to-video"), "missing text-to-video registry entry")


if __name__ == "__main__":
    unittest.main()
