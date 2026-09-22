import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config.settings as settings_module
from config.settings import Settings


class SettingsDownloadDirTests(unittest.TestCase):
    def setUp(self):
        # set_download_dir() persists to disk and mutates process-global env
        # vars / class state, so isolate all of that for the duration of the
        # test and restore it afterward — otherwise running this test
        # overwrites the real user_settings.json (and thus the real app's
        # configured model cache directory) on the developer's machine.
        self._orig_cache_dir = Settings.DEFAULT_CACHE_DIR
        self._orig_env = {
            key: os.environ.get(key)
            for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
        }
        self._tmp_settings_dir = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(
            settings_module,
            "_USER_SETTINGS_FILE",
            Path(self._tmp_settings_dir.name) / "user_settings.json",
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp_settings_dir.cleanup)

    def tearDown(self):
        Settings.DEFAULT_CACHE_DIR = self._orig_cache_dir
        for key, value in self._orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_set_download_dir_updates_default_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "models"

            result = Settings.set_download_dir(str(new_dir))

            self.assertEqual(result, new_dir.resolve())
            self.assertEqual(Settings.DEFAULT_CACHE_DIR, new_dir.resolve())
            self.assertEqual(os.environ.get("HF_HUB_CACHE"), str(new_dir.resolve()))
            self.assertEqual(os.environ.get("HF_HOME"), str(new_dir.resolve()))
            self.assertEqual(os.environ.get("TRANSFORMERS_CACHE"), str(new_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
