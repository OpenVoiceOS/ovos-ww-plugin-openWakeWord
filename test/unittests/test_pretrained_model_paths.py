"""
The plugin picks the pretrained model for its key phrase from the paths that
openWakeWord reports. A directory in the install path can contain a wake word
name (a user home such as /home/alexa, or a project folder called "timer").
The plugin must select by file name, so that such a directory does not make
every pretrained model match the key phrase.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import openwakeword

from ovos_ww_plugin_openwakeword import OwwHotwordPlugin


class TestWakeWordInParentDirectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # One parent directory whose name contains every pretrained model name.
        models_dir = Path(cls.tmp.name, "-".join(openwakeword.MODELS), "models")
        models_dir.mkdir(parents=True)
        cls.expected_names = {}
        cls.relocated = {}
        for name, info in openwakeword.MODELS.items():
            tflite_path = Path(info["model_path"])
            cls.expected_names[name] = tflite_path.stem
            for path in (tflite_path, tflite_path.with_suffix(".onnx")):
                shutil.copy(path, models_dir / path.name)
            cls.relocated[name] = {**info, "model_path": str(models_dir / tflite_path.name)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loads_only_the_key_phrase_model(self):
        with mock.patch.dict(openwakeword.MODELS, self.relocated):
            for name, expected in self.expected_names.items():
                with self.subTest(key_phrase=name):
                    for path in openwakeword.get_pretrained_model_paths("onnx"):
                        self.assertIn(name, str(Path(path).parent))
                    plugin = OwwHotwordPlugin(name.replace("_", " "))
                    self.assertEqual([expected], plugin.model_names)


if __name__ == "__main__":
    unittest.main()
