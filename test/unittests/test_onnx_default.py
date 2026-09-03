import unittest
from unittest.mock import patch

from ovos_ww_plugin_openwakeword import OwwHotwordPlugin, _tflite_available


class TestBothFrameworksLoadExactlyOneModel(unittest.TestCase):
    """(a) both inference frameworks must load exactly one hey_mycroft model."""

    @unittest.skipUnless(_tflite_available(), "tflite-runtime not installed in this interpreter")
    def test_tflite_framework_loads_one_hey_mycroft_model(self):
        engine = OwwHotwordPlugin(key_phrase="hey_mycroft",
                                   config={"inference_framework": "tflite"})
        self.assertEqual(len(engine.model_names), 1)
        self.assertIn("hey_mycroft", engine.model_names[0])

    def test_onnx_framework_loads_one_hey_mycroft_model(self):
        engine = OwwHotwordPlugin(key_phrase="hey_mycroft",
                                   config={"inference_framework": "onnx"})
        self.assertEqual(len(engine.model_names), 1)
        self.assertIn("hey_mycroft", engine.model_names[0])


class TestFrameworkAutoDetection(unittest.TestCase):
    """(b) inference_framework must default to onnx when tflite-runtime is unavailable,
    and to tflite when it is. These patch the detection helper directly (rather than
    relying on the __import__-patching fallback-warning branch) so a mutation that
    guts the actual default-selection logic in __init__ is caught."""

    def test_defaults_to_onnx_when_tflite_runtime_absent(self):
        with patch("ovos_ww_plugin_openwakeword._tflite_available", return_value=False):
            engine = OwwHotwordPlugin(key_phrase="hey_mycroft", config={"threshold": 0.5})
            self.assertEqual(engine.inference_framework, "onnx")

    @unittest.skipUnless(_tflite_available(), "tflite-runtime not installed in this interpreter")
    def test_uses_tflite_when_genuinely_available(self):
        # unpatched: exercises the real _tflite_available() detection end to end
        engine = OwwHotwordPlugin(key_phrase="hey_mycroft", config={"threshold": 0.5})
        self.assertEqual(engine.inference_framework, "tflite")


class TestDownloadScoping(unittest.TestCase):
    """(c) only the model(s) actually needed must be requested for download."""

    def test_download_models_scoped_to_key_phrase(self):
        with patch("ovos_ww_plugin_openwakeword.download_models") as mock_download:
            OwwHotwordPlugin(key_phrase="hey_mycroft", config={"threshold": 0.5})
            mock_download.assert_called_once_with(["hey_mycroft"])

    def test_download_skipped_when_explicit_models_given(self):
        import openwakeword
        framework = "tflite" if _tflite_available() else "onnx"
        pretrained = openwakeword.get_pretrained_model_paths(framework) or []
        explicit = [p for p in pretrained if "hey_mycroft" in p]
        with patch("ovos_ww_plugin_openwakeword.download_models") as mock_download:
            OwwHotwordPlugin(key_phrase="hey_mycroft",
                              config={"models": explicit})
            mock_download.assert_not_called()

    def test_download_scoping_uses_lowercased_key_phrase(self):
        # HotWordEngine.__init__ lowercases key_phrase into self.key_phrase;
        # the scoped download must use that, not the raw constructor argument,
        # or a differently-cased phrase never matches a pretrained model name.
        with patch("ovos_ww_plugin_openwakeword.download_models") as mock_download:
            OwwHotwordPlugin(key_phrase="Hey_Mycroft", config={"threshold": 0.5})
            mock_download.assert_called_once_with(["hey_mycroft"])

    def test_unrecognized_key_phrase_falls_back_to_full_catalog(self):
        # "hey_computer" is not an openwakeword pretrained model name, so the
        # scoped download_models(["hey_computer"]) silently fetches nothing
        # and the pretrained-path filter yields []. Loading
        # openwakeword.Model(wakeword_models=[]) must not crash: it should
        # fall back to downloading the full pretrained catalog and loading
        # openwakeword.Model with wakeword_models=[] (its own documented way
        # of requesting "load every pretrained model").
        with patch("ovos_ww_plugin_openwakeword.download_models") as mock_download, \
                patch("ovos_ww_plugin_openwakeword.openwakeword.Model") as mock_model:
            mock_model.return_value.models = {"alexa": object(), "hey_jarvis": object()}
            OwwHotwordPlugin(key_phrase="hey_computer", config={"threshold": 0.5})
            # first the scoped, doomed-to-be-empty attempt, then the fallback
            mock_download.assert_any_call(["hey_computer"])
            mock_download.assert_any_call()
            self.assertEqual(mock_download.call_count, 2)
            self.assertEqual(mock_model.call_args.kwargs["wakeword_models"], [])


class TestLegacyConfigWarning(unittest.TestCase):
    """Legacy precise-era 'model'/'sensitivity' keys must be flagged, not silently ignored.

    ovos_utils.log.LOG generates a fresh per-callsite logger name on every call
    (not a stable "ovos_ww_plugin_openwakeword" logger), so assertLogs can't
    target it by name; the plugin's LOG reference is mocked directly instead.
    """

    def test_warns_on_legacy_model_key(self):
        with patch("ovos_ww_plugin_openwakeword.LOG") as mock_log:
            OwwHotwordPlugin(key_phrase="hey_mycroft",
                              config={"model": "/some/path.pb"})
        self.assertTrue(any("'model'" in call.args[0] for call in mock_log.warning.call_args_list))

    def test_warns_on_legacy_sensitivity_key(self):
        with patch("ovos_ww_plugin_openwakeword.LOG") as mock_log:
            OwwHotwordPlugin(key_phrase="hey_mycroft",
                              config={"sensitivity": 0.5})
        self.assertTrue(any("'sensitivity'" in call.args[0] for call in mock_log.warning.call_args_list))

    def test_no_warning_for_modern_config(self):
        with patch("ovos_ww_plugin_openwakeword.LOG") as mock_log:
            OwwHotwordPlugin(key_phrase="hey_mycroft", config={"threshold": 0.5})
        mock_log.warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
