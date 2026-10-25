"""
The pretrained "timer" model has seven outputs: a background class and six
timer phrases. openWakeWord keys the predictions of a multi-class model by
class label, not by model name, so the plugin must score such a model over
its class labels and must not count the background class as a label.

"hey_mycroft" is a single-output model and serves as the control: the plugin
must fire exactly when openWakeWord's own score for the model name reaches
the threshold. "alexa" adds a control that fires on the committed clip.
"""
import unittest
import wave
from pathlib import Path

import numpy as np
import openwakeword

from ovos_ww_plugin_openwakeword import OwwHotwordPlugin

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK = 1280  # openWakeWord's fixed input chunk size
THRESHOLD = 0.5


def _read_pcm16_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def _chunks(samples: np.ndarray):
    n = (len(samples) // CHUNK) * CHUNK
    for i in range(0, n, CHUNK):
        yield samples[i:i + CHUNK]


class TestTimerModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plugin = OwwHotwordPlugin("timer", {"threshold": THRESHOLD})

    def test_checks_the_six_timer_labels(self):
        expected = set(openwakeword.model_class_mappings["timer"].values())
        self.assertEqual(6, len(expected))
        self.assertEqual(1, len(self.plugin.model_names))
        labels = self.plugin.model_labels[self.plugin.model_names[0]]
        self.assertEqual(expected, set(labels))
        self.assertEqual(len(expected), len(labels))

    def test_silence_does_not_fire(self):
        silence = np.zeros(CHUNK * 30, dtype=np.int16)
        detections = 0
        for chunk in _chunks(silence):
            self.plugin.update(chunk.tobytes())
            if self.plugin.found_wake_word():
                detections += 1
        self.assertEqual(0, detections)


class TestSingleOutputControl(unittest.TestCase):

    def _assert_fires_like_openwakeword_score(self, key_phrase, audio):
        plugin = OwwHotwordPlugin(key_phrase, {"threshold": THRESHOLD})
        self.assertEqual(1, len(plugin.model_names))
        mdl_name = plugin.model_names[0]

        model_path = [p for p in openwakeword.get_pretrained_model_paths(plugin.inference_framework)
                      if Path(p).name.startswith(key_phrase.replace(" ", "_"))]
        reference = openwakeword.Model(wakeword_models=model_path,
                                       inference_framework=plugin.inference_framework)

        expected, actual = [], []
        for chunk in _chunks(audio):
            fired = reference.predict(chunk)[mdl_name] >= THRESHOLD
            if fired:
                reference.reset()
            expected.append(fired)
            plugin.update(chunk.tobytes())
            actual.append(plugin.found_wake_word())
        self.assertEqual(expected, actual)
        return expected

    def _audio(self):
        # openWakeWord fills its feature buffer with random noise on load, so
        # 4 s of lead-in flush that state before the clip and both models
        # score the clip identically.
        rng = np.random.default_rng(1234)
        return np.concatenate([
            rng.normal(0.0, 60.0, 16000 * 4).astype(np.int16),
            _read_pcm16_mono_16k(DATA_DIR / "alexa_16k.wav"),
            rng.normal(0.0, 60.0, 16000 * 2).astype(np.int16),
        ])

    def test_hey_mycroft_fires_like_openwakeword_score(self):
        self._assert_fires_like_openwakeword_score("hey mycroft", self._audio())

    def test_alexa_fires_like_openwakeword_score(self):
        expected = self._assert_fires_like_openwakeword_score("alexa", self._audio())
        self.assertEqual(1, sum(expected))

    def test_single_output_label_is_the_model_name(self):
        plugin = OwwHotwordPlugin("hey mycroft", {"threshold": THRESHOLD})
        mdl_name = plugin.model_names[0]
        self.assertEqual({mdl_name: [mdl_name]}, plugin.model_labels)


if __name__ == "__main__":
    unittest.main()
