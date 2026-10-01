"""
Regression test for https://github.com/OpenVoiceOS/ovos-ww-plugin-openwakeword/issues/38.

OwwHotwordPlugin.update() used to "flush" openWakeWord's internal state after
a detection by hand-zeroing the tails of raw_data_buffer, feature_buffer and
melspectrogram_buffer. Real microphone audio never contains exact digital
silence, so that fabricated region pushed some bundled models (alexa,
hey_mycroft) back above their detection threshold, which re-entered the same
branch, re-flushed with the same zeros, and looped indefinitely.

This test drives the plugin to a genuine detection with a synthesized "alexa"
utterance, then feeds it several seconds of low-level synthetic noise
(representing room tone / silence) and asserts detection fires exactly once,
never again on the noise that follows.

The positive clip is a synthetic "alexa" utterance produced with edge-tts
(en-US-AriaNeural), downsampled to 16 kHz mono PCM16 with ffmpeg, and committed
at test/data/alexa_16k.wav. No network access is required to run the test; the
plugin itself downloads openWakeWord's pretrained "alexa" model via
huggingface_hub on first use (openwakeword.utils.download_models), same as any
other use of this plugin -- there is no way to exercise real openWakeWord
inference without that.
"""
import unittest
import wave
from pathlib import Path

import numpy as np

from ovos_ww_plugin_openwakeword import OwwHotwordPlugin

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK = 1280  # openWakeWord's fixed input chunk size


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
        yield samples[i:i + CHUNK].tobytes()


class TestNoRefireAfterDetection(unittest.TestCase):
    """openWakeWord must fire once per genuine utterance, not loop on silence."""

    @classmethod
    def setUpClass(cls):
        cls.plugin = OwwHotwordPlugin(key_phrase="alexa")
        cls.positive = _read_pcm16_mono_16k(DATA_DIR / "alexa_16k.wav")
        # 3 s of low-amplitude noise standing in for room tone / silence,
        # fixed seed for a reproducible clip.
        rng = np.random.default_rng(1234)
        cls.room_tone = (rng.normal(0.0, 60.0, 16000 * 3)).astype(np.int16)

    def test_single_detection_no_loop_on_silence(self):
        detections_during_positive = 0
        for chunk in _chunks(self.positive):
            self.plugin.update(chunk)
            if self.plugin.found_wake_word():
                detections_during_positive += 1

        detections_during_room_tone = 0
        for chunk in _chunks(self.room_tone):
            self.plugin.update(chunk)
            if self.plugin.found_wake_word():
                detections_during_room_tone += 1

        self.assertEqual(
            1, detections_during_positive,
            "expected exactly one detection on the positive 'alexa' clip, "
            f"got {detections_during_positive}",
        )
        self.assertEqual(
            0, detections_during_room_tone,
            "openWakeWord re-fired on room tone after a detection -- the "
            "post-detection flush is fabricating an in-distribution false "
            f"positive (got {detections_during_room_tone} extra detections)",
        )


if __name__ == "__main__":
    unittest.main()
