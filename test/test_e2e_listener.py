"""E2E listener test for ovos-ww-plugin-openWakeWord using ovoscope.

Model under test: hey_mycroft_v0.1 (bundled tflite model).

Positive fixture: tests/data/hey_mycroft_test.wav shipped with the
openwakeword source tree — a real recorded utterance that the upstream
test suite uses, yielding detections at >0.98 confidence.

Negative fixture: test/fixtures/command.wav (a generic command clip that
does not contain "hey mycroft").
"""
from __future__ import annotations

import wave
import numpy as np
from pathlib import Path

import pytest

pytest.importorskip("ovoscope", reason="ovoscope not installed")
pytest.importorskip("openwakeword", reason="openwakeword not installed")

from ovoscope.voice_loop import MiniVoiceLoop  # noqa: E402
from ovos_ww_plugin_openwakeword import OwwHotwordPlugin  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
WAKEWORD_WAV = FIXTURE_DIR / "wakeword.wav"   # hey_mycroft_test.wav (bundled)
COMMAND_WAV  = FIXTURE_DIR / "command.wav"    # generic command — negative sample

MODEL_NAME   = "hey_mycroft"
CHUNK_SIZE   = 1280 * 2  # 1280 int16 samples → 2560 bytes per OWW frame


def _wav_to_chunks(path: Path, chunk_bytes: int = CHUNK_SIZE) -> list[bytes]:
    """Read a 16-kHz mono 16-bit WAV and return fixed-size PCM byte chunks."""
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 1, "fixture must be mono"
        assert wf.getframerate() == 16000, "fixture must be 16 kHz"
        assert wf.getsampwidth() == 2, "fixture must be 16-bit"
        raw = wf.readframes(wf.getnframes())
    chunks = []
    for i in range(0, len(raw) - chunk_bytes + 1, chunk_bytes):
        chunks.append(raw[i : i + chunk_bytes])
    return chunks


@pytest.fixture(scope="module")
def oww_engine():
    """Instantiate the plugin once; model download happens here."""
    import openwakeword
    from openwakeword.utils import download_models
    download_models()
    pretrained = openwakeword.get_pretrained_model_paths() or []
    models = [p for p in pretrained if MODEL_NAME in p]
    engine = OwwHotwordPlugin(
        key_phrase=MODEL_NAME,
        config={"models": models, "inference_framework": "tflite", "threshold": 0.5},
    )
    return engine


# ---------------------------------------------------------------------------
# Positive test — bundled real-utterance sample
# ---------------------------------------------------------------------------

class TestPositiveDetection:
    """Feed a real 'hey mycroft' recording through the OWW plugin via ovoscope."""

    def test_wakeword_detected(self, oww_engine):
        """wakeword.wav (hey_mycroft_test.wav from openwakeword repo) must fire."""
        chunks = _wav_to_chunks(WAKEWORD_WAV)
        with MiniVoiceLoop(ww_instances={MODEL_NAME: oww_engine}) as vl:
            msgs = vl.feed_chunks(chunks)
        vl.assert_wakeword_detected(msgs)


# ---------------------------------------------------------------------------
# Negative test — unrelated command audio must NOT trigger the wakeword
# ---------------------------------------------------------------------------

class TestNegativeDetection:
    """Feed a non-wakeword audio clip; no detection must fire."""

    def test_command_not_detected(self, oww_engine):
        """command.wav contains no wake word — engine must stay silent."""
        # Reset engine state between tests
        oww_engine.has_found = False
        oww_engine.audio_buffer = []

        chunks = _wav_to_chunks(COMMAND_WAV)
        with MiniVoiceLoop(ww_instances={MODEL_NAME: oww_engine}) as vl:
            msgs = vl.feed_chunks(chunks)
        vl.assert_wakeword_suppressed(msgs)
