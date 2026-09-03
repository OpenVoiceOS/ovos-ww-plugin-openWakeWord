# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import openwakeword
from openwakeword.utils import download_models
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG


def _tflite_available() -> bool:
    """Check whether tflite-runtime (or full tensorflow-lite) can actually be imported.

    openwakeword's own PyPI metadata requires tflite-runtime unconditionally on
    Linux, but that wheel does not exist for Python >= 3.12, so the package is
    frequently installed without it. openwakeword itself already tolerates this
    at runtime as long as `inference_framework="onnx"` is requested instead.
    """
    try:
        import tflite_runtime.interpreter  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import tensorflow.lite  # noqa: F401
        return True
    except ImportError:
        return False


class OwwHotwordPlugin(HotWordEngine):
    """OpenWakeWord is an open-source wakeword or phrase engine that can be trained on 100% synthetic data.
    It can produce high-quality models for arbitrary words and phrases that perform well across
    a wide range of voices and acoustic environments.
    """

    def __init__(self, key_phrase="hey jarvis", config=None):
        super().__init__(key_phrase, config)

        for legacy_key in ("model", "sensitivity"):
            if legacy_key in self.config:
                LOG.warning(
                    f"ovos-ww-plugin-openwakeword ignores the '{legacy_key}' config key "
                    "(a precise-lite era setting). This plugin uses 'models' (a list of "
                    "model file paths) and 'threshold' instead."
                )

        default_framework = "tflite" if _tflite_available() else "onnx"
        inference_framework = self.config.get('inference_framework', default_framework)
        if inference_framework == "tflite" and not _tflite_available():
            LOG.warning("inference_framework 'tflite' was requested but tflite-runtime "
                        "is not installed; falling back to 'onnx'")
            inference_framework = "onnx"

        explicit_models = self.config.get('models')
        if explicit_models:
            wakeword_models = explicit_models
        else:
            # openwakeword model names use underscores (e.g. "hey_jarvis"),
            # while OVOS key phrases are usually space-separated ("hey jarvis").
            # self.key_phrase is already lowercased by HotWordEngine.__init__.
            normalized_key_phrase = self.key_phrase.replace(" ", "_")
            # Support for openwakeword>=0.6.0, which removes packaged defaults;
            # only fetch the pretrained model(s) that are actually needed
            # instead of the full ~19 MB catalog for both frameworks.
            download_models([normalized_key_phrase])
            pretrained_models = openwakeword.get_pretrained_model_paths(inference_framework) or []
            wakeword_models = [i for i in pretrained_models if normalized_key_phrase in i]
            if not wakeword_models:
                # normalized_key_phrase is not a name openwakeword recognizes as
                # a pretrained model, so the scoped download above silently
                # fetched nothing; fall back to the pre-0.6.0 behavior of
                # downloading and loading the full pretrained catalog.
                LOG.warning(
                    f"'{normalized_key_phrase}' is not a known openwakeword pretrained "
                    "model name; downloading the full pretrained catalog instead. "
                    "Pass an explicit 'models' path in the config to avoid this."
                )
                download_models()
                wakeword_models = []

        self.inference_framework = inference_framework

        # Load openWakeWord model
        self.model = openwakeword.Model(
            wakeword_models=wakeword_models,
            custom_verifier_models=self.config.get('custom_verifier_models', {}),
            custom_verifier_threshold=self.config.get('custom_verifier_threshold', 0.1),
            inference_framework=inference_framework
        )
        self.model_names = list(self.model.models.keys())

        # Define short buffer for audio to ensure correct chunk sizes
        self.audio_buffer = []
        self.has_found = False

    def update(self, chunk: bytes):
        """
        Predict on input audio using openWakeWord models.
        openWakeWord requires that audio be provided in chunks of 1280 samples,
        so a small buffer is used to ensure proper sizes.
        """
        audio_frame = np.frombuffer(chunk, dtype=np.int16).tolist()
        self.audio_buffer.extend(audio_frame)  # build up the buffer until it has enough samples

        if len(self.audio_buffer) >= 1280:
            if isinstance(self.audio_buffer, list):
                self.audio_buffer = np.asarray(self.audio_buffer)
            # Get prediction from openWakeWord
            prediction = self.model.predict(self.audio_buffer)

            # Clear the buffer after each prediction
            self.audio_buffer = []

            # Check for score above threshold
            for mdl_name in self.model_names:
                # https://github.com/dscripka/openwakeword#threshold-scores-for-activation
                if prediction[mdl_name] >= self.config.get("threshold", 0.5):
                    # Set flag indicating that a wakeword was detected
                    self.has_found = True

                    # Flush recent history of openWakeWord internal audio buffer to avoid re-activations
                    n_frames = self.model.model_inputs[mdl_name]
                    self.model.preprocessor.raw_data_buffer.extend([0.0] * n_frames * 1280)
                    self.model.preprocessor.feature_buffer[-n_frames:, :] = np.zeros((n_frames, 96)).astype(np.float32)
                    mel_buf = self.model.preprocessor.melspectrogram_buffer
                    n_mel = min(250, mel_buf.shape[0])
                    mel_buf[-n_mel:, :] = np.zeros((n_mel, 32)).astype(np.float32)

                    break

    def found_wake_word(self) -> bool:
        if self.has_found:
            self.has_found = False
            return True
        return False
