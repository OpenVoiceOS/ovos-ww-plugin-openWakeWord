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

# Imports
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
import openwakeword
import numpy as np

class OwwHotwordPlugin(HotWordEngine):
    """OpenWakeWord is an open-source wakeword or phrase engine that can be trained on 100% synthetic data.
    It can produce high-quality models for arbitrary words and phrases that perform well across
    a wide range of voices and acoustic environments.
    """

    def __init__(self, key_phrase="hey jarvis", config=None, lang="en-us"):
        super().__init__(key_phrase, config, lang)

        # Load openWakeWord model
        pretrained_models = openwakeword.get_pretrained_model_paths()
        self.model = openwakeword.Model(
            wakeword_model_paths=self.config.get('model', [i for i in pretrained_models if key_phrase in i]),
        )
        self.model_names = list(self.model.models.keys())

        # Define short buffer for audio to ensure correct chunk sizes
        self.audio_buffer = []
        self.has_found = False

    def update(self, chunk):
        """
        Predict on input audio using openWakeWord models.
        openWakeWord requires that audio be provided in chunks of 1280 samples,
        so a small buffer is used to ensure proper sizes.
        """
        audio_frame = np.frombuffer(chunk, dtype=np.int16).tolist()
        self.audio_buffer.extend(audio_frame)
        if len(self.audio_buffer) > 1280:
            # Get prediction from openWakeWord
            prediction = self.model.predict(list(self.audio_buffer[0:1280]))

            # Advance the buffer
            self.audio_buffer = self.audio_buffer[1280:]

            # Check for score above threshold
            for mdl_name in self.model_names:
                if prediction[mdl_name] >= self.config.get("threshold"):
                    # Set flag indicating that a wakeword was detected
                    self.has_found = True

                    # Flush recent history of openWakeWord feature buffer to avoid re-activations
                    n_frames = self.model.model_inputs[mdl_name]
                    self.model.preprocessor.feature_buffer[-n_frames:, :] = np.zeros((n_frames, 96)).astype(np.float32)
                    
                    break
                    

    def found_wake_word(self, frame_data):
        if self.has_found:
            self.has_found = False
            return True
        return False
