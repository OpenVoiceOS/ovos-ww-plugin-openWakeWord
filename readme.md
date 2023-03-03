## Description

This is an OVOS plugin for [openWakeWord](https://www.github.com/dscripka/openwakeword), an open-source
wakeword or phrase detection system. It has competitive performance compared to Mycroft Precise or Picovoice Porcupine,
can be trained on 100% synthetic data, and can run on a single Raspberry Pi 3 core.

## Install

`pip install ovos-ww-plugin-openwakeword`

Configure your wake word in mycroft.conf. Do not provide the `model` key to just load the default model ("hey jarvis").

```json
"listener": {
    "wake_word": "hey_jarvis"
},
"hotwords": {
  "hey_jarvis": {
      "module": "ovos-ww-plugin-openwakeword"
  }
} 
```

Additional configuration options:

```json
"listener": {
    "wake_word": "android"
},
"hotwords": {
  "android": {
      "module": "ovos-ww-plugin-openwakeword",
      "model": ["path/to/openwakeword/model/my_model.onnx"], # provide paths to as many models as desired
      "threshold": 0.5,  # the score threshold for activation (higher values means less sensitive)
  }
}
```

See the [openWakeWord](https://www.github.com/dscripka/openwakeword) for more details.