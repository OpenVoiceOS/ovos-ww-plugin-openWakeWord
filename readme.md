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
    "wake_word": "hey_jarvis"
},
"hotwords": {
  "hey_jarvis": {
      "module": "ovos-ww-plugin-openwakeword",
      "model": ["path/to/openwakeword/model/my_model.onnx"],
      "threshold": 0.5,
  }
}
```

For the `model` key, provide paths to as many openWakeWord models as desired and any of them can be used to activate OVOS.

For the `threshold` key, set the score threshold for activation (higher values means less sensitive). The default value of 0.5 is reccomended for most use-cases.

See the [openWakeWord](https://www.github.com/dscripka/openwakeword) repository for more details.