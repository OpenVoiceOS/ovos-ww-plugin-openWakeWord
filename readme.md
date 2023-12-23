## Description

This is an OVOS plugin for [openWakeWord](https://www.github.com/dscripka/openwakeword), an open-source
wakeword or phrase detection system. It has competitive performance compared to Mycroft Precise or Picovoice Porcupine,
can be trained on 100% synthetic data, and can run on a single Raspberry Pi 3 core.

## Install

`pip install ovos-ww-plugin-openwakeword`

Configure your wake word in mycroft.conf. Do not provide the `models` key to just load the default model ("hey jarvis").

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
      "models": ["path/to/openwakeword/model/hey_jarvis.tflite"],
      "inference_framework": "tflite",
      "custom_verifier_models": {"hey_jarvis": "path/to/openwakeword/custom/verifier/model.pkl"},
      "threshold": 0.3,
      "custom_verifier_threshold": 0.1
  }
}
```

For the `models` key, provide paths to as many openWakeWord models (in `.onnx` or `.tflite` format) as desired and any of them can be used to activate OVOS. The `inference_framework` argument should match the type of openwakeword model(s) provided in the in the `models` arguments and can be either `tflite` or `onnx` for Tensorflow Lite and ONNX, respectively. `tflite` (the default for `openWakeWord >=0.5.0`) has better performance on most platforms, but `onnx` may have broader compatibility.

For the `threshold` key, set the score threshold for activation (higher values means less sensitive). The default value of 0.5 is recommended for most use-cases.

The `custom_verifier_model` and `custom_verifier_threshold` arguments are for the [user-specific verification models](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md) that are supported by `openWakeWord>=0.3.0`. Training and using a custom verifier model can significantly improve performance if the included pre-trained models are not sufficient for a given deployment scenario.

See the [openWakeWord](https://www.github.com/dscripka/openwakeword) repository for more details.