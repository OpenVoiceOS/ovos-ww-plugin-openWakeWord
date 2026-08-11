## Description

This plugin adds [openWakeWord](https://www.github.com/dscripka/openwakeword) support to OpenVoiceOS (OVOS). openWakeWord is an open-source wake word detection system. It trains on synthetic data and runs on a single core of a Raspberry Pi 3.

## Install

Run this command:

```bash
pip install ovos-ww-plugin-openwakeword
```

Set your wake word in `mycroft.conf`. Do not set the `models` key if you want the default model ("hey jarvis").

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

## Configuration

The plugin accepts these extra options:

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

- `models`: paths to one or more openWakeWord models, in `.onnx` or `.tflite` format. Any model in the list can activate OVOS.
- `inference_framework`: the format of the models in `models`. Use `tflite` or `onnx`. `tflite` is the default for `openWakeWord >=0.5.0` and gives better performance on most platforms. `onnx` may work on more platforms.

- `threshold`: the score needed to trigger activation. Higher values need a stronger match. The default, 0.5, works for most cases.
- `custom_verifier_model` and `custom_verifier_threshold`: paths and settings for [custom verifier models](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md), supported since `openWakeWord>=0.3.0`. A custom verifier model can improve performance when the included pre-trained models do not fit your deployment.

See the [openWakeWord](https://www.github.com/dscripka/openwakeword) repository for more details.

## Related projects

- [openWakeWord](https://www.github.com/dscripka/openwakeword): the wake word engine this plugin wraps.
- [OVOS Plugin Manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): loads and configures this plugin.
- [ovos-ww-plugin-precise-lite](https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-lite): another OVOS wake word plugin.

## License

Apache-2.0
