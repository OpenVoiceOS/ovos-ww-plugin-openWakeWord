## Description

This is an OVOS plugin for [openWakeWord](https://www.github.com/dscripka/openwakeword), an open-source
wakeword or phrase detection system. It has competitive performance compared to Mycroft Precise or Picovoice Porcupine,
can be trained on 100% synthetic data, and can run on a single Raspberry Pi 3 core.

## Install

`pip install ovos-ww-plugin-openwakeword`

Configure your wake word in mycroft.conf

```json
 "listener": {
      "wake_word": "my_wakeword"
 },
 "hotwords": {
    "my_wakeword": {
        "module": "ovos-ww-plugin-openwakeword",
        "model": "path/to/openwakeword/model/my_model.onnx"
    }
  }
 
```

Advanced configuration

```json
 "listener": {
      "wake_word": "android"
 },
 "hotwords": {
    "android": {
        "module": "ovos-ww-plugin-openwakeword",
        "model": "path/to/openwakeword/model/my_model.onnx",
        "threshold": 0.5,
        "enable_speex_noise_suppression": true,
        "expected_duration": 3
    }
  }
 
```

See the [openWakeWord](https://www.github.com/dscripka/openwakeword) repository for more information on the possible arguments to the plugin.