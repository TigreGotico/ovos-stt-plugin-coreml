# OVOS CoreML STT Plugin

An OVOS Speech-to-Text (STT) plugin that runs speech recognition models natively on Apple devices using CoreML. Currently supports **Parakeet CTC models** with plans to support additional architectures in the future.

## Features

- **On-device inference**: Uses Apple's CoreML framework for efficient, private speech recognition
- **Parakeet CTC support**: Full support for Parakeet TDT CTC models (tested with 110M variant)
- **Greedy CTC decoding**: Fast token-based decoding with blank collapsing
- **Two-stage pipeline**: Separate mel-spectrogram/encoder and CTC decoder components
- **Future expansion**: Architecture designed to support additional ASR models

## Installation

### Prerequisites

- Python 3.8+
- macOS 11.0+ (for CoreML support)

### From PyPI

```bash
pip install ovos-stt-plugin-coreml
```

### From Source

```bash
git clone https://github.com/TigreGotico/ovos-stt-plugin-coreml
cd ovos-stt-plugin-coreml
pip install -e .
```


## Configuration

Download or convert a model, e.g. [parakeet-tdt_ctc-110m](https://huggingface.co/OpenVoiceOS/parakeet-tdt_ctc-110m-coreml). You find some pre-converted models in this [huggingface collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-coreml)

```json
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_ctc_coreml/metadata.json",
      "encoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
      "decoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
      "vocab": "/path/to/parakeet_ctc_coreml/vocab.json"
    }
  }
```

## Standalone Usage

```python
from ovos_stt_plugin_coreml import CoremlSTT
from ovos_plugin_manager.utils.audio import AudioFile

config = {
    "metadata": "parakeet_ctc_coreml/metadata.json",
    "vocab": "parakeet_ctc_coreml/vocab.json",
    "encoder": "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
    "decoder": "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
}

stt = CoremlSTT(config=config)

with AudioFile("sample_16k.wav") as f:
    audio = f.read()

transcript = stt.execute(audio)
print(transcript)
```

## Model Support

### Currently Supported

- **Parakeet TDT CTC 110M** (`nvidia/parakeet-tdt_ctc-110m`)
  - Architecture: Encoder-Decoder with CTC head
  - Input: 16 kHz mono audio
  - Vocabulary: SentencePiece BPE (1024 tokens + 1 blank)
  - Output: Log probabilities over vocabulary


## Requirements

See `requirements.txt`:

- `requests` - HTTP utilities
- `ovos-plugin-manager>=2.1.1,<3.0.0` - Plugin framework
- `ovos-utils>=0.8.4,<1.0.0` - Audio utilities
- `coremltools>=7.1` - CoreML conversion/loading

## Export Utilities

Three helper scripts are provided to export your own models:

1. **`export_pa.py`** - Convert .nemo to .pt
2. **`convert_to_coreml.py`** - Convert torch to CoreML packages
3. **`compile_modelc.py`** - Compile `.mlpackage` to `.mlmodelc` for iOS

## Limitations & Known Issues

- **Fixed audio window**: Models are exported with a fixed 15-second audio window. Longer audio is truncated; shorter audio is zero-padded.
- **No confidence scores**: Currently returns 1.0 confidence for all results.
- **Greedy decoding only**: No beam search or alternative decoding strategies implemented.
- **Single architecture**: Only Parakeet CTC models are supported (for now).

## License

Apache License 2.0 – See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional model architectures
- Confidence score estimation
- Streaming/online decoding
- Performance optimizations
