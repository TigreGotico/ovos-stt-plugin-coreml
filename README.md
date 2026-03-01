# OVOS CoreML STT Plugin

An OVOS Speech-to-Text (STT) plugin that runs speech recognition models natively on Apple devices using CoreML. Currently supports **Parakeet CTC models** with plans to support additional architectures in the future.

## Features

- **On-device inference**: Uses Apple's CoreML framework for efficient, private speech recognition
- **Parakeet CTC support**: Full support for Parakeet TDT CTC models (tested with 110M variant)
- **Two decoding modes**: Fast greedy decoding, or beam search with an ARPA n-gram language model
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

Download or convert a model, e.g. [parakeet-tdt_ctc-110m](https://huggingface.co/OpenVoiceOS/parakeet-tdt_ctc-110m-coreml). Pre-converted models are available in this [Hugging Face collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-coreml).

### Greedy decoding (default)

```json
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_ctc_coreml/metadata.json",
      "encoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
      "decoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
      "vocab":   "/path/to/parakeet_ctc_coreml/vocab.json"
    }
  }
```

### Beam search with ARPA language model

Add the `lm` key pointing to an uncompressed ARPA file. The three tuning parameters are optional.

```json
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_ctc_coreml/metadata.json",
      "encoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
      "decoder": "/path/to/parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
      "vocab":   "/path/to/parakeet_ctc_coreml/vocab.json",
      "lm":         "/path/to/language_model.arpa",
      "lm_weight":  0.3,
      "word_bonus": 1.0,
      "beam_width": 100
    }
  }
```

| Key | Default | Description |
|---|---|---|
| `lm` | — | Path to an uncompressed ARPA LM file. Omit to use greedy decoding. |
| `lm_weight` | `0.3` | LM interpolation weight (α). Higher values favour the LM more strongly. |
| `word_bonus` | `1.0` | Per-word score bonus in nats (β). Counters the OOV penalty and encourages longer hypotheses. |
| `beam_width` | `100` | Number of beams kept per timestep. |

If your LM file is gzip-compressed, decompress it first:

```bash
gzip -d language_model.arpa.gz
```

## Standalone Usage

### Greedy decoding

```python
from ovos_stt_plugin_coreml import CoremlSTT
from ovos_plugin_manager.utils.audio import AudioFile

config = {
    "metadata": "parakeet_ctc_coreml/metadata.json",
    "vocab":    "parakeet_ctc_coreml/vocab.json",
    "encoder":  "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
    "decoder":  "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
}

stt = CoremlSTT(config=config)

with AudioFile("sample_16k.wav") as f:
    audio = f.read()

print(stt.execute(audio))
```

### Beam search with LM

```python
config = {
    "metadata": "parakeet_ctc_coreml/metadata.json",
    "vocab":    "parakeet_ctc_coreml/vocab.json",
    "encoder":  "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
    "decoder":  "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
    "lm":         "/path/to/language_model.arpa",
    "lm_weight":  0.3,
    "word_bonus": 1.0,
    "beam_width": 100,
}

stt = CoremlSTT(config=config)

with AudioFile("sample_16k.wav") as f:
    audio = f.read()

print(stt.execute(audio))
```

## How beam search works

When an LM is configured, greedy argmax decoding is replaced with a **CTC beam search** that tracks the joint acoustic + language model score for each hypothesis prefix:

1. At each timestep the top-40 non-blank tokens are expanded across all active beams.
2. When a token starts with `▁` (SentencePiece word boundary), the completed word is scored against the bigram LM and the score is added to the beam.
3. After the final timestep the last partial word is scored, and the highest-scoring prefix is returned.

The LM is a standard bigram ARPA file with Katz back-off. Unigrams and bigrams are loaded into memory once at startup; inference adds no I/O overhead per utterance.

## Model Support

### Currently Supported

- **Parakeet TDT CTC 110M** (`nvidia/parakeet-tdt_ctc-110m`)
  - Architecture: Encoder-Decoder with CTC head
  - Input: 16 kHz mono audio, up to 15 seconds
  - Vocabulary: SentencePiece BPE (1024 tokens + 1 blank)
  - Output: Log probabilities over vocabulary

## Requirements

See `requirements.txt`:

- `ovos-plugin-manager>=2.1.1,<3.0.0` — Plugin framework
- `ovos-utils>=0.8.4,<1.0.0` — Audio utilities
- `coremltools>=7.1` — CoreML loading

## Export Utilities

Helper scripts are provided in the `scripts/` directory:

- **`export_vocab.py`** — Extract vocabulary from a Parakeet model
- **`convert_to_coreml.py`** — Convert PyTorch models to CoreML packages
- **`compile_modelc.py`** — Compile `.mlpackage` to `.mlmodelc` for iOS deployment
- **`quantize_coreml.py`** — Optional quantization for smaller model files
- **`test_inference.py`** — Quick inference test script

## Limitations & Known Issues

- **Fixed audio window**: Models are exported with a fixed 15-second audio window. Longer audio is truncated; shorter audio is zero-padded.
- **No confidence scores**: Returns 1.0 confidence for all results.
- **Beam search is CPU-bound**: The Python beam search is significantly slower than greedy decoding. For latency-sensitive applications prefer greedy or use a faster LM decoding library.
- **Single architecture**: Only Parakeet CTC models are supported (for now).

## License

Apache License 2.0 – See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional model architectures
- Confidence score estimation
- Streaming/online decoding
- Performance optimizations
