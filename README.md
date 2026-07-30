# OVOS CoreML STT Plugin

An OVOS Speech-to-Text plugin that runs speech recognition models on Apple devices with CoreML. Inference
runs on-device. No audio leaves the machine.

A single plugin class supports four model families. It reads `metadata.json` and detects the architecture
automatically:

| Model family           | Architecture         | Languages                   | Decoding                                      |
|------------------------|----------------------|-----------------------------|-----------------------------------------------|
| **Pure CTC**           | EncDecCTCModelBPE    | English / Vietnamese        | Greedy argmax or CTC beam search + ARPA LM    |
| **Hybrid RNNT-CTC**    | EncDecHybridRNNTCTCBPEModel | English / Japanese   | TDT path preferred (lower WER than CTC head)  |
| **TDT**                | EncDecRNNTBPEModel (num_extra > 0) | EN + 25 European (auto-detected) | Greedy TDT loop (token + duration) |
| **Pure RNNT**          | EncDecRNNTBPEModel (num_extra == 0) | EN / DA / multilingual | TDT loop (duration always 0)     |

80 pre-converted models are published in the
[OpenVoiceOS HuggingFace collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-coreml-69a0fa2e3ccaf5a7690de254).

## Features

- **Zero-config**: omit all model settings and the plugin selects the best int8 model for the configured OVOS language.
- **On-device inference**: CoreML runs entirely on-device. No network calls happen, and no data leaves the machine.
- **Neural Engine dispatch**: when `pyobjc-framework-CoreML` is installed, models compile to `.mlmodelc` and
  load through the native ObjC `MLModel` for ANE/GPU dispatch.
- **Auto-detection**: the plugin reads `metadata.json` and selects the correct decoding path automatically.
- **HuggingFace auto-download**: set `repo_id` instead of `metadata` to download and cache the model automatically.
- **27 languages**: dedicated models for EN/JA/VI/DA/NL/ET/PL/PT/SL, plus 16 more EU languages through the multilingual v3 model.
- **Compute unit control**: the `compute_units` config key accepts `"all"` (default), `"cpu_only"`, `"cpu_and_gpu"`, or `"cpu_and_ne"`.
- **Optional ARPA LM**: CTC models support bigram beam search with a language model for higher accuracy.
- **Quantization support**: FP32, INT8, 4-bit, and 6-bit variants on HuggingFace.

## Installation

### Prerequisites

- Python 3.10+
- macOS 13.0+ (CoreML MLProgram / iOS 17 target)
- Xcode command-line tools (only needed for `.mlmodelc` compilation)

### From PyPI

```bash
pip install ovos-stt-plugin-coreml
```

### Optional: ANE/GPU dispatch via PyObjC

```bash
pip install pyobjc-core pyobjc-framework-CoreML
```

When installed, models are compiled to `.mlmodelc` and loaded via the native CoreML ObjC framework for full
ANE/GPU dispatch. Without PyObjC, coremltools loads `.mlpackage` directly (CPU-biased).

### Optional: HuggingFace auto-download

```bash
pip install huggingface-hub
```

Required for `repo_id` auto-download and zero-config mode.

### From Source

```bash
git clone https://github.com/TigreGotico/ovos-stt-plugin-coreml
cd ovos-stt-plugin-coreml
pip install -e .
```

## Quick Start

### Zero-config (auto-selects best int8 model for your language)

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {}
  }
}
```

The plugin reads `lang` from your OVOS config and picks the best published int8 model automatically:

| Language(s) | Default model |
|---|---|
| `en-*` | `parakeet-tdt-ctc-110m-coreml-int8` (smallest/fastest English) |
| `da-*` | `parakeet-rnnt-110m-da-coreml-int8` |
| `nl-*` | `parakeet-tdt-0.6b-dutch-coreml-int8` |
| `et-*` | `parakeet-tdt-0.6b-estonian-coreml-int8` |
| `pl-*` | `parakeet-tdt-0.6b-polish-coreml-int8` |
| `pt-*` | `parakeet-tdt-0.6b-portuguese-coreml-int8` |
| `sl-*` | `parakeet-tdt-0.6b-slovenian-coreml-int8` |
| `ja-*` | `parakeet-tdt-ctc-0.6b-ja-coreml-int8` |
| `vi-*` | `parakeet-ctc-0.6b-vi-coreml-int8` |
| `de/fr/es/it/cs/bg/el/fi/hr/hu/lt/lv/mt/ro/ru/sk/sv/uk-*` | `parakeet-tdt-0.6b-v3-coreml-int8` (multilingual) |

Requires `pip install huggingface-hub`.

### Specific model via HuggingFace repo_id

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "repo_id": "OpenVoiceOS/parakeet-tdt-1.1b-coreml-int8"
    }
  }
}
```

### Local model (minimal)

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_coreml/metadata.json"
    }
  }
}
```

All component paths and vocab are resolved from the metadata directory automatically.

## Configuration

### Full config reference

| Key                          | Default                     | Description                                             |
|------------------------------|-----------------------------|---------------------------------------------------------|
| `repo_id`                    | none                        | HF repo id. Auto-downloads when `metadata` is absent.  |
| `metadata`                   | **required** (or `repo_id`) | Path to `metadata.json` produced by the export script   |
| `model_type`                 | auto                        | `"ctc"` or `"tdt"`, overrides auto-detection            |
| `compute_units`              | `"all"`                     | `"all"` · `"cpu_only"` · `"cpu_and_gpu"` · `"cpu_and_ne"` |
| `vocab`                      | `<metadata_dir>/vocab.json` | Path to `vocab.json`                                    |
| `encoder`                    | from metadata               | Path to mel encoder `.mlpackage`                        |
| `decoder`                    | from metadata               | CTC decoder, or RNNT prediction net for TDT             |
| `joint_decision_single_step` | from metadata               | TDT/RNNT only, single-step joint `.mlpackage`           |
| `max_symbols_per_step`       | `10`                        | TDT/RNNT only, max token emissions per encoder frame    |
| `lm`                         | none                        | CTC only, path to ARPA LM (enables beam search)         |
| `lm_weight`                  | `0.3`                       | CTC + LM, LM interpolation weight                       |
| `word_bonus`                 | `1.0`                       | CTC + LM, per-word score bonus in nats                  |
| `beam_width`                 | `100`                       | CTC + LM, number of beams kept per timestep             |

### CTC beam search with ARPA LM

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata":   "/path/to/parakeet_ctc_coreml/metadata.json",
      "lm":         "/path/to/language_model.arpa",
      "lm_weight":  0.3,
      "word_bonus": 1.0,
      "beam_width": 100
    }
  }
}
```

If your ARPA file is gzip-compressed, decompress it first: `gzip -d model.arpa.gz`

### Force CPU-only inference (faster for short clips)

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata":      "/path/to/parakeet_coreml/metadata.json",
      "compute_units": "cpu_only"
    }
  }
}
```

## Standalone Usage

```python
from ovos_stt_plugin_coreml import CoremlSTT
from ovos_plugin_manager.utils.audio import AudioFile

# Zero-config — auto-selects int8 model for the configured language
stt = CoremlSTT(config={})

# Specific HF repo
stt = CoremlSTT(config={"repo_id": "OpenVoiceOS/parakeet-tdt-1.1b-coreml-int8"})

# Local model
stt = CoremlSTT(config={"metadata": "parakeet_coreml/metadata.json"})

# CTC with LM
stt = CoremlSTT(config={
    "metadata": "parakeet_ctc_coreml/metadata.json",
    "lm": "/path/to/language_model.arpa",
})

with AudioFile("sample_16k.wav") as f:
    audio = f.read()

print(stt.execute(audio))
```

## How Decoding Works

### CTC greedy

Argmax over log-probabilities at each timestep, followed by standard CTC collapse (remove blanks and consecutive
duplicates). The encoder output is sliced to the actual (unpadded) length before the CTC decoder to avoid
spurious tokens from padding frames.

### CTC beam search (with LM)

1. Top-40 non-blank tokens are expanded across all active beams at each timestep.
2. When a token starts with `▁` (SentencePiece word boundary), the completed word is scored against the bigram ARPA LM.
3. After the final timestep the last partial word is scored and the highest-scoring prefix is returned.

### TDT / RNNT greedy

Token and Duration Transducer decoding runs a per-frame loop:

1. `mel_encoder` encodes the full utterance once into encoder frames `[1, D, T]`.
2. For each encoder frame `t`:
    - Run the LSTM prediction network with the last emitted token.
    - Query `joint_decision_single_step` with the current encoder frame and prediction output.
    - The joint returns a **token id** and a **duration argmax index**.
    - The duration index is mapped to a frame count with `duration_bins` from metadata (for example `[0,1,2,3,4]`).
    - If blank, advance `t` by `max(1, duration)`.
    - If non-blank, emit the token. If `duration > 0`, advance the frame. If `duration == 0`, re-run the prediction net
      and stay on the frame (up to `max_symbols_per_step`).
    - For pure RNNT, duration is always 0, and the loop advances one frame per blank.

## Export Utilities

The `scripts/` directory contains conversion and validation tools:

| Script                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `convert_nvidia_parakeet.py`    | Convert any NVIDIA Parakeet model to CoreML (CTC / Hybrid / TDT / RNNT) |
| `quantize_nvidia_parakeet.py`   | Post-hoc weight compression (INT8 / 4-bit / 6-bit)                  |
| `spot_check_parakeet.py`        | Run inference on all local CoreML repos and print transcripts        |
| `validate_hf_models.sh`         | Parallel spot-check all published OpenVoiceOS HF repos               |
| `convert_to_coreml.py`          | Legacy TDT-only converter                                            |
| `quantize_coreml.py`            | Legacy quantizer                                                     |
| `compile_modelc.py`             | Compile `.mlpackage` → `.mlmodelc` via `xcrun coremlcompiler`        |
| `benchmark_rtf_tdt.py`          | Per-component latency benchmarks and RTF report                      |

### Convert any NVIDIA Parakeet model

```bash
# Auto-detect architecture and download from HuggingFace
python scripts/convert_nvidia_parakeet.py \
  --model-id nvidia/parakeet-tdt-0.6b-v2 \
  --output-dir ./parakeet-tdt-0.6b-v2-coreml

# From a local .nemo file
python scripts/convert_nvidia_parakeet.py \
  --model-id yuriyvnv/parakeet-tdt-0.6b-polish \
  --nemo-path /path/to/model.nemo \
  --output-dir ./parakeet-tdt-0.6b-polish-coreml
```

> **Note:** All conversion scripts default to `--compute-precision FLOAT32`. This setting is required for 0.6b-scale
> conformer encoders. FP16 intermediate activations overflow during attention and matmul with real audio, and
> produce all-NaN encoder output. Silence inputs happen to stay within FP16 range and mask the bug, so always test
> with real speech.

### Post-hoc weight compression

```bash
# INT8 (best accuracy/size tradeoff, ~4× smaller)
python scripts/quantize_nvidia_parakeet.py \
  --input-dir ./parakeet-tdt-0.6b-v2-coreml \
  --output-dir ./parakeet-tdt-0.6b-v2-coreml-int8 \
  --dtype int8

# 4-bit palette (~8× smaller)
python scripts/quantize_nvidia_parakeet.py \
  --input-dir ./parakeet-tdt-0.6b-v2-coreml \
  --output-dir ./parakeet-tdt-0.6b-v2-coreml-4bit \
  --dtype 4bit
```

### Compute unit guidance

| Component                    | Recommended           | Notes                                               |
|------------------------------|-----------------------|-----------------------------------------------------|
| `mel_encoder`                | `ALL` or `CPU_AND_NE` | Conformer encoder maps well to ANE on Apple Silicon |
| `decoder`                    | `CPU_ONLY`            | LSTM state passing is always forced to CPU          |
| `joint_decision_single_step` | `ALL` or `CPU_AND_NE` | Linear projection and activation, ANE-friendly      |
| `ctc_decoder`                | `ALL` or `CPU_AND_NE` | Matrix multiply, ANE-friendly                       |

## Model Support

### NVIDIA models

| Slug                      | Source model                          | Type      | Languages           | Notes                                     |
|---------------------------|---------------------------------------|-----------|---------------------|-------------------------------------------|
| parakeet-ctc-0.6b         | nvidia/parakeet-ctc-0.6b              | CTC       | English             | no FP16 variant (NaN in FP16 compute)     |
| parakeet-ctc-1.1b         | nvidia/parakeet-ctc-1.1b              | CTC       | English             |                                           |
| parakeet-ctc-0.6b-vi      | nvidia/parakeet-ctc-0.6b-Vietnamese   | CTC       | Vietnamese          | no FP16 variant (NaN in FP16 compute)     |
| parakeet-tdt-ctc-110m     | nvidia/parakeet-tdt_ctc-110m          | Hybrid    | English             | default for `en-*`                        |
| parakeet-tdt-ctc-0.6b-ja  | nvidia/parakeet-tdt_ctc-0.6b-ja       | Hybrid    | Japanese            |                                           |
| parakeet-tdt-0.6b-v2      | nvidia/parakeet-tdt-0.6b-v2           | TDT       | English             |                                           |
| parakeet-tdt-0.6b-v3      | nvidia/parakeet-tdt-0.6b-v3           | TDT       | 25 European (auto)  | default for 16 EU langs without dedicated model |
| parakeet-tdt-1.1b         | nvidia/parakeet-tdt-1.1b              | TDT       | English             | highest accuracy English model            |
| parakeet-rnnt-0.6b        | nvidia/parakeet-rnnt-0.6b             | RNNT      | English             | no FP16 variant (NaN in FP16 compute)     |
| parakeet-rnnt-1.1b        | nvidia/parakeet-rnnt-1.1b             | RNNT      | English             |                                           |
| parakeet-rnnt-110m-da     | nvidia/parakeet-rnnt-110m-da-dk       | RNNT      | Danish              |                                           |
| parakeet-rnnt-120m-eou    | nvidia/parakeet_realtime_eou_120m-v1  | RNNT      | English (EOU)       | no 4-bit variant                          |

### Community models

| Slug                        | Source model                           | Type | Languages  |
|-----------------------------|----------------------------------------|------|------------|
| parakeet-tdt-0.6b-dutch     | yuriyvnv/parakeet-tdt-0.6b-dutch       | TDT  | Dutch      |
| parakeet-tdt-0.6b-estonian  | yuriyvnv/parakeet-tdt-0.6b-estonian    | TDT  | Estonian   |
| parakeet-tdt-0.6b-polish    | yuriyvnv/parakeet-tdt-0.6b-polish      | TDT  | Polish     |
| parakeet-tdt-0.6b-portuguese| yuriyvnv/parakeet-tdt-0.6b-portuguese  | TDT  | Portuguese |
| parakeet-tdt-0.6b-slovenian | yuriyvnv/parakeet-tdt-0.6b-slovenian   | TDT  | Slovenian  |

Each slug is published on HuggingFace as `OpenVoiceOS/<slug>-coreml` in up to four quantization variants:
FP32 (base), INT8, 4-bit, 6-bit. Models built on the 0.6b conformer encoder (`ctc-0.6b`, `ctc-0.6b-vi`,
`rnnt-0.6b`) have no FP16 variant, because FP16 activations cause NaN during inference with real audio on that architecture.

### Parakeet TDT v3: supported languages

`bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr` `hu` `it` `lv` `lt` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`

The plugin detects the language automatically from the audio. No language input is accepted or needed.

## Requirements

- `ovos-plugin-manager>=2.1.1,<3.0.0`
- `ovos-utils>=0.8.4,<1.0.0`
- `coremltools>=7.1`
- `huggingface-hub` *(optional, required for `repo_id` auto-download and zero-config mode)*
- `pyobjc-framework-CoreML` *(optional, enables ANE/GPU dispatch through the native ObjC framework)*

## Limitations

- **Fixed 15-second window**: audio longer than 15 s is truncated, and shorter audio is zero-padded. Chunked and
  streaming decoding are not yet supported.
- **No confidence scores**: the plugin returns `1.0` for all results.
- **CTC beam search is CPU-bound**: the Python beam search is slower than greedy. For latency-sensitive use, prefer
  greedy decoding or a native LM decoder.
- **TDT/RNNT decoder loop is sequential**: each joint step depends on the previous one, so the Python loop cannot
  run in parallel.
- **0.6b conformer requires FLOAT32 compute precision**: `parakeet-ctc-0.6b`, `parakeet-ctc-0.6b-vi`, and
  `parakeet-rnnt-0.6b` produce NaN with FP16 activations. All models in the HF collection are exported with
  `compute_precision=FLOAT32`. These models have no FP16 quantized variant for the same reason, so use INT8 instead
  (about 4x smaller, and compute stays FLOAT32).

## Related Projects

- [ovos-stt-plugin-whisper](https://github.com/TigreGotico/ovos-stt-plugin-whisper): OVOS STT plugin for OpenAI Whisper models.
- [ovos-stt-plugin-onnx-asr](https://github.com/TigreGotico/ovos-stt-plugin-onnx-asr): OVOS STT plugin for ONNX-exported ASR models.
- [ovos-stt-plugin-sherpa-onnx](https://github.com/TigreGotico/ovos-stt-plugin-sherpa-onnx): OVOS STT plugin backed by sherpa-onnx.
- [ovos-stt-plugin-rover](https://github.com/TigreGotico/ovos-stt-plugin-rover): OVOS STT plugin that combines results from multiple STT engines with ROVER voting.

## License

Apache License 2.0. See the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional model architectures
- Confidence score estimation
- Streaming/online decoding
- Performance optimizations
