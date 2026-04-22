# OVOS CoreML STT Plugin

An OVOS Speech-to-Text plugin that runs speech recognition models natively on Apple devices using CoreML.

Supports four model families through a single generic plugin class that auto-detects the architecture from
`metadata.json`:

| Model family           | Architecture         | Languages                   | Decoding                                      |
|------------------------|----------------------|-----------------------------|-----------------------------------------------|
| **Pure CTC**           | EncDecCTCModelBPE    | English / Vietnamese        | Greedy argmax or CTC beam search + ARPA LM    |
| **Hybrid RNNT-CTC**    | EncDecHybridRNNTCTCBPEModel | English / Japanese   | CTC path (ctc_decoder preferred)              |
| **TDT**                | EncDecRNNTBPEModel (num_extra > 0) | EN + 25 European (auto-detected) | Greedy TDT loop (token + duration) |
| **Pure RNNT**          | EncDecRNNTBPEModel (num_extra == 0) | EN / DA / multilingual | TDT loop (duration always 0)     |

Pre-converted models (84 published) are in the
[OpenVoiceOS HuggingFace collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-coreml-69a0fa2e3ccaf5a7690de254).

## Features

- **On-device inference** — CoreML runs entirely on-device; no network calls, no data leaves the machine
- **Neural Engine dispatch** — when `pyobjc-framework-CoreML` is installed, models compile to `.mlmodelc` and
  load through the native ObjC `MLModel` for proper ANE/GPU dispatch
- **Auto-detection** — plugin reads `metadata.json` and selects the correct decoding path automatically
- **HuggingFace auto-download** — set `repo_id` instead of `metadata` to download and cache automatically
- **Minimal config** — only the path to `metadata.json` is required; all component paths are resolved from it
- **Compute unit control** — `compute_units` config key: `"all"` (default), `"cpu_only"`, `"cpu_and_gpu"`, `"cpu_and_ne"`
- **Optional ARPA LM** — CTC models support bigram beam search with a language model for higher accuracy
- **Quantization support** — FP32, FP16, INT8, 4-bit, 6-bit variants available on HuggingFace

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

### From Source

```bash
git clone https://github.com/TigreGotico/ovos-stt-plugin-coreml
cd ovos-stt-plugin-coreml
pip install -e .
```

## Quick Start

### HuggingFace auto-download (simplest)

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "repo_id": "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml"
    }
  }
}
```

Requires `pip install huggingface-hub`. The model is cached in `~/.cache/huggingface/hub` (shared with transformers).

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
| `repo_id`                    | —                           | HF repo id; auto-downloads when `metadata` is absent   |
| `metadata`                   | **required** (or `repo_id`) | Path to `metadata.json` produced by the export script   |
| `model_type`                 | auto                        | `"ctc"` or `"tdt"` — override auto-detection            |
| `compute_units`              | `"all"`                     | `"all"` · `"cpu_only"` · `"cpu_and_gpu"` · `"cpu_and_ne"` |
| `vocab`                      | `<metadata_dir>/vocab.json` | Path to `vocab.json`                                    |
| `encoder`                    | from metadata               | Path to mel encoder `.mlpackage`                        |
| `decoder`                    | from metadata               | CTC: ctc decoder; TDT: RNNT prediction net `.mlpackage` |
| `joint_decision_single_step` | from metadata               | TDT/RNNT only — single-step joint `.mlpackage`          |
| `max_symbols_per_step`       | `10`                        | TDT/RNNT only — max token emissions per encoder frame   |
| `lm`                         | —                           | CTC only — path to ARPA LM (enables beam search)        |
| `lm_weight`                  | `0.3`                       | CTC + LM — LM interpolation weight                      |
| `word_bonus`                 | `1.0`                       | CTC + LM — per-word score bonus in nats                 |
| `beam_width`                 | `100`                       | CTC + LM — number of beams kept per timestep            |

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

# Minimal — paths resolved from metadata dir
stt = CoremlSTT(config={"metadata": "parakeet_coreml/metadata.json"})

# HF auto-download
stt = CoremlSTT(config={"repo_id": "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml"})

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

1. `mel_encoder` encodes the full utterance once → encoder frames `[1, D, T]`.
2. For each encoder frame `t`:
    - Run the LSTM prediction network with the last emitted token.
    - Query `joint_decision_single_step` with the current encoder frame and prediction output.
    - The joint returns a **token id** and a **duration argmax index**.
    - The duration index is mapped to a frame count via `duration_bins` from metadata (e.g. `[0,1,2,3,4]`).
    - If blank → advance `t` by `max(1, duration)`.
    - If non-blank → emit token; if `duration > 0` advance frame, if `duration == 0` re-run prediction net and stay on
      frame (up to `max_symbols_per_step`).
    - For pure RNNT: duration is always 0; the loop advances one frame per blank.

## Export Utilities

The `scripts/` directory contains conversion and validation tools:

| Script                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `convert_nvidia_parakeet.py`    | Convert any NVIDIA Parakeet model to CoreML (CTC / Hybrid / TDT / RNNT) |
| `quantize_nvidia_parakeet.py`   | Post-hoc weight compression (FP16 / INT8 / 4-bit / 6-bit)           |
| `spot_check_parakeet.py`        | Run inference on all local CoreML repos and print transcripts        |
| `validate_hf_models.sh`         | Parallel spot-check all 90 OpenVoiceOS HF repos                     |
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

# With FLOAT32 compute precision (required for some RNNT models to avoid NaN)
python scripts/convert_nvidia_parakeet.py \
  --model-id nvidia/parakeet-rnnt-0.6b \
  --output-dir ./parakeet-rnnt-0.6b-coreml \
  --compute-precision FLOAT32
```

### Post-hoc weight compression

```bash
# INT8 (best accuracy/size tradeoff)
python scripts/quantize_nvidia_parakeet.py \
  --input-dir ./parakeet-tdt-0.6b-v2-coreml \
  --output-dir ./parakeet-tdt-0.6b-v2-coreml-int8 \
  --dtype int8

# 4-bit palette (smallest)
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
| `joint_decision_single_step` | `ALL` or `CPU_AND_NE` | Linear projection + activation; ANE-friendly        |
| `ctc_decoder`                | `ALL` or `CPU_AND_NE` | Matrix multiply; ANE-friendly                       |

## Model Support

### NVIDIA models

| Slug                      | Source model                          | Type      | Languages           |
|---------------------------|---------------------------------------|-----------|---------------------|
| parakeet-ctc-0.6b         | nvidia/parakeet-ctc-0.6b              | CTC       | English             |
| parakeet-ctc-1.1b         | nvidia/parakeet-ctc-1.1b              | CTC       | English             |
| parakeet-ctc-0.6b-vi      | nvidia/parakeet-ctc-0.6b-Vietnamese   | CTC       | Vietnamese          |
| parakeet-tdt-ctc-110m     | nvidia/parakeet-tdt_ctc-110m          | Hybrid    | English             |
| parakeet-tdt-ctc-0.6b-ja  | nvidia/parakeet-tdt_ctc-0.6b-ja       | Hybrid    | Japanese            |
| parakeet-tdt-0.6b-v2      | nvidia/parakeet-tdt-0.6b-v2           | TDT       | English             |
| parakeet-tdt-0.6b-v3      | nvidia/parakeet-tdt-0.6b-v3           | TDT       | 25 European (auto)  |
| parakeet-tdt-1.1b         | nvidia/parakeet-tdt-1.1b              | TDT       | English             |
| parakeet-rnnt-0.6b        | nvidia/parakeet-rnnt-0.6b             | RNNT      | English             |
| parakeet-rnnt-1.1b        | nvidia/parakeet-rnnt-1.1b             | RNNT      | English             |
| parakeet-rnnt-110m-da     | nvidia/parakeet-rnnt-110m-da-dk       | RNNT      | Danish              |
| parakeet-rnnt-120m-eou    | nvidia/parakeet_realtime_eou_120m-v1  | RNNT      | English (EOU)       |

### Community models

| Slug                       | Source model                          | Type | Languages  |
|----------------------------|---------------------------------------|------|------------|
| parakeet-tdt-0.6b-dutch    | yuriyvnv/parakeet-tdt-0.6b-dutch      | TDT  | Dutch      |
| parakeet-tdt-0.6b-estonian | yuriyvnv/parakeet-tdt-0.6b-estonian   | TDT  | Estonian   |
| parakeet-tdt-0.6b-polish   | yuriyvnv/parakeet-tdt-0.6b-polish     | TDT  | Polish     |
| parakeet-tdt-0.6b-portuguese | yuriyvnv/parakeet-tdt-0.6b-portuguese | TDT | Portuguese |
| parakeet-tdt-0.6b-slovenian | yuriyvnv/parakeet-tdt-0.6b-slovenian | TDT  | Slovenian  |

Each slug is published on HuggingFace as `OpenVoiceOS/<slug>-coreml` in five quantization variants:
FP32 (base), FP16, INT8, 4-bit, 6-bit.

### Parakeet TDT v3 — supported languages

`bg` `hr` `cs` `da` `nl` `en` `et` `fi` `fr` `de` `el` `hu` `it` `lv` `lt` `mt` `pl` `pt` `ro` `sk` `sl` `es` `sv` `ru` `uk`

Language is **automatically detected** from the audio — no language input is accepted or needed.

## Requirements

- `ovos-plugin-manager>=2.1.1,<3.0.0`
- `ovos-utils>=0.8.4,<1.0.0`
- `coremltools>=7.1`
- `huggingface-hub` *(optional — required for `repo_id` auto-download)*
- `pyobjc-framework-CoreML` *(optional — enables ANE/GPU dispatch via native ObjC framework)*

## Limitations

- **Fixed 15-second window** — audio longer than 15 s is truncated; shorter audio is zero-padded. Chunked/streaming
  decoding is not yet supported.
- **No confidence scores** — returns `1.0` for all results.
- **CTC beam search is CPU-bound** — the Python beam search is slower than greedy. For latency-sensitive use prefer
  greedy decoding or a native LM decoder.
- **TDT/RNNT decoder loop is sequential** — each joint step depends on the previous; the Python loop cannot be
  parallelised.
- **RNNT models require FLOAT32 compute precision** — some models (e.g. `parakeet-rnnt-0.6b`) produce NaN with the
  default FP16 compute precision; converted models in the HF collection use FLOAT32.

## License

Apache License 2.0 – See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional model architectures
- Confidence score estimation
- Streaming/online decoding
- Performance optimizations
