# OVOS CoreML STT Plugin

An OVOS Speech-to-Text plugin that runs speech recognition models natively on Apple devices using CoreML.

Supports two model families through a single generic plugin class that auto-detects the architecture from
`metadata.json`:

| Model family                                 | Architecture    | Languages                   | Decoding                                      |
|----------------------------------------------|-----------------|-----------------------------|-----------------------------------------------|
| **Parakeet CTC** (`parakeet-tdt_ctc-110m`)   | Hybrid RNNT-CTC | English only                | Greedy argmax or CTC beam search + ARPA LM    |
| **Parakeet TDT v3** (`parakeet-tdt-0.6b-v3`) | Pure RNNT / TDT | 25 European (auto-detected) | Greedy TDT loop (token + duration prediction) |

## Features

- **On-device inference** — CoreML runs entirely on-device; no network calls, no data leaves the machine
- **Neural Engine ready** — models can be exported targeting ANE for lower latency on Apple Silicon
- **Auto-detection** — plugin reads `metadata.json` and selects the correct decoding path automatically
- **Minimal config** — only the path to `metadata.json` is required; all component paths are resolved from it
- **Optional ARPA LM** — CTC models support bigram beam search with a language model for higher accuracy
- **Quantization support** — INT8/INT4 linear and palette quantization available via export tooling

## Installation

### Prerequisites

- Python 3.10+
- macOS 13.0+ (for CoreML MLProgram / iOS 17 target support)
- Xcode command-line tools (only needed for `.mlmodelc` compilation)

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

## Obtaining a Model

Pre-converted CoreML models are available in
the [OpenVoiceOS Hugging Face collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-coreml).

To convert a model yourself, see [Export Utilities](#export-utilities) below.

## Configuration

### Parakeet CTC — greedy decoding

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_ctc_coreml/metadata.json",
      "vocab":    "/path/to/parakeet_ctc_coreml/vocab.json",
      "encoder":  "/path/to/parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
      "decoder":  "/path/to/parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage"
    }
  }
}
```

### Parakeet CTC — beam search with ARPA language model

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata":   "/path/to/parakeet_ctc_coreml/metadata.json",
      "vocab":      "/path/to/parakeet_ctc_coreml/vocab.json",
      "encoder":    "/path/to/parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
      "decoder":    "/path/to/parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
      "lm":         "/path/to/language_model.arpa",
      "lm_weight":  0.3,
      "word_bonus": 1.0,
      "beam_width": 100
    }
  }
}
```

If your ARPA file is gzip-compressed, decompress it first: `gzip -d model.arpa.gz`

### Parakeet TDT v3

```json
{
  "stt": {
    "module": "ovos-stt-plugin-coreml",
    "ovos-stt-plugin-coreml": {
      "metadata": "/path/to/parakeet_tdt_coreml/metadata.json",
      "vocab":    "/path/to/parakeet_tdt_coreml/vocab.json",
      "encoder":  "/path/to/parakeet_tdt_coreml/mel_encoder.mlpackage",
      "decoder":  "/path/to/parakeet_tdt_coreml/decoder.mlpackage",
      "joint_decision_single_step": "/path/to/parakeet_tdt_coreml/joint_decision_single_step.mlpackage"
    }
  }
}
```

All component paths (`vocab`, `encoder`, `decoder`, `joint_decision_single_step`) are optional — the plugin resolves
them automatically from the metadata directory when omitted. The minimal config is just:

```json
{"metadata": "/path/to/parakeet_coreml/metadata.json"}
```

---

### Full config reference

| Key                          | Default                     | Description                                             |
|------------------------------|-----------------------------|---------------------------------------------------------|
| `metadata`                   | **required**                | Path to `metadata.json` produced by the export script   |
| `model_type`                 | auto                        | `"ctc"` or `"tdt"` — override auto-detection            |
| `vocab`                      | `<metadata_dir>/vocab.json` | Path to `vocab.json`                                    |
| `encoder`                    | from metadata               | Path to mel encoder `.mlpackage` (CTC and TDT)          |
| `decoder`                    | from metadata               | CTC: ctc decoder; TDT: RNNT prediction net `.mlpackage` |
| `joint_decision_single_step` | from metadata               | TDT only — path to single-step joint `.mlpackage`       |
| `max_symbols_per_step`       | `10`                        | TDT only — max token emissions per encoder frame        |
| `lm`                         | —                           | CTC only — path to ARPA LM (enables beam search)        |
| `lm_weight`                  | `0.3`                       | CTC + LM — LM interpolation weight                      |
| `word_bonus`                 | `1.0`                       | CTC + LM — per-word score bonus in nats                 |
| `beam_width`                 | `100`                       | CTC + LM — number of beams kept per timestep            |

## Standalone Usage

```python
from ovos_stt_plugin_coreml import CoremlSTT
from ovos_plugin_manager.utils.audio import AudioFile

# CTC — greedy decoding
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

```python
# CTC — beam search with ARPA language model
config = {
    "metadata":   "parakeet_ctc_coreml/metadata.json",
    "vocab":      "parakeet_ctc_coreml/vocab.json",
    "encoder":    "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
    "decoder":    "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
    "lm":         "/path/to/language_model.arpa",
    "lm_weight":  0.3,
    "word_bonus": 1.0,
    "beam_width": 100,
}
stt = CoremlSTT(config=config)
```

```python
# TDT
config = {
    "metadata": "parakeet_tdt_coreml/metadata.json",
    "vocab":    "parakeet_tdt_coreml/vocab.json",
    "encoder":  "parakeet_tdt_coreml/mel_encoder.mlpackage",
    "decoder":  "parakeet_tdt_coreml/decoder.mlpackage",
    "joint_decision_single_step": "parakeet_tdt_coreml/joint_decision_single_step.mlpackage",
}
stt = CoremlSTT(config=config)
```

## How Decoding Works

### CTC greedy

Argmax over log-probabilities at each timestep, followed by standard CTC collapse (remove blanks and consecutive
duplicates).

### CTC beam search (with LM)

1. Top-40 non-blank tokens are expanded across all active beams at each timestep.
2. When a token starts with `▁` (SentencePiece word boundary), the completed word is scored against the bigram ARPA LM.
3. After the final timestep the last partial word is scored and the highest-scoring prefix is returned.

The LM is loaded once at startup; per-utterance inference adds no I/O overhead.

### TDT greedy

Token and Duration Transducer decoding runs a per-frame loop:

1. `mel_encoder` encodes the full utterance once → encoder frames `[1, D, T]`.
2. For each encoder frame `t`:
    - Run the LSTM prediction network with the last emitted token.
    - Query `joint_decision_single_step` with the current encoder frame and prediction output.
    - The joint returns a **token id** and a **duration** (0–4 frames).
    - If blank → advance `t` by `max(1, duration)`.
    - If non-blank → emit token; if `duration > 0` advance frame, if `duration == 0` re-run prediction net and stay on
      frame (up to `max_symbols_per_step`).

## Export Utilities

The `parakeet_export/` directory (kept alongside this repo) contains conversion scripts:

| Script                     | Purpose                                                              |
|----------------------------|----------------------------------------------------------------------|
| `convert_to_coreml.py`     | Convert a `.nemo` checkpoint to CoreML `.mlpackage` files            |
| `individual_components.py` | PyTorch wrapper classes and `ExportSettings` shared by the converter |
| `export_pa.py`             | Extract `vocab.json` and `model_info.json` from an RNNT checkpoint   |
| `quantize_coreml.py`       | INT8 / INT4 linear and 4-/6-bit palette quantization                 |
| `compile_modelc.py`        | Compile `.mlpackage` → `.mlmodelc` via `xcrun coremlcompiler`        |
| `benchmark_rtf.py`         | Per-component latency benchmarks and RTF report                      |

### Quick export (Parakeet TDT v3)

```bash
cd parakeet_export

# CPU-only export (safe baseline)
python convert_to_coreml.py --nemo-path parakeet-tdt-0.6b-v3.nemo

# Let CoreML compiler choose ANE dispatch for encoder and joint
python convert_to_coreml.py \
  --nemo-path parakeet-tdt-0.6b-v3.nemo \
  --mel-encoder-cu ALL \
  --encoder-cu ALL \
  --joint-cu ALL \
  --joint-decision-cu ALL \
  --joint-decision-single-step-cu ALL

# Quantize weights (run after export)
python quantize_coreml.py

# Compile to .mlmodelc for on-device deployment
python compile_modelc.py
```

All `--*-cu` flags accept: `CPU_ONLY` · `CPU_AND_NE` · `CPU_AND_GPU` · `ALL`

> **Note:** The LSTM prediction network (`--decoder-cu`) does not dispatch to ANE; leave it at `CPU_ONLY`.

### Compute unit guidance

| Component                    | Recommended           | Notes                                               |
|------------------------------|-----------------------|-----------------------------------------------------|
| `mel_encoder`                | `ALL` or `CPU_AND_NE` | Conformer encoder maps well to ANE on Apple Silicon |
| `encoder` (standalone)       | `ALL` or `CPU_AND_NE` | Same as above                                       |
| `decoder`                    | `CPU_ONLY`            | LSTM state passing not supported on ANE             |
| `joint` / `joint_decision`   | `ALL` or `CPU_AND_NE` | Linear projection + activation; ANE-friendly        |
| `joint_decision_single_step` | `ALL` or `CPU_AND_NE` | Same                                                |
| `preprocessor`               | `CPU_ONLY`            | Mel filterbank; minimal compute, CPU is fine        |

## Model Support

| Model                 | HuggingFace ID                 | Type | Size         | Languages                    | Notes                        |
|-----------------------|--------------------------------|------|--------------|------------------------------|------------------------------|
| Parakeet TDT CTC 110M | `nvidia/parakeet-tdt_ctc-110m` | CTC  | ~110M params | English only                 | Greedy or beam search        |
| Parakeet TDT v3 0.6B  | `nvidia/parakeet-tdt-0.6b-v3`  | TDT  | ~600M params | 25 European (auto-detected)  | Greedy TDT loop              |

### Parakeet TDT v3 — supported languages

`bg` `hr` `cs` `da` `nl` `en` `et` `fi` `fr` `de` `el` `hu` `it` `lv` `lt` `mt` `pl` `pt` `ro` `sk` `sl` `es` `sv` `ru` `uk`

Language is **automatically detected** from the audio — no language input is accepted or needed.

## Requirements

- `ovos-plugin-manager>=2.1.1,<3.0.0`
- `ovos-utils>=0.8.4,<1.0.0`
- `coremltools>=7.1`

## Limitations

- **Fixed 15-second window** — audio longer than 15 s is truncated; shorter audio is zero-padded. Chunked/streaming
  decoding is not yet supported.
- **No confidence scores** — returns `1.0` for all results.
- **CTC beam search is CPU-bound** — the Python beam search is slower than greedy. For latency-sensitive use prefer
  greedy decoding or a native LM decoder.
- **TDT decoder loop is sequential** — each joint step depends on the previous; the Python loop cannot be parallelised.

## License

Apache License 2.0 – See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional model architectures
- Confidence score estimation
- Streaming/online decoding
- Performance optimizations
