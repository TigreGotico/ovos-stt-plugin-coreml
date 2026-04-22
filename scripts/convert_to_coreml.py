#!/usr/bin/env python3
"""Export Parakeet TDT CTC 110M (Mel+Encoder+CTC) to CoreML."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt_ctc-110m"
AUTHOR = "Fluid Inference"


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target.iOS17]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_NEURALENGINE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if normalized == "":
        return None
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute precision '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    cu = compute_units_override if compute_units_override is not None else settings.compute_units
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    return ct.convert(traced, **kwargs)


def _default_trace_audio() -> Optional[Path]:
    local_audio = Path(__file__).resolve().parent / "audio" / "yc_first_minute_16k_15s.wav"
    if local_audio.exists():
        return local_audio
    fallback = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "parakeet-tdt-v2-0.6b"
        / "coreml"
        / "audio"
        / "yc_first_minute_16k_15s.wav"
    )
    if fallback.exists():
        return fallback
    return None


def _prepare_audio(audio_path: Path, sample_rate: int, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not audio_path.exists():
        raise typer.BadParameter(f"Validation audio not found: {audio_path}")
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Validation audio sample rate {sr} does not match model rate {sample_rate}"
        )
    if data.ndim > 1:
        data = data[:, 0]
    if data.size == 0:
        raise typer.BadParameter("Validation audio is empty")
    original_len = int(data.size)
    if data.size < max_samples:
        data = np.pad(data, (0, max_samples - data.size))
    elif data.size > max_samples:
        data = data[:max_samples]
    audio_tensor = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    length = torch.tensor([min(original_len, max_samples)], dtype=torch.int32)
    return audio_tensor, length


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(input_signal=audio_signal, length=length.to(dtype=torch.long))
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(audio_signal=mel, length=mel_length.to(dtype=torch.long))
        return encoded, encoded_lengths


class CTCDecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.module(encoder_output=encoder_output)


class MelEncoderWrapper(torch.nn.Module):
    """Waveform -> mel -> encoder (CTC-ready features)."""

    def __init__(self, preprocessor: PreprocessorWrapper, encoder: EncoderWrapper) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, audio_length: torch.Tensor):
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoder_out, encoder_length = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoder_out, encoder_length.to(dtype=torch.int32)


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Optional[Path] = typer.Option(
        None,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to parakeet-tdt_ctc-110m .nemo checkpoint (skip to auto-download)",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier to download when --nemo-path is omitted",
    ),
    output_dir: Path = typer.Option(
        Path("parakeet_ctc_coreml"),
        help="Directory where mlpackage and metadata will be written",
    ),
    audio_path: Optional[Path] = typer.Option(
        None,
        "--audio-path",
        resolve_path=True,
        help="Path to 15s 16kHz WAV used for tracing (defaults to bundled yc_first_minute_16k_15s.wav)",
    ),
    max_audio_seconds: float = typer.Option(
        15.0,
        "--max-audio-seconds",
        help="Fixed waveform window (seconds) for export",
    ),
    compute_units: str = typer.Option(
        "CPU_AND_NE",
        "--compute-units",
        help="Compute units for conversion: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE",
    ),
    compute_precision: Optional[str] = typer.Option(
        "FLOAT32",
        "--compute-precision",
        help="Export precision: FLOAT32 (default, required for 0.6b conformer — FP16 causes NaN with real audio) or FLOAT16",
    ),
) -> None:
    """Export fused Mel+Encoder+CTC CoreML model (waveform -> log_probs)."""
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=_parse_compute_units(compute_units),
        deployment_target=ct.target.iOS18,
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=max_audio_seconds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading NeMo model ({'file' if nemo_path else 'pretrained'})…")
    if nemo_path is not None:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            str(nemo_path), map_location="cpu"
        )
        checkpoint_meta = {"type": "file", "path": str(nemo_path)}
    else:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_id, map_location="cpu"
        )
        checkpoint_meta = {"type": "pretrained", "model_id": model_id}
    asr_model.eval()

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = int(round(settings.max_audio_seconds * sample_rate))

    trace_audio = audio_path or _default_trace_audio()
    if trace_audio is None:
        raise typer.BadParameter(
            "Provide --audio-path pointing to a 15s 16kHz WAV (no bundled trace audio found)."
        )
    typer.echo(f"Using trace audio: {trace_audio}")
    audio_tensor, audio_length = _prepare_audio(trace_audio, sample_rate, max_samples)

    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    ctc_decoder = CTCDecoderWrapper(asr_model.ctc_decoder.eval())

    with torch.inference_mode():
        mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
        mel_length_ref = mel_length_ref.to(dtype=torch.int32)
        encoder_ref, encoder_length_ref = encoder(mel_ref, mel_length_ref)
        encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)
        log_probs_ref = ctc_decoder(encoder_ref)

    mel_ref = mel_ref.clone()
    mel_length_ref = mel_length_ref.clone()
    encoder_ref = encoder_ref.clone()
    encoder_length_ref = encoder_length_ref.clone()
    log_probs_ref = log_probs_ref.clone()

    typer.echo("Tracing Mel+Encoder (stage 1)…")
    mel_encoder = MelEncoderWrapper(preprocessor, encoder).cpu().eval()
    audio_tensor = audio_tensor.cpu()
    audio_length = audio_length.cpu()
    traced_mel_encoder = torch.jit.trace(mel_encoder, (audio_tensor, audio_length), strict=False)
    traced_mel_encoder.eval()

    mel_inputs = [
        # Use fixed 15s window at export time to avoid runtime shape ambiguity.
        ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
    ]
    mel_outputs = [
        ct.TensorType(name="encoder", dtype=np.float32),
        ct.TensorType(name="encoder_length", dtype=np.int32),
    ]

    mel_model = _coreml_convert(
        traced_mel_encoder,
        mel_inputs,
        mel_outputs,
        settings,
        compute_units_override=settings.compute_units,
    )
    mel_path = output_dir / "parakeet_ctc_mel_encoder.mlpackage"
    _save_mlpackage(
        mel_model,
        mel_path,
        f"Parakeet CTC Mel+Encoder ({settings.max_audio_seconds:.1f}s window)",
    )

    typer.echo("Tracing CTC decoder (stage 2)…")
    traced_ctc = torch.jit.trace(ctc_decoder.cpu().eval(), (encoder_ref.cpu(),), strict=False)
    traced_ctc.eval()
    ctc_inputs = [
        ct.TensorType(
            name="encoder",
            shape=(
                1,
                encoder_ref.shape[1],
                ct.RangeDim(1, int(encoder_ref.shape[2])),
            ),
            dtype=np.float32,
        )
    ]
    ctc_outputs = [ct.TensorType(name="log_probs", dtype=np.float32)]

    ctc_model = _coreml_convert(
        traced_ctc,
        ctc_inputs,
        ctc_outputs,
        settings,
        compute_units_override=settings.compute_units,
    )
    ctc_path = output_dir / "parakeet_ctc_decoder.mlpackage"
    _save_mlpackage(
        ctc_model,
        ctc_path,
        "Parakeet CTC decoder head (encoder -> log_probs)",
    )

    vocab_size = int(getattr(asr_model.ctc_decoder, "num_classes_with_blank", log_probs_ref.shape[-1]) - 1)
    blank_id = vocab_size
    subsampling = int(getattr(asr_model.encoder, "subsampling_factor", 1))

    metadata = {
        "model_id": model_id,
        "sample_rate": sample_rate,
        "max_audio_seconds": settings.max_audio_seconds,
        "max_audio_samples": max_samples,
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "subsampling_factor": subsampling,
        "checkpoint": checkpoint_meta,
        "coreml": {
            "compute_units": settings.compute_units.name,
            "compute_precision": (
                settings.compute_precision.name if settings.compute_precision is not None else "FLOAT32"
            ),
        },
        "components": {
            "mel_encoder": {
                "inputs": {
                    "audio_signal": [1, max_samples],
                    "audio_length": [1],
                },
                "outputs": {
                    "encoder_length": list(_tensor_shape(encoder_length_ref)),
                    "encoder": list(_tensor_shape(encoder_ref)),
                },
                "path": mel_path.name,
            },
            "ctc_decoder": {
                "inputs": {
                    "encoder": list(_tensor_shape(encoder_ref)),
                },
                "outputs": {
                    "log_probs": list(_tensor_shape(log_probs_ref)),
                },
                "path": ctc_path.name,
            },
        },
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    typer.echo(f"Export complete. Metadata written to {metadata_path}")


if __name__ == "__main__":
    app()
