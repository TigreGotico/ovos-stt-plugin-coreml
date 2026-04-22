#!/usr/bin/env python3
"""Convert any NVIDIA Parakeet model to CoreML.

Auto-detects the NeMo model class from the model_id and exports the appropriate
components:

  EncDecCTCModelBPE            → CTC only   (parakeet-ctc-*)
  EncDecHybridRNNTCTCBPEModel  → CTC + TDT  (parakeet-tdt_ctc-*)
  EncDecRNNTBPEModel           → TDT or RNNT (parakeet-tdt-*, parakeet-rnnt-*)

For hybrid models both CTC and TDT heads are exported into a single output dir.
The plugin then auto-selects based on metadata (CTC by default; set
model_type="tdt" in plugin config to force TDT).

Use --ctc-only to skip TDT export on hybrid models (smaller output, faster RTF,
slightly higher WER).

Usage:
    # CTC model
    python convert_nvidia_parakeet.py --model-id nvidia/parakeet-ctc-0.6b --output-dir /out/ctc

    # Hybrid: exports both CTC and TDT heads
    python convert_nvidia_parakeet.py --model-id nvidia/parakeet-tdt_ctc-110m --output-dir /out/hybrid

    # Hybrid: CTC head only (fast inference)
    python convert_nvidia_parakeet.py --model-id nvidia/parakeet-tdt_ctc-110m --output-dir /out/hybrid --ctc-only

    # Pure TDT
    python convert_nvidia_parakeet.py --model-id nvidia/parakeet-tdt-0.6b-v3 --output-dir /out/tdt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
import typer

AUTHOR = "Fluid Inference"
DEPLOYMENT_TARGET = ct.target.iOS17


# ── Model wrappers ────────────────────────────────────────────────────────────

class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_len = self.module(input_signal=audio_signal, length=length.to(torch.long))
        return mel, mel_len


class EncoderWrapper(torch.nn.Module):
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc, enc_len = self.module(audio_signal=mel, length=mel_len.to(torch.long))
        return enc, enc_len


class CTCDecoderWrapper(torch.nn.Module):
    """Wraps either asr_model.decoder (pure CTC) or asr_model.ctc_decoder (hybrid)."""
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.module(encoder_output=encoder_output)


class MelEncoderWrapper(torch.nn.Module):
    """Fused waveform → mel → encoder."""
    def __init__(self, pre: PreprocessorWrapper, enc: EncoderWrapper):
        super().__init__(); self.preprocessor = pre; self.encoder = enc
    def forward(self, audio_signal: torch.Tensor, audio_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_len = self.preprocessor(audio_signal, audio_length)
        enc, enc_len = self.encoder(mel, mel_len.to(torch.int32))
        return enc, enc_len.to(torch.int32)


class DecoderWrapper(torch.nn.Module):
    """RNNT prediction network (stateful LSTM)."""
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, targets: torch.Tensor, target_lengths: torch.Tensor,
                h_in: torch.Tensor, c_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _, new_state = self.module(
            targets=targets.to(torch.long),
            target_length=target_lengths.to(torch.long),
            states=[h_in, c_in],
        )
        return out, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        enc = enc.transpose(1, 2); dec = dec.transpose(1, 2)
        x = self.module.enc(enc).unsqueeze(2) + self.module.pred(dec).unsqueeze(1)
        x = self.module.joint_net[0](x); x = self.module.joint_net[1](x)
        return self.module.joint_net[2](x)


class JointDecisionSingleStep(torch.nn.Module):
    """Single encoder-frame decision. Handles TDT (num_extra > 0) and RNNT (num_extra == 0)."""
    def __init__(self, joint: JointWrapper, vocab_size: int, num_extra: int):
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        logits = self.joint(encoder_step, decoder_step)
        token_logits = logits[..., :self.vocab_with_blank]
        token_ids = torch.argmax(token_logits, dim=-1, keepdim=False).to(torch.int32)
        token_probs = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(token_probs, dim=-1, index=token_ids.long().unsqueeze(-1)).squeeze(-1)
        if self.num_extra > 0:
            duration = torch.argmax(logits[..., -self.num_extra:], dim=-1, keepdim=False).to(torch.int32)
        else:
            # Pure RNNT: no duration head — return 0 (plugin stays on frame until blank)
            duration = torch.zeros_like(token_ids)
        return token_ids, token_prob, duration


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(t: torch.Tensor) -> list: return list(int(d) for d in t.shape)


def _parse_cu(name: str) -> ct.ComputeUnit:
    m = {"ALL": ct.ComputeUnit.ALL, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
         "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE}
    v = name.strip().upper()
    if v not in m: raise typer.BadParameter(f"Unknown compute units '{name}'")
    return m[v]


def _parse_prec(name: Optional[str]) -> Optional[ct.precision]:
    if not name: return None
    m = {"FLOAT32": ct.precision.FLOAT32, "FLOAT16": ct.precision.FLOAT16}
    v = name.strip().upper()
    if v not in m: raise typer.BadParameter(f"Unknown precision '{name}'")
    return m[v]


def _cvt(traced, inputs, outputs, cu, prec=None) -> ct.models.MLModel:
    kwargs = dict(convert_to="mlprogram", inputs=inputs, outputs=outputs,
                  compute_units=cu, skip_model_load=True,
                  minimum_deployment_target=DEPLOYMENT_TARGET)
    if prec: kwargs["compute_precision"] = prec
    return ct.convert(traced, **kwargs)


def _save(model: ct.models.MLModel, path: Path, desc: str) -> None:
    try: model.minimum_deployment_target = DEPLOYMENT_TARGET
    except Exception: pass
    model.short_description = desc; model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True); model.save(str(path))
    typer.echo(f"  → {path}")


def _audio(path: Path, sr: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr: raise typer.BadParameter(f"Audio SR {file_sr} != model SR {sr}")
    if data.ndim > 1: data = data[:, 0]
    orig = len(data)
    data = np.pad(data, (0, max(0, n - len(data))))[:n]
    return torch.from_numpy(data).unsqueeze(0).float(), torch.tensor([min(orig, n)], dtype=torch.int32)


def _vocab(asr_model) -> list:
    try:
        sp = asr_model.tokenizer.tokenizer
        return [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    except Exception:
        return [asr_model.tokenizer.ids_to_tokens([i])[0] for i in range(asr_model.tokenizer.vocab_size)]


def _detect_class(model_id: str) -> str:
    slug = model_id.split("/")[-1].lower()
    if "tdt_ctc" in slug or "tdt-ctc" in slug:
        return "EncDecHybridRNNTCTCBPEModel"
    if "ctc" in slug and "tdt" not in slug and "rnnt" not in slug:
        return "EncDecCTCModelBPE"
    return "EncDecRNNTBPEModel"  # TDT, RNNT, or unknown


def _load_model(model_id: str, nemo_path: Optional[Path], cls_name: Optional[str]):
    from huggingface_hub import snapshot_download
    if not cls_name:
        cls_name = _detect_class(model_id)
    cls = getattr(nemo_asr.models, cls_name)
    typer.echo(f"Loading {cls_name} from {'file' if nemo_path else 'pretrained'}…")
    if nemo_path:
        model = cls.restore_from(str(nemo_path), map_location="cpu")
        checkpoint = {"type": "file", "path": str(nemo_path)}
    else:
        try:
            model = cls.from_pretrained(model_id, map_location="cpu")
        except Exception:
            snap = snapshot_download(model_id)
            nemo_files = sorted(Path(snap).glob("*.nemo"))
            if not nemo_files:
                raise FileNotFoundError(f"No .nemo file found in {snap}")
            typer.echo(f"Falling back to restore_from: {nemo_files[0]}")
            model = cls.restore_from(str(nemo_files[0]), map_location="cpu")
        checkpoint = {"type": "pretrained", "model_id": model_id}
    return model.eval(), cls_name, checkpoint


# ── Export functions ──────────────────────────────────────────────────────────

def _export_mel_encoder(pre, enc, audio_tensor, audio_length, max_samples, cu, prec, output_dir):
    me = MelEncoderWrapper(pre, enc).cpu().eval()
    traced = torch.jit.trace(me, (audio_tensor.cpu(), audio_length.cpu()), strict=False)
    model = _cvt(traced,
        inputs=[ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
                ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="encoder", dtype=np.float32),
                 ct.TensorType(name="encoder_length", dtype=np.int32)],
        cu=cu, prec=prec)
    path = output_dir / "parakeet_mel_encoder.mlpackage"
    _save(model, path, "Parakeet fused Mel+Encoder")
    return path


def _export_ctc_decoder(ctc_head, encoder_ref, cu, prec, output_dir):
    wrapper = CTCDecoderWrapper(ctc_head).cpu().eval()
    traced = torch.jit.trace(wrapper, (encoder_ref.cpu(),), strict=False)
    enc_T = encoder_ref.shape[2]
    model = _cvt(traced,
        inputs=[ct.TensorType(name="encoder",
                              shape=(1, encoder_ref.shape[1], ct.RangeDim(1, enc_T)),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="log_probs", dtype=np.float32)],
        cu=cu, prec=prec)
    path = output_dir / "parakeet_ctc_decoder.mlpackage"
    _save(model, path, "Parakeet CTC decoder head")
    return path


def _export_rnnt_components(dec_wrapper, joint_wrapper, encoder_ref, decoder_ref,
                             targets, target_lengths, zero_state, h_ref, c_ref,
                             vocab_size, num_extra, enc_dim, dec_dim,
                             cu, prec, output_dir):
    """Export decoder LSTM, joint, joint_decision, joint_decision_single_step."""

    # Decoder (LSTM — CPU only)
    traced = torch.jit.trace(dec_wrapper, (targets, target_lengths, zero_state, zero_state), strict=False)
    dec_model = _cvt(traced,
        inputs=[ct.TensorType(name="targets", shape=tuple(targets.shape), dtype=np.int32),
                ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
                ct.TensorType(name="h_in", shape=tuple(zero_state.shape), dtype=np.float32),
                ct.TensorType(name="c_in", shape=tuple(zero_state.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="decoder", dtype=np.float32),
                 ct.TensorType(name="h_out", dtype=np.float32),
                 ct.TensorType(name="c_out", dtype=np.float32)],
        cu=ct.ComputeUnit.CPU_ONLY, prec=prec)  # LSTM stays on CPU
    dec_path = output_dir / "parakeet_decoder.mlpackage"
    _save(dec_model, dec_path, "Parakeet RNNT prediction network (LSTM)")

    # JointDecisionSingleStep
    jds = JointDecisionSingleStep(joint_wrapper, vocab_size=vocab_size, num_extra=num_extra)
    enc_step = encoder_ref[:, :, :1].contiguous()
    dec_step = decoder_ref[:, :, :1].contiguous()
    traced = torch.jit.trace(jds, (enc_step, dec_step), strict=False)
    jds_model = _cvt(traced,
        inputs=[ct.TensorType(name="encoder_step", shape=(1, enc_dim, 1), dtype=np.float32),
                ct.TensorType(name="decoder_step", shape=(1, dec_dim, 1), dtype=np.float32)],
        outputs=[ct.TensorType(name="token_id", dtype=np.int32),
                 ct.TensorType(name="token_prob", dtype=np.float32),
                 ct.TensorType(name="duration", dtype=np.int32)],
        cu=cu, prec=prec)
    jds_path = output_dir / "parakeet_joint_decision_single_step.mlpackage"
    _save(jds_model, jds_path, "Parakeet single-step joint decision (streaming)")

    return dec_path, jds_path


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option("nvidia/parakeet-tdt_ctc-110m", "--model-id"),
    nemo_path: Optional[Path] = typer.Option(None, "--nemo-path", exists=True, resolve_path=True),
    model_class: Optional[str] = typer.Option(
        None, "--model-class",
        help="EncDecCTCModelBPE | EncDecHybridRNNTCTCBPEModel | EncDecRNNTBPEModel (auto-detected if omitted)"),
    output_dir: Path = typer.Option(Path("parakeet_coreml"), "--output-dir"),
    trace_audio: Optional[Path] = typer.Option(
        None, "--trace-audio",
        help="16 kHz WAV for tracing (defaults to yc_first_minute_16k_15s.wav next to script)"),
    max_audio_seconds: float = typer.Option(15.0, "--max-audio-seconds"),
    mel_encoder_cu: str = typer.Option("ALL", "--mel-encoder-cu"),
    joint_cu: str = typer.Option("ALL", "--joint-cu"),
    ctc_cu: str = typer.Option("ALL", "--ctc-cu"),
    compute_precision: Optional[str] = typer.Option(
        "FLOAT32", "--compute-precision",
        help="Export precision: FLOAT32 (default, required for 0.6b conformer — FP16 causes NaN with real audio) or FLOAT16",
    ),
    ctc_only: bool = typer.Option(False, "--ctc-only",
        help="Export only the CTC head (faster RTF, higher WER). Ignored for pure CTC/TDT models."),
    language: str = typer.Option("", "--language", help="Language code stored in metadata."),
) -> None:
    """Export any NVIDIA Parakeet model to CoreML (CTC, Hybrid TDT+CTC, pure TDT, pure RNNT)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    asr_model, cls_name, checkpoint = _load_model(model_id, nemo_path, model_class)

    # FastConformer 1.1b uses local attention windows (att_context_size=[128,128])
    # which produce as_strided ops unsupported by CoreML. Switch to global attention.
    if hasattr(asr_model.encoder, "set_default_att_context_size"):
        asr_model.encoder.set_default_att_context_size([-1, -1])

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = int(round(max_audio_seconds * sample_rate))

    if trace_audio is None:
        trace_audio = Path(__file__).parent / "yc_first_minute_16k_15s.wav"
    if not trace_audio.exists():
        raise typer.BadParameter(f"Trace audio not found: {trace_audio}")

    audio_tensor, audio_length = _audio(trace_audio, sample_rate, max_samples)

    pre = PreprocessorWrapper(asr_model.preprocessor.eval())
    enc = EncoderWrapper(asr_model.encoder.eval())

    with torch.no_grad():
        mel_ref, mel_len_ref = pre(audio_tensor, audio_length)
        mel_len_ref = mel_len_ref.to(torch.int32)
        enc_ref, _ = enc(mel_ref, mel_len_ref)
    enc_ref = enc_ref.clone()

    prec = _parse_prec(compute_precision)
    mel_cu = _parse_cu(mel_encoder_cu)
    jnt_cu = _parse_cu(joint_cu)
    ctc_unit = _parse_cu(ctc_cu)

    # ── 1. Fused Mel+Encoder (always exported) ────────────────────────────────
    typer.echo("Exporting MelEncoder…")
    mel_path = _export_mel_encoder(pre, enc, audio_tensor, audio_length, max_samples, mel_cu, prec, output_dir)

    components: dict = {
        "mel_encoder": {
            "path": mel_path.name,
            "inputs": {"audio_signal": [1, max_samples], "audio_length": [1]},
            "outputs": {"encoder": _ts(enc_ref), "encoder_length": [1]},
        },
    }

    # ── 2a. CTC head (CTC or Hybrid models) ──────────────────────────────────
    is_ctc_model = cls_name == "EncDecCTCModelBPE"
    is_hybrid = cls_name == "EncDecHybridRNNTCTCBPEModel"
    is_rnnt_family = cls_name == "EncDecRNNTBPEModel"

    vocab_size = None
    blank_id = None

    if is_ctc_model or is_hybrid:
        typer.echo("Exporting CTC decoder…")
        ctc_head = asr_model.decoder if is_ctc_model else asr_model.ctc_decoder
        with torch.no_grad():
            log_probs_ref = CTCDecoderWrapper(ctc_head)(enc_ref)
        log_probs_ref = log_probs_ref.clone()

        ctc_path = _export_ctc_decoder(ctc_head, enc_ref, ctc_unit, prec, output_dir)
        n_classes = int(getattr(ctc_head, "num_classes_with_blank", log_probs_ref.shape[-1]))
        vocab_size = n_classes - 1
        blank_id = vocab_size
        components["ctc_decoder"] = {
            "path": ctc_path.name,
            "inputs": {"encoder": _ts(enc_ref)},
            "outputs": {"log_probs": _ts(log_probs_ref)},
        }

    # ── 2b. RNNT / TDT components ────────────────────────────────────────────
    num_extra = 0
    duration_bins = []

    if (is_rnnt_family or (is_hybrid and not ctc_only)):
        num_extra = int(getattr(asr_model.joint, "num_extra_outputs", 0))
        # Read duration_bins from model config (TDT); fall back to [0..num_extra-1]
        _tdt_cfg = getattr(asr_model.cfg.model_defaults, "tdt_durations", None)
        if _tdt_cfg is not None:
            duration_bins = list(_tdt_cfg)
        elif num_extra > 0:
            duration_bins = list(range(num_extra))
        else:
            duration_bins = []
        blank_idx = int(asr_model.decoder.blank_idx)
        dec_hidden = int(asr_model.decoder.pred_hidden)
        dec_layers = int(asr_model.decoder.pred_rnn_layers)

        if vocab_size is None:
            vocab_size = int(asr_model.tokenizer.vocab_size)
            blank_id = vocab_size

        targets = torch.full((1, 1), fill_value=blank_idx, dtype=torch.int32)
        target_lengths = torch.tensor([1], dtype=torch.int32)
        zero_state = torch.zeros(dec_layers, 1, dec_hidden, dtype=torch.float32)

        dec_wrapper = DecoderWrapper(asr_model.decoder.eval())
        joint_wrapper = JointWrapper(asr_model.joint.eval())
        asr_model.decoder._rnnt_export = True

        try:
            with torch.no_grad():
                dec_ref, h_ref, c_ref = dec_wrapper(targets, target_lengths, zero_state, zero_state)
            dec_ref = dec_ref.clone(); h_ref = h_ref.clone(); c_ref = c_ref.clone()

            enc_dim = enc_ref.shape[1]
            dec_dim = dec_ref.shape[1]

            typer.echo("Exporting Decoder + JointDecisionSingleStep…")
            dec_path, jds_path = _export_rnnt_components(
                dec_wrapper, joint_wrapper, enc_ref, dec_ref,
                targets, target_lengths, zero_state, h_ref, c_ref,
                vocab_size, num_extra, enc_dim, dec_dim,
                jnt_cu, prec, output_dir)

            components["decoder"] = {
                "path": dec_path.name,
                "inputs": {
                    "targets": _ts(targets), "target_length": [1],
                    "h_in": _ts(zero_state), "c_in": _ts(zero_state),
                },
                "outputs": {"decoder": _ts(dec_ref), "h_out": _ts(h_ref), "c_out": _ts(c_ref)},
            }
            components["joint_decision_single_step"] = {
                "path": jds_path.name,
                "inputs": {"encoder_step": [1, enc_dim, 1], "decoder_step": [1, dec_dim, 1]},
                "outputs": {"token_id": [1, 1, 1], "token_prob": [1, 1, 1], "duration": [1, 1, 1]},
            }
        finally:
            asr_model.decoder._rnnt_export = False

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = _vocab(asr_model)
    if vocab_size is None:
        vocab_size = len(vocab)
        blank_id = vocab_size
    (output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    # ── Metadata ──────────────────────────────────────────────────────────────
    model_type = "ctc" if (is_ctc_model or (is_hybrid and ctc_only)) else \
                 ("parakeet_tdt_rnnt" if num_extra > 0 else "parakeet_rnnt")
    meta = {
        "model_id": model_id,
        "model_type": model_type,
        "language": language,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "max_audio_samples": max_samples,
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "checkpoint": checkpoint,
        "coreml": {
            "compute_precision": (prec.name if prec else "FLOAT32"),
            "quantization": "none",
        },
        "components": components,
    }
    if num_extra > 0:
        meta["joint_extra_outputs"] = num_extra
        meta["duration_bins"] = duration_bins
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    typer.echo(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    app()
