#!/usr/bin/env python3
"""Spot-check all available Parakeet CoreML repos against a reference audio clip.

For each repo dir that has metadata.json + all mlpackage files, runs inference
and prints the transcript. Pass to confirm models are working after conversion.

Usage:
    python spot_check_parakeet.py
    python spot_check_parakeet.py --repos-root /Volumes/hdd/models/hf-repos
    python spot_check_parakeet.py --audio /path/to/audio.wav --pattern "parakeet-ctc*"

ANE/GPU dispatch:
    When pyobjc-framework-CoreML is installed models are compiled to .mlmodelc
    and loaded via the native CoreML ObjC framework for full ANE/GPU use.
    Install with: pip install pyobjc-core pyobjc-framework-CoreML
"""
from __future__ import annotations

import ctypes
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import coremltools as ct
import numpy as np
import soundfile as sf
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

REPOS_ROOT = Path("/Volumes/hdd/models/hf-repos")
DEFAULT_AUDIO = Path(__file__).parent / "yc_first_minute_16k_15s.wav"
SAMPLE_RATE = 16_000

# ── PyObjC CoreML backend ─────────────────────────────────────────────────────
try:
    import CoreML as _CoreML
    from Foundation import NSURL as _NSURL
    _PYOBJC = True
except ImportError:
    _PYOBJC = False

_CT_TO_ML_CU = {}
if _PYOBJC:
    _CT_TO_ML_CU = {
        ct.ComputeUnit.ALL:       _CoreML.MLComputeUnitsAll,
        ct.ComputeUnit.CPU_ONLY:  _CoreML.MLComputeUnitsCPUOnly,
        ct.ComputeUnit.CPU_AND_GPU: _CoreML.MLComputeUnitsCPUAndGPU,
        ct.ComputeUnit.CPU_AND_NE: _CoreML.MLComputeUnitsCPUAndNeuralEngine,
    }


def _compile_mlpackage(mlpackage_path: str) -> str:
    """Compile .mlpackage → .mlmodelc (cached alongside the package).

    Re-compiles if the .mlpackage is newer than the cached .mlmodelc.
    """
    pkg = Path(mlpackage_path)
    out = pkg.parent / (pkg.stem + ".mlmodelc")
    pkg_mtime = pkg.stat().st_mtime
    if not out.exists() or pkg_mtime > out.stat().st_mtime:
        tmp = ct.utils.compile_model(str(pkg))
        shutil.move(tmp, str(out))
    return str(out)


class _ObjCModel:
    """ObjC MLModel wrapper with numpy bridging for predict()."""

    def __init__(self, objc_model: Any) -> None:
        self._m = objc_model

    def predict(self, inputs: dict) -> dict:
        def _to_mlarray(arr: np.ndarray):
            arr = np.ascontiguousarray(arr)
            dtype_map = {
                np.dtype("float32"): _CoreML.MLMultiArrayDataTypeFloat32,
                np.dtype("float16"): _CoreML.MLMultiArrayDataTypeFloat16,
                np.dtype("int32"):   _CoreML.MLMultiArrayDataTypeInt32,
            }
            ml_dtype = dtype_map[arr.dtype]
            strides = [s // arr.itemsize for s in arr.strides]
            c_ptr = arr.ctypes.data_as(ctypes.c_void_p)
            ml_arr, err = _CoreML.MLMultiArray.alloc(
            ).initWithDataPointer_shape_dataType_strides_deallocator_error_(
                c_ptr, list(arr.shape), ml_dtype, strides, None, None
            )
            if err:
                raise RuntimeError(f"MLMultiArray init failed: {err}")
            return ml_arr, arr  # keep arr alive

        def _from_mlarray(ml_arr: Any) -> np.ndarray:
            shape = tuple(int(d) for d in ml_arr.shape())
            total = 1
            for d in shape:
                total *= d
            dtype_info = {
                _CoreML.MLMultiArrayDataTypeFloat32: (np.float32, 4),
                _CoreML.MLMultiArrayDataTypeFloat16: (np.float16, 2),
                _CoreML.MLMultiArrayDataTypeInt32:   (np.int32,   4),
            }
            np_dtype, itemsize = dtype_info.get(ml_arr.dataType(), (np.float32, 4))
            # Fast: raw bytes via data pointer
            try:
                ptr = ml_arr.dataPointer()
                addr = ptr if isinstance(ptr, int) else int(ptr)
                buf = ctypes.string_at(addr, total * itemsize)
                return np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()
            except Exception:
                pass
            # Fallback: element-wise
            flat = np.empty(total, dtype=np_dtype)
            for i in range(total):
                flat[i] = ml_arr[i]
            return flat.reshape(shape)

        feat_dict = {}
        refs = []
        for name, arr in inputs.items():
            ml_arr, ref = _to_mlarray(arr)
            feat_dict[name] = _CoreML.MLFeatureValue.featureValueWithMultiArray_(ml_arr)
            refs.append(ref)

        provider, err = _CoreML.MLDictionaryFeatureProvider.alloc(
        ).initWithDictionary_error_(feat_dict, None)
        if err:
            raise RuntimeError(f"MLDictionaryFeatureProvider: {err}")

        result, err = self._m.predictionFromFeatures_error_(provider, None)
        if err:
            raise RuntimeError(f"prediction failed: {err}")

        outputs = {}
        for name in result.featureNames():
            fv = result.featureValueForName_(name)
            if fv is not None:
                ml_arr = fv.multiArrayValue()
                if ml_arr is not None:
                    outputs[str(name)] = _from_mlarray(ml_arr)
        return outputs


def _load_model(mlpackage_path: str, compute_units: ct.ComputeUnit) -> Any:
    """Load model via PyObjC (ANE, compiled .mlmodelc) or coremltools (.mlpackage)."""
    if _PYOBJC:
        mlmodelc = _compile_mlpackage(mlpackage_path)
        config = _CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(_CT_TO_ML_CU[compute_units])
        model, err = _CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            _NSURL.fileURLWithPath_(mlmodelc), config, None
        )
        if err or model is None:
            raise RuntimeError(f"Failed to load {mlmodelc}: {err}")
        return _ObjCModel(model)
    # No PyObjC — load .mlpackage directly via coremltools (CPU-biased)
    return ct.models.MLModel(mlpackage_path, compute_units=compute_units)


# ── Audio ─────────────────────────────────────────────────────────────────────

def _load_audio(path: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Audio SR {sr} != {SAMPLE_RATE}")
    if data.ndim > 1:
        data = data[:, 0]
    orig = len(data)
    data = np.pad(data, (0, max(0, max_samples - len(data))))[:max_samples]
    return data[np.newaxis, :].astype(np.float32), np.array([min(orig, max_samples)], dtype=np.int32)


# ── Decoding ──────────────────────────────────────────────────────────────────

def _decode_ctc(log_probs: np.ndarray, vocab: list, blank_id: int) -> str:
    ids = np.argmax(log_probs[0], axis=-1)
    out, prev = [], None
    for t in ids:
        if t != blank_id and t != prev:
            out.append(int(t))
        prev = t
    return "".join(vocab[i] for i in out).replace("▁", " ").strip()


def _duration_frames(dur_idx: int, duration_bins: Optional[list]) -> int:
    """Map joint duration argmax index → frame count via duration_bins."""
    if duration_bins:
        return duration_bins[min(dur_idx, len(duration_bins) - 1)]
    return dur_idx


def _decode_tdt(encoder: np.ndarray, encoder_length: np.ndarray,
                dec_model: Any, joint_model: Any, vocab: list, blank_id: int,
                h_shape: tuple, c_shape: tuple, duration_bins: Optional[list] = None,
                max_sym: int = 10) -> str:
    T = int(encoder_length.flat[0])
    h = np.zeros(h_shape, dtype=np.float32)
    c = np.zeros(c_shape, dtype=np.float32)
    prev = np.array([[blank_id]], dtype=np.int32)
    tlen = np.array([1], dtype=np.int32)
    tokens: list[int] = []
    t = 0

    while t < T:
        dec_out = dec_model.predict({"targets": prev, "target_length": tlen, "h_in": h, "c_in": c})
        dec_feat, h, c = dec_out["decoder"], dec_out["h_out"], dec_out["c_out"]

        syms = 0
        advanced = False
        while not advanced:
            jd = joint_model.predict({"encoder_step": encoder[:, :, t:t + 1], "decoder_step": dec_feat})
            tok = int(jd["token_id"].flat[0])
            dur = _duration_frames(int(jd["duration"].flat[0]), duration_bins)

            if tok == blank_id or syms >= max_sym:
                t += max(1, dur)
                advanced = True
            else:
                tokens.append(tok)
                prev = np.array([[tok]], dtype=np.int32)
                syms += 1
                if dur > 0:
                    t += dur
                    advanced = True
                else:
                    dec_out = dec_model.predict({"targets": prev, "target_length": tlen, "h_in": h, "c_in": c})
                    dec_feat, h, c = dec_out["decoder"], dec_out["h_out"], dec_out["c_out"]

    return "".join(vocab[i] for i in tokens if i < len(vocab)).replace("▁", " ").strip()


# ── Repo check ────────────────────────────────────────────────────────────────

def _check_repo(repo_dir: Path, audio_path: Path) -> Optional[str]:
    """Return transcript string, or None if repo is not ready."""
    meta_path = repo_dir / "metadata.json"
    vocab_path = repo_dir / "vocab.json"

    if not meta_path.exists() or not vocab_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    components = meta.get("components", {})

    for comp in components.values():
        ml_path = repo_dir / comp["path"]
        if not ml_path.exists():
            return None

    vocab = json.loads(vocab_path.read_text())
    max_samples = meta["max_audio_samples"]
    audio_signal, audio_length = _load_audio(audio_path, max_samples)

    mel_enc = _load_model(str(repo_dir / components["mel_encoder"]["path"]),
                          ct.ComputeUnit.ALL)
    enc_out = mel_enc.predict({"audio_signal": audio_signal, "audio_length": audio_length})
    encoder = enc_out["encoder"]
    enc_len = enc_out["encoder_length"]

    if "ctc_decoder" in components:
        ctc = _load_model(str(repo_dir / components["ctc_decoder"]["path"]),
                          ct.ComputeUnit.ALL)
        enc_T = int(enc_len.flat[0])
        log_probs = ctc.predict({"encoder": encoder[:, :, :enc_T]})["log_probs"]
        blank_id = meta.get("blank_id", meta.get("vocab_size", len(vocab)))
        return _decode_ctc(log_probs, vocab, blank_id)

    elif "joint_decision_single_step" in components and "decoder" in components:
        blank_id = meta.get("blank_id", meta.get("vocab_size", len(vocab)))
        dec_inputs = components["decoder"]["inputs"]
        h_shape = tuple(dec_inputs["h_in"])
        c_shape = tuple(dec_inputs["c_in"])
        dec_model = _load_model(str(repo_dir / components["decoder"]["path"]),
                                ct.ComputeUnit.CPU_ONLY)  # LSTM: CPU only
        joint_model = _load_model(str(repo_dir / components["joint_decision_single_step"]["path"]),
                                  ct.ComputeUnit.ALL)
        duration_bins = meta.get("duration_bins")
        return _decode_tdt(encoder, enc_len, dec_model, joint_model, vocab, blank_id,
                           h_shape, c_shape, duration_bins=duration_bins)

    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def spot_check(
    repos_root: Path = typer.Option(REPOS_ROOT, "--repos-root"),
    audio: Path = typer.Option(DEFAULT_AUDIO, "--audio"),
    pattern: str = typer.Option("parakeet-*-coreml*", "--pattern"),
    stop_on_error: bool = typer.Option(False, "--stop-on-error"),
) -> None:
    """Run inference on all available converted Parakeet CoreML repos."""
    if not audio.exists():
        raise typer.BadParameter(f"Audio file not found: {audio}")

    backend = "PyObjC/CoreML (ANE)" if _PYOBJC else "coremltools (CPU)"
    typer.echo(f"Backend: {backend}")

    dirs = sorted(repos_root.glob(pattern))
    if not dirs:
        typer.echo(f"No directories match {repos_root}/{pattern}")
        raise typer.Exit(1)

    passed = failed = skipped = 0

    for repo_dir in dirs:
        name = repo_dir.name
        try:
            t0 = time.perf_counter()
            result = _check_repo(repo_dir, audio)
            elapsed = time.perf_counter() - t0

            if result is None:
                typer.echo(f"[SKIP] {name} — not ready (missing files)")
                skipped += 1
            else:
                typer.echo(f"[ OK ] {name} ({elapsed:.1f}s)")
                typer.echo(f"       {result[:120]}")
                passed += 1

        except Exception as e:
            typer.echo(f"[FAIL] {name}: {e}")
            failed += 1
            if stop_on_error:
                raise typer.Exit(1)

    typer.echo(f"\n{'─' * 60}")
    typer.echo(f"Results: {passed} passed, {skipped} skipped (not ready), {failed} failed")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
