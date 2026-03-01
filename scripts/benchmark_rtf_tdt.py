#!/usr/bin/env python3
"""Benchmark Parakeet TDT v3 CoreML components and report RTF.

Three passes are measured:
  1. mel_encoder   – waveform → encoder output  (once per utterance)
  2. decoder_step  – single prediction-net step  (latency proxy)
  3. joint_step    – single joint-decision step   (latency proxy)

Full RNNT RTF depends on the number of decoder steps per frame, which
varies per utterance. This script benchmarks each component in isolation
so you can estimate end-to-end cost from the per-step timings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf


# ── Helpers ───────────────────────────────────────────────────────────────────

def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


def calculate_stats(times: List[float]) -> Dict[str, float]:
    arr = np.array(times)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def load_audio(audio_path: Path, sample_rate: int, max_samples: int) -> Tuple[np.ndarray, np.ndarray, float]:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        print(f"Warning: sample rate mismatch ({sr} vs {sample_rate})", file=sys.stderr)
    original_len = len(audio)
    duration = original_len / sample_rate
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    audio_signal = audio[np.newaxis, :].astype(np.float32)   # [1, N]
    audio_length = np.array([min(original_len, max_samples)], dtype=np.int32)
    return audio_signal, audio_length, duration


# ── Component benchmark ───────────────────────────────────────────────────────

class ComponentBenchmark:
    def __init__(self, model_dir: Path, meta: Dict):
        self.model_dir = model_dir
        self.sample_rate: int = meta["sample_rate"]
        self.max_samples: int = meta["max_audio_samples"]
        self.vocab_size: int = meta["vocab_size"]
        self.num_extra: int = meta["joint_extra_outputs"]
        self.max_symbol_steps: int = meta.get("max_symbol_steps", 1)

        self.mel_encoder: Optional[ct.models.MLModel] = None
        self.decoder: Optional[ct.models.MLModel] = None
        self.jd_single: Optional[ct.models.MLModel] = None

    def load(self) -> bool:
        def _try_load(name: str) -> Optional[ct.models.MLModel]:
            p = self.model_dir / name
            if not p.exists():
                return None
            try:
                return ct.models.MLModel(str(p))
            except Exception as e:
                print(f"  Warning: could not load {name}: {e}", file=sys.stderr)
                return None

        self.mel_encoder = _try_load("parakeet_mel_encoder.mlpackage")
        self.decoder = _try_load("parakeet_decoder.mlpackage")
        self.jd_single = _try_load("parakeet_joint_decision_single_step.mlpackage")

        if self.mel_encoder is None:
            print(f"  Error: parakeet_mel_encoder.mlpackage not found in {self.model_dir}", file=sys.stderr)
            return False
        return True

    # ── Mel+Encoder ───────────────────────────────────────────────────────────

    def _run_mel_encoder(self, audio_signal: np.ndarray, audio_length: np.ndarray) -> Optional[Dict]:
        assert self.mel_encoder is not None
        t0 = time.perf_counter()
        out = self.mel_encoder.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        })
        elapsed = time.perf_counter() - t0
        return {"time": elapsed, "encoder_shape": out["encoder"].shape, "encoder": out["encoder"]}

    def benchmark_mel_encoder(
        self,
        audio_signal: np.ndarray,
        audio_length: np.ndarray,
        num_runs: int,
    ) -> Dict:
        times, encoder_ref, encoder_shape = [], None, None
        for _ in range(num_runs):
            r = self._run_mel_encoder(audio_signal, audio_length)
            if r is None:
                return {}
            times.append(r["time"])
            encoder_ref = r["encoder"]
            encoder_shape = r["encoder_shape"]
        return {"times": times, "encoder_ref": encoder_ref, "encoder_shape": encoder_shape}

    # ── Decoder single step ───────────────────────────────────────────────────

    def _make_decoder_inputs(self) -> Dict[str, np.ndarray]:
        """Blank-initialised single decoder step inputs."""
        # decoder_layers x 1 x pred_hidden — sizes come from metadata encoder shape
        # We don't store those directly, so derive from a probe run or use safe defaults.
        # The decoder mlpackage input shapes are fixed at export time; we discover them
        # from the model spec to avoid hardcoding.
        assert self.decoder is not None
        spec = self.decoder.get_spec()
        inputs = {}
        for inp in spec.description.input:
            name = inp.name
            shape = tuple(d.size for d in inp.type.multiArrayType.shape.dims) if \
                inp.type.multiArrayType.shape.dims else (1,)
            if name == "targets":
                inputs[name] = np.full(shape, fill_value=self.vocab_size, dtype=np.int32)
            elif name == "target_length":
                inputs[name] = np.array([self.max_symbol_steps], dtype=np.int32)
            else:
                # h_in / c_in
                inputs[name] = np.zeros(shape, dtype=np.float32)
        return inputs

    def benchmark_decoder(self, num_runs: int) -> Dict:
        if self.decoder is None:
            return {}
        try:
            inputs = self._make_decoder_inputs()
        except Exception as e:
            print(f"  Warning: could not build decoder inputs: {e}", file=sys.stderr)
            return {}
        times = []
        decoder_ref = None
        for _ in range(num_runs):
            t0 = time.perf_counter()
            out = self.decoder.predict(inputs)
            times.append(time.perf_counter() - t0)
            decoder_ref = out.get("decoder")
        return {"times": times, "decoder_ref": decoder_ref}

    # ── Joint decision single step ────────────────────────────────────────────

    def benchmark_joint_decision_single_step(
        self,
        encoder_ref: Optional[np.ndarray],
        decoder_ref: Optional[np.ndarray],
        num_runs: int,
    ) -> Dict:
        if self.jd_single is None or encoder_ref is None or decoder_ref is None:
            return {}
        # Slice first frame from encoder and decoder
        enc_step = encoder_ref[:, :, :1].astype(np.float32)
        dec_step = decoder_ref[:, :, :1].astype(np.float32)
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.jd_single.predict({"encoder_step": enc_step, "decoder_step": dec_step})
            times.append(time.perf_counter() - t0)
        return {"times": times}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_component_stats(name: str, times: List[float], audio_duration: Optional[float] = None) -> None:
    stats = calculate_stats(times)
    rtf_str = ""
    if audio_duration:
        rtf = stats["mean"] / audio_duration
        rtf_str = f"  RTF (mean): {rtf:.4f}"
    print(f"\n  {name}:")
    print(f"    mean  {format_ms(stats['mean'])}   min {format_ms(stats['min'])}   "
          f"max {format_ms(stats['max'])}   std {format_ms(stats['std'])}{rtf_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def find_model_dirs(base_dir: Path) -> Dict[str, Path]:
    dirs = {}
    base = base_dir / "parakeet_coreml"
    if base.exists():
        dirs["base"] = base
    quant_root = base_dir / "parakeet_coreml_quantized"
    if quant_root.exists():
        for variant in sorted(quant_root.iterdir()):
            if variant.is_dir():
                dirs[f"quantized/{variant.name}"] = variant
    return dirs


def main() -> None:
    base_dir = Path("/Users/tigregotico/PycharmProjects/ovos-stt-plugin-coreml/parakeet_export").resolve()
    audio_path = base_dir / "yc_first_minute_16k_15s.wav"
    metadata_path = base_dir / "parakeet_coreml" / "metadata.json"
    num_runs = 10

    if not metadata_path.exists():
        print(f"Error: metadata not found at {metadata_path}  (run convert_to_coreml.py first)", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path) as f:
        meta = json.load(f)

    sample_rate: int = meta["sample_rate"]
    max_samples: int = meta["max_audio_samples"]

    if not audio_path.exists():
        print(f"Error: audio not found at {audio_path}", file=sys.stderr)
        sys.exit(1)

    audio_signal, audio_length, audio_duration = load_audio(audio_path, sample_rate, max_samples)

    print_section("Parakeet TDT v3 CoreML Benchmark")
    print(f"  Audio:        {audio_path.name}  ({audio_duration:.3f} s)")
    print(f"  Sample rate:  {sample_rate} Hz")
    print(f"  Runs:         {num_runs}")
    print(f"\nNote: Decoder & joint timings are per-step (not per-utterance).")
    print(f"      Full RNNT RTF = mel_encoder_time + N_steps × (decoder_step + joint_step).")

    model_dirs = find_model_dirs(base_dir)
    if not model_dirs:
        print("Error: no model directories found.", file=sys.stderr)
        sys.exit(1)

    all_results: Dict[str, Dict] = {}

    for label, model_dir in model_dirs.items():
        print_section(f"Model: {label}  ({model_dir.relative_to(base_dir)})")

        bench = ComponentBenchmark(model_dir, meta)
        if not bench.load():
            print("  Skipping (mel_encoder not found).")
            continue

        # ── mel_encoder ───────────────────────────────────────────────────────
        print(f"  Running mel+encoder ({num_runs} passes)...", end=" ", flush=True)
        me_result = bench.benchmark_mel_encoder(audio_signal, audio_length, num_runs)
        print("done")
        if not me_result:
            print("  mel_encoder benchmark failed.", file=sys.stderr)
            continue

        encoder_ref = me_result.get("encoder_ref")
        print_component_stats("mel_encoder", me_result["times"], audio_duration)
        print(f"      encoder output shape: {me_result['encoder_shape']}")

        # ── decoder step ─────────────────────────────────────────────────────
        print(f"  Running decoder step ({num_runs} passes)...", end=" ", flush=True)
        dec_result = bench.benchmark_decoder(num_runs)
        if dec_result:
            print("done")
            print_component_stats("decoder (single step)", dec_result["times"])
        else:
            print("skipped (model not found)")

        decoder_ref = dec_result.get("decoder_ref") if dec_result else None

        # ── joint decision single step ────────────────────────────────────────
        print(f"  Running joint-decision single step ({num_runs} passes)...", end=" ", flush=True)
        jd_result = bench.benchmark_joint_decision_single_step(encoder_ref, decoder_ref, num_runs)
        if jd_result:
            print("done")
            print_component_stats("joint_decision_single_step", jd_result["times"])
        else:
            print("skipped (model or refs not available)")

        all_results[label] = {
            "mel_encoder": me_result.get("times", []),
            "decoder_step": dec_result.get("times", []) if dec_result else [],
            "joint_step": jd_result.get("times", []) if jd_result else [],
            "audio_duration": audio_duration,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_section("RTF Summary (mel_encoder mean / audio duration)")
        print(f"\n  {'Model':<40} {'Mel+Enc (ms)':<18} {'RTF':<10} {'Dec step (ms)':<18} {'Joint step (ms)'}")
        print("  " + "-" * 100)
        for label, res in all_results.items():
            me_stats = calculate_stats(res["mel_encoder"]) if res["mel_encoder"] else {}
            dec_stats = calculate_stats(res["decoder_step"]) if res["decoder_step"] else {}
            jd_stats = calculate_stats(res["joint_step"]) if res["joint_step"] else {}
            me_mean = me_stats.get("mean", 0)
            rtf = me_mean / res["audio_duration"] if res["audio_duration"] else 0
            print(
                f"  {label:<40} "
                f"{me_mean * 1000:>10.2f} ms   "
                f"{rtf:>6.4f}     "
                f"{dec_stats.get('mean', 0) * 1000:>10.2f} ms   "
                f"{jd_stats.get('mean', 0) * 1000:>10.2f} ms"
            )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = base_dir / "benchmark_report.csv"
    with open(csv_path, "w") as f:
        f.write("model,mel_encoder_mean_ms,mel_encoder_min_ms,mel_encoder_max_ms,"
                "rtf_mean,decoder_step_mean_ms,joint_step_mean_ms\n")
        for label, res in all_results.items():
            me = calculate_stats(res["mel_encoder"]) if res["mel_encoder"] else {}
            dec = calculate_stats(res["decoder_step"]) if res["decoder_step"] else {}
            jd = calculate_stats(res["joint_step"]) if res["joint_step"] else {}
            dur = res["audio_duration"]
            f.write(
                f"{label},"
                f"{me.get('mean', 0) * 1000:.3f},"
                f"{me.get('min', 0) * 1000:.3f},"
                f"{me.get('max', 0) * 1000:.3f},"
                f"{me.get('mean', 0) / dur if dur else 0:.6f},"
                f"{dec.get('mean', 0) * 1000:.3f},"
                f"{jd.get('mean', 0) * 1000:.3f}\n"
            )
    print(f"\nCSV report saved to {csv_path}")
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
