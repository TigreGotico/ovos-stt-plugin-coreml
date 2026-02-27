#!/usr/bin/env python3
"""Quantize the Parakeet CTC CoreML models (mel_encoder + ctc_decoder)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    linear_quantize_weights,
    palettize_weights,
)

INPUT_DIR = Path("parakeet_ctc_coreml")
OUTPUT_DIR = Path("parakeet_ctc_coreml_quantized")

VARIANTS = {
    "int8_linear": lambda m: linear_quantize_weights(
        m,
        OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear", granularity="per_channel")
        ),
    ),
    "int4_linear": lambda m: linear_quantize_weights(
        m,
        OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_block", block_size=32, dtype="int4")
        ),
    ),
    "6bit_palettize": lambda m: palettize_weights(
        m,
        OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6)),
    ),
    "4bit_palettize": lambda m: palettize_weights(
        m,
        OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4)),
    ),
}

MODELS = {
    "mel_encoder": "parakeet_ctc_mel_encoder.mlpackage",
    "ctc_decoder": "parakeet_ctc_decoder.mlpackage",
}


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


def quantize_model(src: Path, fn, dst: Path, name: str) -> float:
    print(f"  Loading {src.name}...")
    model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass

    print(f"  Quantizing...")
    q_model = fn(model)
    try:
        q_model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    q_model.save(str(dst))
    size = dir_size_mb(dst)
    print(f"  Saved → {dst}  ({size:.1f} MB)")
    return size


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input dir not found: {INPUT_DIR}")

    baseline_sizes = {k: dir_size_mb(INPUT_DIR / v) for k, v in MODELS.items()}
    print("Baseline sizes:")
    for k, v in baseline_sizes.items():
        print(f"  {k}: {v:.1f} MB")
    print()

    results = {}
    for variant_name, fn in VARIANTS.items():
        print(f"=== Variant: {variant_name} ===")
        variant_dir = OUTPUT_DIR / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        results[variant_name] = {}

        for comp_name, filename in MODELS.items():
            src = INPUT_DIR / filename
            dst = variant_dir / filename
            if not src.exists():
                print(f"  Skipping {filename} (not found)")
                continue
            q_size = quantize_model(src, fn, dst, comp_name)
            base_size = baseline_sizes[comp_name]
            ratio = base_size / q_size if q_size > 0 else 0
            results[variant_name][comp_name] = {
                "size_mb": round(q_size, 2),
                "baseline_mb": round(base_size, 2),
                "compression_ratio": round(ratio, 2),
            }

        # Copy metadata and vocab
        for f in ["metadata.json", "vocab.json"]:
            src_f = INPUT_DIR / f
            if src_f.exists():
                shutil.copy2(src_f, variant_dir / f)
        print()

    summary_path = OUTPUT_DIR / "quantization_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Summary written to {summary_path}")
    print()
    print("Compression ratios:")
    for variant, comps in results.items():
        for comp, m in comps.items():
            print(f"  {variant}/{comp}: {m['baseline_mb']:.1f} MB → {m['size_mb']:.1f} MB  (×{m['compression_ratio']:.2f})")


if __name__ == "__main__":
    main()
