#!/usr/bin/env python3
"""Quantize the Parakeet TDT v3 CoreML components."""
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

INPUT_DIR = Path("../parakeet_export/parakeet_coreml")
OUTPUT_DIR = Path("../parakeet_export/parakeet_coreml_quantized")

VARIANTS = {
    "int8_linear": lambda m: linear_quantize_weights(
        m,
        OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear", granularity="per_channel")
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

# All RNNT components exported by convert_to_coreml.py.
# The preprocessor is tiny and excluded; quantizing it yields negligible savings.
MODELS = {
    "mel_encoder": "parakeet_mel_encoder.mlpackage",
    "encoder": "parakeet_encoder.mlpackage",
    "decoder": "parakeet_decoder.mlpackage",
    "joint": "parakeet_joint.mlpackage",
    "joint_decision": "parakeet_joint_decision.mlpackage",
    "joint_decision_single_step": "parakeet_joint_decision_single_step.mlpackage",
}


def dir_size_mb(path: Path) -> float:
    """
    Compute the total size of all files under the given directory, in megabytes.
    
    Parameters:
    	path (Path): Root directory whose files will be counted recursively. Only regular files are included; if `path` is not a directory, matching files under it (if any) are used.
    
    Returns:
    	total_mb (float): Total size of included files in megabytes.
    """
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


def quantize_model(src: Path, fn, dst: Path) -> float:
    """
    Load a CoreML model from `src`, apply the given quantization function, save the quantized model to `dst`, and return its size in megabytes.
    
    Parameters:
        src (Path): Path to the source CoreML .mlpackage to be quantized.
        fn (Callable[[ct.models.MLModel], ct.models.MLModel]): Function that accepts a loaded CoreML model and returns a quantized CoreML model.
        dst (Path): Destination path where the quantized model will be saved.
    
    Returns:
        float: Size of the saved quantized model directory in megabytes.
    """
    print(f"  Loading {src.name}...")
    model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass

    print(f"  Quantizing...")
    q_model = fn(model)
    try:
        q_model.minimum_deployment_target = ct.target.iOS18
    except Exception:
        pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    q_model.save(str(dst))
    size = dir_size_mb(dst)
    print(f"  Saved → {dst}  ({size:.1f} MB)")
    return size


def main():
    """
    Quantize available CoreML model packages in INPUT_DIR using each configured VARIANTS and write per-variant outputs and a summary.
    
    For each quantization variant, creates an output subdirectory under OUTPUT_DIR, writes quantized .mlpackage files for each model found in INPUT_DIR, copies metadata files (metadata.json, vocab.json) if present, and writes a JSON summary at OUTPUT_DIR/quantization_summary.json. Prints baseline sizes, per-variant results, and compression ratios to the console.
    
    Raises:
        SystemExit: if INPUT_DIR does not exist or no expected .mlpackage files are found in INPUT_DIR.
    """
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input dir not found: {INPUT_DIR}  (run convert_to_coreml.py first)")

    # Only include models that actually exist in the input dir
    available = {k: v for k, v in MODELS.items() if (INPUT_DIR / v).exists()}
    if not available:
        raise SystemExit(f"No expected .mlpackage files found in {INPUT_DIR}")

    baseline_sizes = {k: dir_size_mb(INPUT_DIR / v) for k, v in available.items()}
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

        for comp_name, filename in available.items():
            src = INPUT_DIR / filename
            dst = variant_dir / filename
            q_size = quantize_model(src, fn, dst)
            base_size = baseline_sizes[comp_name]
            ratio = base_size / q_size if q_size > 0 else 0
            results[variant_name][comp_name] = {
                "size_mb": round(q_size, 2),
                "baseline_mb": round(base_size, 2),
                "compression_ratio": round(ratio, 2),
            }

        # Copy metadata alongside each variant for self-contained dirs
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
