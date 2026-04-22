#!/usr/bin/env python3
"""Compress fp32 CoreML mlpackages to int8, float16, 4bit, or 6bit.

Usage:
    python quantize_nvidia_parakeet.py --input-dir /path/fp32 --output-dir /path/int8
    python quantize_nvidia_parakeet.py --input-dir /path/fp32 --output-dir /path/fp16 --dtype float16
    python quantize_nvidia_parakeet.py --input-dir /path/fp32 --output-dir /path/4bit --dtype 4bit
    python quantize_nvidia_parakeet.py --input-dir /path/fp32 --output-dir /path/6bit --dtype 6bit
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

import coremltools as ct
import typer
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    linear_quantize_weights,
    palettize_weights,
)
try:
    from coremltools.optimize.coreml import OpFp16Config
    _HAS_FP16_CONFIG = True
except ImportError:
    _HAS_FP16_CONFIG = False

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
BYTES_IN_MB = 1024 * 1024

DTYPE_META = {
    "int8":    "int8_per_channel_symmetric",
    "float16": "float16_per_channel",
    "4bit":    "4bit_palettize_kmeans",
    "6bit":    "6bit_palettize_kmeans",
}


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / BYTES_IN_MB


def _compress(model: ct.models.MLModel, dtype: str) -> ct.models.MLModel:
    if dtype == "int8":
        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        return linear_quantize_weights(model, config=cfg)
    if dtype == "float16":
        if _HAS_FP16_CONFIG:
            cfg = OptimizationConfig(global_config=OpFp16Config())
        else:
            # Older coremltools: use linear quantizer with float16 numpy dtype
            import numpy as np
            cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=np.float16, granularity="per_channel"))
        return linear_quantize_weights(model, config=cfg)
    if dtype == "4bit":
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
        return palettize_weights(model, config=cfg)
    if dtype == "6bit":
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6))
        return palettize_weights(model, config=cfg)
    raise ValueError(f"Unknown dtype: {dtype}")


@app.command()
def quantize(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
    dtype: str = typer.Option("int8", "--dtype", help="int8 | float16 | 4bit | 6bit"),
) -> None:
    dtype = dtype.lower()
    if dtype not in DTYPE_META:
        raise typer.BadParameter(f"--dtype must be one of: {', '.join(DTYPE_META)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for item in sorted(input_dir.iterdir()):
        if item.suffix == ".mlpackage":
            typer.echo(f"Compressing {item.name} → {dtype}…")
            model = ct.models.MLModel(str(item), compute_units=ct.ComputeUnit.CPU_ONLY)
            orig_mb = _dir_size_mb(item)
            q = _compress(model, dtype)
            out = output_dir / item.name
            if out.exists():
                shutil.rmtree(out)
            q.save(str(out))
            q_mb = _dir_size_mb(out)
            ratio = orig_mb / q_mb if q_mb > 0 else 0
            summary[item.stem] = {
                "original_mb": round(orig_mb, 1),
                f"{dtype}_mb": round(q_mb, 1),
                "compression": round(ratio, 2),
            }
            typer.echo(f"  {orig_mb:.1f} MB → {q_mb:.1f} MB ({ratio:.2f}x)")
        elif item.is_file():
            dst = output_dir / item.name
            try:
                shutil.copy2(item, dst)
            except shutil.SameFileError:
                pass  # dst already exists (hardlinked); skip
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("coreml", {})["quantization"] = DTYPE_META[dtype]
        meta_path.write_text(json.dumps(meta, indent=2))
    (output_dir / "quantization_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nDone. Summary: {output_dir}/quantization_summary.json")


if __name__ == "__main__":
    app()
