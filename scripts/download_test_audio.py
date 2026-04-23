#!/usr/bin/env python3
"""Download reference audio clips for model spot-checking.

Downloads one clip per language from the google/fleurs dataset (CC-BY-4.0,
16 kHz, clean read speech) and saves them as mono 16 kHz WAV files in
--out-dir (default: test_audio/ next to this script).

Languages saved:
    en.wav  — English (en_us)
    ja.wav  — Japanese (ja_jp)
    vi.wav  — Vietnamese (vi_vn)
    da.wav  — Danish (da_dk)
    nl.wav  — Dutch (nl_nl)
    et.wav  — Estonian (et_ee)
    pl.wav  — Polish (pl_pl)
    pt.wav  — Portuguese / Brazilian (pt_br)
    sl.wav  — Slovenian (sl_si)

Usage:
    python download_test_audio.py
    python download_test_audio.py --out-dir /path/to/audio
    python download_test_audio.py --langs en pl pt   # subset only
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

DEFAULT_OUT = Path(__file__).parent / "test_audio"
SAMPLE_RATE = 16_000

# ISO 639-1 code → (HF dataset id, config/split, friendly name)
# FluidInference/fleurs: European languages (16 kHz WAV via HF API)
# OpenSpeechHub/Common-Voice-17-Ja: Japanese (streaming, Parquet)
# tsdocode/common_voice_13_0_vi_pseudo_labelled: Vietnamese (streaming, Parquet)
LANG_CONFIGS: dict[str, tuple[str, str, str]] = {
    "en": ("FluidInference/fleurs",                          "en_us", "English"),
    "ja": ("OpenSpeechHub/Common-Voice-17-Ja",               "",      "Japanese"),
    "vi": ("tsdocode/common_voice_13_0_vi_pseudo_labelled",  "vi",    "Vietnamese"),
    "da": ("FluidInference/fleurs",                          "da_dk", "Danish"),
    "nl": ("FluidInference/fleurs",                          "nl_nl", "Dutch"),
    "et": ("FluidInference/fleurs",                          "et_ee", "Estonian"),
    "pl": ("FluidInference/fleurs",                          "pl_pl", "Polish"),
    "pt": ("FluidInference/fleurs",                          "pt_br", "Portuguese (Brazilian)"),
    "sl": ("FluidInference/fleurs",                          "sl_si", "Slovenian"),
}


def _fetch_fleurs_wav(hf_config: str, min_seconds: float = 5.0) -> Optional[np.ndarray]:
    """Download one WAV directly from FluidInference/fleurs via the HF API (no full download)."""
    import io
    import urllib.request

    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        typer.echo("  ERROR: 'huggingface_hub' not installed. Run: pip install huggingface-hub", err=True)
        return None

    hf_dataset = "FluidInference/fleurs"
    try:
        items = list(api.list_repo_tree(hf_dataset, repo_type="dataset", path_in_repo=hf_config))
    except Exception as exc:
        typer.echo(f"  ERROR listing {hf_dataset}/{hf_config}: {exc}", err=True)
        return None

    wav_paths = sorted(
        item.path for item in items
        if hasattr(item, "path") and item.path.lower().endswith(".wav")
    )
    if not wav_paths:
        typer.echo(f"  ERROR: no WAV files in {hf_dataset}/{hf_config}", err=True)
        return None

    for wav_path in wav_paths[:10]:
        url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{wav_path}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr[:, 0]
            if len(arr) / sr < min_seconds:
                continue
            arr = _maybe_resample(arr, sr)
            return arr[: SAMPLE_RATE * 20].astype(np.float32)
        except Exception:
            continue

    typer.echo(f"  ERROR: no suitable clip in {hf_dataset}/{hf_config}", err=True)
    return None


def _fetch_parquet_audio(hf_dataset: str, subdir: str = "data", min_seconds: float = 5.0) -> Optional[np.ndarray]:
    """Download one Parquet shard, read raw audio bytes, decode with soundfile.

    Common Voice and similar datasets store audio as raw WAV/FLAC bytes in the
    Parquet 'audio' column — no torchcodec or FFmpeg required.
    """
    import io

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        typer.echo(f"  ERROR: missing dependency ({exc}). Run: pip install pyarrow huggingface-hub", err=True)
        return None

    api = HfApi()
    try:
        items = list(api.list_repo_tree(hf_dataset, repo_type="dataset", path_in_repo=subdir))
    except Exception:
        # Try root level
        try:
            items = list(api.list_repo_tree(hf_dataset, repo_type="dataset", path_in_repo=""))
        except Exception as exc:
            typer.echo(f"  ERROR listing {hf_dataset}: {exc}", err=True)
            return None

    parquet_paths = sorted(
        item.path for item in items
        if hasattr(item, "path") and item.path.endswith(".parquet")
    )
    if not parquet_paths:
        typer.echo(f"  ERROR: no Parquet files found in {hf_dataset}", err=True)
        return None

    # Download only the first shard
    local = hf_hub_download(hf_dataset, parquet_paths[0], repo_type="dataset")
    pf = pq.ParquetFile(local)

    for rg_idx in range(min(pf.num_row_groups, 5)):
        batch = pf.read_row_group(rg_idx).to_pydict()
        audio_col = batch.get("audio", [])
        for entry in audio_col:
            if not isinstance(entry, dict):
                continue
            raw: bytes = entry.get("bytes") or b""
            if not raw:
                continue
            try:
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                if arr.ndim > 1:
                    arr = arr[:, 0]
                if len(arr) / max(sr, 1) < min_seconds:
                    continue
                arr = _maybe_resample(arr, sr)
                return arr[: SAMPLE_RATE * 20].astype(np.float32)
            except Exception:
                continue

    typer.echo(f"  ERROR: no suitable clip found in first shard of {hf_dataset}", err=True)
    return None


def _maybe_resample(arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return arr
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        return resample_poly(arr, SAMPLE_RATE // g, sr // g).astype(np.float32)
    except ImportError:
        if sr != SAMPLE_RATE:
            raise RuntimeError(
                f"Audio sample rate {sr} Hz != {SAMPLE_RATE} Hz and scipy is not installed. "
                "Install it with: pip install scipy"
            )
        return arr


def _fetch_clip(hf_dataset: str, hf_config: str, min_seconds: float = 5.0) -> Optional[np.ndarray]:
    """Dispatch to the right fetch strategy based on the dataset."""
    if hf_dataset == "FluidInference/fleurs":
        return _fetch_fleurs_wav(hf_config, min_seconds)
    # Parquet-backed dataset: read raw audio bytes directly (no torchcodec needed)
    subdir = hf_config if hf_config else "data"
    return _fetch_parquet_audio(hf_dataset, subdir=subdir, min_seconds=min_seconds)


@app.command()
def download(
    out_dir: Path = typer.Option(DEFAULT_OUT, "--out-dir", help="Directory to save audio files."),
    langs: Optional[list[str]] = typer.Option(None, "--langs",
        help="Space/comma-separated ISO 639-1 codes to download (default: all)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-download even if file exists."),
) -> None:
    """Download per-language reference audio clips from google/fleurs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise --langs (accepts both "en pl" and "en,pl")
    requested: set[str] = set()
    if langs:
        for item in langs:
            for code in item.replace(",", " ").split():
                requested.add(code.strip().lower())
    if not requested:
        requested = set(LANG_CONFIGS.keys())

    unknown = requested - set(LANG_CONFIGS.keys())
    if unknown:
        typer.echo(f"Unknown language codes: {', '.join(sorted(unknown))}. "
                   f"Supported: {', '.join(sorted(LANG_CONFIGS.keys()))}", err=True)
        raise typer.Exit(1)

    ok = failed = skipped = 0

    for code in sorted(requested):
        hf_dataset, hf_config, name = LANG_CONFIGS[code]
        out_path = out_dir / f"{code}.wav"

        if out_path.exists() and not overwrite:
            typer.echo(f"[SKIP] {out_path.name} already exists ({name})")
            skipped += 1
            continue

        hf_dataset, hf_config, name = LANG_CONFIGS[code]
        typer.echo(f"[....] {code}.wav  ({name}) …", nl=False)
        sys.stdout.flush()

        clip = _fetch_clip(hf_dataset, hf_config)
        if clip is None:
            typer.echo(" FAILED")
            failed += 1
            continue

        sf.write(str(out_path), clip, SAMPLE_RATE, subtype="PCM_16")
        duration = len(clip) / SAMPLE_RATE
        typer.echo(f" OK  ({duration:.1f}s)")
        ok += 1

    typer.echo(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed")
    typer.echo(f"Audio saved to: {out_dir}")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
