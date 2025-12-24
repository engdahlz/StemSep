"""
Smoke test for local (community) YAML + weights models.

Goals
-----
- Terminating script (no servers/watchers)
- Verifies that local YAML config + weight file can be loaded
- Runs a *short* separation on a *short* excerpt to validate end-to-end wiring
- Uses the same codepaths as the app where possible:
  1) Prefer audio-separator via StemSepApp's SimpleSeparator (official path)
  2) Allows fallback (SimpleSeparator has internal fallback to RoformerSeparator)

Usage examples
--------------
# Basic:
python StemSepApp/scripts/smoke_test_local_model.py ^
  --models-dir "D:\\StemSep Models" ^
  --model-id "gabox-denoise-debleed" ^
  --input "StemSepApp/dev_samples/good_morning.mp3" ^
  --out "StemSepApp/dev_samples/_smoke_out"

# CPU only:
python StemSepApp/scripts/smoke_test_local_model.py --device cpu ...

# Clip only first 10 seconds:
python StemSepApp/scripts/smoke_test_local_model.py --clip-seconds 10 ...

Exit codes
----------
0: success
2: validation/config error (missing files etc.)
3: separation failed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"YAML did not load as a dict: {path}")
    return data


def _find_weight_file(models_dir: Path, model_id: str) -> Optional[Path]:
    for ext in (".ckpt", ".pth", ".pt", ".safetensors", ".onnx"):
        p = models_dir / f"{model_id}{ext}"
        if p.exists():
            return p
    return None


def _validate_inputs(
    models_dir: Path, model_id: str, input_path: Path
) -> Tuple[Path, Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"--models-dir does not exist: {models_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")

    yaml_path = models_dir / f"{model_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing YAML: {yaml_path}")

    weight_path = _find_weight_file(models_dir, model_id)
    if weight_path is None:
        raise FileNotFoundError(
            f"Missing weight file for model_id='{model_id}' in {models_dir} "
            f"(tried .ckpt/.pth/.pt/.safetensors/.onnx)"
        )

    # Basic YAML sanity checks
    cfg = _read_yaml(yaml_path)
    if "model" not in cfg and not any(
        k in cfg for k in ("num_bands", "freqs_per_bands")
    ):
        _eprint(
            "WARNING: YAML has no top-level 'model' section and does not look like a flat Roformer config."
        )
    if "training" not in cfg:
        _eprint(
            "WARNING: YAML has no 'training' section; audio-separator uses it for stem naming/residual logic."
        )

    return yaml_path, weight_path


def _maybe_clip_to_wav(
    input_path: Path, out_dir: Path, clip_seconds: Optional[float]
) -> Path:
    """
    Creates a short WAV excerpt for deterministic testing.

    - Avoids full-file processing
    - Avoids stressing GPU/CPU unnecessarily
    """
    if not clip_seconds or clip_seconds <= 0:
        return input_path

    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "clip-seconds requested but required packages are missing (soundfile/numpy/librosa). "
            f"Underlying error: {e}"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / f"{input_path.stem}__clip_{int(clip_seconds)}s.wav"

    # Load with librosa for broad codec support. This may invoke ffmpeg for mp3.
    y, sr = librosa.load(str(input_path), sr=None, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    max_samples = int(sr * clip_seconds)
    y = y[:, :max_samples]

    # Write WAV float32
    sf.write(str(clip_path), y.T.astype("float32"), sr, subtype="FLOAT")
    return clip_path


def _progress(pct: float, msg: str) -> None:
    pct_s = f"{pct:6.1f}%"
    print(f"[progress] {pct_s} {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test local YAML+weights model separation"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Path to local models directory (YAML + weights)",
    )
    parser.add_argument(
        "--model-id", required=True, help="Model id (basename without extension)"
    )
    parser.add_argument("--input", required=True, help="Input audio path (mp3/wav/...)")
    parser.add_argument("--out", required=True, help="Output directory for stems")
    parser.add_argument(
        "--device",
        default=None,
        choices=(None, "cpu", "cuda"),
        help="Force device (cpu/cuda)",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=10.0,
        help="Clip input to first N seconds (0 disables)",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=352800,
        help="Segment size (passed to separator)",
    )
    parser.add_argument(
        "--overlap", type=int, default=8, help="Overlap (passed to separator)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (passed to separator)"
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model_id = args.model_id
    input_path = Path(args.input)
    out_dir = Path(args.out)

    try:
        yaml_path, weight_path = _validate_inputs(models_dir, model_id, input_path)
        print(f"[ok] YAML:    {yaml_path}")
        print(f"[ok] WEIGHT:  {weight_path}")
    except Exception as e:
        _eprint(f"[fail] Input validation failed: {e}")
        return 2

    # Optional clipping to keep tests fast/terminating
    try:
        test_input = _maybe_clip_to_wav(input_path, out_dir, args.clip_seconds)
        print(f"[ok] TEST INPUT: {test_input}")
    except Exception as e:
        _eprint(f"[fail] Failed to prepare test input: {e}")
        return 2

    # Ensure repo-relative imports work when running from project root
    # Expected execution: python StemSepApp/scripts/smoke_test_local_model.py ...
    # Add StemSepApp/src to sys.path if needed.
    this_file = Path(__file__).resolve()
    stemsepapp_dir = this_file.parents[1]  # StemSepApp/
    src_dir = stemsepapp_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Import and run separation via SimpleSeparator (official audio-separator path + fallback)
    try:
        from audio.simple_separator import SimpleSeparator  # type: ignore
    except Exception as e:
        _eprint(f"[fail] Could not import StemSep SimpleSeparator: {e}")
        return 2

    # If device is forced, propagate via env to keep this script simple.
    # SimpleSeparator chooses cuda if available; for CPU forcing, we disable CUDA visibility.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[info] Forced CPU by setting CUDA_VISIBLE_DEVICES=''")
    elif args.device == "cuda":
        # Leave as-is; if CUDA isn't available torch will fall back anyway.
        print("[info] Requested CUDA (will fall back if unavailable)")

    separator = SimpleSeparator(models_dir=str(models_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        result = separator.separate(
            audio_path=str(test_input),
            model_id=model_id,
            output_dir=str(out_dir),
            output_format="wav",
            segment_size=int(args.segment_size),
            overlap=int(args.overlap),
            batch_size=int(args.batch_size),
            progress_callback=_progress,
        )
    except Exception as e:
        _eprint(f"[fail] Separation failed: {e}")
        return 3
    finally:
        dt = time.perf_counter() - t0
        print(f"[info] Elapsed: {dt:.2f}s")

    if not isinstance(result, dict) or not result:
        _eprint(f"[fail] Separation returned no outputs: {result!r}")
        return 3

    print("[ok] Outputs:")
    missing = 0
    for stem, path in result.items():
        p = Path(path)
        print(f" - {stem}: {p}")
        if not p.exists():
            missing += 1

    if missing:
        _eprint(f"[fail] {missing} output files were reported but not found on disk.")
        return 3

    print("[success] Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
