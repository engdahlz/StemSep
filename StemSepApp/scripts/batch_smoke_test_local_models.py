"""
Batch smoke test for local (community) YAML + weights models.

This script is intentionally "heavier":
- For each model that has a local YAML + weights file, it runs a short separation
  on a short clipped excerpt (default: 5 seconds).
- It reports whether the *official* audio-separator path succeeded or whether
  we had to fall back to StemSep's internal RoformerSeparator implementation.
- It writes per-model outputs to disk for manual inspection.

Why this exists
---------------
We want "all models work" AND prefer official implementations for quality and confidence.
This batch runner helps you:
- Identify which models still require fallback
- Quickly reproduce failures and gather logs
- Iterate toward fixing upstream/official loading issues without guessing

Requirements
------------
- Run from a venv that has the project deps installed (torch, librosa, soundfile, etc.)
- Access to your models directory (YAML + weights)
- Input audio file (mp3/wav/...)

Usage (PowerShell examples)
---------------------------
python StemSepApp\\scripts\\batch_smoke_test_local_models.py `
  --models-dir "D:\\StemSep Models" `
  --input "C:\\Users\\engdahlz\\StemSep-V3\\StemSepApp\\dev_samples\\good_morning.mp3" `
  --out-root "C:\\Users\\engdahlz\\StemSep-V3\\StemSepApp\\dev_samples\\_batch_smoke_out" `
  --clip-seconds 5 `
  --max-models 0

CPU-only:
python StemSepApp\\scripts\\batch_smoke_test_local_models.py --device cpu ...

Notes
-----
- This script prefers using StemSep's SimpleSeparator (which prefers official audio-separator
  but may fall back). To *classify* official vs fallback accurately, we run the two paths
  explicitly and compare:
  1) Official path: audio-separator MDXCSeparator wiring via SimpleSeparator._load_model_direct
  2) Fallback path: StemSep audio.zfturbo_wrapper.RoformerSeparator

Exit codes
----------
0: All tested models succeeded via official path
1: Some models required fallback but still succeeded overall
2: One or more models failed entirely (both official and fallback failed)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def sanitize_filename(name: str) -> str:
    # Keep it stable and filesystem-safe
    name = name.strip()
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return "unnamed"
    return name[:160]


def find_weight_file(models_dir: Path, model_id: str) -> Optional[Path]:
    for ext in (".ckpt", ".pth", ".pt", ".safetensors", ".onnx"):
        p = models_dir / f"{model_id}{ext}"
        if p.exists():
            return p
    return None


def list_local_models(models_dir: Path) -> List[str]:
    """
    Returns model_ids for which:
      - <model_id>.yaml exists
      - and at least one supported weight file exists
    """
    model_ids: List[str] = []
    for yaml_path in sorted(models_dir.glob("*.yaml")):
        model_id = yaml_path.stem
        weight = find_weight_file(models_dir, model_id)
        if weight is None:
            continue
        model_ids.append(model_id)
    return model_ids


def ensure_repo_src_on_path() -> None:
    """
    Make `StemSepApp/src` importable when running from repo root.
    This script lives at StemSepApp/scripts/...
    """
    this_file = Path(__file__).resolve()
    stemsepapp_dir = this_file.parents[1]  # StemSepApp/
    src_dir = stemsepapp_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def clip_to_wav(input_path: Path, out_dir: Path, clip_seconds: float) -> Path:
    """
    Writes a clipped WAV excerpt. Uses librosa + soundfile.
    """
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / f"{input_path.stem}__clip_{int(clip_seconds)}s.wav"

    y, sr = librosa.load(str(input_path), sr=None, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    max_samples = int(sr * clip_seconds)
    y = y[:, :max_samples]

    sf.write(str(clip_path), y.T.astype("float32"), sr, subtype="FLOAT")
    return clip_path


# -----------------------------
# Result model
# -----------------------------


@dataclass
class ModelRunResult:
    model_id: str
    official_ok: bool
    fallback_ok: bool
    overall_ok: bool
    official_error: Optional[str] = None
    fallback_error: Optional[str] = None
    official_outputs: Optional[Dict[str, str]] = None
    fallback_outputs: Optional[Dict[str, str]] = None
    elapsed_seconds: float = 0.0


# -----------------------------
# Separation runners
# -----------------------------


def run_official_path(
    models_dir: Path,
    model_id: str,
    input_wav: Path,
    out_dir: Path,
    segment_size: int,
    overlap: int,
    batch_size: int,
) -> Dict[str, str]:
    """
    Runs the "official" audio-separator path for a local YAML+weights model.

    Implementation details:
    - We instantiate StemSep's SimpleSeparator and call its internal _load_model_direct
      to set separator.model_instance to audio-separator's MDXCSeparator with injected local config.
    - Then we run separator.separate() (audio-separator) to generate files.

    Any exception means official path failed.
    """
    ensure_repo_src_on_path()
    from audio_separator.separator import Separator  # type: ignore

    from audio.simple_separator import SimpleSeparator  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    ss = SimpleSeparator(models_dir=str(models_dir))

    separator = Separator(
        log_level=20,  # INFO
        model_file_dir=str(models_dir),
        output_dir=str(out_dir),
        output_format="wav",
        use_autocast=ss.use_autocast,
        mdxc_params={
            "segment_size": int(segment_size),
            "overlap": int(overlap),
            "batch_size": int(batch_size),
        },
    )

    # This is the "official" path we want to validate: MDXCSeparator + RoformerLoader.
    ss._load_model_direct(separator, model_id)  # type: ignore[attr-defined]

    # Now run separation using audio-separator
    output_files = separator.separate(str(input_wav))
    if not output_files:
        raise RuntimeError("audio-separator returned no outputs")

    # Build result map stem->path
    results: Dict[str, str] = {}
    for f in output_files:
        p = out_dir / f
        results[Path(f).stem] = str(p)
    return results


def run_fallback_path(
    models_dir: Path,
    model_id: str,
    input_wav: Path,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Runs StemSep's internal fallback path (RoformerSeparator).
    Any exception means fallback failed.
    """
    ensure_repo_src_on_path()
    from audio.zfturbo_wrapper import RoformerSeparator  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    # Device is chosen inside RoformerSeparator; respects CUDA availability.
    sep = RoformerSeparator(models_dir=models_dir)
    return sep.separate(
        audio_path=str(input_wav), model_id=model_id, output_dir=str(out_dir)
    )


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch smoke test local YAML+weights models with short-clip separation"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Local models directory containing YAML + weights",
    )
    parser.add_argument("--input", required=True, help="Input audio file (mp3/wav/...)")
    parser.add_argument(
        "--out-root", required=True, help="Output root directory for per-model outputs"
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Seconds to clip from input for testing",
    )
    parser.add_argument(
        "--max-models", type=int, default=0, help="Limit number of models (0 = all)"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=(None, "cpu", "cuda"),
        help="Force device for the overall run. cpu sets CUDA_VISIBLE_DEVICES=''.",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=352800,
        help="Segment size for official path (MDXC)",
    )
    parser.add_argument(
        "--overlap", type=int, default=8, help="Overlap for official path (MDXC)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for official path (MDXC)"
    )
    parser.add_argument(
        "--only-model",
        default=None,
        help="Run only a single model_id (useful for focused debugging)",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional path to write JSON report (default: <out-root>/report.json)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    input_path = Path(args.input)
    out_root = Path(args.out_root)

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[info] Forced CPU by setting CUDA_VISIBLE_DEVICES=''")
    elif args.device == "cuda":
        print("[info] Requested CUDA (will fall back if unavailable)")

    if not models_dir.exists():
        eprint(f"[fail] --models-dir does not exist: {models_dir}")
        return 2
    if not input_path.exists():
        eprint(f"[fail] --input does not exist: {input_path}")
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare clip once and reuse
    try:
        clip_wav = clip_to_wav(
            input_path, out_root / "_clip_cache", float(args.clip_seconds)
        )
        print(f"[ok] Test clip: {clip_wav}")
    except Exception as e:
        eprint(f"[fail] Failed to create clip wav: {e}")
        return 2

    model_ids = list_local_models(models_dir)
    if args.only_model:
        model_ids = [m for m in model_ids if m == args.only_model]
    if args.max_models and args.max_models > 0:
        model_ids = model_ids[: int(args.max_models)]

    if not model_ids:
        eprint("[fail] No local models found (need <id>.yaml + weight file).")
        return 2

    print(f"[info] Found {len(model_ids)} local models to test.")

    results: List[ModelRunResult] = []
    t_batch0 = time.perf_counter()

    for idx, model_id in enumerate(model_ids, start=1):
        model_label = f"{idx}/{len(model_ids)} {model_id}"
        print("\n" + "=" * 80)
        print(f"[run] {model_label}")

        per_model_dir = out_root / sanitize_filename(model_id)
        official_dir = per_model_dir / "official"
        fallback_dir = per_model_dir / "fallback"

        t0 = time.perf_counter()
        official_ok = False
        fallback_ok = False
        official_err = None
        fallback_err = None
        official_outputs = None
        fallback_outputs = None

        # Official first
        try:
            official_outputs = run_official_path(
                models_dir=models_dir,
                model_id=model_id,
                input_wav=clip_wav,
                out_dir=official_dir,
                segment_size=int(args.segment_size),
                overlap=int(args.overlap),
                batch_size=int(args.batch_size),
            )
            # Verify files exist
            missing = [p for p in official_outputs.values() if not Path(p).exists()]
            if missing:
                raise RuntimeError(
                    f"Official path reported outputs that do not exist: {missing[:3]}"
                )
            official_ok = True
            print(f"[ok] Official path succeeded ({len(official_outputs)} outputs).")
        except Exception as e:
            official_err = str(e)
            print(f"[warn] Official path failed: {official_err}")

        # Fallback if needed (or also run always? we run only if official fails to reduce load)
        if not official_ok:
            try:
                fallback_outputs = run_fallback_path(
                    models_dir=models_dir,
                    model_id=model_id,
                    input_wav=clip_wav,
                    out_dir=fallback_dir,
                )
                missing = [p for p in fallback_outputs.values() if not Path(p).exists()]
                if missing:
                    raise RuntimeError(
                        f"Fallback path reported outputs that do not exist: {missing[:3]}"
                    )
                fallback_ok = True
                print(
                    f"[ok] Fallback path succeeded ({len(fallback_outputs)} outputs)."
                )
            except Exception as e:
                fallback_err = str(e)
                print(f"[fail] Fallback path failed: {fallback_err}")

        dt = time.perf_counter() - t0
        overall_ok = official_ok or fallback_ok

        results.append(
            ModelRunResult(
                model_id=model_id,
                official_ok=official_ok,
                fallback_ok=fallback_ok,
                overall_ok=overall_ok,
                official_error=official_err,
                fallback_error=fallback_err,
                official_outputs=official_outputs,
                fallback_outputs=fallback_outputs,
                elapsed_seconds=dt,
            )
        )

    dt_batch = time.perf_counter() - t_batch0

    # Summaries
    n = len(results)
    n_official = sum(1 for r in results if r.official_ok)
    n_fallback_only = sum(1 for r in results if (not r.official_ok) and r.fallback_ok)
    n_failed = sum(1 for r in results if not r.overall_ok)

    print("\n" + "=" * 80)
    print("[summary]")
    print(f"Total models tested: {n}")
    print(f"Official OK:        {n_official}")
    print(f"Fallback OK only:   {n_fallback_only}")
    print(f"Failed:             {n_failed}")
    print(f"Elapsed (batch):    {dt_batch:.2f}s")

    # Write JSON report
    report_path = (
        Path(args.json_report) if args.json_report else (out_root / "report.json")
    )
    report_obj = {
        "models_dir": str(models_dir),
        "input": str(input_path),
        "clip_wav": str(clip_wav),
        "clip_seconds": float(args.clip_seconds),
        "segment_size": int(args.segment_size),
        "overlap": int(args.overlap),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "stats": {
            "total": n,
            "official_ok": n_official,
            "fallback_ok_only": n_fallback_only,
            "failed": n_failed,
            "elapsed_seconds": dt_batch,
        },
        "results": [asdict(r) for r in results],
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")
        print(f"[ok] Wrote report: {report_path}")
    except Exception as e:
        eprint(f"[warn] Failed to write JSON report: {e}")

    # Exit codes:
    # 0 if all were official OK
    # 1 if some needed fallback but none failed entirely
    # 2 if any failed
    if n_failed > 0:
        return 2
    if n_fallback_only > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
