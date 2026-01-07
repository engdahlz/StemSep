import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelAuditRow:
    model_id: str
    architecture: str
    installed: bool
    expected_local_filename: Optional[str]
    expected_files: List[str]
    missing_files: List[str]
    notes: List[str]


@dataclass
class SmokeResult:
    model_id: str
    ok: bool
    elapsed_s: float
    error: Optional[str]
    output_files: Optional[Dict[str, str]]


def _read_models_registry(registry_path: Path) -> List[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict) and m.get("id")]


def _run_py(venv_python: Path, code: str) -> Tuple[int, str, str]:
    p = subprocess.run(
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout, p.stderr


def _audit_models(models_dir: Path, registry_path: Path, stemsepapp_venv_python: Path) -> List[ModelAuditRow]:
    # Use StemSepApp ModelManager because it contains registry semantics + repair helpers.
    code = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'''%STEMSEP_SRC%''')

from stemsep.models.model_manager import ModelManager

models_dir = Path(r'''%MODELS_DIR%''')
registry_path = Path(r'''%REGISTRY_PATH%''')

mm = ModelManager(models_dir=models_dir)

# Read registry ids in order
models = json.loads(registry_path.read_text(encoding='utf-8')).get('models', [])
ids = [m.get('id') for m in models if isinstance(m, dict) and m.get('id')]

out = []
for mid in ids:
    try:
        report = mm.ensure_model_ready(mid, auto_repair_filenames=True, copy_instead_of_rename=True)
    except Exception as e:
        report = {'ok': False, 'model_id': mid, 'installed': False, 'error': str(e), 'expected_files': []}

    expected_files = report.get('expected_files') if isinstance(report, dict) else []
    if not isinstance(expected_files, list):
        expected_files = []

    missing = []
    for p in expected_files:
        try:
            if not Path(p).exists():
                missing.append(p)
        except Exception:
            missing.append(p)

    out.append({
        'model_id': mid,
        'ok': bool(report.get('ok', False)) if isinstance(report, dict) else False,
        'installed': bool(report.get('installed', False)) if isinstance(report, dict) else False,
        'architecture': str(report.get('architecture') or ''),
        'expected_local_filename': report.get('expected_filename'),
        'expected_files': expected_files,
        'missing_files': missing,
        'error': report.get('error'),
    })

print(json.dumps(out, ensure_ascii=False))
"""

    code = (
        code.replace("%MODELS_DIR%", str(models_dir))
        .replace("%REGISTRY_PATH%", str(registry_path))
        .replace("%STEMSEP_SRC%", str((Path(__file__).resolve().parents[2] / 'StemSepApp' / 'src').resolve()))
    )

    rc, out, err = _run_py(stemsepapp_venv_python, code)
    if rc != 0:
        raise RuntimeError(f"audit failed (rc={rc}) stderr={err.strip()}")

    rows_raw = json.loads(out)
    rows: List[ModelAuditRow] = []
    for r in rows_raw:
        notes = []
        if r.get("error"):
            notes.append(str(r.get("error")))
        rows.append(
            ModelAuditRow(
                model_id=str(r.get("model_id")),
                architecture=str(r.get("architecture") or ""),
                installed=bool(r.get("installed")),
                expected_local_filename=r.get("expected_local_filename"),
                expected_files=[str(x) for x in (r.get("expected_files") or [])],
                missing_files=[str(x) for x in (r.get("missing_files") or [])],
                notes=notes,
            )
        )

    return rows


def _smoke_one(
    repo_root: Path,
    backend_exe: Path,
    assets_dir: Path,
    models_dir: Path,
    stemsepapp_venv_python: Path,
    input_audio: Path,
    output_base: Path,
    model_id: str,
    device: str,
    timeout_min: int,
) -> SmokeResult:
    env = os.environ.copy()
    # Ensure Rust backend uses correct Python for inference
    env["STEMSEP_PYTHON"] = str(stemsepapp_venv_python)
    env["STEMSEP_PROXY_PYTHON"] = "0"
    env["STEMSEP_INFERENCE_SCRIPT"] = str(repo_root / "scripts" / "inference.py")

    cmd = [
        str(repo_root / ".venv" / "Scripts" / "python.exe"),
        str(repo_root / "scripts" / "run_backend_separation.py"),
        "--backend",
        str(backend_exe),
        "--assets-dir",
        str(assets_dir),
        "--models-dir",
        str(models_dir),
        "--input",
        str(input_audio),
        "--output-dir",
        str(output_base / model_id),
        "--device",
        device,
        "--model-id",
        model_id,
        "--timeout-min",
        str(timeout_min),
    ]

    t0 = time.time()
    # Hard timeout so a single model can't stall the entire batch.
    timeout_s = max(60, int(timeout_min) * 60 + 120)
    try:
        p = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        tail_out = (getattr(e, "stdout", "") or "")[-2000:]
        tail_err = (getattr(e, "stderr", "") or "")[-2000:]
        msg = f"timeout after ~{timeout_s}s"
        if tail_err or tail_out:
            msg += "\n" + (tail_err + "\n" + tail_out).strip()
        return SmokeResult(model_id=model_id, ok=False, elapsed_s=elapsed, error=msg, output_files=None)

    if p.returncode == 0:
        # Try to parse the last complete event JSON from stdout
        output_files = None
        try:
            # run_backend_separation prints 'complete: {json}' then 'Done.'
            # We scan for the last line starting with 'complete:'
            for line in reversed(p.stdout.splitlines()):
                if line.strip().startswith("complete:"):
                    raw = line.split("complete:", 1)[1].strip()
                    output_files = json.loads(raw)
                    break
        except Exception:
            output_files = None

        return SmokeResult(model_id=model_id, ok=True, elapsed_s=elapsed, error=None, output_files=output_files)

    err = (p.stderr or "")[-4000:]
    out = (p.stdout or "")[-4000:]
    combined = (err + "\n" + out).strip() or f"exit={p.returncode}"
    return SmokeResult(model_id=model_id, ok=False, elapsed_s=elapsed, error=combined, output_files=None)


def main() -> int:
    ap = argparse.ArgumentParser(description="StemSep model healthcheck")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--registry", default=str(Path("StemSepApp") / "assets" / "models.json.bak"))
    ap.add_argument("--input", default=str(Path("Audio") / "target_15s.wav"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default=str(Path("stemsep-backend") / "target" / "debug" / "stemsep-backend.exe"))
    ap.add_argument("--assets-dir", default=str(Path("StemSepApp") / "assets"))
    ap.add_argument("--timeout-min", type=int, default=20)

    ap.add_argument("--audit-only", action="store_true", help="Only audit installed files; do not run smoke tests")
    ap.add_argument("--smoke", action="store_true", help="Run 15s smoke-test per installed model")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of smoke tests (0 = no limit)")
    ap.add_argument("--out", default=str(Path("_backend_out") / "model_healthcheck_report.json"))

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    models_dir = Path(args.models_dir)
    registry_path = (repo_root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry)
    input_audio = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    backend_exe = (repo_root / args.backend).resolve() if not Path(args.backend).is_absolute() else Path(args.backend)
    assets_dir = (repo_root / args.assets_dir).resolve() if not Path(args.assets_dir).is_absolute() else Path(args.assets_dir)

    stemsepapp_py = repo_root / "StemSepApp" / ".venv" / "Scripts" / "python.exe"

    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")
    if not registry_path.exists():
        raise SystemExit(f"registry not found: {registry_path}")
    if not input_audio.exists():
        raise SystemExit(f"input not found: {input_audio}")
    if not backend_exe.exists():
        raise SystemExit(f"backend not found: {backend_exe}")
    if not stemsepapp_py.exists():
        raise SystemExit(f"StemSepApp venv python not found: {stemsepapp_py}")

    audit = _audit_models(models_dir=models_dir, registry_path=registry_path, stemsepapp_venv_python=stemsepapp_py)

    report: Dict[str, Any] = {
        "meta": {
            "models_dir": str(models_dir),
            "registry": str(registry_path),
            "input": str(input_audio),
            "device": args.device,
            "backend": str(backend_exe),
            "assets_dir": str(assets_dir),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "audit": [asdict(r) for r in audit],
        "smoke": [],
    }

    if args.audit_only and not args.smoke:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote report: {out_path}")
        return 0

    if args.smoke:
        output_base = repo_root / "_backend_out" / "model_healthcheck_smoke"
        output_base.mkdir(parents=True, exist_ok=True)

        installed = [r for r in audit if r.installed]
        if args.limit and args.limit > 0:
            installed = installed[: args.limit]

        results: List[SmokeResult] = []
        for i, row in enumerate(installed, 1):
            print(f"[{i}/{len(installed)}] smoke: {row.model_id}")
            res = _smoke_one(
                repo_root=repo_root,
                backend_exe=backend_exe,
                assets_dir=assets_dir,
                models_dir=models_dir,
                stemsepapp_venv_python=stemsepapp_py,
                input_audio=input_audio,
                output_base=output_base,
                model_id=row.model_id,
                device=args.device,
                timeout_min=args.timeout_min,
            )
            results.append(res)
            status = "OK" if res.ok else "FAIL"
            print(f"  -> {status} ({res.elapsed_s:.1f}s)")

        report["smoke"] = [asdict(r) for r in results]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
