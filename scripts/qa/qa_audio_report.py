import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _read_text_tail(path: str, max_bytes: int = 50_000_000) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read()
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _iter_json_objects_from_log(text: str):
    # backend-stdio.log contains lines like:
    # 2025-... [rust stdout] {json}
    # and sometimes raw json lines.
    for line in text.splitlines():
        if "{" not in line or "}" not in line:
            continue
        # best-effort: pull the last {...} block on the line
        m = re.search(r"(\{.*\})\s*$", line)
        if not m:
            continue
        payload = m.group(1)
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def find_output_files_in_backend_log(
    backend_log_path: str,
    recipe_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    text = _read_text_tail(backend_log_path)

    best: Optional[Dict[str, Any]] = None
    for obj in _iter_json_objects_from_log(text):
        if obj.get("type") != "separation_complete":
            continue
        if job_id and obj.get("job_id") != job_id:
            continue
        meta = obj.get("meta") or {}
        if recipe_id and meta.get("recipe_id") != recipe_id:
            continue
        if "output_files" not in obj:
            continue
        best = obj

    if not best:
        raise SystemExit(
            f"Could not find separation_complete for job_id={job_id!r} recipe_id={recipe_id!r} in {backend_log_path}"
        )

    out = best.get("output_files") or {}
    if not isinstance(out, dict) or not out:
        raise SystemExit(f"Found separation_complete but output_files missing/empty: keys={list(best.keys())}")

    found_job_id = str(best.get("job_id"))
    out_files: Dict[str, str] = {}
    for k, v in out.items():
        if isinstance(v, str):
            out_files[str(k)] = v

    return found_job_id, out_files


@dataclass
class Audio:
    sr: int
    data: "Any"  # numpy.ndarray


def _load_audio(path: str) -> Audio:
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return Audio(sr=int(sr), data=data)
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore
        from scipy.io import wavfile  # type: ignore

        sr, data = wavfile.read(path)
        # Normalize to float32 [-1,1]
        if data.dtype == np.int16:
            data_f = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data_f = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data_f = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data_f = data.astype(np.float32)
        if data_f.ndim == 1:
            data_f = data_f[:, None]
        return Audio(sr=int(sr), data=data_f)
    except Exception as e:
        raise SystemExit(
            f"Failed to read WAV via soundfile and scipy.io.wavfile. Install soundfile or scipy. Error: {e}"
        )


def _resample_if_needed(audio: Audio, target_sr: int) -> Audio:
    if audio.sr == target_sr:
        return audio

    try:
        import numpy as np  # type: ignore
        from scipy.signal import resample_poly  # type: ignore

        # Rational approximation for polyphase resampling
        g = math.gcd(audio.sr, target_sr)
        up = target_sr // g
        down = audio.sr // g
        y = resample_poly(audio.data, up=up, down=down, axis=0)
        y = np.asarray(y, dtype=np.float32)
        return Audio(sr=target_sr, data=y)
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore

        y = []
        for ch in range(audio.data.shape[1]):
            y_ch = librosa.resample(audio.data[:, ch], orig_sr=audio.sr, target_sr=target_sr)
            y.append(y_ch)
        y = np.stack(y, axis=1).astype(np.float32)
        return Audio(sr=target_sr, data=y)
    except Exception as e:
        raise SystemExit(f"Need resampling but scipy.signal.resample_poly and librosa are unavailable. Error: {e}")


def _trim_to_min(*arrays):
    import numpy as np  # type: ignore

    n = min(a.shape[0] for a in arrays)
    return [np.asarray(a[:n], dtype=np.float32) for a in arrays]


def _rms(x) -> float:
    import numpy as np  # type: ignore

    return float(np.sqrt(np.mean(np.square(x), axis=0)).mean())


def _peak(x) -> float:
    import numpy as np  # type: ignore

    return float(np.max(np.abs(x)))


def _db(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(eps, x))


def _corr(a, b) -> float:
    import numpy as np  # type: ignore

    # average channel correlation
    corrs = []
    for ch in range(a.shape[1]):
        x = a[:, ch]
        y = b[:, ch]
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
        if denom <= 0:
            continue
        corrs.append(float(np.dot(x0, y0) / denom))
    return float(sum(corrs) / max(1, len(corrs)))


def _snr_db(ref, err) -> float:
    import numpy as np  # type: ignore

    num = float(np.sum(ref * ref))
    den = float(np.sum(err * err))
    if den <= 0:
        return float("inf")
    return 10.0 * math.log10(max(1e-20, num) / max(1e-20, den))


def _si_sdr_db(ref, est) -> float:
    import numpy as np  # type: ignore

    # SI-SDR per channel, averaged
    vals = []
    for ch in range(ref.shape[1]):
        s = ref[:, ch]
        x = est[:, ch]
        s_energy = float(np.dot(s, s))
        if s_energy <= 0:
            continue
        alpha = float(np.dot(x, s) / s_energy)
        s_target = alpha * s
        e_noise = x - s_target
        num = float(np.dot(s_target, s_target))
        den = float(np.dot(e_noise, e_noise))
        if den <= 0:
            vals.append(float("inf"))
        else:
            vals.append(10.0 * math.log10(max(1e-20, num) / max(1e-20, den)))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def qa_report_data(original: str, instrumental: str, vocals: str) -> Dict[str, Any]:
    import numpy as np  # type: ignore

    orig = _load_audio(original)
    inst = _load_audio(instrumental)
    voc = _load_audio(vocals)

    # Choose a common SR (prefer stems SR)
    target_sr = inst.sr
    orig_rs = _resample_if_needed(orig, target_sr)
    voc_rs = _resample_if_needed(voc, target_sr)

    # Channel alignment: if original has different channel count, attempt best-effort
    def align_channels(a: np.ndarray, c: int) -> np.ndarray:
        if a.shape[1] == c:
            return a
        if a.shape[1] == 1 and c == 2:
            return np.repeat(a, 2, axis=1)
        if a.shape[1] == 2 and c == 1:
            return a.mean(axis=1, keepdims=True)
        # fallback: truncate or pad repeats
        if a.shape[1] > c:
            return a[:, :c]
        reps = c // a.shape[1]
        rem = c % a.shape[1]
        out = np.concatenate([np.tile(a, (1, reps)), a[:, :rem]], axis=1)
        return out

    channels = max(orig_rs.data.shape[1], inst.data.shape[1], voc_rs.data.shape[1])
    o = align_channels(orig_rs.data, channels)
    i = align_channels(inst.data, channels)
    v = align_channels(voc_rs.data, channels)

    o, i, v = _trim_to_min(o, i, v)
    recon = i + v
    err = recon - o

    dur_s = o.shape[0] / float(target_sr)

    # Metrics
    o_rms = _rms(o)
    r_rms = _rms(recon)
    e_rms = _rms(err)

    stems: Dict[str, Dict[str, Any]] = {}

    def stem_block(name: str, x: np.ndarray):
        peak = _peak(x)
        rms = _rms(x)
        clip_1 = int(np.sum(np.abs(x) > 1.0))
        clip_0999 = int(np.sum(np.abs(x) > 0.999))
        stems[name] = {
            "peak_abs": peak,
            "rms": rms,
            "rms_dbfs": _db(rms),
            "crest_factor": peak / max(1e-12, rms),
            "samples_abs_gt_1_0": clip_1,
            "samples_abs_gt_0_999": clip_0999,
        }

    stem_block("Original", o)
    stem_block("Instrumental", i)
    stem_block("Vocals", v)
    stem_block("Recombined (inst+voc)", recon)
    stem_block("Recombine error (recon-original)", err)

    consistency = {
        "recombined_vs_original_correlation": _corr(recon, o),
        "error_rms": e_rms,
        "error_rms_dbfs": _db(e_rms),
        "snr_db": _snr_db(o, err),
        "si_sdr_db": _si_sdr_db(o, recon),
        "gain_delta_db": _db(r_rms) - _db(o_rms),
        "instrumental_vs_vocals_correlation": _corr(i, v),
    }

    return {
        "schema_version": "1.0",
        "mode": "recombine_consistency",
        "inputs": {
            "original": original,
            "instrumental": instrumental,
            "vocals": vocals,
        },
        "analysis": {
            "sample_rate": target_sr,
            "channels": channels,
            "duration_seconds": dur_s,
        },
        "stems": stems,
        "consistency": consistency,
        "notes": [
            "Recombined vs original should be close to 1.0 correlation for a clean split; small deviations are expected after phase-fix/denoise.",
            "Non-zero recombine error is normal; large gain drift or heavy clipping is usually a quality concern.",
        ],
    }


def qa_report(original: str, instrumental: str, vocals: str) -> str:
    data = qa_report_data(original, instrumental, vocals)
    report_lines = []
    report_lines.append("# StemSep QA report")
    report_lines.append("")
    report_lines.append("## Inputs")
    report_lines.append(f"- Original: {data['inputs']['original']}")
    report_lines.append(f"- Instrumental: {data['inputs']['instrumental']}")
    report_lines.append(f"- Vocals: {data['inputs']['vocals']}")
    report_lines.append("")
    report_lines.append("## Basic")
    report_lines.append(f"- Sample rate used for analysis: {data['analysis']['sample_rate']} Hz")
    report_lines.append(f"- Channels (analysis-aligned): {data['analysis']['channels']}")
    report_lines.append(f"- Duration compared: {data['analysis']['duration_seconds']:.2f} s")
    report_lines.append("")

    report_lines.append("## Stem stats")
    for name, stats in data["stems"].items():
        report_lines.append(f"### {name}")
        report_lines.append(f"- Peak abs: {stats['peak_abs']:.6f}")
        report_lines.append(f"- RMS: {stats['rms']:.6f} ({stats['rms_dbfs']:.2f} dBFS)")
        report_lines.append(f"- Crest factor: {stats['crest_factor']:.2f}")
        report_lines.append(f"- Samples |x|>1.0: {stats['samples_abs_gt_1_0']}")
        report_lines.append(f"- Samples |x|>0.999: {stats['samples_abs_gt_0_999']}")
        report_lines.append("")

    consistency = data["consistency"]
    report_lines.append("## Consistency checks")
    report_lines.append(
        f"- Recombined vs original correlation: {consistency['recombined_vs_original_correlation']:.6f}"
    )
    report_lines.append(
        f"- Error RMS: {consistency['error_rms']:.8f} ({consistency['error_rms_dbfs']:.2f} dBFS)"
    )
    report_lines.append(f"- SNR (original vs error): {consistency['snr_db']:.2f} dB")
    report_lines.append(
        f"- SI-SDR (recombined vs original): {consistency['si_sdr_db']:.2f} dB"
    )
    report_lines.append(
        f"- Gain delta (RMS recon/orig): {consistency['gain_delta_db']:.2f} dB"
    )
    report_lines.append(
        f"- Instrumental vs vocals correlation (heuristic): {consistency['instrumental_vs_vocals_correlation']:.6f}"
    )
    report_lines.append("")
    report_lines.append("## Notes")
    for note in data["notes"]:
        report_lines.append(f"- {note}")

    return "\n".join(report_lines)


def main():
    p = argparse.ArgumentParser(description="Objective QA report for StemSep outputs")
    p.add_argument("--original", required=True, help="Path to original mix WAV")
    p.add_argument("--instrumental", help="Path to instrumental stem WAV")
    p.add_argument("--vocals", help="Path to vocals stem WAV")
    p.add_argument(
        "--backend-log",
        default=os.path.join(os.environ.get("APPDATA", ""), "electron-poc", "backend-stdio.log"),
        help="Path to Electron backend-stdio.log (default: %%APPDATA%%/electron-poc/backend-stdio.log)",
    )
    p.add_argument("--job-id", help="job_id to resolve stems from backend log")
    p.add_argument("--recipe-id", default="golden_ultimate_inst", help="recipe_id to resolve stems")
    p.add_argument("--out", help="Write report to this markdown file")
    p.add_argument("--json-out", help="Write report data to this JSON file")
    args = p.parse_args()

    instrumental = args.instrumental
    vocals = args.vocals

    if not instrumental or not vocals:
        job_id, outs = find_output_files_in_backend_log(args.backend_log, recipe_id=args.recipe_id, job_id=args.job_id)
        instrumental = outs.get("instrumental")
        vocals = outs.get("vocals")
        if not instrumental or not vocals:
            raise SystemExit(f"Resolved job {job_id} but missing instrumental/vocals in output_files: {outs}")
        print(f"Resolved stems from backend log: job_id={job_id}")
        print(f"- instrumental: {instrumental}")
        print(f"- vocals: {vocals}")

    for path in (args.original, instrumental, vocals):
        if not os.path.exists(path):
            raise SystemExit(f"File does not exist: {path}")

    report_data = qa_report_data(args.original, instrumental, vocals)
    report = qa_report(args.original, instrumental, vocals)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote report: {args.out}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
            f.write("\n")
        print(f"Wrote JSON report: {args.json_out}")

    print("\n" + report)


if __name__ == "__main__":
    main()
