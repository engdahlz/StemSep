#!/usr/bin/env python3
"""
Quick diagnostics for StemSep environment.
Checks: Python, Torch/CUDA, Tkinter, audio libs, ffmpeg, SDL, and basic GPU info.
Run: python scripts/diagnostics.py
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<error: {e}>"


def main():
    print("== StemSep Diagnostics ==")
    print("Python:", sys.version)
    print("Platform:", platform.platform())

    # Torch / CUDA
    try:
        import torch
        print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("CUDA version:", getattr(torch.version, "cuda", "unknown"))
            print("GPU count:", torch.cuda.device_count())
            if torch.cuda.device_count() > 0:
                dev = torch.cuda.get_device_properties(0)
                print("GPU 0:", dev.name, f"VRAM: {round(dev.total_memory/(1024**3),2)} GB")
    except Exception as e:
        print("Torch import failed:", e)

    # Tk
    try:
        import tkinter  # noqa: F401
        import customtkinter  # noqa: F401
        print("Tkinter/CustomTkinter: OK")
    except Exception as e:
        print("Tkinter failed:", e)

    # Audio libs
    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        print("Audio libs (librosa/soundfile): OK")
    except Exception as e:
        print("Audio libs failed:", e)

    # ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ver = run_cmd([ffmpeg, "-version"]).splitlines()[0]
        print("ffmpeg:", ver)
    else:
        print("ffmpeg: not found (install via your package manager)")

    # SDL (pygame runtime libs)
    try:
        import pygame  # noqa: F401
        print("pygame: OK")
    except Exception as e:
        print("pygame import failed:", e)

    # ALSA devices hint (non-fatal)
    if os.path.isdir("/proc/asound"):  # Linux
        cards = [p for p in os.listdir("/proc/asound") if p.startswith("card")]
        print("ALSA cards:", ", ".join(cards) if cards else "none detected")

    print("== End Diagnostics ==")


if __name__ == "__main__":
    main()

