"""Torch compatibility helpers for evolving AMP/SDPA APIs."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def cuda_autocast(enabled: bool = True) -> Any:
    """Return a CUDA autocast context compatible across Torch versions."""
    amp_mod = getattr(torch, "amp", None)
    amp_autocast = getattr(amp_mod, "autocast", None) if amp_mod else None
    if callable(amp_autocast):
        return amp_autocast("cuda", enabled=enabled)

    legacy_cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    legacy_autocast = (
        getattr(legacy_cuda_amp, "autocast", None) if legacy_cuda_amp else None
    )
    if callable(legacy_autocast):
        return legacy_autocast(enabled=enabled)

    return nullcontext()


def sdpa_kernel_context(
    *,
    enable_flash: bool,
    enable_math: bool,
    enable_mem_efficient: bool,
) -> Any:
    """Return SDPA kernel selection context for old/new Torch APIs."""
    attn_mod = getattr(getattr(torch, "nn", None), "attention", None)
    sdpa_kernel = getattr(attn_mod, "sdpa_kernel", None) if attn_mod else None
    sdp_backend = getattr(attn_mod, "SDPBackend", None) if attn_mod else None

    if callable(sdpa_kernel) and sdp_backend is not None:
        backends = []
        if enable_flash and hasattr(sdp_backend, "FLASH_ATTENTION"):
            backends.append(sdp_backend.FLASH_ATTENTION)
        if enable_math and hasattr(sdp_backend, "MATH"):
            backends.append(sdp_backend.MATH)
        if enable_mem_efficient:
            # Torch names this backend EFFICIENT_ATTENTION in newer APIs.
            if hasattr(sdp_backend, "EFFICIENT_ATTENTION"):
                backends.append(sdp_backend.EFFICIENT_ATTENTION)
            elif hasattr(sdp_backend, "MEM_EFFICIENT_ATTENTION"):
                backends.append(sdp_backend.MEM_EFFICIENT_ATTENTION)
        if backends:
            return sdpa_kernel(backends=backends)
        return nullcontext()

    cuda_mod = getattr(torch.backends, "cuda", None)
    legacy_sdp = getattr(cuda_mod, "sdp_kernel", None) if cuda_mod else None
    if callable(legacy_sdp):
        return legacy_sdp(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient,
        )

    return nullcontext()
