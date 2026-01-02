"""
Unit tests for overlap normalization and audio-separator overlap ratio mapping.

These tests focus on *semantics* and deterministic mapping rules, not on GPU/audio I/O.

Why this exists:
- UI/recipes may express overlap either as:
  - a fraction in [0, 1) (overlap fraction), or
  - a guide-style integer overlap (2/3/4/8...) matching MSST/UVR "num_overlap"/divisor.
- The core engine normalizes overlap to an integer overlap >= 2 for chunking (legacy path).
- The audio-separator wrapper (SimpleSeparator) expects overlap as a ratio in [0, 1),
  and converts guide-style integers N -> (N-1)/N.

Notes:
- We intentionally keep these tests lightweight and independent of files/models.
- We do NOT import audio-separator itself; we only test our mapping logic.
"""

from __future__ import annotations

import math

import pytest

from stemsep.audio.separation_engine import _resolve_overlap_int


def _to_audio_separator_overlap_ratio(overlap_value) -> float:
    """
    Mirror of SimpleSeparator.separate(...) overlap normalization.

    - If overlap <= 1.0: treat as already a ratio in [0..1)
    - If overlap > 1.0: treat as guide-style integer overlap N, map to ratio (N-1)/N
    - Clamp: ratio < 0 => 0, ratio >= 1 => 0.999
    - If overlap can't be parsed: default ratio 0.25 (matches SimpleSeparator fallback)

    We keep this helper local to tests to avoid importing implementation-only details.
    """
    try:
        ov = float(overlap_value)
    except Exception:
        ov = 0.25

    if ov <= 0:
        return 0.25
    if ov <= 1.0:
        return ov

    n = ov
    ratio = (n - 1.0) / n

    if ratio < 0.0:
        ratio = 0.0
    if ratio >= 1.0:
        ratio = 0.999

    return float(ratio)


@pytest.mark.parametrize(
    "overlap,expected",
    [
        (None, 4),  # default is balanced
        (-1, 4),  # invalid -> balanced
        (0.0, 2),  # fraction -> maps to >=2
        (0.01, 2),
        (0.25, 2),  # round(1/(1-0.25))=1 -> clamped up to 2
        (0.5, 2),  # 1/(1-0.5)=2
        (0.66, 3),  # ~2.94 -> 3
        (0.75, 4),  # 1/(1-0.75)=4
        (0.875, 8),  # 1/(1-0.875)=8
        (1.0, 2),  # treated as integer-ish, rounded then clamped to >=2
        (2, 2),
        (3, 3),
        (4, 4),
        (8, 8),
        (50, 50),
        (51, 50),  # clamped
    ],
)
def test_resolve_overlap_int_normalizes_to_guide_style_int(overlap, expected):
    assert _resolve_overlap_int(overlap) == expected


@pytest.mark.parametrize(
    "overlap_int,expected_ratio",
    [
        (2, 0.5),
        (3, 2.0 / 3.0),
        (4, 0.75),
        (8, 0.875),
        (50, 49.0 / 50.0),
    ],
)
def test_audio_separator_ratio_from_integer_overlap(overlap_int, expected_ratio):
    ratio = _to_audio_separator_overlap_ratio(overlap_int)
    assert math.isclose(ratio, expected_ratio, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize(
    "overlap_fraction,expected_ratio",
    [
        # SimpleSeparator treats overlap <= 0 as invalid and falls back to 0.25
        (0.0, 0.25),
        (0.1, 0.1),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (0.99, 0.99),
    ],
)
def test_audio_separator_ratio_accepts_fraction_passthrough(
    overlap_fraction, expected_ratio
):
    ratio = _to_audio_separator_overlap_ratio(overlap_fraction)
    assert math.isclose(ratio, float(expected_ratio), rel_tol=0, abs_tol=1e-12)


def test_engine_to_audio_separator_semantics_round_trip_examples():
    """
    This test ties the two worlds together:

    If the engine receives an overlap fraction, we normalize to an integer overlap,
    and then audio-separator sees the corresponding ratio.

    Example:
      overlap_fraction=0.75 -> overlap_int=4 -> overlap_ratio=(4-1)/4=0.75
    """
    overlap_fraction = 0.75
    overlap_int = _resolve_overlap_int(overlap_fraction)
    assert overlap_int == 4

    ratio = _to_audio_separator_overlap_ratio(overlap_int)
    assert math.isclose(ratio, 0.75, rel_tol=0, abs_tol=1e-12)

    overlap_fraction = 0.875
    overlap_int = _resolve_overlap_int(overlap_fraction)
    assert overlap_int == 8

    ratio = _to_audio_separator_overlap_ratio(overlap_int)
    assert math.isclose(ratio, 0.875, rel_tol=0, abs_tol=1e-12)
