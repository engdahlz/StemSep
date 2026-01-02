"""
Audio-separator Roformer implementations with extended parameter support.

Downloaded from: https://github.com/nomadkaraoke/python-audio-separator
Extended to support:
- skip_final_norm: Skips the final RMSNorm layer (used by some models like unwa-inst)
"""

from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer

__all__ = ['BSRoformer', 'MelBandRoformer']
