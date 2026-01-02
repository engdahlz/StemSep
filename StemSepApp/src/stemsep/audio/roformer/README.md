# Roformer Models - Local Implementation

This directory contains Roformer model implementations downloaded from 
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 
with extended parameter support.

## Files
- `attend.py` - Attention mechanism
- `bs_roformer.py` - BSRoformer with `skip_final_norm` support  
- `mel_band_roformer.py` - MelBandRoformer with `skip_final_norm` support

## Extended Parameters
Added `skip_final_norm` parameter to both models - when `True`, the final 
RMSNorm layer is replaced with `nn.Identity()`. This is required by some 
trained models like `unwa-inst-v1e-plus`.

## Known Limitation: 12 Unexpected Keys

When loading certain Roformer checkpoints, 12 unexpected keys are logged:

```
Unexpected keys in checkpoint: 12 keys
['layers.0.0.norm.gamma', 'layers.0.1.norm.gamma', 
 'layers.1.0.norm.gamma', 'layers.1.1.norm.gamma', ...]
```

**Why this happens:**
- These are norm parameters inside each `Transformer` block (not `final_norm`)
- Our `Transformer` class uses `norm_output=False`, so these norms aren't created
- The checkpoint contains weights for these internal norms that we don't use

**Impact:** None observed. Separation quality is unaffected.

**Note:** The official [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 
library has the same behavior - they also don't include these norm layers 
in their Transformer implementation.

## Source
Downloaded from: https://github.com/nomadkaraoke/python-audio-separator/tree/main/audio_separator/separator/uvr_lib_v5/roformer

Modified: December 2025
