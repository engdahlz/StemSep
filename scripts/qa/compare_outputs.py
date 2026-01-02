#!/usr/bin/env python3
"""Compare audio files from working vs broken outputs."""
import numpy as np
import soundfile as sf
from pathlib import Path

# Paths
working_dir = Path('test_zfturbo_demix_output')
broken_dir = Path('test_engine_output')
original_file = r'C:\Users\engdahlz\Downloads\Good Morning #newmusic #singersongwriter #jakeminch #music #george #shorts [TubeRipper.cc].mp3'

import librosa

print("=" * 60)
print("AUDIO COMPARISON: Working vs Broken")
print("=" * 60)

# Load original
print("\n1. Loading original file...")
orig, sr = librosa.load(original_file, sr=44100, mono=False)
if orig.ndim == 1:
    orig = np.stack([orig, orig])
print(f"   Original: shape={orig.shape}, sr={sr}")
print(f"   Original stats: mean={orig.mean():.6f}, std={orig.std():.6f}, min={orig.min():.6f}, max={orig.max():.6f}")

# Compare working outputs
print("\n2. Working outputs (test_zfturbo_demix_output/):")
for f in working_dir.glob('*.wav'):
    audio, sr = sf.read(f)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    print(f"   {f.name}: shape={audio.shape}, sr={sr}")
    print(f"      stats: mean={audio.mean():.6f}, std={audio.std():.6f}, min={audio.min():.6f}, max={audio.max():.6f}")

# Compare broken outputs
print("\n3. Broken outputs (test_engine_output/):")
for f in broken_dir.glob('*.wav'):
    audio, sr = sf.read(f)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    print(f"   {f.name}: shape={audio.shape}, sr={sr}")
    print(f"      stats: mean={audio.mean():.6f}, std={audio.std():.6f}, min={audio.min():.6f}, max={audio.max():.6f}")

# Compare specific files
print("\n4. Detailed comparison of instrumental/other:")
try:
    working_inst, _ = sf.read(working_dir / 'other.wav')
    broken_inst, _ = sf.read(broken_dir / 'instrumental.wav')
    
    print(f"   Working shape: {working_inst.shape}")
    print(f"   Broken shape: {broken_inst.shape}")
    
    # Check if shapes match
    if working_inst.shape == broken_inst.shape:
        diff = np.abs(working_inst - broken_inst)
        print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
        print(f"   Correlation: {np.corrcoef(working_inst.flatten(), broken_inst.flatten())[0,1]:.6f}")
    else:
        print("   Shapes don't match!")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n5. Detailed comparison of vocals:")
try:
    working_voc, _ = sf.read(working_dir / 'vocals.wav')
    broken_voc, _ = sf.read(broken_dir / 'vocals.wav')
    
    print(f"   Working shape: {working_voc.shape}")
    print(f"   Broken shape: {broken_voc.shape}")
    
    if working_voc.shape == broken_voc.shape:
        diff = np.abs(working_voc - broken_voc)
        print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
        print(f"   Correlation: {np.corrcoef(working_voc.flatten(), broken_voc.flatten())[0,1]:.6f}")
    else:
        print("   Shapes don't match!")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
