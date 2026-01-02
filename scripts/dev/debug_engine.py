#!/usr/bin/env python3
"""Debug what separation_engine actually does."""
import sys
sys.path.insert(0, 'StemSepApp/src')

import yaml
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

MODEL_ID = 'unwa-inst-v1e-plus'
INPUT_FILE = r'C:\Users\engdahlz\Downloads\Good Morning #newmusic #singersongwriter #jakeminch #music #george #shorts [TubeRipper.cc].mp3'
OUTPUT_DIR = Path('test_debug_output')
OUTPUT_DIR.mkdir(exist_ok=True)

models_dir = Path.home() / '.stemsep' / 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== DEBUG: What separation_engine does ===\n")

# 1. Load config (same as engine)
print("1. Loading YAML config...")
yaml_path = models_dir / f'{MODEL_ID}.yaml'
with open(yaml_path) as f:
    yaml_config = yaml.unsafe_load(f)

print(f"   audio.chunk_size: {yaml_config['audio']['chunk_size']}")
print(f"   inference.num_overlap: {yaml_config['inference']['num_overlap']}")
print(f"   training.target_instrument: {yaml_config['training']['target_instrument']}")

# 2. Load audio (same as engine - using sf.read first, then librosa)
print("\n2. Loading audio (engine way)...")
try:
    audio_data, sr = sf.read(INPUT_FILE)
    if audio_data.ndim == 1:
        audio_engine = audio_data[np.newaxis, :]
    else:
        audio_engine = audio_data.T
except Exception:
    audio_engine, sr = librosa.load(INPUT_FILE, sr=None, mono=False)

print(f"   Engine audio shape: {audio_engine.shape}, sr={sr}")
print(f"   Engine audio stats: mean={audio_engine.mean():.6f}, std={audio_engine.std():.6f}")

# 3. Load audio (test_separation_unwa.py way - librosa with sr=44100)
print("\n3. Loading audio (simple test way)...")
audio_simple, sr_simple = librosa.load(INPUT_FILE, sr=44100, mono=False)
if audio_simple.ndim == 1:
    audio_simple = np.stack([audio_simple, audio_simple])

print(f"   Simple audio shape: {audio_simple.shape}, sr={sr_simple}")
print(f"   Simple audio stats: mean={audio_simple.mean():.6f}, std={audio_simple.std():.6f}")

# 4. Compare
print("\n4. Comparing audio inputs:")
print(f"   Shapes equal: {audio_engine.shape == audio_simple.shape}")
if audio_engine.shape == audio_simple.shape:
    diff = np.abs(audio_engine - audio_simple)
    print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
else:
    print(f"   Engine: {audio_engine.shape}, Simple: {audio_simple.shape}")
    # If sample rate differs, resample
    if sr != 44100:
        print(f"   WARNING: Engine sample rate is {sr}, not 44100!")

# Import zfturbo_demix
from audio.zfturbo_demix import demix as zfturbo_demix
from models.model_factory import ModelFactory

# 5. Load model
print("\n5. Loading model...")
model = ModelFactory.create_model('Mel-Roformer', yaml_config['model'])
ckpt = torch.load(models_dir / f'{MODEL_ID}.ckpt', map_location='cpu', weights_only=False)
if 'state_dict' in ckpt:
    state = ckpt['state_dict']
elif 'state' in ckpt:
    state = ckpt['state']
else:
    state = ckpt
model.load_state_dict(state, strict=True)
model.to(DEVICE)
model.eval()
print(f"   Model loaded")

# 6. Run demix with ENGINE audio
print("\n6. Running demix with ENGINE audio...")
waveforms_engine = zfturbo_demix(yaml_config, model, audio_engine, DEVICE, pbar=False)
for name, wav in waveforms_engine.items():
    print(f"   {name}: shape={wav.shape}, std={wav.std():.6f}")

# 7. Run demix with SIMPLE audio  
print("\n7. Running demix with SIMPLE audio...")
waveforms_simple = zfturbo_demix(yaml_config, model, audio_simple, DEVICE, pbar=False)
for name, wav in waveforms_simple.items():
    print(f"   {name}: shape={wav.shape}, std={wav.std():.6f}")

# 8. Save both for comparison
print("\n8. Saving outputs...")
for name, wav in waveforms_engine.items():
    sf.write(OUTPUT_DIR / f'{name}_engine_audio.wav', wav.T if wav.ndim > 1 else wav, 44100)
for name, wav in waveforms_simple.items():
    sf.write(OUTPUT_DIR / f'{name}_simple_audio.wav', wav.T if wav.ndim > 1 else wav, 44100)

print("\nDone!")
