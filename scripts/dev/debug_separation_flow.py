
import sys
import os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
import logging

# Setup paths
PROJECT_ROOT = Path("c:/Users/engdahlz/StemSep-V3")
APP_ROOT = PROJECT_ROOT / "StemSepApp"
sys.path.append(str(APP_ROOT / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugSep")

from models.model_factory import ModelFactory
from audio.separation_engine import SeparationEngine

# Configuration
TEST_FILE = r"C:\Users\engdahlz\Downloads\Jake Minch - Fingers and Clothes (Official Lyric Video) [TubeRipper.cc].wav"
MODEL_ID = "unwa-inst-v1e-plus"
TARGET_SR = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION = 30 # Process 30 seconds

async def run_debug():
    logger.info(f"Using device: {DEVICE}")
    
    # 1. Load Audio
    logger.info(f"Loading {TEST_FILE}...")
    audio, sr = sf.read(TEST_FILE)
    audio = audio.T # (channels, samples)
    logger.info(f"Original Audio: shape={audio.shape}, sr={sr}, mean={audio.mean():.6f}, std={audio.std():.6f}")

    # 2. Resample
    if sr != TARGET_SR:
        logger.info(f"Resampling to {TARGET_SR} Hz...")
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled_channels.append(librosa.resample(audio[ch], orig_sr=sr, target_sr=TARGET_SR))
        audio = np.stack(resampled_channels, axis=0)
        logger.info(f"Resampled Audio: shape={audio.shape}, mean={audio.mean():.6f}, std={audio.std():.6f}")
        
    # Crop to 30s
    max_samples = TARGET_SR * DURATION
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
        logger.info(f"Cropped to 30s: {audio.shape}")

    # Save input for comparison
    sf.write("DEBUG_input_resampled.wav", audio.T, TARGET_SR)

    # 3. Load Model
    logger.info(f"Loading model {MODEL_ID}...")
    models_dir = Path(os.environ['USERPROFILE']) / ".stemsep" / "models"
    config_path = models_dir / f"{MODEL_ID}.yaml"
    ckpt_path = models_dir / f"{MODEL_ID}.ckpt"
    
    import yaml
    with open(config_path, 'r') as f:
        # Using full Loader to support python/tuple tags
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Extract model config
    model_config = config.get('model', config)
    target_instrument = config.get('training', {}).get('target_instrument', 'unknown')
    
    # DEBUG FIX: Override depth to match checkpoint
    logger.info("DEBUG: Overriding mask_estimator_depth to 3")
    model_config['mask_estimator_depth'] = 3
    
    logger.info(f"Target Instrument: {target_instrument}")
    
    # Load weights first to inspect structure
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    logger.info("Inspecting checkpoint to deduce config...")
    
    # Dump handled by loop below if needed

    # 1. Deduce mask_estimator_depth
    # Structure: mask_estimators.0.to_freqs.[BAND_IDX].0.[LAYER_IDX].weight
    # The .0. comes from the outer Sequential(MLP(...), GLU) wrapper
    
    max_layer_idx = 0
    found_any = False
    for key in state_dict.keys():
        if "mask_estimators.0.to_freqs.0.0." in key and ".weight" in key:
            found_any = True
            parts = key.split('.')
            # mask_estimators.0.to_freqs.0.0.[LAYER_IDX].weight
            # index 0,1,2,3,4,5
            try:
                layer_idx = int(parts[5])
                if layer_idx > max_layer_idx:
                    max_layer_idx = layer_idx
            except:
                pass
    
    # MLP structure: Linear(0), Act(1), Linear(2), Act(3), Linear(4) ...
    # depth = (max_layer_idx / 2) + 1
    deduced_depth = (max_layer_idx // 2) + 1
    logger.info(f"Deducing mask_estimator_depth: {deduced_depth} (max layer idx: {max_layer_idx})")
    model_config['mask_estimator_depth'] = deduced_depth
    
    # 2. Deduce freqs_per_bands
    last_layer_idx = max_layer_idx
    deduced_bands = []
    
    num_bands = 0
    while True:
        # Pattern: ...to_freqs.[BAND].0.[LAST_LAYER].weight
        key = f"mask_estimators.0.to_freqs.{num_bands}.0.{last_layer_idx}.weight"
        if key not in state_dict:
            break
        
        weight = state_dict[key]
        out_dim = weight.shape[0]
        
        # output = dim_in * 2
        # dim_in = output / 2
        # freq = dim_in / 4 (stereo)
        freq = out_dim // 2 // 4
        deduced_bands.append(freq)
        num_bands += 1
        
    logger.info(f"Deducea {len(deduced_bands)} bands.")
    logger.info(f"Bands tuple: {tuple(deduced_bands)}")
    logger.info(f"Total bins: {sum(deduced_bands)}")
    
    model_config['freqs_per_bands'] = tuple(deduced_bands)
    
    # Setup target sum for padding
    target_sum = model_config.get('stft_n_fft', 2048) // 2 + 1 # 1025
    current_sum = sum(deduced_bands)
    
    if current_sum < target_sum:
        diff = target_sum - current_sum
        logger.info(f"Adding filler band of size {diff} to reach {target_sum}")
        deduced_bands.append(diff)
        model_config['freqs_per_bands'] = tuple(deduced_bands)
    elif current_sum > target_sum:
         logger.warning(f"Sum {current_sum} > Target {target_sum}. Weird.")

    # Save to recovered_config.yaml
    import yaml
    output_config = {
        'model': {
            'mask_estimator_depth': deduced_depth,
            'freqs_per_bands': tuple(deduced_bands),
            'dim': model_config.get('dim', 384),
            'stft_n_fft': model_config.get('stft_n_fft', 2048),
            'stft_hop_length': model_config.get('stft_hop_length', 441),
            'num_stems': model_config.get('num_stems', 1),
            # Add other critical fields
        },
        'training': {
             'target_instrument': target_instrument
        }
    }
    with open('recovered_config.yaml', 'w') as f:
        yaml.dump(output_config, f)
        
    logger.info("Saved recovered_config.yaml. Done.")
    return # Stop here
    
    # model = ModelFactory.create_model("Mel-Roformer", model_config) ...
    
    # 4. Inference
    logger.info("Running inference...")
    # Use SeparationEngine's _perform_separation logic manually
    segment_size = 352800
    chunk_size = segment_size
    overlap = 0.25
    stride = int(chunk_size * (1 - overlap))
    
    # Prepare
    length = audio.shape[1]
    final_result_sum = np.zeros((1, 2, length)) # (stems, channels, samples) - Assuming 1 stem model
    weight_accumulator = np.zeros(length)
    window = np.hanning(chunk_size)
    
    # Pad
    pad_length = chunk_size - (length % stride)
    padded_audio = np.pad(audio, ((0, 0), (0, pad_length)))
    padded_length = padded_audio.shape[1]
    
    current_shift_outputs = np.zeros((1, 2, padded_length))
    accum_weights = np.zeros(padded_length)
    
    num_chunks = (padded_length - chunk_size) // stride + 1
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * stride
            end = start + chunk_size
            chunk = padded_audio[:, start:end]
            
            chunk_tensor = torch.from_numpy(chunk).float().to(DEVICE).unsqueeze(0)
            output = model(chunk_tensor)
            
            output = output.squeeze(0).cpu().numpy() # (stems, channels, chunk)
            if output.ndim == 2: output = output[np.newaxis, ...] # Add stem dim if missing
            
            # De-padding output logic simplified for debug
            for c in range(2):
                current_shift_outputs[0, c, start:end] += output[0, c, :] * window
            accum_weights[start:end] += window
            
            if i % 5 == 0: logger.info(f"Chunk {i}/{num_chunks}")

    # Normalize
    accum_weights[accum_weights < 1e-6] = 1.0
    current_shift_outputs /= accum_weights
    
    # Crop back
    final_output = current_shift_outputs[:, :, :length]
    model_output_audio = final_output[0] # (channels, samples)
    
    logger.info(f"Model Output: shape={model_output_audio.shape}, mean={model_output_audio.mean():.6f}, std={model_output_audio.std():.6f}")

    # Save Model Output
    sf.write("DEBUG_model_output.wav", model_output_audio.T, TARGET_SR)
    
    # 5. Calculate Residual
    logger.info("Calculating residual...")
    residual = audio - model_output_audio
    logger.info(f"Residual: shape={residual.shape}, mean={residual.mean():.6f}, std={residual.std():.6f}")
    
    sf.write("DEBUG_residual.wav", residual.T, TARGET_SR)
    logger.info("Done! Check DEBUG output files.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_debug())
