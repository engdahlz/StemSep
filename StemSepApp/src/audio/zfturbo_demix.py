"""
ZFTurbo Official Demix Function
Copied EXACTLY from: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/model_utils.py
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(
    config,
    model: torch.nn.Module,
    mix: np.ndarray,
    device: torch.device,
    pbar: bool = False
) -> Dict[str, np.ndarray]:
    """
    Perform audio source separation with a given model.
    EXACT copy from ZFTurbo's Music-Source-Separation-Training.
    
    Args:
        config: Configuration dict with audio and inference parameters.
        model: Source separation model for inference.
        mix: Input audio array of shape (channels, time).
        device: Device on which to run inference (CPU or CUDA).
        pbar: If True, show a progress bar during chunk processing.

    Returns:
        Dictionary mapping instrument names to separated waveforms.
    """
    
    mix = torch.tensor(mix, dtype=torch.float32)
    
    # Helper to get config value (supports both dict and object access)
    def get_cfg(section, key, default=None):
        if isinstance(config, dict):
            return config.get(section, {}).get(key, default)
        elif hasattr(config, section):
            sec = getattr(config, section)
            if hasattr(sec, key):
                return getattr(sec, key)
        return default
    
    # Get parameters from config - match official YAML structure
    chunk_size = get_cfg('audio', 'chunk_size', 485100)
    num_overlap = get_cfg('inference', 'num_overlap', 2)
    batch_size = get_cfg('inference', 'batch_size', 1)
    
    target_instrument = get_cfg('training', 'target_instrument', None)
    if target_instrument:
        instruments = [target_instrument]
    else:
        instruments = get_cfg('training', 'instruments', ['other'])
    
    num_instruments = len(instruments)
        
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = _getWindowingArray(chunk_size, fade_size)
    
    # Add padding for generic mode to handle edge artifacts
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    use_amp = True  # Mixed precision

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            
            if pbar:
                from tqdm import tqdm
                progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False)
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                
                if chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowing_array.clone()
                    if i - step == 0:  # First audio chunk, no fadein
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                        counter[..., start:start + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding
            if length_init > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary
    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}
    return ret_data
