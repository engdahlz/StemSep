# coding: utf-8
"""
Official ZFTurbo Model Utils - Copied directly from:
https://github.com/ZFTurbo/Music-Source-Separation-Training

This file contains the exact official demix() and supporting functions.
DO NOT MODIFY - keep in sync with upstream.
"""
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional

try:
    import torch.distributed as dist
except:
    dist = None


def demix(
    config,  # ConfigDict or dict
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str = 'generic',
    pbar: bool = False
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform audio source separation with a given model.

    Supports both Demucs-specific and generic processing modes, including
    overlapping chunk-based inference with optional progress bar display.
    Handles padding, fading, and batching to reduce artifacts during separation.

    Args:
        config: Configuration object with audio and inference parameters.
        model: Source separation model for inference.
        mix: Input audio tensor of shape (channels, time).
        device: Device on which to run inference (CPU or CUDA).
        model_type: Type of model (e.g., 'htdemucs', 'mdx23c').
        pbar: If True, show a progress bar during chunk processing.

    Returns:
        Dictionary mapping instrument names to separated waveforms, or
        NumPy array of separated audio if single instrument (Demucs mode).
    """

    def _should_print():
        if dist is None:
            return True
        return not dist.is_initialized() or dist.get_rank() == 0

    should_print = _should_print()

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'

    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = _get_config_value(config, 'training.samplerate', 44100) * _get_config_value(config, 'training.segment', 10)
        num_instruments = len(_get_config_value(config, 'training.instruments', []))
        num_overlap = _get_config_value(config, 'inference.num_overlap', 4)
        step = chunk_size // num_overlap
    else:
        # Generic mode for Roformer, MDX23C, SCNet, etc.
        chunk_size = _get_config_value(config, 'inference.chunk_size', None)
        if chunk_size is None:
            chunk_size = _get_config_value(config, 'audio.chunk_size', 131584)
        
        instruments = prefer_target_instrument(config)
        num_instruments = len(instruments)
        num_overlap = _get_config_value(config, 'inference.num_overlap', 4)

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = _get_config_value(config, 'inference.batch_size', 1)
    use_amp = _get_config_value(config, 'training.use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            
            if pbar and should_print:
                from tqdm.auto import tqdm
                progress_bar = tqdm(
                    total=mix.shape[1], desc="Processing audio chunks", leave=False
                )
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
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

                    if mode == "generic":
                        window = windowing_array.clone()  # clone() fixes clicks at chunk edges
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

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

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = _get_config_value(config, 'training.instruments', ['vocals'])
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with linear fade-in and fade-out.

    Args:
        window_size: Total size of the window.
        fade_size: Size of the fade-in and fade-out regions.

    Returns:
        torch.Tensor of shape (window_size,) containing the windowing array.
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def prefer_target_instrument(config) -> List[str]:
    """
    Return the list of target instruments based on the configuration.
    """
    target_instrument = _get_config_value(config, 'training.target_instrument', None)
    if target_instrument:
        return [target_instrument]
    else:
        return _get_config_value(config, 'training.instruments', ['vocals'])


def apply_tta(
    config,
    model: torch.nn.Module,
    mix: torch.Tensor,
    waveforms_orig: Union[dict, np.ndarray],
    device: torch.device,
    model_type: str
) -> Union[dict, np.ndarray]:
    """
    Enhance source separation results using Test-Time Augmentation (TTA).

    Applies channel reversal and polarity inversion, reprocesses with the model,
    and combines results by averaging.
    """
    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _get_config_value(config, key_path: str, default=None):
    """
    Get a value from config by dot-separated path.
    Works with both dict and ConfigDict/namespace objects.
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict):
            if key in value:
                value = value[key]
            else:
                return default
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            # Try dict-like access for ConfigDict
            try:
                value = value[key]
            except (KeyError, TypeError):
                return default
    
    return value


def normalize_audio(audio: np.ndarray):
    """
    Normalize an audio signal using mean and standard deviation.
    """
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Reverse normalization on an audio signal.
    """
    return audio * norm_params["std"] + norm_params["mean"]
