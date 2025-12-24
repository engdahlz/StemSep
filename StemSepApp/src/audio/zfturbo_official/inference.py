# coding: utf-8
"""
ZFTurbo Official Inference Engine

This is a clean separation engine based entirely on ZFTurbo's 
Music-Source-Separation-Training inference.py.

https://github.com/ZFTurbo/Music-Source-Separation-Training
"""
__author__ = 'Based on Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Tuple

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import yaml


class ZFTurboInference:
    """
    Official ZFTurbo-style inference engine.
    
    This is a minimal, clean implementation following exactly the patterns
    from ZFTurbo's inference.py for maximum quality.
    """
    
    def __init__(self, models_dir: Path = None, device: str = None):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir or Path.home() / '.stemsep' / 'models'
        
        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.logger.info(f"ZFTurboInference initialized with device: {self.device}")
    
    def load_config(self, model_id: str) -> dict:
        """Load model YAML configuration."""
        yaml_path = self.models_dir / f'{model_id}.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")
        
        with open(yaml_path) as f:
            # Use safe_load for security (per Gemini Code Assist review)
            config = yaml.safe_load(f)
        
        return config
    
    def load_model(self, model_id: str, config: dict) -> nn.Module:
        """Load model checkpoint."""
        # Determine model type from config - prioritize config structure over model_type string
        model_cfg = config.get('model', config)
        
        # Detect Roformer type by checking config keys (ZFTurbo pattern):
        # - freqs_per_bands (explicit list) -> BSRoformer
        # - num_bands (auto-generated mel bands) -> MelBandRoformer
        if 'freqs_per_bands' in model_cfg:
            model_type = 'bs_roformer'
            print(f"*** DETECTED BSRoformer (has freqs_per_bands) ***")
            self.logger.info(f"Detected BSRoformer (has freqs_per_bands)")
        elif 'num_bands' in model_cfg:
            model_type = 'mel_band_roformer'
            self.logger.info(f"Detected MelBandRoformer (has num_bands)")
        else:
            # Fallback to training.model_type or architecture field
            model_type = config.get('training', {}).get('model_type')
            if not model_type:
                arch = model_cfg.get('architecture', '').lower()
                if 'bs' in arch or 'band-split' in arch:
                    model_type = 'bs_roformer'
                elif 'mel' in arch:
                    model_type = 'mel_band_roformer'
                elif 'mdx' in arch:
                    model_type = 'mdx23c'
                elif 'scnet' in arch:
                    model_type = 'scnet'
                else:
                    model_type = 'mel_band_roformer'  # Safe default
            self.logger.info(f"Model type from fallback: {model_type}")
        
        # Import model class based on type
        if model_type in ['mel_band_roformer', 'Mel-Roformer']:
            from models.model_factory import ModelFactory
            model = ModelFactory.create_model('Mel-Roformer', config['model'])
        elif model_type in ['bs_roformer', 'BS-Roformer']:
            from models.model_factory import ModelFactory
            model = ModelFactory.create_model('BS-Roformer', config['model'])
        elif model_type in ['mdx23c', 'MDX23C']:
            from models.model_factory import ModelFactory
            model = ModelFactory.create_model('MDX23C', config)
        elif model_type in ['scnet', 'SCNet']:
            from models.model_factory import ModelFactory
            model = ModelFactory.create_model('SCNet', config['model'])
        else:
            # Try generic model factory
            from models.model_factory import ModelFactory
            model = ModelFactory.create_model(model_type, config.get('model', config))
        
        # Load checkpoint
        ckpt_path = self.models_dir / f'{model_id}.ckpt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats (ZFTurbo pattern)
        if 'state' in checkpoint:
            state_dict = checkpoint['state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle known key mismatches
        # (12 unexpected "layers.X.X.norm.gamma" keys - same as audio-separator/UVR)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            self.logger.warning(f"Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            self.logger.warning(f"Unexpected keys: {len(result.unexpected_keys)} (expected for some models)")
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model {model_id} loaded on {self.device}")
        return model
    
    def load_audio(self, file_path: str, sample_rate: int = 44100) -> np.ndarray:
        """
        Load audio file exactly like ZFTurbo's run_folder.
        
        Returns float32 array of shape (channels, samples).
        """
        # Load with target sample rate directly
        mix, sr = librosa.load(file_path, sr=sample_rate, mono=False)
        
        # Handle mono -> stereo
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
        if mix.shape[0] == 1:
            mix = np.concatenate([mix, mix], axis=0)
        
        # Ensure float32
        mix = mix.astype(np.float32)
        
        self.logger.info(f"Loaded audio: shape={mix.shape}, dtype={mix.dtype}, sr={sr}")
        return mix
    
    def demix(
        self,
        config: dict,
        model: nn.Module,
        mix: np.ndarray,
        pbar: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Official ZFTurbo demix implementation.
        
        Uses the exact algorithm from zfturbo_demix.py (copied from ZFTurbo's repo).
        """
        # Import the official demix function
        from audio.zfturbo_demix import demix as official_demix
        
        # Call with our device
        return official_demix(config, model, mix, self.device, pbar)
    
    def _get_windowing_array(self, window_size: int, fade_size: int) -> torch.Tensor:
        """Generate windowing array with linear fade-in/fade-out."""
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        
        window = torch.ones(window_size)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window
    
    async def separate(
        self,
        file_path: str,
        model_id: str,
        output_dir: str,
        stems: List[str] = None,
        use_tta: bool = False,
        progress_callback: Callable = None
    ) -> Tuple[Dict[str, str], str]:
        """
        Separate audio file into stems.
        
        Args:
            file_path: Path to input audio file
            model_id: Model ID to use
            output_dir: Directory to save output files
            stems: List of stems to output (default: all from model)
            use_tta: Use test-time augmentation
            progress_callback: Callback for progress updates
            
        Returns:
            Tuple of (dict of stem name -> output path, device name)
        """
        def notify(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)
        
        notify(0, "Loading config...")
        
        # Load config
        config = self.load_config(model_id)
        sample_rate = config.get('audio', {}).get('sample_rate', 44100)
        
        notify(5, "Loading audio...")
        
        # Load audio
        mix = self.load_audio(file_path, sample_rate)
        
        notify(15, "Loading model...")
        
        # Load model
        model = self.load_model(model_id, config)
        
        notify(25, "Running separation...")
        
        # Run demix
        waveforms = self.demix(config, model, mix, pbar=True)
        
        # TTA if requested
        if use_tta:
            notify(60, "Applying TTA...")
            waveforms = self._apply_tta(config, model, mix, waveforms)
        
        notify(80, "Computing residual...")
        
        # Compute residual stems
        target_stems = stems or ['instrumental', 'vocals']
        output_waveforms = {}
        
        # Debug logging
        self.logger.info(f"Demix returned waveforms: {list(waveforms.keys())}")
        self.logger.info(f"Target stems: {target_stems}")
        
        # Handle different output formats from demix (case-insensitive)
        for stem_name, stem_audio in waveforms.items():
            stem_lower = stem_name.lower()
            if stem_lower == 'other':
                output_waveforms['instrumental'] = stem_audio
            else:
                output_waveforms[stem_lower] = stem_audio
        
        self.logger.info(f"Output waveforms after mapping: {list(output_waveforms.keys())}")
        
        # Compute residual for missing stems
        if 'vocals' in target_stems and 'vocals' not in output_waveforms:
            if 'instrumental' in output_waveforms:
                output_waveforms['vocals'] = mix - output_waveforms['instrumental']
                self.logger.info("Computed vocals as residual")
        
        if 'instrumental' in target_stems and 'instrumental' not in output_waveforms:
            if 'vocals' in output_waveforms:
                output_waveforms['instrumental'] = mix - output_waveforms['vocals']
                self.logger.info("Computed instrumental as residual")
        
        notify(90, "Saving outputs...")
        
        # Save outputs
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_paths = {}
        
        for stem_name in target_stems:
            if stem_name in output_waveforms:
                output_path = Path(output_dir) / f'{stem_name}.wav'
                sf.write(output_path, output_waveforms[stem_name].T, sample_rate)
                output_paths[stem_name] = str(output_path)
        
        notify(100, "Complete")
        
        device_name = 'GPU' if 'cuda' in str(self.device) else 'CPU'
        return output_paths, device_name
    
    def _apply_tta(
        self,
        config: dict,
        model: nn.Module,
        mix: np.ndarray,
        waveforms_orig: dict
    ) -> dict:
        """Apply test-time augmentation."""
        # Channel inversion and polarity inversion
        augmentations = [mix[::-1].copy(), -1.0 * mix.copy()]
        
        for i, aug_mix in enumerate(augmentations):
            waveforms = self.demix(config, model, aug_mix, pbar=False)
            for stem in waveforms:
                if i == 0:
                    waveforms_orig[stem] = waveforms_orig.get(stem, 0) + waveforms[stem][::-1].copy()
                else:
                    waveforms_orig[stem] = waveforms_orig.get(stem, 0) - waveforms[stem]
        
        # Average
        for stem in waveforms_orig:
            waveforms_orig[stem] /= len(augmentations) + 1
        
        return waveforms_orig


# Test function - run with: python inference.py <input_file> <model_id> [output_dir]
async def test():
    """Test the inference engine."""
    import argparse
    parser = argparse.ArgumentParser(description='Test ZFTurbo inference')
    parser.add_argument('input_file', nargs='?', help='Input audio file')
    parser.add_argument('model_id', nargs='?', default='unwa-inst-v1e-plus', help='Model ID')
    parser.add_argument('output_dir', nargs='?', default='test_output', help='Output directory')
    args = parser.parse_args()
    
    if not args.input_file:
        print("Usage: python inference.py <input_file> [model_id] [output_dir]")
        print("Example: python inference.py song.mp3 unwa-inst-v1e-plus ./output")
        return
    
    engine = ZFTurboInference()
    
    def progress(pct, msg):
        print(f"[{pct:3.0f}%] {msg}")
    
    result, device = await engine.separate(
        file_path=args.input_file,
        model_id=args.model_id,
        output_dir=args.output_dir,
        progress_callback=progress
    )
    
    print(f"\nSuccess! Device: {device}")
    for stem, path in result.items():
        audio, _ = sf.read(path)
        print(f"  {stem}: std={audio.std():.6f}")


if __name__ == "__main__":
    # Add path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(test())

