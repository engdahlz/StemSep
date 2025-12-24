import numpy as np
import logging
from typing import List, Dict, Literal, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class EnsembleBlender:
    """
    Handles the blending of audio stems from multiple sources (models).
    Supports different blending algorithms like Weighted Average and Max Spec.
    """
    
    # ==================== ARTIFACT CLEANUP PRESETS ====================
    # Based on NotebookLM recommendations for common artifacts
    
    # Phase Fixer presets for Roformer "buzzing" noise
    # Per NotebookLM: low_hz=500, high_hz=5000, weight=2.0 (vs default 0.8)
    PHASE_FIXER_PRESETS = {
        "roformer_buzzing": {
            "low_hz": 500,
            "high_hz": 5000,
            "high_freq_weight": 2.0,  # Aggressive (default is 0.8)
            "description": "Fix metallic buzzing from Roformer models"
        },
        "gentle": {
            "low_hz": 500,
            "high_hz": 5000,
            "high_freq_weight": 0.8,
            "description": "Standard phase fixing, less aggressive"
        },
        "wide": {
            "low_hz": 250,
            "high_hz": 8000,
            "high_freq_weight": 1.5,
            "description": "Wider frequency range for extreme cases"
        }
    }
    
    # Demudder presets for spectral holes / "mud"
    # Per NotebookLM: High-pass 100-250 Hz to protect bass
    DEMUDDER_PRESETS = {
        "rock_metal": {
            "high_pass_hz": 100,
            "description": "For dense genres (rock/metal), preserve bass"
        },
        "standard": {
            "high_pass_hz": 250,
            "description": "Standard demudding, safe for most genres"
        },
        "aggressive": {
            "high_pass_hz": 500,
            "description": "Aggressive cleanup, may thin the low end"
        },
        "vocal_cleanup": {
            "high_pass_hz": 500,
            "description": "Remove rumble/low-freq bleed from vocals"
        }
    }
    
    # ==================== PER-STEM ALGORITHM PRESETS ====================
    # Per user guide: Different algorithms work better for different stems
    # - Max Spec: Best for vocals (fuller, more detail)
    # - Min Spec: Best for instrumental (less vocal bleed)
    # - Average: Highest SDR, safest default
    
    STEM_ALGORITHM_PRESETS = {
        "best_instrumental": {
            "instrumental": "max_spec",
            "vocals": "average",
            "description": "Max detail on instrumental, balanced vocals"
        },
        "best_vocals": {
            "vocals": "max_spec", 
            "instrumental": "average",
            "description": "Fuller vocals with max detail"
        },
        "bleedless_inst": {
            "instrumental": "min_spec",
            "vocals": "max_spec",
            "description": "Cleanest instrumental (less vocal bleed)"
        },
        "bleedless_vocals": {
            "vocals": "min_spec",
            "instrumental": "max_spec", 
            "description": "Cleanest vocals (less instrument bleed)"
        },
        "balanced_sdr": {
            "instrumental": "average",
            "vocals": "average",
            "description": "Highest SDR, safest default for both"
        },
        "max_both": {
            "instrumental": "max_spec",
            "vocals": "max_spec",
            "description": "Maximum detail on both stems (may increase bleed)"
        }
    }
    
    @classmethod
    def get_stem_algorithm_preset(cls, preset: str = "balanced_sdr") -> Dict:
        """
        Get per-stem algorithm settings for a preset.
        
        Returns:
            Dict with stem names as keys and algorithm names as values
        """
        return cls.STEM_ALGORITHM_PRESETS.get(preset, cls.STEM_ALGORITHM_PRESETS["balanced_sdr"])
    
    def __init__(self):
        self.logger = logging.getLogger('StemSep.Blender')

    def _ensure_channels_first(self, audio: np.ndarray) -> np.ndarray:
        a = np.asarray(audio)
        if a.ndim == 1:
            return a[None, :]
        if a.ndim == 2 and a.shape[0] > a.shape[1]:
            return a.T
        return a
    
    @classmethod
    def get_phase_fixer_settings(cls, preset: str = "roformer_buzzing") -> Dict:
        """
        Get Phase Fixer settings for a preset.
        
        Per NotebookLM: Use 'roformer_buzzing' preset (weight 2.0, 500-5000 Hz)
        to fix metallic buzzing from Roformer models.
        
        Returns:
            Dict with low_hz, high_hz, high_freq_weight
        """
        return cls.PHASE_FIXER_PRESETS.get(preset, cls.PHASE_FIXER_PRESETS["roformer_buzzing"])
    
    @classmethod
    def get_demudder_settings(cls, preset: str = "standard") -> Dict:
        """
        Get Demudder settings for a preset.
        
        Per NotebookLM: Use 'rock_metal' (100 Hz HP) for dense genres,
        'vocal_cleanup' (500 Hz HP) for vocal tracks.
        
        Returns:
            Dict with high_pass_hz
        """
        return cls.DEMUDDER_PRESETS.get(preset, cls.DEMUDDER_PRESETS["standard"])
    
    def phase_fix_workflow(
        self,
        fullness_instrumental: np.ndarray,
        reference_vocal: np.ndarray,
        reference_instrumental: np.ndarray,
        low_hz: int = 500,
        high_hz: int = 5000,
        high_freq_weight: float = 2.0,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Full 4-step Phase Fix workflow per NotebookLM/user guide recommendations.
        
        This runs automatically as part of ensemble separation when Phase Fix is enabled.
        
        Steps:
        1. (Already done) Fullness model run → fullness_instrumental
        2. (Already done) Reference model run → reference_vocal + reference_instrumental
        3. Phase Fix: Copy phase from reference_vocal to fullness_instrumental
        4. Final Ensemble: Max Spec blend of phase_fixed + reference_instrumental
        
        Args:
            fullness_instrumental: Instrumental from fullness model (e.g., Resurrection, HyperACE)
            reference_vocal: Vocal from clean/reference model (e.g., Becruily Vocal)
            reference_instrumental: Instrumental from reference model (for final ensemble)
            low_hz: Low frequency cutoff for phase fix (default 500)
            high_hz: High frequency cutoff for phase fix (default 5000)
            high_freq_weight: Phase blend weight (default 2.0 for aggressive fix)
            sample_rate: Audio sample rate
            
        Returns:
            Final phase-fixed and ensembled instrumental
        """
        self.logger.info(f"Phase Fix Workflow: {low_hz}-{high_hz}Hz, weight={high_freq_weight}")

        fullness_instrumental = self._ensure_channels_first(fullness_instrumental)
        reference_vocal = self._ensure_channels_first(reference_vocal)
        reference_instrumental = self._ensure_channels_first(reference_instrumental)
        
        # Ensure consistent shapes
        min_len = min(
            fullness_instrumental.shape[-1],
            reference_vocal.shape[-1],
            reference_instrumental.shape[-1]
        )
        fullness_instrumental = fullness_instrumental[..., :min_len]
        reference_vocal = reference_vocal[..., :min_len]
        reference_instrumental = reference_instrumental[..., :min_len]
        
        # Step 3: Phase Fix (copy phase from vocal to instrumental)
        self.logger.info("Phase Fix Workflow Step 3: Applying phase correction...")
        phase_fixed = self._phase_fix_blend(
            fullness_instrumental,
            reference_vocal,
            low_hz=low_hz,
            high_hz=high_hz,
            high_freq_weight=high_freq_weight,
            sample_rate=sample_rate
        )
        
        # Step 4: Final Ensemble with Max Spec
        self.logger.info("Phase Fix Workflow Step 4: Max Spec ensemble...")
        final_result = self._max_spec_blend([phase_fixed, reference_instrumental])
        
        self.logger.info("Phase Fix Workflow complete!")
        return final_result


    def blend_stems(
        self,
        stem_map: Dict[str, List[np.ndarray]], 
        weights: List[float] = None, 
        algorithm: Literal["average", "max_spec", "min_spec", "phase_fix", "frequency_split"] = "average",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Blends multiple versions of stems into a single output.

        Args:
            stem_map: Dictionary mapping stem name (e.g., 'vocals') to a list of audio arrays (numpy).
                      All arrays in the list must have the same shape.
            weights: List of weights corresponding to the input arrays. 
                     If None, equal weights are used. Only used for 'average' algorithm.
            algorithm: Blending algorithm to use.
            kwargs: Additional arguments for specific algorithms (e.g. split_freq).

        Returns:
            Dictionary mapping stem name to the blended audio array.
        """
        blended_stems = {}

        for stem_name, audio_list in stem_map.items():
            if not audio_list:
                continue
            
            # Ensure all arrays are numpy arrays
            audio_list = [np.array(a) for a in audio_list]
            
            # Transpose to (channels, samples) if needed
            # soundfile.read returns (samples, channels)
            for i in range(len(audio_list)):
                if audio_list[i].ndim == 2 and audio_list[i].shape[0] > audio_list[i].shape[1]:
                    # If dim0 > dim1 (e.g. 44100 > 2), it's likely (samples, channels)
                    # We want (channels, samples)
                    audio_list[i] = audio_list[i].T
            
            # Check shapes
            base_shape = audio_list[0].shape
            for i, audio in enumerate(audio_list):
                if audio.shape != base_shape:
                    # Trim to min length
                    min_len = min(a.shape[-1] for a in audio_list)
                    audio_list = [a[..., :min_len] for a in audio_list]
                    break

            if algorithm == "average":
                blended = self._average_blend(audio_list, weights)
                        
            elif algorithm == "max_spec":
                if TORCH_AVAILABLE:
                    blended = self._max_spec_blend(audio_list)
                else:
                    self.logger.warning("Torch not available for Max Spec, falling back to Average")
                    blended = self._average_blend(audio_list, weights)
                    
            elif algorithm == "min_spec":
                 if TORCH_AVAILABLE:
                    blended = self._min_spec_blend(audio_list)
                 else:
                    self.logger.warning("Torch not available for Min Spec, falling back to Average")
                    blended = self._average_blend(audio_list, weights)
            
            elif algorithm == "phase_fix":
                if TORCH_AVAILABLE:
                    if len(audio_list) < 2:
                        self.logger.warning("Phase Fix requires at least 2 sources (Magnitude, Phase Ref). Falling back to Average.")
                        blended = self._average_blend(audio_list, weights)
                    else:
                        # Source 0 = Magnitude (Fullness), Source 1 = Phase (Clean/Reference)
                        blended = self._phase_fix_blend(
                            audio_list[0], 
                            audio_list[1],
                            low_hz=kwargs.get('low_hz', 500),
                            high_hz=kwargs.get('high_hz', 5000),
                            high_freq_weight=kwargs.get('high_freq_weight', 2.0)
                        )
                else:
                    self.logger.warning("Torch not available for Phase Fix, falling back to Average")
                    blended = self._average_blend(audio_list, weights)

            elif algorithm == "frequency_split":
                if TORCH_AVAILABLE:
                    split_freq = kwargs.get('split_freq', 750)
                    blended = self._frequency_split_blend(audio_list, split_freq)
                else:
                    self.logger.warning("Torch not available for Frequency Split, falling back to Average")
                    blended = self._average_blend(audio_list, weights)

            else:
                self.logger.warning(f"Unknown algorithm: {algorithm}, falling back to Average")
                blended = self._average_blend(audio_list, weights)

            blended_stems[stem_name] = blended

        return blended_stems

    def blend_stems_mixed(
        self,
        stem_map: Dict[str, List[np.ndarray]],
        algorithms: Dict[str, str],
        weights: List[float] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Blend stems with per-stem algorithm selection.
        
        Per NotebookLM recommendations:
        - Max/Min: Best instrumental (max for vocals extraction, min for inst)
        - Min/Max: Best acapella (min for vocals, max for inst)
        - Avg/Avg: Highest SDR, safest default
        
        Args:
            stem_map: Dict mapping stem name to list of audio arrays
            algorithms: Dict mapping stem name to algorithm 
                       e.g. {"vocals": "max_spec", "instrumental": "min_spec"}
            weights: Optional weights for average algorithm
            
        Returns:
            Blended stems dict
        """
        blended_stems = {}
        
        for stem_name, audio_list in stem_map.items():
            if not audio_list:
                continue
            
            # Get algorithm for this stem (default to average)
            algorithm = algorithms.get(stem_name, "average")
            
            # Adjust weights for this stem - if source count doesn't match global weights,
            # pass None to let blend_stems use equal weights (fallback behavior)
            stem_weights = weights if weights and len(weights) == len(audio_list) else None
            
            # Use existing blend_stems for single stem
            single_stem_map = {stem_name: audio_list}
            result = self.blend_stems(single_stem_map, stem_weights, algorithm, **kwargs)
            
            if stem_name in result:
                blended_stems[stem_name] = result[stem_name]
        
        return blended_stems

    def validate_ensemble_size(self, num_models: int) -> Dict:
        """
        Validate ensemble size and calculate volume compensation.
        
        Per NotebookLM: Don't use more than 4-5 models, quality drops after.
        For DAW mixing: -3 dB per added stem to replicate Average algorithm.
        
        Args:
            num_models: Number of models in ensemble
            
        Returns:
            Dict with validation result and compensation info
        """
        max_recommended = 5
        
        result = {
            "valid": num_models <= max_recommended,
            "num_models": num_models,
            "max_recommended": max_recommended,
            "db_compensation": -3 * num_models,  # Per NotebookLM formula
            "linear_gain": 10 ** ((-3 * num_models) / 20),  # Convert to linear
        }
        
        if num_models > max_recommended:
            result["warning"] = f"Using {num_models} models may degrade quality. Max recommended: {max_recommended}"
            self.logger.warning(result["warning"])
        
        return result

    def get_recommended_algorithm(self, target_stem: str, priority: str = "balanced") -> str:
        """
        Get recommended ensemble algorithm for a target stem.
        
        Per NotebookLM:
        - Vocals: max_spec (fuller, more detail)
        - Instrumentals: min_spec (less vocal bleed)
        - Balanced/SDR: average
        
        Args:
            target_stem: "vocals" or "instrumental"
            priority: "fullness", "bleedless", or "balanced"
            
        Returns:
            Recommended algorithm name
        """
        recommendations = {
            "vocals": {
                "fullness": "max_spec",
                "bleedless": "min_spec",
                "balanced": "average"
            },
            "instrumental": {
                "fullness": "max_spec",  # Risk: more vocal bleed
                "bleedless": "min_spec",
                "balanced": "average"
            }
        }
        
        stem_recs = recommendations.get(target_stem, recommendations["vocals"])
        return stem_recs.get(priority, "average")

    def _average_blend(self, audio_list: List[np.ndarray], weights: Optional[List[float]]) -> np.ndarray:
        """Weighted average blending"""
        if weights is None:
            return np.mean(audio_list, axis=0)
        
        # Handle weight mismatch gracefully - can occur when different models
        # produce different stems (e.g., one model only produces vocals)
        if len(weights) != len(audio_list):
            self.logger.warning(
                f"Weight count ({len(weights)}) doesn't match source count ({len(audio_list)}). "
                f"Using equal weights instead."
            )
            return np.mean(audio_list, axis=0)
        
        # Normalize weights
        norm_weights = np.array(weights) / np.sum(weights)
        
        # Compute weighted sum - ensure float type to prevent dtype casting errors
        blended = np.zeros_like(audio_list[0], dtype=np.float64)
        for audio, w in zip(audio_list, norm_weights):
            blended += audio.astype(np.float64) * w
            
        return blended

    def _frequency_split_blend(self, audio_list: List[np.ndarray], split_freq: int = 750) -> np.ndarray:
        """
        Combines Low Frequencies from Model A with High Frequencies from Model B via STFT Masking.
        Model 0 = Low Band (Bass/Body)
        Model 1 = High Band (Detail/Treble)
        """
        if len(audio_list) != 2:
            self.logger.warning(f"Frequency Split requires exactly 2 models, got {len(audio_list)}. Fallback to average.")
            return self._average_blend(audio_list, None)

        stack = np.stack(audio_list, axis=0)
        num_models, channels, samples = stack.shape
        final_audio = np.zeros((channels, samples))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for c in range(channels):
            # Model 0 (Low), Model 1 (High)
            wav_low = torch.from_numpy(stack[0, c, :]).float().to(device)
            wav_high = torch.from_numpy(stack[1, c, :]).float().to(device)
            
            n_fft = 4096
            hop_length = 1024
            window = torch.hann_window(n_fft).to(device)
            
            # STFT
            spec_low = torch.stft(wav_low, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            spec_high = torch.stft(wav_high, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            
            # Calculate cutoff bin
            # Freq resolution = sr / n_fft
            # bin = freq / (sr / n_fft) = freq * n_fft / sr
            sample_rate = 44100 # Assuming 44.1k
            bin_idx = int(split_freq * n_fft / sample_rate)
            
            # Crossover Feathering (Soft Split)
            # Avoid hard cuts which cause phase artifacts (clicks/dips)
            # Create a smooth transition mask
            width_hz = 200 # 200Hz crossover width
            width_bins = int(width_hz * n_fft / sample_rate)
            half_width = width_bins // 2
            
            start_bin = max(0, bin_idx - half_width)
            end_bin = min(spec_low.shape[0], bin_idx + half_width)

            # Create Low-Pass Mask (1.0 -> 0.0)
            # Shape: (freq, 1) to broadcast over time
            mask = torch.ones((spec_low.shape[0], 1), device=device)
            mask[end_bin:] = 0.0

            if end_bin > start_bin:
                # Linear ramp from 1.0 to 0.0
                ramp = torch.linspace(1.0, 0.0, end_bin - start_bin, device=device).unsqueeze(1)
                mask[start_bin:end_bin] = ramp

            # Combine: Low * mask + High * (1-mask)
            # Linear mixing in complex domain handles phase transition smoothly
            out_spec = spec_low * mask + spec_high * (1.0 - mask)
            
            # iSTFT
            out_wav = torch.istft(out_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=samples)
            
            final_audio[c, :] = out_wav.cpu().numpy()
            
        return final_audio

    def _max_spec_blend(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Max Spectrogram blending using PyTorch STFT.
        Takes the bin with maximum magnitude across models, preserving phase of that max bin.
        """
        # Stack: (num_models, channels, samples)
        stack = np.stack(audio_list, axis=0)
        num_models, channels, samples = stack.shape
        
        final_audio = np.zeros((channels, samples))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for c in range(channels):
            # Convert to tensor: (num_models, samples)
            wavs = torch.from_numpy(stack[:, c, :]).float().to(device)
            
            # STFT params
            n_fft = 4096
            hop_length = 1024
            window = torch.hann_window(n_fft).to(device)
            
            # Compute STFT for all models
            # Shape: (num_models, freq, time) complex
            specs = torch.stft(wavs, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            
            # Get Magnitude
            mags = torch.abs(specs)
            
            # Find index of max magnitude
            # max_indices: (freq, time)
            max_mags, max_indices = torch.max(mags, dim=0)
            
            # Gather the complex values corresponding to the max magnitude
            # specs shape: (num_models, F, T)
            # max_indices shape: (F, T)
            
            index = max_indices.unsqueeze(0) # (1, F, T)
            selected_spec = torch.gather(specs, 0, index).squeeze(0) # (F, T)
            
            # ISTFT
            out_wav = torch.istft(selected_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=samples)
            
            final_audio[c, :] = out_wav.cpu().numpy()
            
        return final_audio

    def _min_spec_blend(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Min Spectrogram blending using PyTorch STFT.
        Useful for aggressive bleed reduction (e.g. taking min of instrumental to remove vocal bleed).
        """
        stack = np.stack(audio_list, axis=0)
        num_models, channels, samples = stack.shape
        
        final_audio = np.zeros((channels, samples))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for c in range(channels):
            wavs = torch.from_numpy(stack[:, c, :]).float().to(device)
            
            n_fft = 4096
            hop_length = 1024
            window = torch.hann_window(n_fft).to(device)
            
            specs = torch.stft(wavs, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            mags = torch.abs(specs)
            
            # Min magnitude
            min_mags, min_indices = torch.min(mags, dim=0)
            
            index = min_indices.unsqueeze(0)
            selected_spec = torch.gather(specs, 0, index).squeeze(0)
            
            out_wav = torch.istft(selected_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=samples)
            
            final_audio[c, :] = out_wav.cpu().numpy()
            
        return final_audio

    def _phase_fix_blend(
        self, 
        mag_source: np.ndarray, 
        phase_source: np.ndarray,
        low_hz: int = 500,
        high_hz: int = 5000,
        high_freq_weight: float = 2.0,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Enhanced Phase Fix: Combines Magnitude from Source A with Phase from Source B
        in a specified frequency range, with configurable blending weight.
        
        Per user guide recommendations:
        - Standard: 500-5000 Hz (fixes Roformer metallic buzzing)
        - HyperACE: 3000-5000 Hz
        - Wide: 250-8000 Hz (for extreme cases)
        
        Args:
            mag_source: Audio with desired magnitude (fullness)
            phase_source: Audio with clean phase reference
            low_hz: Low frequency cutoff for phase fix (Hz)
            high_hz: High frequency cutoff for phase fix (Hz)  
            high_freq_weight: Blend weight (0.0=ignore phase_source, 1.0=replace, >1.0=aggressive)
            sample_rate: Audio sample rate
            
        Returns:
            Phase-fixed audio
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mag_source = self._ensure_channels_first(mag_source)
        phase_source = self._ensure_channels_first(phase_source)

        # Match channel count
        min_ch = min(mag_source.shape[0], phase_source.shape[0])
        mag_source = mag_source[:min_ch, :]
        phase_source = phase_source[:min_ch, :]

        # Ensure same length
        min_len = min(mag_source.shape[-1], phase_source.shape[-1])
        mag_source = mag_source[..., :min_len]
        phase_source = phase_source[..., :min_len]

        n_fft = 4096
        hop_length = 1024

        # Minimum length check for STFT
        if mag_source.shape[-1] < n_fft * 2:
            self.logger.warning(
                f"Audio too short for Phase Fix ({mag_source.shape[-1]} samples < {n_fft*2}). Skipping phase correction."
            )
            return mag_source

        wav_mag = torch.from_numpy(mag_source).float().to(device)
        wav_phase = torch.from_numpy(phase_source).float().to(device)

        channels, samples = wav_mag.shape
        final_audio = np.zeros((channels, samples))

        window = torch.hann_window(n_fft).to(device)

        # Calculate frequency bin indices
        max_bin = n_fft // 2 + 1
        freq_per_bin = sample_rate / n_fft
        low_bin = max(0, int(low_hz / freq_per_bin))
        high_bin = min(max_bin, int(high_hz / freq_per_bin))

        if high_bin <= low_bin:
            self.logger.warning(
                f"Phase Fix range invalid ({low_hz}-{high_hz}Hz -> bins {low_bin}-{high_bin}). Skipping phase correction."
            )
            return mag_source

        self.logger.info(
            f"Phase Fix: {low_hz}-{high_hz}Hz (bins {low_bin}-{high_bin}), weight={high_freq_weight}"
        )
        
        for c in range(channels):
            # STFT
            spec_mag = torch.stft(wav_mag[c], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            spec_phase = torch.stft(wav_phase[c], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            
            # Get magnitude and phase
            magnitude = torch.abs(spec_mag)
            angle_mag = torch.angle(spec_mag)
            angle_phase = torch.angle(spec_phase)
            
            # Create blended phase: use original outside range, blend inside range
            blended_angle = angle_mag.clone()
            
            # Apply phase from reference in the target frequency range
            # Weight > 1.0 means more aggressive replacement
            blend_factor = max(0.0, min(high_freq_weight, 1.0))  # Cap at 1.0 for blending
            
            # Blend phase in frequency range [low_bin:high_bin]
            blended_angle[low_bin:high_bin, :] = (
                (1.0 - blend_factor) * angle_mag[low_bin:high_bin, :] +
                blend_factor * angle_phase[low_bin:high_bin, :]
            )
            
            # Reconstruct spectrum
            new_spec = torch.polar(magnitude, blended_angle)
            
            # iSTFT
            out_wav = torch.istft(new_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=samples)
            
            final_audio[c, :] = out_wav.cpu().numpy()
            
        return final_audio

    # ==================== DEMUDDER METHODS ====================
    # Post-processing workflow to fix "spectral holes" in instrumentals
    # Best for: Metal, Rock, EDM. Less recommended for acoustic/light tracks.
    
    def demudder_phase_remix(
        self, 
        vocals: np.ndarray, 
        instrumental: np.ndarray, 
        original: np.ndarray,
        model_callback,
        high_pass_hz: int = 200
    ) -> np.ndarray:
        """
        Manual Demudder: Phase Remix workflow (pioneered by Aufr33).
        
        Steps:
        1. Invert vocals phase and mix with instrumental
        2. Apply high-pass filter (demudding not needed at low freqs)
        3. Re-separate the mixture
        4. Mix extracted vocals with original (without inversion)
        
        Args:
            vocals: Extracted vocal stem (channels, samples)
            instrumental: Extracted instrumental stem
            original: Original input audio
            model_callback: async function(audio) -> dict with 'vocals' key
            high_pass_hz: High-pass cutoff (100-250, or 900 for aggressive)
            
        Returns:
            Demudded instrumental
        """
        import asyncio
        
        self.logger.info(f"Starting Phase Remix demudder (high_pass={high_pass_hz}Hz)")
        
        # Ensure same length
        min_len = min(vocals.shape[-1], instrumental.shape[-1], original.shape[-1])
        vocals = vocals[..., :min_len]
        instrumental = instrumental[..., :min_len]
        original = original[..., :min_len]
        
        # Step 1: Invert vocals and mix with instrumental
        inverted_vocals = -vocals
        mixture = instrumental + inverted_vocals
        
        # Step 2: Apply high-pass filter (optional, protects bass)
        if high_pass_hz > 0:
            mixture = self._apply_high_pass(mixture, high_pass_hz)
        
        # Step 3: Re-separate (get extracted vocals from mixture)
        # This is async, caller must handle
        result = asyncio.get_event_loop().run_until_complete(model_callback(mixture))
        extracted_vocals = result.get('vocals', mixture)
        
        # Step 4: Mix with original (no inversion) = demudded instrumental
        demudded = original - extracted_vocals
        
        self.logger.info("Phase Remix demudder complete")
        return demudded

    def demudder_phase_rotate(self, audio: np.ndarray, model_callback) -> np.ndarray:
        """
        Demudder: Phase Rotate workflow (recommended as "fullest sounding").
        
        Steps:
        1. Swap Left/Right channels
        2. Shift phase by 90 degrees
        3. Invert against original mixture
        4. Second inference pass
        
        Note: May cause OOM on AMD/Intel 4GB GPUs even with small chunks.
        
        Args:
            audio: Input audio (channels, samples)
            model_callback: async function(audio) -> separation result
            
        Returns:
            Phase-rotated result
        """
        import asyncio
        
        self.logger.info("Starting Phase Rotate demudder")
        
        # Step 1: Swap L/R
        if audio.shape[0] >= 2:
            swapped = np.stack([audio[1], audio[0]], axis=0)
        else:
            swapped = audio
        
        # Step 2: 90° phase shift via Hilbert transform
        from scipy.signal import hilbert
        analytic = hilbert(swapped, axis=-1)
        phase_shifted = np.imag(analytic).astype(np.float32)
        
        # Step 3: Invert against original
        inverted = audio - phase_shifted
        
        # Step 4: Second inference 
        result = asyncio.get_event_loop().run_until_complete(model_callback(inverted))
        
        self.logger.info("Phase Rotate demudder complete")
        return result.get('instrumental', inverted)

    def demudder_combine(
        self, 
        original_inst: np.ndarray,
        phase_rotate_result: np.ndarray,
        phase_remix_result: np.ndarray,
        weights: tuple = (0.4, 0.3, 0.3)
    ) -> np.ndarray:
        """
        Demudder: Combine method.
        Weighted mix of original instrumental, Phase Rotate, and Phase Remix results.
        
        Args:
            original_inst: Original instrumental output
            phase_rotate_result: Output from demudder_phase_rotate
            phase_remix_result: Output from demudder_phase_remix
            weights: Tuple of weights (original, rotate, remix). Default balanced.
            
        Returns:
            Combined demudded instrumental
        """
        w_orig, w_rotate, w_remix = weights
        total = w_orig + w_rotate + w_remix
        
        # Normalize
        w_orig /= total
        w_rotate /= total
        w_remix /= total
        
        # Match lengths
        min_len = min(
            original_inst.shape[-1],
            phase_rotate_result.shape[-1],
            phase_remix_result.shape[-1]
        )
        
        combined = (
            original_inst[..., :min_len] * w_orig +
            phase_rotate_result[..., :min_len] * w_rotate +
            phase_remix_result[..., :min_len] * w_remix
        )
        
        self.logger.info(f"Demudder combine: weights=({w_orig:.2f}, {w_rotate:.2f}, {w_remix:.2f})")
        return combined

    def _apply_high_pass(self, audio: np.ndarray, cutoff_hz: int, sr: int = 44100) -> np.ndarray:
        """Apply high-pass filter to audio."""
        from scipy.signal import butter, sosfilt
        
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        sos = butter(4, normalized_cutoff, btype='high', output='sos')
        
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            filtered[ch] = sosfilt(sos, audio[ch])
        
        return filtered

