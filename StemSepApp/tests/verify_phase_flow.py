import sys
import os
import logging
import unittest
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.ensemble_blender import EnsembleBlender

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhaseFlowTest")

class TestPhaseFlow(unittest.TestCase):

    def test_phase_fix_math(self):
        """Verify that Phase Fix blends Magnitude from A and Phase from B."""
        logger.info("Testing Phase Fix Math...")
        
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Signal A: Target Magnitude (Amp 1.0), Wrong Phase (shifted pi/2)
        # cos(x) = sin(x + pi/2). So using cos simulates phase shift.
        sig_a = 1.0 * np.cos(2 * np.pi * 440 * t) 
        
        # Signal B: Reference Phase (Amp 0.5), Correct Phase (sin)
        sig_b = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Prepare input (2 channels stereo)
        input_a = np.stack([sig_a, sig_a])
        input_b = np.stack([sig_b, sig_b])
        
        blender = EnsembleBlender()
        
        if not torch.cuda.is_available():
            logger.warning("Running on CPU")
            
        # Run Blend
        output = blender._phase_fix_blend(input_a, input_b, low_hz=0)
        
        # Analysis
        # We check a small segment in the middle to avoid boundary effects
        mid = int(len(t) / 2)
        segment = output[0, mid:mid+100]
        ref_sin = np.sin(2 * np.pi * 440 * t)[mid:mid+100]
        
        # Correlation with Sine (Phase Ref)
        # If phase is correct, correlation should be high (1.0 or -1.0)
        # If phase is Cosine (Source A), correlation with Sine would be 0.
        corr_phase = np.corrcoef(segment, ref_sin)[0, 1]
        logger.info(f"Correlation with Reference Phase (Sine): {corr_phase:.4f}")
        
        # Amplitude check
        # RMS of output should match RMS of Source A (1.0) not Source B (0.5)
        rms_out = np.sqrt(np.mean(segment**2))
        rms_a = np.sqrt(np.mean((1.0 * np.cos(2 * np.pi * 440 * t)[mid:mid+100])**2))
        
        logger.info(f"RMS Output: {rms_out:.4f}")
        logger.info(f"RMS Target (Source A): {rms_a:.4f}")
        
        # Assertions
        self.assertGreater(abs(corr_phase), 0.9, "Output phase does not match reference phase")
        self.assertAlmostEqual(rms_out, rms_a, delta=0.1, msg="Output magnitude does not match target magnitude")

if __name__ == '__main__':
    unittest.main()
