import sys
import os
import logging
import unittest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.separation_engine import SeparationEngine

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    sf = None  # type: ignore
    _HAS_SOUNDFILE = False

try:
    import librosa  # noqa: F401
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UTF8Test")

class TestUTF8Separation(unittest.TestCase):

    @unittest.skipUnless(
        _HAS_SOUNDFILE and _HAS_LIBROSA,
        "Optional dependencies ('soundfile', 'librosa') not installed; skipping UTF-8 filename test",
    )
    def test_utf8_filename(self):
        """Verify that SeparationEngine can handle unicode filenames."""
        logger.info("Testing UTF-8 Filename Handling...")
        
        # Create dummy audio
        sr = 44100
        data = np.random.uniform(-0.5, 0.5, (sr * 2, 2)) # 2 seconds stereo
        
        # Unicode filename
        filename = "Tëst Fïle 🎵.wav"
        sf.write(filename, data, sr)
        
        engine = SeparationEngine(gpu_enabled=False)
        
        # Mock model loading to avoid huge downloads/inference
        # We inject a dummy model that returns input as output
        import torch
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                # Return (stems, channels, samples)
                # Input: (batch, channels, samples)
                # Output: (2, channels, samples)
                return torch.stack([x, x], dim=1).squeeze(0)
                
        async def mock_load(*args, **kwargs): return DummyModel()
        engine._load_model = mock_load
        engine._get_model_info = lambda *args: {"architecture": "Dummy", "stems": ["vocals", "instrumental"]}
        
        # Run separation
        try:
            outputs, device = asyncio.run(engine.separate(
                file_path=filename,
                model_id="dummy",
                output_dir="utf8_output",
                segment_size=44100*10 # Large enough to avoid chunks logic complexity
            ))
            
            logger.info(f"Separation successful. Outputs: {outputs}")
            self.assertTrue(os.path.exists(outputs["vocals"]))
            
        except Exception as e:
            self.fail(f"Separation failed with unicode filename: {e}")
        finally:
            # Cleanup
            if os.path.exists(filename): os.remove(filename)
            import shutil
            if os.path.exists("utf8_output"): shutil.rmtree("utf8_output")

import asyncio
if __name__ == '__main__':
    unittest.main()
