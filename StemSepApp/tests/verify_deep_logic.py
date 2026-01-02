import sys
import os
import logging
import tempfile
import shutil
import unittest
from pathlib import Path

# Setup paths
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root / "StemSepApp" / "src"))

try:
    import soundfile  # noqa: F401
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False

if _HAS_SOUNDFILE:
    from core.separation_manager import SeparationManager
    from models.model_manager import ModelManager
else:  # pragma: no cover
    SeparationManager = None  # type: ignore
    ModelManager = None  # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StemSepTest')

@unittest.skipUnless(
    _HAS_SOUNDFILE,
    "Optional dependency 'soundfile' not installed; skipping SeparationManager logic tests",
)
class TestDeepLogic(unittest.TestCase):

    def setUp(self):
        self.model_manager = ModelManager()
        self.separation_manager = SeparationManager(self.model_manager, gpu_enabled=False)

    def test_01_create_job_arguments(self):
        """Verify create_job passes arguments correctly to create_ensemble_job."""
        print("\n--- Testing create_job Argument Passing ---")
        
        # We create a dummy ensemble config
        ensemble_config = [
            {"model_id": "unwa-inst-v1e", "weight": 1.0},
            {"model_id": "bs-roformer-viperx-1297", "weight": 1.0}
        ]
        
        # We mock the internal create_ensemble_job to trap arguments
        original_create_ensemble = self.separation_manager.create_ensemble_job
        
        captured_args = {}
        def mock_create_ensemble(*args, **kwargs):
            captured_args.update(kwargs)
            return "job_id_123"
            
        self.separation_manager.create_ensemble_job = mock_create_ensemble
        
        try:
            # Call create_job with specific problematic arguments
            self.separation_manager.create_job(
                file_path="test.wav",
                model_id="test_ensemble",
                output_dir="out",
                ensemble_config=ensemble_config,
                overlap=0.5, # Specific overlap
                segment_size=123456, # Specific size
                ensemble_algorithm="max_spec"
            )
            
            # Verify what reached the function
            self.assertEqual(captured_args.get('overlap'), 0.5, "Overlap argument mismatch!")
            self.assertEqual(captured_args.get('segment_size'), 123456, "Segment size argument mismatch!")
            self.assertEqual(captured_args.get('algorithm'), "max_spec", "Algorithm argument mismatch!")
            
            print("✅ create_job arguments passed correctly (No more overlap/algorithm swap)")
            
        finally:
            self.separation_manager.create_ensemble_job = original_create_ensemble

    def test_02_recommended_settings_defaulting_single_model(self):
        """Verify model recommended_settings apply when caller uses 'unset' defaults."""
        job_id = self.separation_manager.create_job(
            file_path="test.wav",
            model_id="bs-roformer-viperx-1297",
            output_dir="out",
            overlap=0.25,
            segment_size=0,
            batch_size=1,
            tta=False,
        )

        job = self.separation_manager.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(getattr(job, 'segment_size', None), 352800)
        # Registry uses overlap as integer divisor (e.g., 4)
        self.assertEqual(getattr(job, 'overlap', None), 4)
        # On CPU, we deliberately do not auto-increase batch_size
        self.assertEqual(getattr(job, 'batch_size', None), 1)
        self.assertEqual(getattr(job, 'tta', None), False)

if __name__ == '__main__':
    unittest.main()