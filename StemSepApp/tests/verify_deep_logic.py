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

from models.model_factory import ModelFactory
from core.separation_manager import SeparationManager
from models.model_manager import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StemSepTest')

class TestDeepLogic(unittest.TestCase):
    
    def setUp(self):
        self.model_manager = ModelManager()
        # We don't need GPU for config checks, but we need it if we init engines.
        # Mocking GPU for logic verification is safer/faster.
        self.separation_manager = SeparationManager(self.model_manager, gpu_enabled=False)
        
    def test_01_model_factory_unwa_config(self):
        """Verify Unwa Inst V1e gets the correct Mel-Roformer config (60 bands)"""
        print("\n--- Testing Unwa Config ---")
        config = ModelFactory.get_model_config("unwa-inst-v1e")
        
        self.assertEqual(config.get('dim'), 384, "Unwa dim should be 384 (Verified via HuggingFace config)")
        self.assertEqual(config.get('dim_freqs_in'), 1025, "Unwa dim_freqs_in should be 1025 (Verified via HuggingFace config)")
        # Official YAML uses num_bands=60, not explicit freqs_per_bands list
        self.assertEqual(config.get('num_bands'), 60, "Unwa should have num_bands=60")
        print("✅ Unwa config looks correct (Mel-Roformer 60 bands)")

    def test_02_model_factory_viperx_config(self):
        """Verify ViperX gets correct BS-Roformer config"""
        print("\n--- Testing ViperX Config ---")
        config = ModelFactory.get_model_config("bs-roformer-viperx-1297")
        
        self.assertEqual(config.get('dim'), 512, "ViperX dim should be 512")
        self.assertEqual(config.get('dim_freqs_in'), 1025, "ViperX dim_freqs_in should be 1025 (BS)")
        print("✅ ViperX config looks correct")

    def test_03_model_factory_vr_config(self):
        """Verify VR models get VR config"""
        print("\n--- Testing VR Config ---")
        config = ModelFactory.get_model_config("5_hp-karaoke-uvr")
        self.assertEqual(config.get('dim'), 32, "VR dim should be 32")
        
        # Test instantiation (to verify imports work)
        try:
            model = ModelFactory.create_model("VR", config)
            print("✅ VR Model instantiated successfully (Imports work)")
        except Exception as e:
            self.fail(f"Failed to instantiate VR model: {e}")

    def test_04_create_job_arguments(self):
        """Verify create_job passes arguments correctly to create_ensemble_job"""
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

if __name__ == '__main__':
    unittest.main()