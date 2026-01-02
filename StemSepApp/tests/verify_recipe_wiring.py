import sys
import os
import logging
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StemSepRecipeWiringTest")


@unittest.skipUnless(
    _HAS_SOUNDFILE,
    "Optional dependency 'soundfile' not installed; skipping recipe wiring tests",
)
class TestRecipeWiring(unittest.TestCase):
    def setUp(self):
        self.model_manager = ModelManager()
        self.separation_manager = SeparationManager(self.model_manager, gpu_enabled=False)

    def test_01_demudder_recipe_forwards_post_processing(self):
        """Verify recipe post_processing is forwarded into create_ensemble_job."""

        captured = {}
        original = self.separation_manager.create_ensemble_job

        def mock_create_ensemble_job(*args, **kwargs):
            captured.update(kwargs)
            return "job_id_123"

        self.separation_manager.create_ensemble_job = mock_create_ensemble_job
        try:
            self.separation_manager.create_job(
                file_path="test.wav",
                model_id="recipe_demudder_metal",
                output_dir="out",
            )
            self.assertEqual(
                captured.get("post_processing"),
                "demudder_phase_rotate",
                "Demudder recipe should forward post_processing into create_ensemble_job",
            )
        finally:
            self.separation_manager.create_ensemble_job = original

    def test_02_chained_recipe_normalizes_steps_and_passes_output_settings(self):
        """Verify chained recipes execute via pipeline path with normalized schema."""

        captured = {}
        original = self.separation_manager.create_pipeline_job

        def mock_create_pipeline_job(*args, **kwargs):
            captured.update(kwargs)
            return "job_id_456"

        self.separation_manager.create_pipeline_job = mock_create_pipeline_job
        try:
            self.separation_manager.create_job(
                file_path="test.wav",
                model_id="chain_ultra_clean_vocal",
                output_dir="out",
                normalize=True,
                bit_depth="24",
            )

            pipeline_config = captured.get("pipeline_config")
            self.assertIsInstance(pipeline_config, list)
            self.assertGreaterEqual(len(pipeline_config), 3)

            step1, step2, step3 = pipeline_config[0], pipeline_config[1], pipeline_config[2]
            self.assertEqual(step1.get("step_name"), "isolation")
            self.assertNotIn("input_source", step1)
            self.assertEqual(step1.get("output"), "vocals")

            self.assertEqual(step2.get("step_name"), "dereverb")
            self.assertEqual(step2.get("input_source"), "isolation.vocals")
            self.assertEqual(step2.get("output"), "vocals")

            self.assertEqual(step3.get("step_name"), "lead_isolation")
            self.assertEqual(step3.get("input_source"), "dereverb.vocals")
            self.assertEqual(step3.get("output"), "lead_vocal")

            self.assertEqual(captured.get("normalize"), True)
            self.assertEqual(captured.get("bit_depth"), "24")
        finally:
            self.separation_manager.create_pipeline_job = original


if __name__ == "__main__":
    unittest.main()
