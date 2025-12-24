"""
Test script for ensemble separation with per-stem algorithms.
Tests the new stem_algorithms and phase_fix_params features.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "StemSepApp" / "src"))

from models.model_manager import ModelManager
from core.separation_manager import SeparationManager
from core.logger import setup_logging

async def test_ensemble_separation():
    """Test ensemble separation with per-stem algorithms."""
    setup_logging()
    logger = logging.getLogger("EnsembleTest")
    
    # Audio file to test
    input_file = r"c:\Users\engdahlz\StemSep-V3\Good Morning #newmusic #singersongwriter #jakeminch #music #george #shorts [TubeRipper.cc].mp3"
    output_dir = r"c:\Users\engdahlz\StemSep-V3\test_ensemble_output"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize Managers - use correct models directory
    logger.info("Initializing ModelManager and SeparationManager...")
    models_dir = Path(r"D:\StemSep Models")  # Where models are actually installed
    model_manager = ModelManager(models_dir=models_dir)
    separation_manager = SeparationManager(model_manager, gpu_enabled=True)
    
    # Define ensemble config with 2 models
    # Using models that should be available
    ensemble_config = [
        {"model_id": "bs-roformer-viperx-1297", "weight": 1.0},  # Good for vocals
        {"model_id": "mel-band-roformer-kim", "weight": 1.0}     # Good for instrumentals
    ]
    
    # Per-stem algorithm configuration (new feature!)
    stem_algorithms = {
        "vocals": "max_spec",       # Full detail on vocals
        "instrumental": "min_spec"  # Bleedless instrumental
    }
    
    logger.info("=" * 60)
    logger.info("ENSEMBLE TEST: Per-Stem Algorithms")
    logger.info("=" * 60)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Models: {[m['model_id'] for m in ensemble_config]}")
    logger.info(f"Stem Algorithms: {stem_algorithms}")
    logger.info("=" * 60)
    
    try:
        # Create ensemble job with per-stem algorithms
        job_id = separation_manager.create_ensemble_job(
            file_path=input_file,
            ensemble_config=ensemble_config,
            output_dir=output_dir,
            stems=["vocals", "instrumental"],
            algorithm="average",  # Global fallback
            stem_algorithms=stem_algorithms,  # NEW FEATURE!
            output_format="wav",
            normalize=True
        )
        
        logger.info(f"Ensemble job created: {job_id}")
        
        # Get job and set progress callback
        job = separation_manager.get_job(job_id)
        
        def progress_handler(progress, message, device=None):
            dev_str = f" [{device}]" if device else ""
            print(f"\rProgress: {progress:.1f}% - {message}{dev_str}", end="", flush=True)
        
        job.progress_callback = progress_handler
        
        # Start job
        logger.info("Starting ensemble separation...")
        success = separation_manager.start_job(job_id)
        
        if not success:
            logger.error("Failed to start job!")
            return False
        
        # Wait for completion - poll for status changes
        # Include more statuses and add timeout
        waiting_statuses = ["pending", "queued", "running", "validating", "processing", "blending"]
        timeout = 120  # 2 min timeout
        start_wait = asyncio.get_event_loop().time()
        while job.status in waiting_statuses:
            await asyncio.sleep(0.5)  # Poll faster
            if asyncio.get_event_loop().time() - start_wait > timeout:
                logger.error(f"Timeout waiting for job completion! Status: {job.status}")
                break
        # Extra wait to ensure job thread has finished
        logger.info(f"Polling ended. Current status: {job.status}. Waiting 5s for job thread...")
        await asyncio.sleep(5)
        
        print()  # Newline after progress
        
        # Debug: log the final job status
        logger.info(f"Final job status: {job.status}")
        logger.info(f"Final job error: {job.error}")
        logger.info(f"Final job output_files: {job.output_files}")
        
        if job.status == "completed":
            logger.info("=" * 60)
            logger.info("ENSEMBLE SEPARATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            for stem, path in job.output_files.items():
                logger.info(f"  ✓ {stem}: {path}")
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"Ensemble separation failed: {job.error}")
            return False
            
    except Exception as e:
        import traceback
        logger.critical(f"Test failed with error: {e}")
        logger.critical(f"Full traceback:\n{traceback.format_exc()}")
        return False
    finally:
        separation_manager.shutdown()

if __name__ == "__main__":
    result = asyncio.run(test_ensemble_separation())
    sys.exit(0 if result else 1)
