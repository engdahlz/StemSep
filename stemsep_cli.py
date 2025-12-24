import argparse
import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "StemSepApp" / "src"))

from models.model_manager import ModelManager
from core.separation_manager import SeparationManager
from core.logger import setup_logging

async def main():
    parser = argparse.ArgumentParser(description="StemSep CLI - Headless Audio Separation")
    
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("--model", required=True, help="Model ID or Preset Name (e.g. 'vocals_fast', 'bs-roformer-viperx-1297')")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (overrides --gpu)")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "flac"], help="Output format")
    parser.add_argument("--overlap", type=float, help="Overlap (0.1-0.9)")
    parser.add_argument("--segment_size", type=int, help="Segment size (e.g. 352800)")
    parser.add_argument("--shifts", type=int, default=1, help="Number of shifts (TTA)")
    parser.add_argument("--invert", action="store_true", help="Enable spectral inversion (Residual Mode)")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("StemSepCLI")
    
    # Determine device
    gpu_enabled = True
    if args.cpu:
        gpu_enabled = False
    elif not args.gpu:
        # Default to auto-detect, but SeparationManager defaults to True if not specified
        # We'll stick to True unless --cpu
        pass
        
    logger.info(f"Initializing StemSep CLI (GPU: {gpu_enabled})")
    
    # Initialize Managers
    model_manager = ModelManager()
    separation_manager = SeparationManager(model_manager, gpu_enabled=gpu_enabled)
    
    # Resolve Model/Preset
    model_id = args.model
    logger.info(f"Target Model/Preset: {model_id}")
    
    # Check if model exists or needs download
    # Note: SeparationManager handles presets internally, but doesn't auto-download in create_job.
    # The engine downloads if missing.
    
    try:
        # Prepare job arguments
        job_kwargs = {
            "file_path": args.input_file,
            "model_id": model_id,
            "output_dir": args.output,
            "device": "cuda" if gpu_enabled else "cpu",
            "shifts": args.shifts,
            "output_format": args.format,
            "tta": (args.shifts > 1),
            "invert": args.invert
        }
        
        # Only pass optional args if provided, to allow manager defaults
        if args.overlap is not None:
            job_kwargs["overlap"] = args.overlap
        if args.segment_size is not None:
            job_kwargs["segment_size"] = args.segment_size

        # Create Job
        job_id = separation_manager.create_job(**job_kwargs)
        
        logger.info(f"Job created: {job_id}")
        
        # Setup progress monitoring
        job = separation_manager.get_job(job_id)
        
        def progress_handler(progress, message, device=None):
            dev_str = f" [{device}]" if device else ""
            print(f"\rProgress: {progress:.1f}% - {message}{dev_str}", end="", flush=True)
            
        job.progress_callback = progress_handler
        
        # Start Job
        print(f"Starting separation of '{args.input_file}'...")
        success = separation_manager.start_job(job_id)
        
        if not success:
            logger.error("Failed to start job.")
            sys.exit(1)
            
        # Wait for completion
        # Since start_job runs in a thread pool, we need to wait here.
        # We can poll the job status.
        while job.status in ["pending", "queued", "running", "validating", "processing"]:
            await asyncio.sleep(0.5)
            
        print() # Newline after progress
        
        if job.status == "completed":
            logger.info("Separation completed successfully!")
            for stem, path in job.output_files.items():
                logger.info(f"  - {stem}: {path}")
        else:
            logger.error(f"Separation failed: {job.error}")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        separation_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
