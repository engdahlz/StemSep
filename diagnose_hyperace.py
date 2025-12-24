import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("StemSepApp/src").absolute()))

from models.model_factory import ModelFactory
from models.model_manager import ModelManager

def diagnose():
    print("Diagnosing HyperACE model loading...")
    
    # 1. Get config
    model_id = "unwa-hyperace"
    print(f"Fetching config for {model_id}...")
    
    # Manually constructing config as ModelFactory would
    # Using the exact same config block I added to ModelFactory
    config = {
        "architecture": "BS-Roformer-HyperACE",
        "dim": 256, "depth": 12, "stereo": True, "num_stems": 1, "time_transformer_depth": 1, "freq_transformer_depth": 1, "linear_transformer_depth": 0,
        "freqs_per_bands": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 128, 129),
        "dim_head": 64, "heads": 8, "attn_dropout": 0.0, "ff_dropout": 0.0, "flash_attn": True, "dim_freqs_in": 1025,
        "stft_n_fft": 2048, "stft_hop_length": 512, "stft_win_length": 2048, "stft_normalized": False,
        "mask_estimator_depth": 2, "multi_stft_resolution_loss_weight": 1.0, "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256), "multi_stft_hop_size": 147, "multi_stft_normalized": False,
        "mlp_expansion_factor": 4, "use_torch_checkpoint": True, "skip_connection": False,
        "target_name": "instrumental"
    }

    # 2. Instantiate Model
    print("Instantiating model architecture...")
    try:
        model = ModelFactory.create_model("BS-Roformer-HyperACE", config)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # 3. Load Checkpoint
    # Use the custom path seen in logs
    manager = ModelManager(models_dir=Path(r"D:\StemSep Models"))
    if not manager.is_model_installed(model_id):
        print(f"Model {model_id} not installed. Please download it first.")
        return
        
    ckpt_path = manager.installed_models[model_id].file_path
    print(f"Loading checkpoint from: {ckpt_path}")
    
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        print(f"Checkpoint keys: {len(state_dict)}")
        print(f"Model keys: {len(model.state_dict())}")
        
        # 4. Compare Keys
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        missing_in_model = ckpt_keys - model_keys
        missing_in_ckpt = model_keys - ckpt_keys
        
        print("\n--- ANALYSIS ---")
        if not missing_in_model and not missing_in_ckpt:
            print("PERFECT MATCH! Keys align exactly.")
        else:
            print(f"Keys in Checkpoint but NOT in Model (Unexpected): {len(missing_in_model)}")
            if len(missing_in_model) > 0:
                print("Examples:", list(missing_in_model)[:5])
                
            print(f"Keys in Model but NOT in Checkpoint (Missing/Unloaded): {len(missing_in_ckpt)}")
            if len(missing_in_ckpt) > 0:
                print("Examples:", list(missing_in_ckpt)[:5])
                
            # Attempt fuzzy matching diagnosis
            print("\n--- FUZZY MATCH DIAGNOSIS ---")
            # Check for prefix issues (e.g. mask_estimators.0 vs model.mask_estimators.0)
            sample_ckpt = list(ckpt_keys)[0]
            sample_model = list(model_keys)[0]
            print(f"Sample Checkpoint Key: {sample_ckpt}")
            print(f"Sample Model Key:      {sample_model}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    diagnose()
