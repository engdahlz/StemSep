#!/usr/bin/env python3
"""
Script to add recommended_settings to all model JSON files.
Based on research from model YAML configs.
"""
import json
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "assets" / "models"

# Settings by architecture and model family
SETTINGS = {
    # Gabox/Unwa Mel-Roformers use larger chunk size
    "gabox_mel": {"segment_size": 485100, "overlap": 0.5},
    "unwa_mel": {"segment_size": 485100, "overlap": 0.5},
    "amane_mel": {"segment_size": 485100, "overlap": 0.5},
    "mesk_mel": {"segment_size": 485100, "overlap": 0.5},
    
    # Standard Mel-Roformers (Kim, Aufr33, Becruily standalone)
    "mel_standard": {"segment_size": 352800, "overlap": 0.5},
    
    # BS-Roformer
    "bs_roformer": {"segment_size": 352800, "overlap": 0.5},
    
    # MDX23C
    "mdx23c": {"segment_size": 352800, "overlap": 0.5},
    
    # SCNet
    "scnet": {"segment_size": 352800, "overlap": 0.5},
    
    # VR/UVR models (different segment approach)
    "vr_uvr": {"segment_size": 256, "overlap": 0.5},
    
    # MDX-Net (ONNX)
    "mdx_net": {"segment_size": 256, "overlap": 0.5},
}

def get_settings_for_model(model_id: str, architecture: str) -> dict:
    """Determine settings based on model ID and architecture."""
    model_lower = model_id.lower()
    arch_lower = architecture.lower() if architecture else ""
    
    # HTDemucs uses internal chunking - no settings needed
    if "demucs" in model_lower or "demucs" in arch_lower:
        return None
    
    # Gabox models
    if model_lower.startswith("gabox"):
        return SETTINGS["gabox_mel"]
    
    # Unwa models
    if model_lower.startswith("unwa"):
        return SETTINGS["unwa_mel"]
    
    # Amane models
    if model_lower.startswith("amane"):
        return SETTINGS["amane_mel"]
    
    # Mesk models
    if model_lower.startswith("mesk"):
        return SETTINGS["mesk_mel"]
    
    # BS-Roformer
    if "bs-roformer" in model_lower or arch_lower == "bs-roformer":
        return SETTINGS["bs_roformer"]
    
    # Mel-Roformer (Becruily, Aufr33, Kim, other standalone)
    if "mel" in model_lower or "mel-roformer" in arch_lower:
        return SETTINGS["mel_standard"]
    
    # Becruily standalone models
    if model_lower.startswith("becruily"):
        return SETTINGS["mel_standard"]
    
    # Anvuew models
    if model_lower.startswith("anvuew"):
        return SETTINGS["mel_standard"]
    
    # Aufr33 models
    if model_lower.startswith("aufr33"):
        return SETTINGS["mel_standard"]
    
    # Sucial models (VR architecture)
    if model_lower.startswith("sucial"):
        return SETTINGS["vr_uvr"]
    
    # UVR models
    if model_lower.startswith("uvr") or model_lower.startswith("5_hp"):
        return SETTINGS["vr_uvr"]
    
    # MDX23C
    if "mdx23c" in model_lower or arch_lower == "mdx23c":
        return SETTINGS["mdx23c"]
    
    # SCNet
    if "scnet" in model_lower or arch_lower == "scnet":
        return SETTINGS["scnet"]
    
    # Kim Vocal (MDX-Net ONNX)
    if model_lower.startswith("kim") and "mdx" in arch_lower:
        return SETTINGS["mdx_net"]
    
    # Reverb-HQ
    if "reverb" in model_lower:
        return SETTINGS["mel_standard"]
    
    # Default fallback
    print(f"  ! Using default for {model_id} ({architecture})")
    return SETTINGS["mel_standard"]


def update_model_json(json_path: Path) -> bool:
    """Add recommended_settings to a model JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_id = data.get("id", json_path.stem)
        architecture = data.get("architecture", "")
        
        settings = get_settings_for_model(model_id, architecture)
        
        if settings is None:
            print(f"  - Skipping {model_id} (internal chunking)")
            return False
        
        # Add recommended_settings if not present or update if exists
        if "recommended_settings" not in data:
            data["recommended_settings"] = settings
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"  + Added settings to {model_id}: {settings}")
            return True
        else:
            print(f"  = {model_id} already has recommended_settings")
            return False
            
    except Exception as e:
        print(f"  ! Error processing {json_path}: {e}")
        return False


def main():
    print(f"Updating model JSONs in: {MODELS_DIR}")
    print("-" * 50)
    
    updated = 0
    skipped = 0
    errors = 0
    
    for json_file in sorted(MODELS_DIR.glob("*.json")):
        result = update_model_json(json_file)
        if result:
            updated += 1
        elif result is False:
            skipped += 1
    
    print("-" * 50)
    print(f"Done! Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
