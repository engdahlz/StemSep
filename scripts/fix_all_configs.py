# coding: utf-8
"""
Batch extract correct freqs_per_bands from all broken model checkpoints.
This script extracts the configuration from the checkpoint weights and
generates corrected YAML files.

Note: After running, use scripts/dev/diagnose_model.py to verify config/weights alignment.
"""

import os
import sys
from pathlib import Path

# Add StemSepApp to path
sys.path.insert(0, str(Path(__file__).parent / "StemSepApp" / "src"))

import torch
import yaml

MODELS_DIR = Path.home() / ".stemsep" / "models"

# Models that need fixing (from diagnostic report)
BROKEN_MODELS = [
    "becruily-guitar",
    "becruily-vocal",
    "gabox-inst-fv7z",
    "gabox-inst-fv8b",
    "mel-band-roformer-kim",
    "mel-roformer-debleed",
    "unwa-inst-fno",
    "unwa-inst-v1e",
]


def extract_freqs_from_checkpoint(ckpt_path):
    """Extract freqs_per_bands from checkpoint weight shapes."""
    print(f"\n{'=' * 60}")
    print(f"Extracting from: {ckpt_path.name}")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Get state dict
    state_dict = ckpt
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "state" in ckpt:
        state_dict = ckpt["state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]

    # Extract freqs_per_bands from band_split.to_features weights
    freqs_per_bands = []
    band_idx = 0

    while True:
        gamma_key = f"band_split.to_features.{band_idx}.0.gamma"
        if gamma_key in state_dict:
            freq_count = state_dict[gamma_key].shape[0]
            freqs_per_bands.append(freq_count)
            band_idx += 1
        else:
            break

    if not freqs_per_bands:
        print(
            "  ⚠️ Could not find band_split.to_features - might be different architecture"
        )
        return None

    total_freqs = sum(freqs_per_bands)
    num_bands = len(freqs_per_bands)

    print(f"  Found {num_bands} bands")
    print(f"  Total frequencies (dim_freqs_in): {total_freqs}")

    # Extract other key parameters
    config = {
        "freqs_per_bands": freqs_per_bands,
        "dim_freqs_in": total_freqs,
        "num_bands": num_bands,
    }

    # Try to determine model dimension from transformer weights
    for key in state_dict:
        if "time_transformer" in key and "to_qkv" in key and key.endswith(".weight"):
            # Shape is typically (dim*3, dim)
            dim = state_dict[key].shape[1]
            config["dim"] = dim
            print(f"  Detected dim: {dim}")
            break

    # Check for mask_estimator_depth
    max_depth = 0
    for key in state_dict:
        if "mask_estimators" in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "mask_estimators" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        max_depth = max(max_depth, idx + 1)
                    except ValueError:
                        pass
    if max_depth > 0:
        config["mask_estimator_depth"] = max_depth
        print(f"  Detected mask_estimator_depth: {max_depth}")

    # Check for skip_final_norm
    if "final_norm.gamma" not in state_dict:
        config["skip_final_norm"] = True
        print(f"  skip_final_norm: True (no final_norm in checkpoint)")

    # Calculate stft_n_fft from dim_freqs_in
    # Formula: n_fft // 2 + 1 = dim_freqs_in
    # So: n_fft = (dim_freqs_in - 1) * 2
    stft_n_fft = (total_freqs - 1) * 2
    config["stft_n_fft"] = stft_n_fft
    config["stft_win_length"] = stft_n_fft
    print(f"  Calculated stft_n_fft: {stft_n_fft}")

    return config


def update_yaml_with_config(yaml_path, extracted_config):
    """Update existing YAML file with extracted configuration."""
    print(f"\n  Updating: {yaml_path}")

    if not yaml_path.exists():
        print(f"  ⚠️ YAML file not found: {yaml_path}")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        # Use unsafe_load to handle !!python/tuple tags in existing files
        yaml_content = yaml.unsafe_load(f) or {}

    # Get or create model section
    model_config = yaml_content.get("model", {})

    # Update with extracted values
    updates = []

    if "dim_freqs_in" in extracted_config:
        old = model_config.get("dim_freqs_in")
        model_config["dim_freqs_in"] = extracted_config["dim_freqs_in"]
        if old != extracted_config["dim_freqs_in"]:
            updates.append(f"dim_freqs_in: {old} → {extracted_config['dim_freqs_in']}")

    if "stft_n_fft" in extracted_config:
        old = model_config.get("stft_n_fft")
        model_config["stft_n_fft"] = extracted_config["stft_n_fft"]
        model_config["stft_win_length"] = extracted_config["stft_win_length"]
        if old != extracted_config["stft_n_fft"]:
            updates.append(f"stft_n_fft: {old} → {extracted_config['stft_n_fft']}")

    if "mask_estimator_depth" in extracted_config:
        old = model_config.get("mask_estimator_depth")
        model_config["mask_estimator_depth"] = extracted_config["mask_estimator_depth"]
        if old != extracted_config["mask_estimator_depth"]:
            updates.append(
                f"mask_estimator_depth: {old} → {extracted_config['mask_estimator_depth']}"
            )

    if "skip_final_norm" in extracted_config:
        old = model_config.get("skip_final_norm")
        model_config["skip_final_norm"] = extracted_config["skip_final_norm"]
        if old != extracted_config["skip_final_norm"]:
            updates.append(
                f"skip_final_norm: {old} → {extracted_config['skip_final_norm']}"
            )

    if "freqs_per_bands" in extracted_config:
        model_config["freqs_per_bands"] = tuple(extracted_config["freqs_per_bands"])
        updates.append(
            f"freqs_per_bands: {len(extracted_config['freqs_per_bands'])} bands"
        )

    yaml_content["model"] = model_config

    # Write back
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    if updates:
        for u in updates:
            print(f"    ✓ {u}")
        return True
    else:
        print("    (no changes needed)")
        return False


def main():
    print("=" * 60)
    print("BATCH CONFIG FIXER FOR BROKEN MODELS")
    print("=" * 60)

    fixed_count = 0
    failed_count = 0

    for model_id in BROKEN_MODELS:
        # Find checkpoint file
        ckpt_patterns = [
            MODELS_DIR / f"{model_id}.ckpt",
            MODELS_DIR / f"{model_id}.th",
            MODELS_DIR / f"{model_id}.pth",
        ]

        ckpt_path = None
        for pattern in ckpt_patterns:
            if pattern.exists():
                ckpt_path = pattern
                break

        if not ckpt_path:
            print(f"\n⚠️ Checkpoint not found for: {model_id}")
            failed_count += 1
            continue

        # Extract config from checkpoint
        extracted = extract_freqs_from_checkpoint(ckpt_path)

        if not extracted:
            print(f"  ⚠️ Could not extract config from: {model_id}")
            failed_count += 1
            continue

        # Update YAML
        yaml_path = MODELS_DIR / f"{model_id}.yaml"
        if update_yaml_with_config(yaml_path, extracted):
            fixed_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Fixed: {fixed_count} models")
    print(f"⚠️ Failed/Skipped: {failed_count} models")

    if fixed_count > 0:
        print("\nRun scripts/dev/diagnose_model.py again to verify config match!")


if __name__ == "__main__":
    main()
