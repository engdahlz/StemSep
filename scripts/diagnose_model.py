"""
Diagnostic script to compare checkpoint keys with model architecture.
This helps identify why certain models fail to separate properly.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "StemSepApp" / "src"))

# Use official ZFTurbo implementations
from models.architectures.zfturbo_bs_roformer import BSRoformer, MelBandRoformer

def load_config(config_path: Path) -> dict:
    """Load YAML config with Python tuple support."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def analyze_checkpoint(model_id: str, models_dir: Path):
    """Analyze checkpoint vs model architecture."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_id}")
    print('='*60)
    
    # Find files
    config_path = models_dir / f"{model_id}.yaml"
    checkpoint_path = models_dir / f"{model_id}.ckpt"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load config
    print(f"\n📄 Loading config from: {config_path}")
    config = load_config(config_path)
    model_config = config.get('model', {})
    
    # Show key config params
    print(f"\n📋 Model Config:")
    for key in ['dim', 'depth', 'stft_n_fft', 'mask_estimator_depth', 'skip_final_norm', 'num_bands']:
        if key in model_config:
            print(f"   {key}: {model_config[key]}")
    
    # Determine which architecture to use based on config
    use_mel = 'num_bands' in model_config
    ModelClass = MelBandRoformer if use_mel else BSRoformer
    arch_name = "MelBandRoformer" if use_mel else "BSRoformer"
    
    # Build model architecture
    print(f"\n🏗️  Building model architecture ({arch_name})...")
    try:
        # Get valid init params for the selected class
        import inspect
        valid_params = set(inspect.signature(ModelClass.__init__).parameters.keys())
        valid_params.discard('self')
        
        # Filter config to only valid params
        filtered_config = {k: v for k, v in model_config.items() if k in valid_params}
        
        # Show what we're filtering out
        ignored = set(model_config.keys()) - valid_params
        if ignored:
            print(f"   ⚠️ Ignoring unknown params: {ignored}")
        
        model = ModelClass(**filtered_config)
        model_state = model.state_dict()
        print(f"   Model has {len(model_state)} parameters")
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            ckpt_state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            ckpt_state = checkpoint['model_state_dict']
        elif 'state' in checkpoint:
            ckpt_state = checkpoint['state']
        else:
            ckpt_state = checkpoint
        print(f"   Checkpoint has {len(ckpt_state)} parameters")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Compare keys
    print(f"\n🔍 Comparing keys...")
    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    
    matching = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys
    
    print(f"\n📊 Results:")
    print(f"   ✅ Matching keys: {len(matching)}")
    print(f"   ❌ Missing in checkpoint: {len(missing_in_ckpt)}")
    print(f"   ⚠️  Extra in checkpoint (ignored): {len(extra_in_ckpt)}")
    
    # Check shape mismatches
    shape_mismatches = []
    for key in matching:
        model_shape = model_state[key].shape
        ckpt_shape = ckpt_state[key].shape
        if model_shape != ckpt_shape:
            shape_mismatches.append((key, model_shape, ckpt_shape))
    
    if shape_mismatches:
        print(f"\n🔴 SHAPE MISMATCHES ({len(shape_mismatches)}):")
        for key, m_shape, c_shape in shape_mismatches[:10]:  # Show first 10
            print(f"   {key}:")
            print(f"      Model:      {m_shape}")
            print(f"      Checkpoint: {c_shape}")
        if len(shape_mismatches) > 10:
            print(f"   ... and {len(shape_mismatches) - 10} more")
    else:
        print(f"\n✅ No shape mismatches!")
    
    # Show missing keys
    if missing_in_ckpt:
        print(f"\n🔴 MISSING IN CHECKPOINT:")
        for key in sorted(missing_in_ckpt)[:10]:
            print(f"   - {key}")
        if len(missing_in_ckpt) > 10:
            print(f"   ... and {len(missing_in_ckpt) - 10} more")
    
    # Calculate match percentage
    total_model_params = len(model_keys)
    loadable = len(matching) - len(shape_mismatches)
    match_pct = (loadable / total_model_params) * 100 if total_model_params > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {loadable}/{total_model_params} params loadable ({match_pct:.1f}%)")
    if match_pct < 90:
        print(f"⚠️  LOW MATCH RATE - Model likely to fail!")
    elif match_pct < 100:
        print(f"⚠️  Some params missing - May affect quality")
    else:
        print(f"✅ Full match - Should work correctly")
    print('='*60)
    
    return {
        'matching': len(matching),
        'missing': len(missing_in_ckpt),
        'extra': len(extra_in_ckpt),
        'shape_mismatches': len(shape_mismatches),
        'match_pct': match_pct
    }

if __name__ == "__main__":
    models_dir = Path.home() / ".stemsep" / "models"
    
    # Find all model configs
    configs = list(models_dir.glob("*.yaml"))
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS - {len(configs)} models found")
    print('='*60)
    
    results = {}
    for config_path in sorted(configs):
        model_id = config_path.stem
        ckpt_path = models_dir / f"{model_id}.ckpt"
        if ckpt_path.exists():
            result = analyze_checkpoint(model_id, models_dir)
            if result:
                results[model_id] = result
        else:
            print(f"\n⚠️  {model_id}: No checkpoint file found")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for model_id, r in sorted(results.items()):
        status = "✅" if r['match_pct'] >= 99 else "⚠️" if r['match_pct'] >= 90 else "❌"
        print(f"{status} {model_id}: {r['match_pct']:.1f}% match")
