"""Fix Mel-Roformer model configs by extracting correct parameters from checkpoints."""
import torch
import yaml
from pathlib import Path

models_dir = Path.home() / '.stemsep/models'

for yaml_path in sorted(models_dir.glob('*.yaml')):
    model_id = yaml_path.stem
    ckpt_path = models_dir / f'{model_id}.ckpt'
    
    if not ckpt_path.exists():
        continue
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    
    # Count MLP layers in mask_estimators (look at first band, first mask_estimator)
    mlp_layers = sum(1 for k in state if 'mask_estimators.0.to_freqs.0.0' in k and k.endswith('.bias'))
    # depth = number of Linear layers
    actual_depth = mlp_layers
    
    # Check for final_norm
    has_final_norm = 'final_norm.gamma' in state
    
    # Load config
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_config = config.get('model', {})
    config_depth = model_config.get('mask_estimator_depth', 2)
    config_skip_norm = model_config.get('skip_final_norm', False)
    
    needs_update = False
    
    if config_depth != actual_depth:
        print(f'{model_id}: mask_estimator_depth {config_depth} -> {actual_depth}')
        model_config['mask_estimator_depth'] = actual_depth
        needs_update = True
    
    if has_final_norm and config_skip_norm:
        print(f'{model_id}: skip_final_norm True -> False (checkpoint has final_norm)')
        model_config['skip_final_norm'] = False
        needs_update = True
    elif not has_final_norm and not config_skip_norm:
        print(f'{model_id}: skip_final_norm False -> True (checkpoint lacks final_norm)')
        model_config['skip_final_norm'] = True
        needs_update = True
    
    if needs_update:
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f'  -> Updated {yaml_path.name}')

print("\nDone!")
