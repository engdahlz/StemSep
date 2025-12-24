"""Restore all YAML configs by downloading from official sources."""
import json
import requests
from pathlib import Path

assets_dir = Path('StemSepApp/assets/models')
models_dir = Path.home() / '.stemsep/models'

for json_file in assets_dir.glob('*.json'):
    with open(json_file, 'r') as f:
        model = json.load(f)
    
    config_url = model.get('links', {}).get('config')
    if not config_url:
        continue
    
    model_id = model['id']
    yaml_path = models_dir / f'{model_id}.yaml'
    
    # Download fresh copy from source
    try:
        resp = requests.get(config_url, timeout=10)
        if resp.status_code == 200:
            yaml_path.write_text(resp.text, encoding='utf-8')
            print(f'Downloaded: {model_id}')
        else:
            print(f'Failed ({resp.status_code}): {model_id}')
    except Exception as e:
        print(f'Error: {model_id} - {e}')

print('\nDone!')
