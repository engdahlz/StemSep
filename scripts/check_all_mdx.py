"""Check all MDX models against UVR model_data.json"""
import json
import hashlib
from pathlib import Path

# Load UVR config
mdx_data_path = Path("StemSepApp/assets/mdx_model_data.json")
with open(mdx_data_path) as f:
    mdx_data = json.load(f)

# Find all MDX model definitions
models_dir = Path("StemSepApp/assets/models")
mdx_models = list(models_dir.glob("*mdx*.json")) + list(models_dir.glob("*MDX*.json"))

print(f"Found {len(mdx_models)} MDX model definitions\n")

for model_json in sorted(mdx_models):
    with open(model_json) as f:
        model_def = json.load(f)
    
    model_id = model_def.get('id', model_json.stem)
    model_name = model_def.get('name', 'Unknown')
    checkpoint_url = model_def.get('links', {}).get('checkpoint', '')
    
    # Check if downloaded
    local_file = Path.home() / ".stemsep/models" / f"{model_id}.onnx"
    
    print(f"📦 {model_id}")
    print(f"   Name: {model_name}")
    print(f"   URL: {checkpoint_url[:60]}..." if len(checkpoint_url) > 60 else f"   URL: {checkpoint_url}")
    
    if local_file.exists():
        md5 = hashlib.md5(open(local_file, 'rb').read()).hexdigest()
        print(f"   Local: ✅ Exists ({local_file.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"   MD5: {md5}")
        
        if md5 in mdx_data:
            cfg = mdx_data[md5]
            print(f"   UVR Config: ✅ FOUND - dim_f={cfg.get('mdx_dim_f_set')}, n_fft={cfg.get('mdx_n_fft_scale_set')}")
        else:
            print(f"   UVR Config: ❌ NOT FOUND")
    else:
        print(f"   Local: ❌ Not downloaded")
    print()
