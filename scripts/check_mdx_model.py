"""Check MDX model hash and lookup config from UVR model_data.json"""
import json
import hashlib
from pathlib import Path

# Load UVR config
mdx_data_path = Path("StemSepApp/assets/mdx_model_data.json")
with open(mdx_data_path) as f:
    mdx_data = json.load(f)

print(f"Loaded {len(mdx_data)} model configs from UVR model_data.json")

# Check our model
model_path = Path.home() / ".stemsep/models/uvr-mdx-net-inst-hq-5.onnx"
if model_path.exists():
    print(f"\nModel file: {model_path}")
    print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Compute MD5
    with open(model_path, 'rb') as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
    print(f"MD5 hash: {md5_hash}")
    
    # Lookup in UVR data
    if md5_hash in mdx_data:
        config = mdx_data[md5_hash]
        print(f"\n✅ FOUND in UVR model_data.json!")
        print(f"   Config: {json.dumps(config, indent=4)}")
    else:
        print(f"\n❌ NOT FOUND in UVR model_data.json")
        print(f"\nSearching for similar instrumental models...")
        for h, cfg in mdx_data.items():
            if cfg.get('primary_stem') == 'Instrumental' and 'mdx_dim_f_set' in cfg:
                if cfg.get('mdx_dim_f_set') == 2048:
                    print(f"  {h}: dim_f={cfg.get('mdx_dim_f_set')}, n_fft={cfg.get('mdx_n_fft_scale_set')}")
else:
    print(f"Model not found: {model_path}")
