"""Download official ZFTurbo architectures (MDX23C, SCNet)."""
import requests
from pathlib import Path

base_url = 'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/'

files_to_download = [
    ('models/mdx23c_tfc_tdf_v3.py', 'mdx23c.py'),
    ('models/scnet/scnet.py', 'scnet.py'),
    ('models/scnet/separation.py', 'scnet_separation.py'),
]

out_dir = Path('StemSepApp/src/models/architectures/zfturbo_models')
out_dir.mkdir(exist_ok=True)

for src_path, dest_name in files_to_download:
    url = base_url + src_path
    resp = requests.get(url)
    if resp.status_code == 200:
        (out_dir / dest_name).write_text(resp.text, encoding='utf-8')
        print(f'Downloaded: {src_path} -> {dest_name}')
    else:
        print(f'Failed ({resp.status_code}): {src_path}')

# Create __init__.py
init_content = '''"""Official ZFTurbo model implementations."""
# MDX23C uses TFC_TDF_net class internally
from .mdx23c import TFC_TDF_net
from .scnet import SCNet
'''
(out_dir / '__init__.py').write_text(init_content)
print('Created __init__.py')
print('Done!')
