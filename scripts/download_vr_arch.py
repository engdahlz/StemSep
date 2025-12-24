"""Download official VR architecture from tsurumeso/vocal-remover."""
import requests
from pathlib import Path

base_url = 'https://raw.githubusercontent.com/tsurumeso/vocal-remover/main/'

files_to_download = [
    ('lib/nets.py', 'vr_nets.py'),
    ('lib/layers.py', 'vr_layers.py'),
    ('lib/spec_utils.py', 'vr_spec_utils.py'),
]

out_dir = Path('StemSepApp/src/models/architectures/zfturbo_vr')
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
init_content = '''"""Official tsurumeso VR architecture implementations."""
from .vr_nets import CascadedNet
from .vr_layers import Conv2DBNActiv, Encoder, ASPPModule
'''
(out_dir / '__init__.py').write_text(init_content)
print('Created __init__.py')
print('Done!')
