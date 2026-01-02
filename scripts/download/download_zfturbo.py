"""Download complete ZFTurbo bs_roformer module."""
import requests
from pathlib import Path

base_url = 'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/models/bs_roformer/'

files = [
    '__init__.py',
    'attend.py',
    'attend_sage.py',
    'bs_roformer.py',
    'mel_band_roformer.py',
]

out_dir = Path('StemSepApp/src/models/architectures/zfturbo_bs_roformer')
out_dir.mkdir(exist_ok=True)

for filename in files:
    url = base_url + filename
    resp = requests.get(url)
    if resp.status_code == 200:
        content = resp.text
        # Fix imports from models.bs_roformer.* to local imports
        content = content.replace('from models.bs_roformer.attend', 'from .attend')
        content = content.replace('from models.bs_roformer.attend_sage', 'from .attend_sage')
        (out_dir / filename).write_text(content, encoding='utf-8')
        print(f'Downloaded: {filename}')
    else:
        print(f'Failed ({resp.status_code}): {filename}')

# Update __init__.py to export main classes
init_content = '''"""Official ZFTurbo bs_roformer implementations."""
from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer
'''
(out_dir / '__init__.py').write_text(init_content)
print('Updated __init__.py')
print('Done!')
