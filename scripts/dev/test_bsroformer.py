from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
import torch
import yaml
from pathlib import Path
import numpy as np
import librosa

config_path = Path(r'D:\StemSep Models\unwa-inst-v1e-plus.yaml')
ckpt_path = Path(r'D:\StemSep Models\unwa-inst-v1e-plus.ckpt')

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_cfg = config.get('model', config)
print('Creating BSRoformer...')
print('Has freqs_per_bands:', 'freqs_per_bands' in model_cfg)

bsroformer_params = {k: v for k, v in model_cfg.items() if k not in ['architecture', 'num_bands', 'sample_rate', 'skip_final_norm']}
if 'freqs_per_bands' in bsroformer_params:
    bsroformer_params['freqs_per_bands'] = tuple(bsroformer_params['freqs_per_bands'])

model = BSRoformer(**bsroformer_params)
print('Model params:', sum(p.numel() for p in model.parameters()))

checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
result = model.load_state_dict(state_dict, strict=False)
print('Missing keys:', len(result.missing_keys))
print('Unexpected keys:', len(result.unexpected_keys))
if result.unexpected_keys:
    print('First unexpected:', result.unexpected_keys[:3])

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

y, sr = librosa.load('test_audio.wav', sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])
chunk = torch.tensor(y[:, :485100], dtype=torch.float32).unsqueeze(0).to(device)
print('Input RMS:', chunk.pow(2).mean().sqrt().item())

with torch.no_grad():
    with torch.cuda.amp.autocast():
        output = model(chunk)

print('Output RMS:', output.pow(2).mean().sqrt().item())
print('Max output:', output.abs().max().item())
