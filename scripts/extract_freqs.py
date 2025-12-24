"""Extract exact freqs_per_bands from checkpoint."""
import torch
from pathlib import Path

ckpt_path = Path.home() / ".stemsep" / "models" / "unwa-inst-v1e-plus.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

# Extract freqs_per_bands from band_split.to_features weights
# Each weight shape is (out_dim, in_dim)
# in_dim = 2 * freq_per_band * audio_channels where audio_channels=2 for stereo
# So in_dim = 4 * freq_per_band, meaning freq_per_band = in_dim / 4

freqs_per_bands = []
for i in range(60):  # 60 bands
    key = f"band_split.to_features.{i}.1.weight"
    if key in ckpt:
        in_dim = ckpt[key].shape[1]
        freq = in_dim // 4  # 2 channels * 2 (real+imag for complex)
        freqs_per_bands.append(freq)
    else:
        print(f"Missing band {i}")

print(f"Extracted freqs_per_bands ({len(freqs_per_bands)} bands):")
print(f"  {freqs_per_bands}")
print(f"\nSum: {sum(freqs_per_bands)}")
print(f"Expected dim_freqs_in: {sum(freqs_per_bands)} (should match stft_n_fft/2 + 1)")

# Generate YAML tuple format
print(f"\nYAML format:")
print("  freqs_per_bands: !!python/tuple")
for f in freqs_per_bands:
    print(f"  - {f}")
