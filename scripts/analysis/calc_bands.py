
import librosa
import numpy as np

def calculate_freqs_per_bands(n_fft=2048, n_mels=60, sr=44100, fmin=0.0, fmax=None):
    # Mel-Roformer uses a specific way to split bands.
    # Often it simply groups STFT bins into Mel bands.
    
    # Calculate mel filterbank to see boundaries?
    # Or strict mapping of bins?
    
    # Usually:
    # 1. create Mel filters
    # 2. find determining bins
    
    # Or simpler:
    # intervals are based on mel scale points
    
    fmax = fmax or sr / 2
    
    # Create mel points
    mel_min = librosa.hz_to_mel(fmin)
    mel_max = librosa.hz_to_mel(fmax)
    
    mels = np.linspace(mel_min, mel_max, n_mels + 1)
    hz_points = librosa.mel_to_hz(mels)
    
    # Convert Hz to bins
    # bin = freq * n_fft / sr
    bins = np.floor(hz_points * n_fft / sr).astype(int)
    
    # Force alignment
    bins[0] = 0
    bins[-1] = n_fft // 2 + 1 # 1025
    
    # Calculate widths
    widths = np.diff(bins)
    
    # Fix 0-width bands if any (shouldn't be with 60 bands)
    widths[widths == 0] = 1
    
    # Adjust last bin to make sum correct
    current_sum = np.sum(widths)
    target = n_fft // 2 + 1
    
    if current_sum != target:
        diff = target - current_sum
        widths[-1] += diff
        
    print(f"Calculated {len(widths)} bands.")
    print(f"Tuple: {tuple(widths)}")
    print(f"Sum: {np.sum(widths)} (Target: 1025)")

if __name__ == "__main__":
    calculate_freqs_per_bands()
