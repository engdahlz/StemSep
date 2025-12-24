from audio_separator.separator import Separator
print("Audio Separator imported successfully")
# Unfortunately audio-separator doesn't expose a simple list of architectures publicly easily,
# but we can check if we can instantiate it.
sep = Separator()
print(f"Separator version: {sep.separator_version}")
