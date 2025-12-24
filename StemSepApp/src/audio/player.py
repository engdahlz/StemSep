"""
Audio player for playback of separated stems
"""

import logging
from pathlib import Path
from typing import Optional
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except ImportError:
    SIMPLEAUDIO_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Constants for playback timing
PLAYBACK_START_DELAY = 0.1  # Seconds to wait before cleanup to ensure playback starts

class AudioPlayer:
    """Audio player for playback of audio files"""

    def __init__(self):
        self.logger = logging.getLogger('StemSep')
        self.current_file = None
        self.is_playing = False
        self.position = 0

        # Initialize pygame mixer if available
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
        elif PYDUB_AVAILABLE:
            # Initialize with pydub
            pass

        self.logger.info("AudioPlayer initialized")

    def play(self, file_path: str, start_position: float = 0.0) -> bool:
        """Play an audio file"""
        try:
            if not Path(file_path).exists():
                self.logger.error(f"Audio file not found: {file_path}")
                return False

            self.stop()
            self.current_file = file_path
            self.position = start_position

            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play(start=start_position)
                self.is_playing = True
                return True
            elif PYDUB_AVAILABLE:
                self._play_with_pydub(file_path, start_position)
                return True
            else:
                self.logger.error("No audio playback library available")
                return False

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            return False

    def stop(self):
        """Stop playback"""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.is_playing = False
        self.position = 0

    def pause(self):
        """Pause playback"""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.music.pause()
            self.is_playing = False

    def resume(self):
        """Resume playback"""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.music.unpause()
            self.is_playing = True

    def get_position(self) -> float:
        """Get current playback position in seconds"""
        if PYGAME_AVAILABLE and self.is_playing:
            return pygame.mixer.music.get_busy() and pygame.mixer.music.get_pos() / 1000.0 or 0.0
        return self.position

    def get_duration(self, file_path: str) -> Optional[float]:
        """Get duration of audio file in seconds"""
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                return len(audio) / 1000.0
            except Exception as e:
                self.logger.error(f"Error getting duration: {e}")
        return None

    def _play_with_pydub(self, file_path: str, start_position: float):
        """Play using pydub (fallback when pygame not available)
        
        Note: This method includes a small delay (PLAYBACK_START_DELAY) to ensure
        the audio buffer is loaded before cleanup. This may briefly block the calling
        thread but is acceptable as this is a fallback path used only when pygame
        is unavailable. For production use, pygame is the recommended audio backend.
        """
        audio = AudioSegment.from_file(file_path)
        start_ms = int(start_position * 1000)
        audio = audio[start_ms:]

        # Use context manager for temp file to ensure cleanup
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="wav")

        try:
            # Play using simpleaudio if available
            if SIMPLEAUDIO_AVAILABLE:
                wave_obj = sa.WaveObject.from_wave_file(temp_path)
                play_obj = wave_obj.play()
                self.is_playing = True
                # Delay needed to ensure audio buffer is loaded before cleanup
                time.sleep(PLAYBACK_START_DELAY)
            else:
                self.logger.warning("simpleaudio not available, cannot play with pydub")
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not clean up temp file: {e}")

    def is_file_playing(self) -> bool:
        """Check if currently playing"""
        if PYGAME_AVAILABLE:
            return pygame.mixer.music.get_busy()
        return self.is_playing

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
