"""
StemSep - Advanced Audio Stem Separation Application
A Windows application for separating audio into individual stems using local AI models.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow
from core.config import Config
from core.logger import setup_logging
from core.gpu_detector import GPUDetector
from core.fonts import ensure_font_permissions
from models.model_manager import ModelManager

class StemSepApp:
    """Main application class for StemSep"""

    def __init__(self):
        """Initialize the application"""
        # Setup logging
        self.logger = setup_logging()

        # Load configuration
        self.config = Config()

        # Setup appearance mode
        ctk.set_appearance_mode(self.config.get('ui.theme', 'dark'))
        ctk.set_default_color_theme(self.config.get('ui.color_theme', 'blue'))

        # Ensure font readability on Linux to avoid CustomTkinter fallback rendering
        try:
            ensure_font_permissions(self.logger)
        except Exception:
            pass

        # Detect GPU
        self.gpu_detector = GPUDetector()
        gpu_info = self.gpu_detector.get_gpu_info()

        self.logger.info("StemSep Application Starting")
        self.logger.info(f"GPU Detected: {gpu_info}")

        # Initialize model manager
        self.model_manager = ModelManager()

        # Create main window
        self.root = ctk.CTk()
        self.root.title("StemSep - Audio Stem Separation")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Set window icon if available
        try:
            icon_path = Path(__file__).parent.parent / 'assets' / 'icon.ico'
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception as e:
            self.logger.warning(f"Could not set window icon: {e}")

        # Initialize main window UI
        self.main_window = MainWindow(
            root=self.root,
            config=self.config,
            gpu_info=gpu_info,
            model_manager=self.model_manager
        )

    def run(self):
        """Start the application main loop"""
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            self.logger.info("StemSep Application Shutting Down")

def main():
    """Main entry point"""
    try:
        app = StemSepApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
