"""
Main window for StemSep application
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import logging

from .landing_page import LandingPage
from .manual_page import ManualPage
from .active_separations_window import ActiveSeparationsWindow
from .models_page import ModelsPage
from .settings_dialog import SettingsDialog

class MainWindow:
    """Main application window"""

    def __init__(self, root, config, gpu_info, model_manager):
        self.root = root
        self.config = config
        self.gpu_info = gpu_info
        self.model_manager = model_manager
        self.logger = logging.getLogger('StemSep')

        self.mode = self._determine_initial_mode()
        self.current_frame = None
        self.active_separations_window = None
        self.models_window = None

        self._create_ui()

    def _create_ui(self):
        """Create the main UI"""
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create header
        self._create_header()

        # Create navigation
        self._create_navigation()

        # Create content area
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Show page based on mode
        self._navigate_to_current_mode()

    def _create_header(self):
        """Create application header"""
        header_frame = ctk.CTkFrame(self.main_container, height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="StemSep",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=20)

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Advanced Audio Stem Separation",
            font=ctk.CTkFont(size=16)
        )
        subtitle_label.pack(side=tk.LEFT, padx=(0, 20), pady=20)

        # Mode selector
        mode_values = ["Presets", "Manual"]
        self.mode_selector = ctk.CTkSegmentedButton(
            header_frame,
            values=mode_values,
            command=self._on_mode_change,
            width=200
        )
        self.mode_selector.pack(side=tk.LEFT, padx=(0, 20), pady=20)
        self.mode_selector.set("Presets" if self.mode == 'presets' else "Manual")

        # Theme toggle
        theme_button = ctk.CTkButton(
            header_frame,
            text="Toggle Theme",
            width=100,
            command=self._toggle_theme
        )
        theme_button.pack(side=tk.RIGHT, padx=20, pady=20)

        # Settings button
        settings_button = ctk.CTkButton(
            header_frame,
            text="Settings",
            width=100,
            command=self._open_settings
        )
        settings_button.pack(side=tk.RIGHT, padx=(0, 10), pady=20)

    def _create_navigation(self):
        """Create navigation bar"""
        nav_frame = ctk.CTkFrame(self.main_container, height=50)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        nav_frame.pack_propagate(False)

        # Navigation buttons
        self.landing_btn = ctk.CTkButton(
            nav_frame,
            text="Presets",
            width=120,
            command=self.show_landing_page
        )
        self.landing_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.manual_btn = ctk.CTkButton(
            nav_frame,
            text="Manual",
            width=120,
            command=self.show_manual_page
        )
        self.manual_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.separations_btn = ctk.CTkButton(
            nav_frame,
            text="Active Separations",
            width=120,
            command=self.show_active_separations
        )
        self.separations_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.models_btn = ctk.CTkButton(
            nav_frame,
            text="Models",
            width=120,
            command=self.show_models
        )
        self.models_btn.pack(side=tk.LEFT, padx=10, pady=10)

    def _determine_initial_mode(self) -> str:
        """Determine which mode to start in"""
        saved_mode = self.config.get('ui.last_mode') if self.config.get('ui.remember_last_mode', True) else None
        default_mode = self.config.get('ui.default_mode', 'presets')
        mode = saved_mode or default_mode
        return mode if mode in ('presets', 'manual') else 'presets'

    def _on_mode_change(self, value: str):
        """Handle segmented button mode changes"""
        target_mode = value.lower()
        if target_mode not in ('presets', 'manual'):
            return
        if target_mode == self.mode:
            return
        if target_mode == 'presets':
            self.show_landing_page()
        else:
            self.show_manual_page()

    def _navigate_to_current_mode(self):
        """Navigate to the page matching the current mode"""
        if self.mode == 'manual':
            self.show_manual_page()
        else:
            self.show_landing_page()

    def _toggle_theme(self):
        """Toggle between dark and light theme"""
        current_mode = ctk.get_appearance_mode()
        new_mode = "Light" if current_mode == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.config.set('ui.theme', new_mode.lower())
        self.config.save()

    def _open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.root, self.config)
        dialog.show()

    def clear_content(self):
        """Clear the content area"""
        if self.current_frame:
            self.current_frame.destroy()
            self.current_frame = None

    def show_landing_page(self):
        """Show the landing page"""
        self.clear_content()
        self.mode = 'presets'
        self.mode_selector.set("Presets")
        self._record_last_mode()

        self.landing_btn.configure(fg_color=("gray", "blue"))
        self.manual_btn.configure(fg_color="transparent")
        self.separations_btn.configure(fg_color="transparent")
        self.models_btn.configure(fg_color="transparent")

        self.current_frame = LandingPage(
            self.content_frame,
            self.config,
            self.gpu_info,
            self.model_manager
        )
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_manual_page(self):
        """Show the manual configuration page"""
        self.clear_content()
        self.mode = 'manual'
        self.mode_selector.set("Manual")
        self._record_last_mode()

        self.landing_btn.configure(fg_color="transparent")
        self.manual_btn.configure(fg_color=("gray", "blue"))
        self.separations_btn.configure(fg_color="transparent")
        self.models_btn.configure(fg_color="transparent")

        self.current_frame = ManualPage(
            self.content_frame,
            self.config,
            self.model_manager
        )
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_active_separations(self):
        """Show the active separations window"""
        self.clear_content()
        self.landing_btn.configure(fg_color="transparent")
        self.manual_btn.configure(fg_color="transparent")
        self.separations_btn.configure(fg_color=("gray", "blue"))
        self.models_btn.configure(fg_color="transparent")

        self.current_frame = ActiveSeparationsWindow(
            self.content_frame
        )
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_models(self):
        """Show the models page"""
        self.clear_content()
        self.landing_btn.configure(fg_color="transparent")
        self.manual_btn.configure(fg_color="transparent")
        self.separations_btn.configure(fg_color="transparent")
        self.models_btn.configure(fg_color=("gray", "blue"))

        self.current_frame = ModelsPage(
            self.content_frame,
            self.model_manager
        )
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def _record_last_mode(self):
        """Persist the last used mode if enabled"""
        if not self.config.get('ui.remember_last_mode', True):
            return
        self.config.set('ui.last_mode', self.mode)
        self.config.save()
