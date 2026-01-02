"""
Settings dialog for configuring the application
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os

class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog window"""

    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config

        self.title("Settings")
        self.geometry("600x500")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Center window
        self.center_window()

    def center_window(self):
        """Center the dialog on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _create_ui(self):
        """Create the settings UI"""
        # Title
        title_label = ctk.CTkLabel(
            self,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 20))

        # Create notebook (tabs)
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # UI Tab
        ui_tab = self.notebook.add("UI")
        self._create_ui_tab(ui_tab)

        # Processing Tab
        processing_tab = self.notebook.add("Processing")
        self._create_processing_tab(processing_tab)

        # Paths Tab
        paths_tab = self.notebook.add("Paths")
        self._create_paths_tab(paths_tab)

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=100,
            command=self._on_cancel
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))

        save_btn = ctk.CTkButton(
            button_frame,
            text="Save",
            width=100,
            command=self._on_save
        )
        save_btn.pack(side=tk.RIGHT)

    def _create_ui_tab(self, parent):
        """Create UI settings tab"""
        # Theme
        theme_frame = ctk.CTkFrame(parent)
        theme_frame.pack(fill=tk.X, padx=20, pady=20)

        ctk.CTkLabel(
            theme_frame,
            text="Theme",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        self.theme_var = tk.StringVar(value=self.config.get('ui.theme', 'dark'))
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            variable=self.theme_var,
            values=["dark", "light"]
        )
        theme_menu.pack(pady=10)

        mode_frame = ctk.CTkFrame(parent)
        mode_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            mode_frame,
            text="Default Mode",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        self.default_mode_var = tk.StringVar(value=self.config.get('ui.default_mode', 'presets'))
        mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.default_mode_var,
            values=["presets", "manual"]
        )
        mode_menu.pack(pady=(0, 10))

        self.remember_mode_var = tk.BooleanVar(value=self.config.get('ui.remember_last_mode', True))
        remember_chk = ctk.CTkCheckBox(
            mode_frame,
            text="Remember last used mode",
            variable=self.remember_mode_var
        )
        remember_chk.pack(pady=(0, 10))

    def _create_processing_tab(self, parent):
        """Create processing settings tab"""
        # Chunk Size
        chunk_frame = ctk.CTkFrame(parent)
        chunk_frame.pack(fill=tk.X, padx=20, pady=20)

        ctk.CTkLabel(
            chunk_frame,
            text="Processing Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        self.auto_chunk_var = tk.BooleanVar(value=bool(self.config.get('processing.auto_chunk', True)))
        auto_chunk_chk = ctk.CTkCheckBox(
            chunk_frame,
            text="Automatically choose chunk size based on system RAM",
            variable=self.auto_chunk_var,
            command=self._on_auto_chunk_toggle
        )
        auto_chunk_chk.pack(anchor="w", padx=20, pady=(10, 5))

        ctk.CTkLabel(chunk_frame, text="Chunk Size:").pack(anchor="w", padx=20, pady=(10, 5))

        self.chunk_size_var = tk.StringVar(value=str(self.config.get('processing.chunk_size', 160000)))
        self.chunk_size_entry = ctk.CTkEntry(chunk_frame, textvariable=self.chunk_size_var, width=200)
        self.chunk_size_entry.pack(anchor="w", padx=20)

        ctk.CTkLabel(
            chunk_frame,
            text="Lower values use less VRAM but may be slower",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=20, pady=(0, 10))

        ctk.CTkLabel(chunk_frame, text="Processing Quality:").pack(anchor="w", padx=20, pady=(5, 5))

        self.quality_var = tk.StringVar(value=str(self.config.get('processing.quality', 'hq')))
        quality_menu = ctk.CTkOptionMenu(
            chunk_frame,
            variable=self.quality_var,
            values=["eco", "balanced", "hq", "max"]
        )
        quality_menu.pack(anchor="w", padx=20, pady=(0, 10))

        self._update_chunk_entry_state()

        # Overlap
        ctk.CTkLabel(chunk_frame, text="Overlap:").pack(anchor="w", padx=20, pady=(5, 5))

        self.overlap_var = tk.StringVar(value=str(self.config.get('processing.overlap', 0.75)))
        overlap_entry = ctk.CTkEntry(chunk_frame, textvariable=self.overlap_var, width=200)
        overlap_entry.pack(anchor="w", padx=20)

        # CPU Threads
        ctk.CTkLabel(chunk_frame, text="CPU Threads:").pack(anchor="w", padx=20, pady=(15, 5))

        self.cpu_threads_var = tk.StringVar(
            value=str(self.config.get('processing.cpu_threads', os.cpu_count() or 4))
        )
        threads_entry = ctk.CTkEntry(chunk_frame, textvariable=self.cpu_threads_var, width=200)
        threads_entry.pack(anchor="w", padx=20)

    def _create_paths_tab(self, parent):
        """Create paths settings tab"""
        # Models Directory
        models_frame = ctk.CTkFrame(parent)
        models_frame.pack(fill=tk.X, padx=20, pady=20)

        ctk.CTkLabel(
            models_frame,
            text="Directories",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(models_frame, text="Models Directory:").pack(anchor="w", padx=20, pady=(15, 5))

        path_frame = ctk.CTkFrame(models_frame, fg_color="transparent")
        path_frame.pack(fill=tk.X, padx=20)

        self.models_dir_var = tk.StringVar(
            value=str(self.config.get('paths.models_dir', '~/.stemsep/models'))
        )
        models_entry = ctk.CTkEntry(path_frame, textvariable=self.models_dir_var)
        models_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ctk.CTkButton(
            path_frame,
            text="Browse",
            width=80,
            command=lambda: self._browse_directory(self.models_dir_var)
        )
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Output Directory
        ctk.CTkLabel(models_frame, text="Default Output Directory:").pack(
            anchor="w", padx=20, pady=(15, 5)
        )

        path_frame2 = ctk.CTkFrame(models_frame, fg_color="transparent")
        path_frame2.pack(fill=tk.X, padx=20)

        self.output_dir_var = tk.StringVar(
            value=str(self.config.get('paths.output_dir', '~/StemSep_Output'))
        )
        output_entry = ctk.CTkEntry(path_frame2, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn2 = ctk.CTkButton(
            path_frame2,
            text="Browse",
            width=80,
            command=lambda: self._browse_directory(self.output_dir_var)
        )
        browse_btn2.pack(side=tk.RIGHT, padx=(10, 0))

    def _on_auto_chunk_toggle(self):
        """Handle auto chunk checkbox updates"""
        self._update_chunk_entry_state()

    def _update_chunk_entry_state(self):
        """Enable or disable manual chunk entry based on auto setting"""
        if getattr(self, 'chunk_size_entry', None) is None:
            return
        state = tk.DISABLED if self.auto_chunk_var.get() else tk.NORMAL
        self.chunk_size_entry.configure(state=state)

    def _browse_directory(self, var):
        """Browse for directory"""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def _on_cancel(self):
        """Handle cancel button"""
        self.destroy()

    def _on_save(self):
        """Handle save button"""
        # Save UI settings
        self.config.set('ui.theme', self.theme_var.get())

        # Save processing settings
        try:
            chunk_value = int(self.chunk_size_var.get())
        except ValueError:
            chunk_value = int(self.config.get('processing.chunk_size', 160000))
            self.chunk_size_var.set(str(chunk_value))

        self.config.set('processing.chunk_size', chunk_value)
        self.config.set('processing.auto_chunk', bool(self.auto_chunk_var.get()))
        self.config.set('processing.quality', self.quality_var.get())
        self.config.set('processing.overlap', float(self.overlap_var.get()))
        self.config.set('processing.cpu_threads', int(self.cpu_threads_var.get()))

        # Save mode preferences
        self.config.set('ui.default_mode', self.default_mode_var.get())
        self.config.set('ui.remember_last_mode', bool(self.remember_mode_var.get()))

        # Save paths
        self.config.set('paths.models_dir', self.models_dir_var.get())
        self.config.set('paths.output_dir', self.output_dir_var.get())

        # Save to file
        self.config.save()

        # Close dialog
        self.destroy()
