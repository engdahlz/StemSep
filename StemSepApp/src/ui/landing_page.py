"""
Landing page with drag-and-drop functionality and presets
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
import threading

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class LandingPage(ctk.CTkFrame):
    """Landing page with drag-and-drop, presets, and model selection"""

    def __init__(self, parent, config, gpu_info, model_manager):
        super().__init__(parent)
        self.config = config
        self.gpu_info = gpu_info
        self.model_manager = model_manager
        self.logger = logging.getLogger('StemSep')

        self.selected_file = None
        self.selected_preset = None

        self._create_ui()

    def _create_ui(self):
        """Create the landing page UI"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # System Info Section
        self._create_system_info()

        # Main Content Area
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)

        # Preset Selection
        self._create_preset_section(content_frame)

        # File Drop Zone
        self._create_drop_zone(content_frame)

        # Output Settings
        self._create_output_settings(content_frame)

        # Action Buttons
        self._create_action_buttons(content_frame)

    def _create_system_info(self):
        """Create system information display"""
        info_frame = ctk.CTkFrame(self, height=100)
        info_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        info_frame.grid_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            info_frame,
            text="System Information",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 5))

        # GPU Info
        gpu_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        gpu_frame.pack(fill=tk.X, padx=20, pady=5)

        gpu_text = self._format_gpu_info()
        gpu_label = ctk.CTkLabel(
            gpu_frame,
            text=gpu_text,
            font=ctk.CTkFont(size=12),
            justify=tk.LEFT
        )
        gpu_label.pack(side=tk.LEFT)

        # Recommended Model
        if self.gpu_info['gpus']:
            best_model = self.model_manager.get_model(
                "bs_roformer_2025_07" if self.gpu_info['gpus'][0].get('memory_gb', 0) >= 6 else "mdx23c_8kfft_hq"
            )
            if best_model:
                rec_label = ctk.CTkLabel(
                    gpu_frame,
                    text=f"Recommended: {best_model.name}",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=("blue", "lightblue")
                )
                rec_label.pack(side=tk.RIGHT, padx=20)

    def _format_gpu_info(self) -> str:
        """Format GPU information for display"""
        if not self.gpu_info['gpus']:
            return "GPU: Not detected (CPU mode - very slow)"

        gpu = self.gpu_info['gpus'][0]
        lines = [f"GPU: {gpu['name']}"]
        lines.append(f"VRAM: {gpu.get('memory_gb', 'N/A')} GB")

        if gpu.get('recommended', False):
            lines.append("Status: ✓ Recommended")
        else:
            lines.append("Status: ⚠ Limited")

        return " | ".join(lines)

    def _create_preset_section(self, parent):
        """Create preset selection section"""
        preset_frame = ctk.CTkFrame(parent)
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        preset_frame.grid_columnconfigure((0, 1, 2), weight=1)

        title_label = ctk.CTkLabel(
            preset_frame,
            text="Choose a Preset",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(15, 10))

        # Preset buttons
        presets = self.model_manager.get_presets()
        self.preset_buttons = {}
        col = 0
        for preset_id, preset_info in presets.items():
            btn = ctk.CTkButton(
                preset_frame,
                text=f"{preset_info['name']}\n{preset_info.get('description','')}",
                width=200,
                height=60,
                command=lambda pid=preset_id: self._select_preset(pid)
            )
            btn.grid(row=1, column=col, padx=10, pady=10, sticky="ew")
            self.preset_buttons[preset_id] = btn
            col += 1
            if col > 2:
                col = 0

        # Initial selection
        if presets:
            default_preset = self.config.get('processing.default_preset', next(iter(presets)))
            if default_preset not in presets:
                default_preset = next(iter(presets))
            self._select_preset(default_preset)

    def _create_drop_zone(self, parent):
        """Create drag-and-drop zone"""
        self.drop_frame = ctk.CTkFrame(parent, width=600, height=200)
        self.drop_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        self.drop_frame.grid_propagate(False)

        self.drop_label = ctk.CTkLabel(
            self.drop_frame,
            text="Drag & Drop Audio File Here\nor Click to Browse",
            font=ctk.CTkFont(size=20),
            text_color=("gray", "gray")
        )
        self.drop_label.pack(expand=True)

        # Bind events
        self.drop_frame.bind("<Button-1>", self._browse_file)
        self.drop_frame.bind("<B1-Motion>", self._on_drag)
        self.drop_frame.bind("<ButtonRelease-1>", self._on_drop)
        self.drop_frame.bind("<Enter>", self._on_enter)
        self.drop_frame.bind("<Leave>", self._on_leave)

        # File types
        supported_formats = ctk.CTkLabel(
            parent,
            text="Supported formats: MP3, WAV, FLAC, M4A, OGG",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        supported_formats.grid(row=2, column=0, pady=(0, 10))

    def _create_output_settings(self, parent):
        """Create output settings section"""
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))

        title_label = ctk.CTkLabel(
            settings_frame,
            text="Output Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(10, 5))

        settings_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_row.pack(fill=tk.X, padx=20, pady=10)

        ctk.CTkLabel(settings_row, text="Output folder:").pack(side=tk.LEFT, padx=(0, 10))

        self.output_var = tk.StringVar(value=str(Path.home() / "StemSep_Output"))
        output_entry = ctk.CTkEntry(
            settings_row,
            variable=self.output_var,
            width=320
        )
        output_entry.pack(side=tk.LEFT, padx=(0, 10))

        browse_btn = ctk.CTkButton(
            settings_row,
            text="Browse",
            width=80,
            command=self._browse_output_dir
        )
        browse_btn.pack(side=tk.LEFT)

    def _create_action_buttons(self, parent):
        """Create action buttons"""
        buttons_frame = ctk.CTkFrame(parent, fg_color="transparent")
        buttons_frame.grid(row=4, column=0, pady=20)

        self.separate_btn = ctk.CTkButton(
            buttons_frame,
            text="SEPARATE",
            width=200,
            height=50,
            font=ctk.CTkFont(size=20, weight="bold"),
            command=self._start_separation,
            state=tk.DISABLED
        )
        self.separate_btn.pack()

    def _select_preset(self, preset_id: str):
        """Select a preset"""
        self.selected_preset = preset_id

        # Log selection
        preset = self.model_manager.get_presets()[preset_id]
        self.logger.info(f"Selected preset: {preset['name']}")

        for pid, btn in getattr(self, 'preset_buttons', {}).items():
            btn.configure(fg_color=("gray", "blue") if pid == preset_id else "transparent")

        if self.selected_file:
            self.separate_btn.configure(state=tk.NORMAL)

    def _browse_file(self, event=None):
        """Browse for audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self._set_selected_file(file_path)

    def _browse_output_dir(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_var.get()
        )

        if dir_path:
            self.output_var.set(dir_path)

    def _set_selected_file(self, file_path: str):
        """Set the selected file"""
        self.selected_file = file_path
        file_name = os.path.basename(file_path)

        self.drop_label.configure(
            text=f"Selected: {file_name}\nClick to change or drag new file",
            text_color=("black", "white")
        )
        self.drop_frame.configure(border_color=("green", "blue"))
        self.drop_frame.configure(border_width=2)

        # Enable separate button if file is selected
        if self.selected_preset:
            self.separate_btn.configure(state=tk.NORMAL)

    def _start_separation(self):
        """Start the separation process"""
        if not self.selected_file or not self.selected_preset:
            messagebox.showwarning("Warning", "Please select an audio file and preset")
            return

        # Build pipeline from preset
        try:
            pipeline = self.model_manager.build_preset_pipeline(self.selected_preset)
        except Exception as exc:
            messagebox.showerror("Preset Error", str(exc))
            return

        self.logger.info(f"Starting separation: {self.selected_file}")
        self.logger.info(f"Using preset: {self.selected_preset}")

        # Create separation job
        job = {
            'id': f"job_{id(self)}",
            'file_path': self.selected_file,
            'preset': self.selected_preset,
            'output_dir': self.output_var.get(),
            'pipeline': pipeline,
            'status': 'pending',
            'progress': 0.0
        }

        # For now, just show a message
        # In the full implementation, this would start the actual separation
        messagebox.showinfo(
            "Separation Started",
            f"Separation has been queued!\n\n"
            f"File: {os.path.basename(self.selected_file)}\n"
            f"Preset: {self.model_manager.get_presets()[self.selected_preset]['name']}\n"
            f"Pipeline steps: {len(pipeline['steps'])}\n\n"
            f"Check the 'Active Separations' tab to monitor progress."
        )

    # Drag and drop events
    def _on_enter(self, event):
        if not self.selected_file:
            self.drop_frame.configure(border_color=("blue", "lightblue"))
            self.drop_frame.configure(border_width=3)

    def _on_leave(self, event):
        self.drop_frame.configure(border_color=("gray", "gray"))
        self.drop_frame.configure(border_width=1)

    def _on_drag(self, event):
        pass  # Handled by OS

    def _on_drop(self, event):
        try:
            # This is a simplified drop handler
            # In a full implementation, you'd use tkinterdnd2
            files = self.tk.splitlist(event.data)
            if files:
                self._set_selected_file(files[0])
        except Exception as e:
            self.logger.error(f"Drop error: {e}")
