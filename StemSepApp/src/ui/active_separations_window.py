"""
Active Separations window showing real-time progress
"""

import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from typing import List, Dict
import threading
import time

class ActiveSeparationsWindow(ctk.CTkFrame):
    """Window showing active separations with real-time updates"""

    def __init__(self, parent):
        super().__init__(parent)
        self.separations = []
        self.update_interval = 100  # milliseconds

        self._create_ui()
        self._start_update_loop()

    def _create_ui(self):
        """Create the separations window UI"""
        # Title
        title_label = ctk.CTkLabel(
            self,
            text="Active Separations",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        # Separations list
        self.separations_frame = ctk.CTkScrollableFrame(self)
        self.separations_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Empty state
        self.empty_label = ctk.CTkLabel(
            self.separations_frame,
            text="No active separations",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.empty_label.pack(pady=50)

    def add_separation(self, job_data: Dict):
        """Add a new separation job"""
        separation_card = SeparationCard(
            self.separations_frame,
            job_data,
            on_cancel=self._cancel_separation,
            on_play=self._play_separation
        )
        separation_card.pack(fill=tk.X, pady=10)

        self.separations.append(separation_card)

        # Hide empty state
        if self.empty_label:
            self.empty_label.pack_forget()

    def _cancel_separation(self, job_id: str):
        """Cancel a separation job"""
        # Implementation would cancel the actual job
        print(f"Cancelling job {job_id}")

    def _play_separation(self, file_path: str):
        """Play a completed separation"""
        # Implementation would play the audio file
        print(f"Playing {file_path}")

    def _start_update_loop(self):
        """Start the update loop for progress tracking"""
        def update():
            for separation in self.separations:
                separation.update_progress()
            self.after(self.update_interval, update)

        update()

class SeparationCard(ctk.CTkFrame):
    """Card showing separation progress and controls"""

    def __init__(self, parent, job_data, on_cancel, on_play):
        super().__init__(parent, height=150)
        self.pack_propagate(False)

        self.job_data = job_data
        self.on_cancel = on_cancel
        self.on_play = on_play

        self.start_time = time.time()
        self.estimated_duration = 300  # 5 minutes default

        self._create_ui()
        self._simulate_progress()

    def _create_ui(self):
        """Create the separation card UI"""
        # File name
        self.file_label = ctk.CTkLabel(
            self,
            text=self.job_data.get('file_name', 'Unknown'),
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.file_label.pack(pady=(10, 5))

        # Status and preset
        info_text = f"{self.job_data.get('preset', 'Unknown')} | {self.job_data.get('model', 'Unknown model')}"
        self.info_label = ctk.CTkLabel(
            self,
            text=info_text,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.info_label.pack()

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=15)
        self.progress_bar.set(0)

        # Progress text
        self.progress_text = ctk.CTkLabel(
            self,
            text="0% - Starting...",
            font=ctk.CTkFont(size=12)
        )
        self.progress_text.pack()

        # Time info
        time_frame = ctk.CTkFrame(self, fg_color="transparent")
        time_frame.pack(fill=tk.X, padx=20, pady=(5, 10))

        self.elapsed_label = ctk.CTkLabel(
            time_frame,
            text="Elapsed: 00:00",
            font=ctk.CTkFont(size=11)
        )
        self.elapsed_label.pack(side=tk.LEFT)

        self.remaining_label = ctk.CTkLabel(
            time_frame,
            text="Remaining: --:--",
            font=ctk.CTkFont(size=11)
        )
        self.remaining_label.pack(side=tk.RIGHT)

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=100,
            command=self._on_cancel_clicked,
            fg_color="red",
            hover_color="darkred"
        )
        self.cancel_btn.pack(side=tk.RIGHT)

        self.play_btn = ctk.CTkButton(
            button_frame,
            text="Play",
            width=100,
            command=self._on_play_clicked,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.RIGHT, padx=(0, 10))

    def _simulate_progress(self):
        """Simulate progress for demonstration"""
        # This would be replaced with actual progress updates
        # in a real implementation
        pass

    def update_progress(self):
        """Update the progress display"""
        # In a real implementation, this would check the actual job status
        pass

    def _on_cancel_clicked(self):
        """Handle cancel button click"""
        self.on_cancel(self.job_data['id'])

    def _on_play_clicked(self):
        """Handle play button click"""
        # Get the output file path
        output_files = self.job_data.get('output_files', [])
        if output_files:
            self.on_play(output_files[0])
