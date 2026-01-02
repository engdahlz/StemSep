"""
Models page for downloading and managing models
"""

import asyncio
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List

import aiohttp
import customtkinter as ctk

from stemsep.models.model_manager import ModelInfo


class ModelsPage(ctk.CTkFrame):
    """Page for managing models"""

    def __init__(self, parent, model_manager):
        super().__init__(parent)
        self.model_manager = model_manager
        self.download_tasks = {}
        self.model_cards: Dict[str, ModelCard] = {}
        self.batch_install_running = False
        self.install_progress_var = tk.StringVar(master=self, value="")
        self.batch_results_success: List[str] = []
        self.batch_results_fail: List[str] = []
        self.batch_results_errors: Dict[str, str] = {}
        self.batch_total_count: int = 0

        self._create_ui()
        self.model_manager.add_download_callback(self._handle_download_event)
        self._refresh_model_list()

    def _create_ui(self):
        """Create the models page UI"""
        # Title
        title_label = ctk.CTkLabel(
            self, text="Models", font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        # Filter bar
        filter_frame = ctk.CTkFrame(self)
        filter_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        filter_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(filter_frame, text="Filter:").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )

        self.filter_var = tk.StringVar(value="all")
        filter_menu = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.filter_var,
            values=["all", "BS-Roformer", "Mel-Roformer", "MDX23C", "SCNet"],
            command=self._on_filter_change,
        )
        filter_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        install_status = ctk.CTkLabel(
            filter_frame, textvariable=self.install_progress_var, text_color="gray"
        )
        install_status.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        self.install_all_button = ctk.CTkButton(
            filter_frame,
            text="Install All",
            width=120,
            command=self._on_install_all_clicked,
        )
        self.install_all_button.grid(row=0, column=3, padx=10, pady=10, sticky="e")

        # Models list
        self.models_frame = ctk.CTkScrollableFrame(self)
        self.models_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    def _refresh_model_list(self):
        """Refresh the models list"""
        # Get models
        models = self.model_manager.get_available_models()

        # Filter if needed
        filter_val = self.filter_var.get()
        if filter_val != "all":
            models = [m for m in models if m.architecture == filter_val]

        # Instead of destroying all widgets, show/hide existing ones
        # This is much faster than recreating everything
        model_ids = {m.id for m in models}

        # Hide cards not in filtered list
        for model_id, card in self.model_cards.items():
            if model_id in model_ids:
                card.pack(fill=tk.X, pady=10)
            else:
                card.pack_forget()

        # Create new cards for models we haven't seen yet
        for model in models:
            if model.id not in self.model_cards:
                card = self._create_model_card(model)
                self.model_cards[model.id] = card

    def _create_model_card(self, model: ModelInfo):
        """Create a model card"""
        card = ModelCard(
            self.models_frame,
            model,
            self.model_manager,
            on_install=self._on_install_clicked,
            on_remove=self._on_remove_clicked,
        )
        card.pack(fill=tk.X, pady=10)
        return card

    def _on_filter_change(self, value):
        """Handle filter change"""
        self._refresh_model_list()

    def _on_install_clicked(self, model_id: str):
        """Handle install button click"""

        # Start download in background
        def download():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.model_manager.download_model(model_id))
            loop.close()

        thread = threading.Thread(target=download, daemon=True)
        thread.start()

        messagebox.showinfo("Download Started", f"Downloading {model_id}...")

    def _on_remove_clicked(self, model_id: str):
        """Handle remove button click"""
        result = messagebox.askyesno(
            "Confirm Removal", f"Are you sure you want to remove {model_id}?"
        )

        if result:
            self.model_manager.remove_model(model_id)
            self._refresh_model_list()

    def _on_install_all_clicked(self):
        """Install all missing models"""
        if self.batch_install_running:
            return

        missing_models = [
            m.id for m in self.model_manager.get_available_models() if not m.installed
        ]

        if not missing_models:
            messagebox.showinfo(
                "All Installed", "All available models are already installed."
            )
            return

        self.batch_install_running = True
        self.batch_total_count = len(missing_models)
        self.batch_results_success.clear()
        self.batch_results_fail.clear()
        self.batch_results_errors.clear()
        self.install_all_button.configure(state=tk.DISABLED, text="Installing...")
        self.install_progress_var.set(
            f"Starting batch ({len(missing_models)} models)..."
        )

        def download_all():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def runner():
                try:
                    async with aiohttp.ClientSession() as session:
                        for index, model_id in enumerate(missing_models, start=1):
                            self._update_batch_status(index, len(missing_models))
                            await self.model_manager.download_model(
                                model_id, session=session
                            )
                finally:
                    self._end_batch_status()

            try:
                loop.run_until_complete(runner())
            finally:
                loop.close()

        threading.Thread(target=download_all, daemon=True).start()
        messagebox.showinfo("Batch Download", "Started downloading all missing models.")

    def _update_batch_status(self, current: int, total: int):
        self.after(
            0,
            lambda: self.install_progress_var.set(
                f"Downloading model {current}/{total}..."
            ),
        )

    def _end_batch_status(self):
        def _reset():
            self.batch_install_running = False
            self.install_all_button.configure(state=tk.NORMAL, text="Install All")
            self.install_progress_var.set("")
            self._refresh_model_list()
            total = self.batch_total_count
            success = len(self.batch_results_success)
            failures = len(self.batch_results_fail)
            if total:
                details = [f"Installed {success}/{total} models."]
                if failures:
                    failed_list = ", ".join(self.batch_results_fail)
                    details.append(f"Failed: {failed_list}")
                    if self.batch_results_errors:
                        details.append("")
                        details.append("Issues:")
                        for mid, err in self.batch_results_errors.items():
                            details.append(f"- {mid}: {err}")
                messagebox.showinfo("Batch Download Complete", "\n".join(details))
            self.batch_total_count = 0
            self.batch_results_success.clear()
            self.batch_results_fail.clear()
            self.batch_results_errors.clear()

        self.after(0, _reset)

    def _handle_download_event(self, event_type: str, model_id: str, value):
        self.after(0, lambda: self._process_download_event(event_type, model_id, value))

    def _process_download_event(self, event_type: str, model_id: str, value):
        card = self.model_cards.get(model_id)

        if event_type == "start":
            if card:
                card.set_downloading()
        elif event_type == "progress":
            if card:
                try:
                    progress = float(value)
                except (TypeError, ValueError):
                    progress = 0.0
                card.update_progress(progress)
        elif event_type == "complete":
            if card:
                card.set_complete()
            if self.batch_install_running:
                self.batch_results_success.append(model_id)
            if not self.batch_install_running:
                self.install_progress_var.set("")
                self.install_all_button.configure(state=tk.NORMAL, text="Install All")
                self._refresh_model_list()
        elif event_type == "error":
            if card:
                card.set_error(str(value))
            if self.batch_install_running:
                self.batch_results_fail.append(model_id)
                if value:
                    self.batch_results_errors[model_id] = str(value)
            else:
                self.install_all_button.configure(state=tk.NORMAL, text="Install All")
                self.batch_install_running = False
                if value:
                    messagebox.showerror(
                        "Download Failed", f"Failed to download {model_id}: {value}"
                    )


class ModelCard(ctk.CTkFrame):
    """Card showing model information"""

    def __init__(self, parent, model: ModelInfo, model_manager, on_install, on_remove):
        super().__init__(parent, height=200)
        self.pack_propagate(False)

        self.model = model
        self.model_manager = model_manager
        self.on_install = on_install
        self.on_remove = on_remove
        self.progress_var = tk.StringVar(master=self, value="")

        self.install_button = None
        self.remove_button = None
        self.status_label = None
        self.size_label = None
        self.progress_label = None
        self.button_frame = None

        self._create_ui()

    def _create_ui(self):
        """Create the model card UI"""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)

        # Name
        name_label = ctk.CTkLabel(
            self, text=self.model.name, font=ctk.CTkFont(size=18, weight="bold")
        )
        name_label.grid(row=0, column=1, sticky="w", padx=20, pady=(15, 5))

        # Description
        desc_label = ctk.CTkLabel(
            self, text=self.model.description, font=ctk.CTkFont(size=12), wraplength=500
        )
        desc_label.grid(row=1, column=1, sticky="w", padx=20, pady=5)

        # Metrics
        metrics_text = (
            f"SDR: {self.model.sdr} | "
            f"Fullness: {self.model.fullness} | "
            f"Bleedless: {self.model.bleedless}"
        )
        metrics_label = ctk.CTkLabel(
            self, text=metrics_text, font=ctk.CTkFont(size=12), text_color="gray"
        )
        metrics_label.grid(row=2, column=1, sticky="w", padx=20, pady=5)

        # Info
        info_text = (
            f"Architecture: {self.model.architecture} | "
            f"VRAM: {self.model.vram_required}GB | "
            f"Speed: {self.model.speed}"
        )
        info_label = ctk.CTkLabel(
            self, text=info_text, font=ctk.CTkFont(size=11), text_color="gray"
        )
        info_label.grid(row=3, column=1, sticky="w", padx=20, pady=5)

        # Buttons
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=4, column=1, sticky="e", padx=20, pady=15)

        if self.model.installed:
            # Remove button
            self.remove_button = ctk.CTkButton(
                self.button_frame,
                text="Remove",
                width=100,
                command=self._on_remove_clicked,
                fg_color="red",
                hover_color="darkred",
            )
            self.remove_button.pack(side=tk.RIGHT)

            # Status
            self.status_label = ctk.CTkLabel(
                self.button_frame,
                text="✓ Installed",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="green",
            )
            self.status_label.pack(side=tk.RIGHT, padx=15)
        else:
            # Install button
            self.install_button = ctk.CTkButton(
                self.button_frame,
                text="Download",
                width=100,
                command=self._on_install_clicked,
            )
            self.install_button.pack(side=tk.RIGHT)

            # Size
            if self.model.file_size:
                size_mb = self.model.file_size / (1024 * 1024)
                self.size_label = ctk.CTkLabel(
                    self.button_frame,
                    text=f"{size_mb:.1f} MB",
                    font=ctk.CTkFont(size=12),
                    text_color="gray",
                )
                self.size_label.pack(side=tk.RIGHT, padx=15)

        self.progress_label = ctk.CTkLabel(
            self.button_frame,
            textvariable=self.progress_var,
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )

    def _on_remove_clicked(self):
        """Handle remove button click"""
        self.on_remove(self.model.id)

    def _ensure_progress_visible(self):
        if self.progress_label and not self.progress_label.winfo_manager():
            self.progress_label.pack(side=tk.RIGHT, padx=15)

    def set_downloading(self):
        if self.install_button:
            self.install_button.configure(state=tk.DISABLED, text="Downloading...")
        if self.size_label and self.size_label.winfo_manager():
            self.size_label.pack_forget()
        self._ensure_progress_visible()
        self.progress_var.set("Starting...")

    def update_progress(self, progress: float):
        self._ensure_progress_visible()
        self.progress_var.set(f"{progress:.0f}%")

    def set_complete(self):
        self._ensure_progress_visible()
        self.progress_var.set("✓ Done")
        if self.install_button:
            self.install_button.configure(state=tk.DISABLED, text="Installed")

    def set_error(self, message: str):
        self._ensure_progress_visible()
        self.progress_var.set("Error")
        if self.install_button:
            self.install_button.configure(state=tk.NORMAL, text="Retry Download")
        if self.size_label and not self.size_label.winfo_manager():
            self.size_label.pack(side=tk.RIGHT, padx=15)
