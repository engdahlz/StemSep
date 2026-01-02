"""
Manual configuration page for StemSep
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List

import customtkinter as ctk


class ManualPage(ctk.CTkFrame):
    """Manual configuration interface for advanced users"""

    def __init__(self, parent, config, model_manager):
        super().__init__(parent)
        self.config = config
        self.model_manager = model_manager

        self.selected_file: str | None = None
        self.selected_models: Dict[str, tk.BooleanVar] = {}
        self.model_weight_vars: Dict[str, tk.StringVar] = {}
        self.model_weight_entries: Dict[str, ctk.CTkEntry] = {}
        self.stem_vars: Dict[str, tk.BooleanVar] = {}
        self.strategy_var = tk.StringVar(
            value=self.config.get("manual.default_strategy", "equal")
        )
        self.normalize_var = tk.BooleanVar(
            value=self.config.get("manual.normalize_weights", True)
        )

        self._create_ui()

    # UI construction -----------------------------------------------------
    def _create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._create_file_section()
        self._create_model_section()
        self._create_stem_section()
        self._create_action_bar()
        self._update_start_state()

    def _create_file_section(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Audio file:", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=10)

        self.file_label = ctk.CTkLabel(frame, text="No file selected", anchor="w")
        self.file_label.grid(row=0, column=1, sticky="ew", padx=15)

        browse_btn = ctk.CTkButton(
            frame, text="Browse", width=100, command=self._browse_file
        )
        browse_btn.grid(row=0, column=2, padx=15)

    def _create_model_section(self):
        wrapper = ctk.CTkFrame(self)
        wrapper.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        wrapper.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            wrapper, text="Models", font=ctk.CTkFont(size=16, weight="bold")
        )
        header.pack(anchor="w", padx=15, pady=(15, 5))

        subtext = ctk.CTkLabel(
            wrapper,
            text="Select one or more installed models. Non-installed models are disabled.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        subtext.pack(anchor="w", padx=15, pady=(0, 10))

        self.model_list = ctk.CTkScrollableFrame(wrapper, height=200)
        self.model_list.pack(fill="x", expand=True, padx=15, pady=(0, 15))
        self.model_list.grid_columnconfigure(2, weight=1)

        available_models = self.model_manager.get_available_models()
        installed_ids = {m.id for m in self.model_manager.get_installed_models()}

        for idx, model in enumerate(available_models):
            row = ctk.CTkFrame(self.model_list)
            row.grid(row=idx, column=0, sticky="ew", pady=3)
            row.grid_columnconfigure(1, weight=1)

            var = tk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(
                row,
                text=model.name,
                variable=var,
                command=lambda mid=model.id: self._on_model_toggle(mid),
            )
            chk.grid(row=0, column=0, padx=5, pady=5, sticky="w")

            status = "Installed" if model.id in installed_ids else "Not installed"
            status_label = ctk.CTkLabel(
                row,
                text=status,
                text_color=("green" if model.id in installed_ids else "gray"),
            )
            status_label.grid(row=0, column=1, padx=5, sticky="w")

            weight_var = tk.StringVar(value="1.0")
            entry = ctk.CTkEntry(row, textvariable=weight_var, width=70)
            entry.grid(row=0, column=2, padx=5)
            if self.strategy_var.get() == "equal":
                entry.configure(state="disabled")

            self.selected_models[model.id] = var
            self.model_weight_vars[model.id] = weight_var
            self.model_weight_entries[model.id] = entry

            weight_var.trace_add(
                "write", lambda *_args, mid=model.id: self._on_weight_change(mid)
            )

            if model.id not in installed_ids:
                chk.configure(state="disabled")
                entry.configure(state="disabled")

        strategy_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
        strategy_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(strategy_frame, text="Ensemble strategy:").pack(
            side="left", padx=(0, 10)
        )
        strategy_menu = ctk.CTkOptionMenu(
            strategy_frame,
            variable=self.strategy_var,
            values=["equal", "manual"],
            command=self._on_strategy_change,
        )
        strategy_menu.pack(side="left")

        normalize_chk = ctk.CTkCheckBox(
            strategy_frame, text="Normalize weights", variable=self.normalize_var
        )
        normalize_chk.pack(side="left", padx=(20, 0))

    def _create_stem_section(self):
        wrapper = ctk.CTkFrame(self)
        wrapper.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            wrapper, text="Target stems", font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))

        self.stem_list = ctk.CTkScrollableFrame(wrapper)
        self.stem_list.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self._refresh_stem_options()

    def _create_action_bar(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        frame.grid_columnconfigure(0, weight=1)

        output_label = ctk.CTkLabel(frame, text="Output folder:")
        output_label.grid(row=0, column=0, sticky="w", padx=15, pady=10)

        self.output_var = tk.StringVar(value=str(Path.home() / "StemSep_Output"))
        output_entry = ctk.CTkEntry(frame, width=320, textvariable=self.output_var)
        output_entry.grid(row=0, column=1, sticky="w", padx=(0, 10))

        browse_btn = ctk.CTkButton(
            frame, text="Browse", width=90, command=self._browse_output_dir
        )
        browse_btn.grid(row=0, column=2)

        self.start_btn = ctk.CTkButton(
            frame,
            text="Queue separation",
            width=200,
            command=self._start_manual_separation,
            state=tk.DISABLED,
        )
        self.start_btn.grid(row=0, column=3, padx=15)

    # Helpers --------------------------------------------------------------
    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.configure(text=os.path.basename(file_path))
            self._update_start_state()

    def _browse_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select Output Folder", initialdir=self.output_var.get()
        )
        if directory:
            self.output_var.set(directory)

    def _on_model_toggle(self, model_id: str):
        self._refresh_stem_options()
        self._update_weight_state()
        self._update_start_state()

    def _on_strategy_change(self, value):
        self._update_weight_state()
        self._update_start_state()

    def _update_weight_state(self):
        manual = self.strategy_var.get() == "manual"
        for model_id, entry_widget in self.model_weight_entries.items():
            widget_state = (
                "normal"
                if manual and self.selected_models[model_id].get()
                else "disabled"
            )
            entry_widget.configure(state=widget_state)
        self._normalize_weights_if_needed()

    def _on_weight_change(self, model_id: str):
        if self.strategy_var.get() == "manual":
            self._normalize_weights_if_needed()
        self._update_start_state()

    def _refresh_stem_options(self):
        for widget in self.stem_list.winfo_children():
            widget.destroy()
        self.stem_vars.clear()

        selected_stems = set()
        for model_id, selected_var in self.selected_models.items():
            if selected_var.get():
                model = self.model_manager.get_model(model_id)
                if model:
                    selected_stems.update(model.stems)

        if not selected_stems:
            info = ctk.CTkLabel(
                self.stem_list,
                text="Select at least one model to view available stems",
                text_color="gray",
            )
            info.pack(anchor="w", padx=5, pady=5)
            self._update_start_state()
            return

        for stem in sorted(selected_stems):
            var = tk.BooleanVar(value=True)
            chk = ctk.CTkCheckBox(
                self.stem_list,
                text=stem,
                variable=var,
                command=self._update_start_state,
            )
            chk.pack(anchor="w", padx=5, pady=3)
            self.stem_vars[stem] = var

        self._update_start_state()

    def _normalize_weights_if_needed(self):
        if self.strategy_var.get() != "manual" or not self.normalize_var.get():
            return
        total = 0.0
        selected_ids = [mid for mid, var in self.selected_models.items() if var.get()]
        for mid in selected_ids:
            try:
                total += float(self.model_weight_vars[mid].get())
            except ValueError:
                continue
        if total <= 0:
            return
        for mid in selected_ids:
            try:
                weight = float(self.model_weight_vars[mid].get()) / total
                self.model_weight_vars[mid].set(f"{weight:.3f}")
            except ValueError:
                self.model_weight_vars[mid].set("0.0")

    def _update_start_state(self):
        has_file = self.selected_file is not None
        selected_ids = [mid for mid, var in self.selected_models.items() if var.get()]
        has_model = len(selected_ids) > 0
        has_stem = any(var.get() for var in self.stem_vars.values())
        if self.strategy_var.get() == "manual":
            try:
                [float(self.model_weight_vars[mid].get()) for mid in selected_ids]
                weights_valid = True
            except ValueError:
                weights_valid = False
        else:
            weights_valid = True
        self.start_btn.configure(
            state=tk.NORMAL
            if has_file and has_model and has_stem and weights_valid
            else tk.DISABLED
        )

    # Pipeline building ---------------------------------------------------
    def _build_manual_pipeline(self) -> Dict[str, List[Dict]]:
        selected_ids = [mid for mid, var in self.selected_models.items() if var.get()]
        stems = [stem for stem, var in self.stem_vars.items() if var.get()]
        strategy = self.strategy_var.get()

        if strategy == "equal" or len(selected_ids) == 1:
            models = []
            weight = (
                1.0 / len(selected_ids) if strategy == "equal" and selected_ids else 1.0
            )
            for mid in selected_ids:
                models.append(
                    {
                        "id": mid,
                        "weight": weight,
                        "name": self.model_manager.get_model(mid).name,
                    }
                )
            step_type = "ensemble" if len(selected_ids) > 1 else "separate"
        else:
            models = []
            for mid in selected_ids:
                try:
                    weight = float(self.model_weight_vars[mid].get())
                except ValueError:
                    weight = 0.0
                models.append(
                    {
                        "id": mid,
                        "weight": weight,
                        "name": self.model_manager.get_model(mid).name,
                    }
                )
            step_type = "ensemble" if len(selected_ids) > 1 else "separate"
            if self.normalize_var.get():
                total = sum(m["weight"] for m in models)
                if total > 0:
                    for m in models:
                        m["weight"] = m["weight"] / total

        return {
            "is_manual": True,
            "steps": [
                {
                    "index": 0,
                    "type": step_type,
                    "label": "Manual ensemble"
                    if len(models) > 1
                    else "Manual separation",
                    "models": models,
                    "stems": stems,
                    "params": {},
                }
            ],
        }

    # Actions -------------------------------------------------------------
    def _start_manual_separation(self):
        if not self.selected_file:
            messagebox.showwarning("Missing file", "Select an audio file to continue.")
            return

        selected_ids = [mid for mid, var in self.selected_models.items() if var.get()]
        if not selected_ids:
            messagebox.showwarning(
                "No models selected", "Select at least one installed model."
            )
            return

        if not any(var.get() for var in self.stem_vars.values()):
            messagebox.showwarning(
                "No stems selected", "Select at least one stem to extract."
            )
            return

        pipeline = self._build_manual_pipeline()
        messagebox.showinfo(
            "Manual separation",
            "Manual pipeline has been queued.\n\n"
            f"File: {os.path.basename(self.selected_file)}\n"
            f"Models: {', '.join(selected_ids)}\n"
            f"Stems: {', '.join(pipeline['steps'][0]['stems'])}",
        )

    # Tk callbacks --------------------------------------------------------
    def update(self):  # type: ignore[override]
        super().update()
        self._update_start_state()
