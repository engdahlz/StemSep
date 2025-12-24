# Notes / Artifacts (Developer Convenience)

This folder contains **non-essential artifacts** that were moved out of the repository root to keep the project tidy.

These files are **not required to run the app** and are not part of the core source code. They exist mainly for historical context, debugging, and reproducibility.

## Contents

### `guide_full_text.txt`

- **What it is:** A flattened/plain-text extraction of the full “Instrumental, vocal & other stems separation & mix_master guide…” content.
- **Why it exists:** Useful for quick searching, quoting, or feeding into tooling without parsing PDF/DOCX.
- **How to regenerate:**
  - If you have the original PDF/DOCX in the repo root, you can re-extract text using any of:
    - A PDF-to-text tool (e.g., `pdftotext`)
    - A DOCX-to-text converter
    - Manual export from a word processor
  - Keep the output as UTF-8 text.

### `recovered_config.yaml`

- **What it is:** A recovered/experimental YAML config captured during troubleshooting.
- **Why it exists:** Serves as a reference point when diagnosing config schema issues or when repairing model configs.
- **How to regenerate:**
  - This is typically produced during debugging/config recovery, not a standard build step.
  - If you need to recreate it:
    - Run the relevant maintenance scripts under `scripts/` (e.g., config download/fix/diagnose tools),
    - Or export/save the config that was recovered during a debug session.

## What belongs here vs. in `scripts/`

- Put **outputs and artifacts** here (logs, recovered configs, extracted text).
- Put **executable utilities** in `scripts/`.

## Safety / Git hygiene

- These files may be large or project-specific.
- Prefer not to depend on them at runtime.
- If you add new artifacts, consider adding a short note in this README explaining:
  - where it came from
  - how to regenerate it
  - whether it is safe to delete
