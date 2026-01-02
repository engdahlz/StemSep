# StemSep Model Mirror — Release Assets Staging Folder

This folder (`StemSep/reports/release_assets_stage/`) is the local **staging area** for model artifacts that will be uploaded to the **snapshot GitHub Release** used by the app’s v2 model registry.

Goal:
- Ensure **all required model files exist** with **deterministic filenames**
- Upload them as **GitHub Release assets** to the snapshot tag (e.g. `models-v2-2026-01-01`)
- Patch the v2 registry so it points only to the snapshot URLs

---

## 1) Deterministic filename policy (IMPORTANT)

The mirror uses deterministic filenames so the registry can be stable and reproducible:

- **Checkpoint**: `{model_id}{ext}` (e.g. `bandit-plus.ckpt`, `mdx-net-hq4.onnx`, `drypd-side-extraction.pth`)
- **Config**: `{model_id}.yaml` (e.g. `bandit-plus.yaml`)

Do **not** upload upstream/original filenames (like `model_bandit_plus_dnr_sdr_11.47.chpt`) as the “official” filenames.
Those may exist in the release as legacy/variants, but the registry should reference the deterministic names.

---

## 2) What should end up in this folder?

The source of truth for what needs to exist is:

- `StemSep/reports/mirror_upload_manifest.json`

That file lists all required deterministic filenames and their expected mirror URLs.

This folder should contain (eventually) every missing required file from the manifest, saved with the **exact deterministic filename**.

---

## 3) How to stage files (recommended workflow)

### Option A — Automated staging (recommended)
Use the staging fetcher script from repo root:

1) Ensure you have Python 3.10+ available

2) Run:
- `python scripts/registry/fetch_missing_release_assets.py`

What it does:
- Reads `reports/mirror_upload_manifest.json` to find required filenames
- Uses `reports/link_audit.json` plus `reports/hf_link_suggestions.json` to find candidate upstream URLs
- Performs a **HEAD preflight** first (default) to avoid huge downloads that will fail
- Downloads missing files into this folder using deterministic filenames
- Produces a report:
  - `StemSep/reports/fetch_missing_release_assets_report.json`

Notes:
- Redirects are allowed (needed for Hugging Face and GitHub release asset delivery).
- If some assets cannot be resolved automatically, they will show up as `failed` or `unresolved` in the report.

### Option B — Manual staging (when automation can’t find a source)
If the script reports `unresolved` items:
1) Find a permissible upstream source (GitHub or Hugging Face, respecting licenses/terms)
2) Download the file
3) Place it in this folder with the deterministic filename required by the manifest

---

## 4) Uploading staged assets to the snapshot GitHub Release

Target mirror release:
- Repo: `engdahlz/stemsep-models`
- Tag: `models-v2-2026-01-01` (snapshot)

Upload rules:
- Upload the deterministic filenames from this folder as release assets.
- Prefer **adding missing deterministic assets** first (conservative approach).
- Avoid deleting legacy/variant filenames until everything is verified.

After uploading, verify that the following URL pattern works for each file:
- `https://github.com/engdahlz/stemsep-models/releases/download/models-v2-2026-01-01/<filename>`

Example:
- `.../bandit-plus.ckpt`
- `.../bandit-plus.yaml`

---

## 5) Verification checklist (before patching registry)

1) Confirm staging folder contains all required deterministic filenames.
2) Confirm the GitHub Release contains those exact filenames.
3) Run link auditing (or equivalent verification) so all mirrored URLs resolve (200).
4) Patch the v2 model registry to reference only the snapshot URLs.

---

## 6) Common pitfalls

- Wrong extension:
  - `.ckpt` vs `.chpt` vs `.pth` vs `.onnx`
  - The deterministic extension must match what the runtime expects.
- Incorrect filename casing:
  - GitHub asset names are case-sensitive for URL paths.
- Hugging Face signed blob links:
  - Never store signed blob URLs in registry. Use HF `resolve/...` URLs as upstream sources only.
- Partial downloads:
  - Make sure `.part` files do not get uploaded.

---

## 7) Outputs / artifacts you should expect

- Staged model files in this directory
- `StemSep/reports/fetch_missing_release_assets_report.json` summarizing:
  - downloaded / failed / unresolved
  - sha256 per staged file (where computed)

---

If you’re unsure whether the staging content is correct, re-check:
- `StemSep/reports/mirror_upload_manifest.json`
and ensure the filenames in this directory match the manifest exactly.
