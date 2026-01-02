# Vendored Upstream Reference Repositories

This directory contains **vendored upstream source snapshots** that are kept for:
- reference and cross-checking behavior against upstream implementations
- troubleshooting regressions by comparing with known-good upstream code
- occasionally syncing “exact copy” modules used in `StemSepApp/src/`

These vendored sources are **not required** to run the application.

## Document library (reference documents)

In addition to upstream code snapshots, this folder also contains a small **document library** under:

- `docs/vendor/library/`

Convention:

- One folder per document (stable id / slug)
- Subfolders per format (e.g. `pdf/`, `md/`, `txt/`)

This allows us to keep the canonical PDF alongside extracted text / structured indexes without duplicating content.

## What’s in here

### `ZFTurbo Repo/`
A snapshot of ZFTurbo’s *Music-Source-Separation-Training* repository.

Why we keep it:
- StemSep includes components that are based on (or explicitly copied from) ZFTurbo’s reference inference/training code.
- Having a local snapshot makes it easier to compare changes and confirm behavior without relying on external network access.

Important:
- Treat this as **read-only reference** unless you are intentionally updating the snapshot.
- Do not import from this directory at runtime. Runtime code should live under `StemSepApp/src/`.

## Runtime code vs. vendored code

- **Runtime / app code**: `StemSepApp/src/`
- **Registry and presets**: `StemSepApp/assets/`
- **Vendored upstream reference** (this folder): `docs/vendor/`

If you need to implement or adjust behavior:
1. Identify the upstream logic you want to match in the vendored repo.
2. Apply changes in the real runtime modules under `StemSepApp/src/`.
3. Add comments/doc links pointing to the upstream source location for traceability.

## Syncing / Updating a vendored snapshot

If you decide to update the vendored snapshot, do it deliberately:
- Prefer updating as a single, well-described commit (e.g., “Update vendored ZFTurbo snapshot to commit XYZ”).
- Record upstream reference:
  - upstream repository URL
  - tag/commit SHA (preferred) or release version
  - date of snapshot
- After updating:
  - run validators (e.g. registry validator)
  - run a small separation smoke test using a known audio file
  - compare outputs if you’re touching inference-related code

## Licensing

Vendored upstream repositories may have their own licenses.
If you redistribute this repository, you are responsible for ensuring compliance with:
- upstream licenses in the vendored directories
- model licenses referenced in the model registry (Hugging Face / GitHub releases / etc.)

## Notes

This directory exists to keep the repo root clean while still retaining critical upstream context.
If disk size becomes an issue, consider removing vendored snapshots and replacing them with:
- a pinned commit reference in documentation
- instructions for cloning the upstream repo when needed