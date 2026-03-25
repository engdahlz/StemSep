# Remote Catalog v4

This folder contains the published remote-first StemSep runtime catalog and its
verification material.

Files:

- `catalog.runtime.remote.json`
- `catalog.runtime.remote.json.sig`
- `catalog.public.ed25519.txt`

## Authoring workflow

1. Add or update catalog data under `../catalog-fragments/`.
2. Rebuild the runtime catalog:

   `python scripts/compile_model_catalog_v3.py`

3. Publish the remote payload and signature:

   `python scripts/registry/publish_remote_catalog.py --private-key <path-to-ed25519-private-key>`

4. Commit the updated runtime files so the app can fetch them from GitHub Raw.

## Important rules

- The remote catalog is the only control point. New models should be added to
  the catalog, not hardcoded in Electron or Python.
- Artifact sources must be deterministic. Raw Google Drive folders, Proton
  shares, creator pages and repositories are not installable unless the catalog
  provides enough locator metadata for a Rust resolver to identify the exact
  file.
- The private signing key must never be committed. Only the public key and
  detached signature belong in the repository.
