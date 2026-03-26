# StemSep Remote-First Catalog

This directory contains the authored source for StemSep's remote-first runtime catalog.

The running app and Rust backend now operate on the v4 runtime chain:

- `StemSepApp/assets/catalog.runtime.json`
- `StemSepApp/assets/catalog.runtime.bootstrap.json`
- `StemSepApp/assets/registry/remote/catalog.runtime.remote.json`

The dedicated `StemSep-catalog` repo is the operational publication target for the signed
remote payload. The copies under `StemSepApp/assets/registry/remote/` exist so the app repo
can validate and bootstrap the same signed payload locally.

## Files

- `catalog-fragments/`
  - Authoritative fragments for models, recipes, workflows, sources and external records.
  - Add/update catalog entries here; do not hardcode new download links in app code.

- `catalog.v3.source.json`
  - Base source document merged with fragments during compilation.

- `remote/`
  - App-repo mirror of the currently signed remote runtime payload, signature and public key.
  - These files must stay byte-for-byte in sync with `StemSep-catalog`.

## Generator workflow (single source of truth)

1) Update or ingest fragments:
- edit `catalog-fragments/...` directly, or
- run an ingest helper under `scripts/registry/`

2) Verify sources:
- `python scripts/registry/verify_catalog_v3_sources.py --update-source-fragments`

3) Compile runtime outputs:
- `python scripts/compile_model_catalog_v3.py`

4) Publish and sign:
- `python scripts/registry/release_catalog.py --private-key <pem> --catalog-repo-root ..\\StemSep-catalog`

## Important rules

- Do **not** hand-edit `catalog.runtime*.json` or `remote/catalog.runtime.remote.json`.
- Keep authored changes in fragments/source inputs and regenerate.
- `StemSepApp/assets/models.json.bak` is legacy audit input only. It is no longer the operational runtime registry and no longer blocks the default quality gate.
- If a source cannot be resolved deterministically (for example raw Proton/Drive folder shares), mark it `manual` or `reference` instead of forcing it into auto-download.

## Notes

The project still keeps external guide findings as research input, but the shipping runtime truth is now the signed v4 catalog. Selection-first installation, provenance, fallback and verification all flow from that remote-first catalog chain.
