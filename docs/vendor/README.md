# Vendor References

This folder keeps the minimal local references that are still useful for StemSep development.

## Kept locally

- `deton24_guide/latest.txt`
  - Current local snapshot of the community guide used for registry/workflow curation.
- `msst_reference/`
  - Small, read-only MSST/ZFTurbo reference set with the upstream README, a few key docs, the main inference/ensemble scripts, and representative configs.

## Rules

- Do not import runtime code from `docs/vendor/`.
- Keep this folder small and intentionally curated.
- Prefer text/config references over media dumps, screenshots and per-page extraction artifacts.

## Runtime source of truth

- Runtime code: `StemSepApp/src/`
- Registry and recipes: `StemSepApp/assets/`
- Vendor references: `docs/vendor/`
