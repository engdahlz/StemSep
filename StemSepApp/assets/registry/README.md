# StemSep Model Registry (v2)

This directory contains the **source-of-truth** model registry in a robust v2 format, plus its JSON Schema.

The running app currently reads the legacy registry file:

- `StemSepApp/assets/models.json.bak`

…but that file is **generated** from the v2 source to keep the system maintainable long-term.

## Files

- `models.v2.source.json`
  - **Edit this** when adding/updating models.
  - Designed to express:
    - deterministic runtime routing (which engine is allowed to run a model)
    - compatibility constraints (e.g. “MSST-only”, “incompatible with UVR/audio-separator”)
    - special requirements (pinned Python deps, manual steps, file expectations)
    - phase-fix metadata (e.g. which models are valid phase-fix references)
    - recommended settings with explicit semantics

- `models.v2.schema.json`
  - JSON Schema for `models.v2.source.json`.
  - Used for validation to prevent drift and silent breakage.

## Generator workflow (single source of truth)

1) Update the v2 registry source:
- Edit `models.v2.source.json`

2) (Optional) Validate against schema:
- Validation is supported by the generator script when a JSON Schema validator is available in your Python environment.

3) Generate the legacy registry used by the app:
- Run from repo root:
  - `python scripts/generate_model_registry.py --pretty`

This writes:
- `StemSepApp/assets/models.json.bak`

## Important rules

- Do **not** hand-edit `StemSepApp/assets/models.json.bak`.
- Prefer **additive** changes to keep backward compatibility.
- If the guide/recommendations mention a constraint (e.g. MSST-only, special inference script, dependency pin), encode it explicitly in the v2 source registry so it can be enforced by preflight/routing and surfaced in the UI.

## Notes

The project treats the deton24 guide’s recommendations as **MUST** (strict-spec mode). The v2 registry exists specifically to encode those requirements in a deterministic, machine-checkable way, instead of relying on ad-hoc code paths or user guesswork.