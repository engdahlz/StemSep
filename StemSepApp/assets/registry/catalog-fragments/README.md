# Catalog Fragments

This directory is the remote-first authoring surface for adding new catalog entries without editing the monolithic `catalog.v3.source.json` by hand.

Supported fragment folders:

- `models/`
- `recipes/`
- `workflows/`
- `sources/`
- `external-records/`

Each JSON file should contain exactly one object. During compilation, fragment entries override monolithic entries when they share the same stable identifier:

- models: `model_id` or `id`
- recipes: `selection_id` or `id`
- workflows: `selection_id` or `id`
- sources: `source_id` or `id`
- external records: `record_id` or `id`

The compile pipeline still uses `catalog.v3.source.json` as the base dataset today, but new additions should prefer fragments so they can be reviewed, verified, and published independently.

`sources/` is the preferred authoring surface for installer-grade artifacts. Model artifacts may still embed inline `sources[]` for backward compatibility, but the v4.1 pipeline normalizes those into top-level source records and prefers explicit `source_id` references.
