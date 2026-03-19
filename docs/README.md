# Docs Index

`docs/` holds reference material that helps the product and runtime stay aligned without mixing source code with vendor research.

## Layout

| Path | Purpose |
| --- | --- |
| `backend_contract/` | Schemas and contracts shared with the backend bridge |
| `guide/` | Model and guide mapping notes used during registry/runtime work |
| `vendor/` | Curated upstream references that should stay versioned with the repo |

## Recommended entry points

- Backend schema: [backend_contract/backend_api_v1.schema.json](./backend_contract/backend_api_v1.schema.json)
- Guide mapping: [guide/model_id_mapping.md](./guide/model_id_mapping.md)
- Vendor reference index: [vendor/README.md](./vendor/README.md)

## Conventions

- Keep product documentation concise and close to the code that owns it.
- Store heavyweight external references under `docs/vendor/`.
- Avoid dropping temporary research dumps or generated artifacts directly into `docs/`.
