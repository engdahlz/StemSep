# Contributing

Thanks for your interest in contributing!

## Quick start

- Install Node.js (recommended: Node 20 LTS).
- Install UI deps:
  - `npm run install:ui`
- Run the Electron UI:
  - `npm run electron:dev`

## What to work on

- Check the issue tracker (bugs, feature requests).
- If you’re unsure, open an issue first and describe the change you want to make.

## Development workflow

- Keep changes small and focused.
- Prefer a single source of truth (types/config/registry IDs) — avoid duplicating mappings.
- Add/adjust tests when behavior changes.

### UI checks

From repo root:

- Typecheck: `npm run ui:typecheck`
- Tests: `npm run ui:test`

## Pull requests

Please include:

- What changed and why
- How you tested it (commands + OS)
- Any screenshots for UI changes

## Code style

- Follow existing formatting.
- Avoid large refactors mixed with functional changes.

## Notes on models

This repository can download model weights from third-party sources.
Make sure any new model entries include:

- A stable `model_id`
- A working download URL
- Clear notes about licensing/usage constraints where known
