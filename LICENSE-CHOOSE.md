# Choosing a license

StemSep currently does not ship a finalized open-source license text.

## Why this file exists

We don’t want to accidentally publish under the wrong terms.
Pick a license intentionally, then add it as `LICENSE` at the repo root.

## Common options

- **MIT**: permissive, very common for developer tools.
- **Apache-2.0**: permissive + explicit patent grant.
- **GPL-3.0**: strong copyleft (derivatives must be GPL).

If you’re unsure, MIT or Apache-2.0 are usually the simplest choices.

## Important: model weights

This repo can download model weights from third parties.
Those weights often have their own licenses/terms.

A project license applies to this repository’s code and docs — it does **not** automatically grant rights to redistribute or use third-party model weights.

## What to do

1. Decide on a license (MIT / Apache-2.0 / GPL-3.0 / other).
2. Add the chosen license text to a new root file named `LICENSE`.
3. Update the README “License” section to match.
