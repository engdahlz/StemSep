# Scripts

`scripts/` contains developer tooling, not product code.

## Layout

- `scripts/inference.py`: Python runner spawned by the Rust backend
- `scripts/registry/`: registry, guide-sync and catalog maintenance
- `scripts/dev/`: debugging and investigation tools
- `scripts/qa/`: QA/reporting helpers
- `scripts/download/`: model/bootstrap helpers
- `scripts/mdx/`: MDX validation helpers
- `scripts/analysis/`: experimental analysis utilities

## Stable root scripts

- `inference.py`
- `generate_model_registry.py`
- `verify_backend.py`
- `backend_headless_download.py`
- `backend_headless_separate.py`
- `run_backend_separation.py`
- `run_py.ps1`
- `run_py.sh`
- `run_pytest.ps1`
- `setup_backend_venv.ps1`

Everything else should prefer a subfolder.

## Common commands

```powershell
.\scripts\run_py.ps1 .\scripts\registry\validate_model_registry.py
.\scripts\run_py.ps1 .\scripts\registry\sync_guide_knowledge.py
.\scripts\run_pytest.ps1 -q StemSepApp/tests
```

## Notes

- Run scripts from the repo root unless a script says otherwise.
- Many registry/download tools mutate files under `StemSepApp/assets/`.
- Generated reports belong under ignored output folders, not in git.
