# Walkthrough: Backend Verification and Codebase Cleanup

## Quick QA (Dec 2025) – Simple presets + Multi-step
- New Separation → drop/click to choose file → Configure Separation.
- Simple Mode:
	- Pick a preset (use Goal/Mode/Search).
	- Toggle `Multi-step ON/OFF` to include/exclude recipe/pipeline/chained presets.
	- Toggle `Details ON/OFF` to show/hide technical model IDs.
	- Verify multi-step presets show “Multi-step preset” and a collapsed “What happens?” step list.
- Missing models:
	- Select a preset that requires uninstalled models → Start should be disabled.
	- Click “Go to Model Browser” (or “Download Missing Models”) and verify you land on the Models area/details.
- Run:
	- Start Separation → progress updates → Results preview plays.
	- Discard should remove staged preview stems (cache cleanup).

## Overview
This session focused on verifying the core backend logic, resolving model loading issues, and cleaning up the codebase.

## Key Achievements

### 1. Backend Verification
- **Model Loading**: Fixed `TypeError: BSRoformer.__init__() got an unexpected keyword argument 'n_fft'` by updating `ModelFactory` to match the new `bs_roformer.py` implementation.
- **Inference**: Resolved tensor shape mismatches and padding errors in `SeparationEngine`.
- **Verification Script**: Successfully ran `verify_core_engine.py`, confirming that the backend can load models and perform separation.

### 2. Codebase Cleanup
- **Audit**: Identified and removed temporary files, unused scripts, and debug artifacts.
- **Organization**: Moved test scripts to a dedicated `tests` directory in `electron-poc`.
- **Cleanup**: Deleted `verify_*.py` scripts and temporary output folders from the root.

### 3. Application Startup Fix
- **Issue**: After cleanup, the Electron app failed to start with `TypeError: Cannot read properties of undefined (reading 'getPath')`.
- **Root Cause**: The `ELECTRON_RUN_AS_NODE` environment variable was set to `1`, causing `require('electron')` to return the executable path string instead of the API object.
- **Resolution**: Unset `ELECTRON_RUN_AS_NODE` and verified successful application startup.

## Verification Results
- **Backend**: `verify_core_engine.py` passed.
- **Frontend**: Electron application starts successfully, spawns the Python bridge, and initializes the UI.
- **Communication**: IPC between Electron and Python is functioning (confirmed by logs).

## Next Steps
- Proceed with functional testing of the application via the UI.
