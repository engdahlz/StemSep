# First-class Model Registry Schema Proposal (StemSep)
**Status:** DRAFT (spec-aligned)  
**Owner:** Engineering (AI-assisted)  
**Audience:** Engineers (backend + inference + UI)  
**Policy:** deton24 guide recommendations are treated as **MUST**.  
**Primary goal:** Maximum quality + maximum model/feature coverage with deterministic routing, validation, and clear UX.

---

## 1) Why this exists

The guide repeatedly includes constraints like:

- “incompatible with UVR | use MSST”
- “requires replacing `bs_roformer.py`”
- “pin `neuraloperator==1.0.2`”
- “PyTorch 2.6+ needs `safe_globals` / `strict=False`”
- Phase Fix / Phase Swapper requires specific reference models

These must not live as ad-hoc code branches, string heuristics, or tribal knowledge. To meet strict-spec quality goals, the model registry must be able to represent:

1) **Where** a model can run (runtime routing)  
2) **What** it needs (dependencies, files, manual steps)  
3) **How** to run it well (recommended settings with unambiguous semantics)  
4) **Whether** it can serve special roles (Phase Fix reference, post-processing, etc.)  
5) **What** the app must block/auto-route/auto-configure

---

## 2) Design principles (MUST)

1. **Deterministic routing**
   - A model’s allowed runtimes MUST be explicit (`runtime.allowed`), not inferred from `id` strings.

2. **No silent degradation**
   - If requirements aren’t met, the app MUST fail fast with actionable guidance.
   - If a model cannot run under the selected runtime/device, the app MUST block or auto-route with explicit confirmation/logging.

3. **Backwards compatibility**
   - Existing fields in `models.json.bak` MUST remain valid.
   - New schema must be additive and default-safe.

4. **Architecture-specific parameter semantics must be explicit**
   - “overlap” means different things across engines; registry must specify semantics.

5. **Strict-spec defaults**
   - “Recommended” settings from registry MUST be applied by default unless user explicitly overrides.

---

## 3) Proposed schema (additive fields)

This proposal assumes the registry remains:

- `StemSep/StemSepApp/assets/models.json.bak`
- top-level: `{ "models": [ ... ] }`

Each model entry currently has (example):
- `id`, `name`, `architecture`, `type`, `description`
- `links.checkpoint`, `links.config`
- `vram_required`, `sdr`, `fullness`, `bleedless`
- `recommended_settings` (already present)

### 3.1 `runtime` (new, required for strict mode)
```json
"runtime": {
  "allowed": ["audio-separator", "stemsep-legacy", "msst"],
  "preferred": "audio-separator",
  "blocking_reason": "Incompatible with UVR/audio-separator; requires MSST custom bs_roformer.py"
}
```

**Meaning**
- `allowed`: list of runtimes that MAY execute this model.
- `preferred`: runtime to use by default when multiple are allowed.
- `blocking_reason`: user-facing short reason when a runtime is not allowed.

**Notes**
- Use canonical runtime IDs:
  - `audio-separator`: vendored audio-separator based execution
  - `stemsep-legacy`: in-repo PyTorch model loading via ModelFactory/legacy code path
  - `msst`: MSST-style inference (may require additional repository code / different loader)

### 3.2 `compatibility` (new, optional but strongly recommended)
```json
"compatibility": {
  "uvr_compatible": true,
  "audio_separator_compatible": true,
  "msst_compatible": true,
  "known_issues": [
    "DirectML/AMD may throw 'Invalid parameter' in UVR for some models; prefer MSST/CPU fallback"
  ]
}
```

This exists to preserve common guide constraints in a user-friendly way.  
It must not replace `runtime.allowed`; it complements it.

### 3.3 `requirements` (new, optional; MUST be used when applicable)
```json
"requirements": {
  "python": { "min": "3.10", "max": "3.12" },
  "torch": {
    "min": "2.3.0",
    "max": null,
    "cuda": ["cu118", "cu121"],
    "rocm": []
  },
  "pip": {
    "packages": [
      "neuraloperator==1.0.2"
    ]
  },
  "files": {
    "expects": [],
    "provides": []
  },
  "manual_steps": [
    "MSST: replace models/bs_roformer/bs_roformer.py with HyperACE variant",
    "MSST: add new model_type to keep older BSRoformers functional"
  ]
}
```

**Semantics**
- If `pip.packages` non-empty → preflight MUST verify they are importable/installed for the selected python environment.
- `manual_steps` MUST be surfaced in UI and logs; job must be blocked unless user confirms they have completed them (or the app can perform them automatically, which should be a separate explicit feature).

### 3.4 `settings_semantics` (new, optional but recommended)
Because `recommended_settings.overlap` currently mixes meanings (divisor vs ratio vs seconds), we need semantics:

```json
"settings_semantics": {
  "recommended_settings": {
    "overlap": {
      "meaning_by_runtime": {
        "audio-separator": "divisor_or_seconds",
        "stemsep-legacy": "ratio_0_to_1",
        "msst": "chunk_overlap_divisor"
      },
      "notes": "For roformer in audio-separator path, overlap may be derived from segment_seconds / divisor; for legacy, ratio 0..1."
    }
  }
}
```

This prevents drift and makes UI labeling unambiguous.

### 3.5 `phase_fix` (new, optional; MUST for Phase Fix reference rules)
```json
"phase_fix": {
  "is_valid_reference": true,
  "recommended_params": {
    "lowHz": 500,
    "highHz": 5000,
    "highFreqWeight": 2.0
  },
  "recommended_usage": [
    "Use as reference/source for Phase Fix to reduce Roformer metallic noise"
  ]
}
```

**Semantics**
- If a feature requires a “Phase Fix reference model”, preflight MUST require `is_valid_reference=true`.
- If guide implies only specific models work, those models must be flagged as valid references.

### 3.6 `post_processing` roles/capabilities (new, optional)
```json
"capabilities": {
  "stems": ["vocals", "instrumental"],
  "roles": ["separation", "phase_fix_reference", "denoise", "dereverb", "decrowd"],
  "supports_mono": true,
  "supports_stereo": true
}
```

This is used to:
- power UI filtering (“show only dereverb models”)
- prevent invalid pipelines (e.g. applying mono-only model incorrectly)

### 3.7 `artifacts` (new, optional)
To standardize local filenames and multi-file models:

```json
"artifacts": {
  "primary": { "kind": "checkpoint", "filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt" },
  "config": { "kind": "yaml", "filename": "model_bs_roformer_ep_317_sdr_12.9755.yaml" },
  "additional": []
}
```

This avoids guessing basenames from URLs and helps support models with multiple files.

---

## 4) Enforcement plan (MUST)

### 4.1 Preflight: single source of truth
Preflight MUST:
- validate audio path exists
- validate model exists in registry
- validate `runtime.allowed` contains the runtime implied by the execution path
- validate requirements:
  - required local artifacts exist (or are downloadable)
  - python/torch/pip constraints (best-effort)
- validate Phase Fix / pipeline references:
  - reference model must be flagged `phase_fix.is_valid_reference=true`

**Output**
Preflight response MUST include:
- `can_proceed: bool`
- `errors[]`
- `warnings[]`
- `resolved.runtime`
- `resolved.settings` (with semantics applied)
- `resolved.requirements_status`

### 4.2 Routing: runtime selection rules
Given a requested `model_id` and optional user preference:
1) If user explicitly chose a runtime:
   - MUST check it’s in `runtime.allowed`, else block
2) Else:
   - MUST choose `runtime.preferred` if present, else first in `runtime.allowed`

No string heuristics (“if ‘hyperace’ in id”) should be needed once the registry is filled.

### 4.3 UI: make constraints visible
UI MUST:
- show runtime compatibility (UVR/audio-separator vs MSST-only)
- show “special requirements” in plain language
- block invalid selections early (before download or job start)
- provide a “why blocked” explanation using `blocking_reason` and `manual_steps`

---

## 5) Migration plan (incremental, safe)

### Phase 1 — Add schema fields without enforcement
- Add fields (`runtime`, `requirements`, `phase_fix`, `capabilities`, `settings_semantics`, `artifacts`) to registry entries gradually.
- Keep old behavior working.
- Add a registry validator script to ensure schema correctness.

### Phase 2 — Enforce in preflight only (soft block)
- Preflight reports missing requirements but may allow override only if a dev flag is set.
- Collect real-world errors.

### Phase 3 — Enforce in execution (hard block)
- Start jobs only when preflight passes.
- Remove ad-hoc heuristics once registry coverage is complete.

---

## 6) Example model entries

### 6.1 Standard Roformer (audio-separator preferred)
```json
{
  "id": "bs-roformer-viperx-1297",
  "name": "BS-Roformer-Viperx-1297",
  "architecture": "BS-Roformer",
  "links": { "checkpoint": "https://...", "config": "https://..." },
  "recommended_settings": { "segment_size": 352800, "batch_size": 2, "tta": false, "overlap": 4 },
  "runtime": { "allowed": ["audio-separator", "stemsep-legacy"], "preferred": "audio-separator" },
  "compatibility": { "uvr_compatible": true, "audio_separator_compatible": true, "msst_compatible": true },
  "phase_fix": { "is_valid_reference": true, "recommended_params": { "lowHz": 500, "highHz": 5000, "highFreqWeight": 2.0 } },
  "capabilities": { "stems": ["vocals", "instrumental"], "roles": ["separation", "phase_fix_reference"] }
}
```

### 6.2 MSST-only model requiring pinned dependency + manual step
```json
{
  "id": "bs-roformer-hyperace-v2-inst",
  "name": "BS-Roformer HyperACE v2 (Instrumental)",
  "architecture": "BS-Roformer",
  "links": { "checkpoint": "https://...", "config": "https://..." },
  "runtime": {
    "allowed": ["msst"],
    "preferred": "msst",
    "blocking_reason": "Incompatible with UVR/audio-separator; requires HyperACE bs_roformer.py"
  },
  "requirements": {
    "pip": { "packages": ["neuraloperator==1.0.2"] },
    "manual_steps": [
      "MSST: replace models/bs_roformer/bs_roformer.py with HyperACE variant",
      "MSST: add a new model_type to avoid breaking older BSRoformers"
    ]
  },
  "compatibility": { "uvr_compatible": false, "audio_separator_compatible": false, "msst_compatible": true }
}
```

---

## 7) Open questions (to resolve during guide extraction)

1) What exact overlap semantics does the guide prescribe per engine (divisor vs ratio vs seconds)?
2) Which exact models are valid Phase Fix references (must become `phase_fix.is_valid_reference=true`)?
3) Output policy: do we mandate float WAV always for “pro/nulling mode”, and how is this exposed in UI?
4) Do we need “device constraints” per model (e.g. DirectML/AMD issues) as first-class constraints?

---

## 8) Next steps (implementation tasks)

1) Define canonical runtime IDs and implement deterministic routing.
2) Extend registry JSON schema and create a validator.
3) Implement preflight checks for:
   - runtime allowed
   - required artifacts
   - dependency checks (best-effort)
   - phase-fix reference validity
4) Update UI to surface compatibility and requirements.
5) Populate registry fields for:
   - MSST-only models
   - phase-fix reference models
   - post-processing models (denoise/dereverb/decrowd/demudder)

---