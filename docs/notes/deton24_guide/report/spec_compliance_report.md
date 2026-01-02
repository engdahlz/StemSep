# deton24 guide → Spec Compliance Report (StemSep)

**Status:** IN PROGRESS  
**Policy:** Guide recommendations are treated as **MUST** (strict-spec mode).  
**Guide source (canonical in repo):** `docs/vendor/deton24_guide/deton24_guide.txt`  
**Repo:** StemSep (Electron UI + Rust backend + Python inference)  
**Generated/maintained by:** Engineering review (AI-assisted)

**Reference formats (optional):**
- `docs/vendor/deton24_guide/deton24_guide.pdf`
- `docs/vendor/deton24_guide/deton24_guide.docx`


---

## 1) Executive summary

This report evaluates StemSep’s implementation against the deton24 guide as the authoritative **spec**.

- **Goal:** Max quality + max feature/model coverage, while avoiding ambiguity and long-term drift.
- **Method:** Extract requirements from guide → map to code → mark compliance → propose changes + verification.

**Top risks (early signals):**
- Phase Fix semantics likely diverge from guide intent (see D-001).
- Model compatibility (UVR/audio-separator vs MSST-only) appears partially enforced via ad-hoc special-casing rather than systematic metadata and routing.

---

## 2) Scope & non-goals

### In scope
- End-to-end separation flows: UI → backend → inference → outputs
- Model installation/registry behavior (download/resolve/config)
- All quality-impacting features mentioned in guide that StemSep aims to support:
  - Phase Fix / Phase Swapper
  - Ensemble blending (Average / Max Spec / Min Spec) including per-stem algorithms
  - Overlap / segment_size / chunk_size semantics per engine
  - Post-processing (demudder variants, denoise, dereverb, bleed suppression, etc.)
  - Output invariants (duration/length, sample rate assumptions, bit depth, format conversion, metadata)

### Non-goals (for this report)
- UI polish / styling
- Performance benchmarking beyond “must not violate quality constraints”
- Legal/licensing evaluation of model redistribution (not addressed here)

---

## 3) Compliance legend

### Status
- ✅ **Compliant** — matches guide **exactly**
- ⚠️ **Partial** — implemented, but semantics/defaults differ or edge cases missing
- ❌ **Non-compliant** — required behavior missing or contradicted
- ❓ **Unknown** — not yet fully traced/extracted

### Evidence levels
- **E0**: Assumption
- **E1**: Code location identified
- **E2**: Code behavior verified against requirement
- **E3**: Deterministic test confirms behavior (golden input / expected output)

---

## 4) System architecture (authoritative execution path)

### Primary path (current)
1. Electron (renderer) → Electron main  
2. Electron main → Rust backend via NDJSON over stdio  
3. Rust backend spawns Python `scripts/inference.py` per separation job  
4. Python runner uses `StemSepApp/src`:
   - `core/separation_manager.py`
   - `audio/separation_engine.py` (selects engine: audio-separator path vs legacy)
   - `audio/ensemble_blender.py` (blending + phase fix)
5. Python emits events:
   - `separation_started`, `separation_progress`, `separation_complete`, `separation_error`
6. Rust backend forwards events (forces job_id), UI consumes and presents.

---

## 5) Requirements index (high-level)

This section is a table of the extracted requirements. Each requirement has a detailed entry below.

| ID | Domain | Title | Status | Evidence |
|----|--------|-------|--------|----------|
| REQ-MODELS-001 | models | Compatibility metadata must be represented and enforced (UVR/audio-separator vs MSST-only) | ⚠️ | E1 |
| REQ-MODELS-003 | models | First-class model registry schema for runtime/compatibility/requirements MUST exist and be enforced | ❌ | E0 |
| REQ-MODELS-002 | models | MSST-only models: special install requirements must be representable and enforceable | ❓ | E0 |
| REQ-PHASEFIX-001 | phase_fix | Phase Fix workflow must match guide exactly | ⚠️ | E1 |
| REQ-PHASEFIX-002 | phase_fix | Phase Fix reference model constraints (only specific models work as references) | ❓ | E0 |
| REQ-OUTPUT-001 | output | Output duration/length invariants must hold (validate/retry deterministically) | ⚠️ | E1 |
| REQ-OUTPUT-002 | output | Float output / clipping rules must match guide (and MVSEP behavior where relied upon) | ❓ | E0 |

> Note: This table will expand substantially as the full guide is extracted.

---

## 6) Detailed requirements

### REQ-MODELS-003: First-class model registry schema for runtime/compatibility/requirements MUST exist and be enforced
- **Domain:** models
- **Guide location:** Numerous recurring constraints and recommendations, including (examples):
  - “incompatible with UVR | use MSST”
  - models requiring custom inference files (e.g. replaced `bs_roformer.py`)
  - pinned dependencies (e.g. `neuraloperator==1.0.2`)
  - PyTorch/load behavior constraints (safe globals, `strict=False`)
- **Requirement (MUST):**
  - MUST represent model execution constraints and requirements as first-class registry metadata (not scattered ad-hoc special-cases).
  - MUST enforce these constraints during:
    - model listing/selection (UI/UX)
    - preflight (blocking before long-running jobs)
    - job execution routing (correct engine/runtime)
    - model download/import/installation validation
- **Registry schema (MUST)**
  - Each model entry MUST support the following fields (names can differ, but semantics must match exactly):

    1) `runtime`
       - **Purpose:** declare which inference runtime(s) may execute the model.
       - Allowed values (example): `["audio-separator"]`, `["msst"]`, `["legacy"]`, or `["audio-separator","msst"]`
       - MUST be used to route execution (no guessing from `model_id` strings).

    2) `compatibility`
       - **Purpose:** make hard constraints explicit and machine-checkable.
       - Example shape:
         - `uvr_compatible: boolean`
         - `audio_separator_compatible: boolean`
         - `msst_compatible: boolean`
         - `notes: string?` (human-readable)

    3) `requirements`
       - **Purpose:** encode additional requirements that must be satisfied for correct quality/function.
       - Example shape:
         - `python: { min_version?: string, max_version?: string }`
         - `torch: { min_version?: string, max_version?: string, cuda?: string[] , rocm?: string[] }`
         - `pip: { packages?: string[] }` (e.g. `["neuraloperator==1.0.2"]`)
         - `files: { expects?: string[], provides?: string[] }`
         - `manual_steps?: string[]` (must be shown to user; cannot be silently skipped)

    4) `phase_fix`
       - **Purpose:** encode phase-fix-specific constraints.
       - Example shape:
         - `is_valid_reference: boolean`
         - `recommended_reference_for?: string[]` (model ids or architecture tags)
         - `recommended_params?: { lowHz?: number, highHz?: number, highFreqWeight?: number }`

    5) `recommended_settings` (already exists)
       - MUST have unambiguous semantics (e.g. overlap divisor vs ratio vs seconds must be defined per runtime/architecture).

- **Current implementation (observed):**
  - Registry contains `recommended_settings`, but runtime/compatibility/requirements are not confirmed as first-class fields and are partially inferred via ad-hoc logic.
- **Status:** ❌ Non-compliant (E0) — required schema + enforcement not yet implemented.
- **Impact:** Extremely high
  - Without first-class metadata, the app cannot reliably provide “max quality + max model coverage” in strict-spec mode.
- **Proposed fix direction (not yet implemented):**
  - Extend `assets/models.json.bak` schema with the fields above.
  - Update:
    - backend `separation_preflight` to validate registry requirements and runtime routing deterministically
    - Python `ModelManager` to expose these fields
    - UI to display constraints and block invalid selections
- **Verification plan:**
  - Unit tests: parsing/validation of registry entries; routing decisions are deterministic.
  - Integration tests: selecting an MSST-only model without requirements yields a clear actionable error.
  - Regression tests: ensure legacy models still run unchanged when their runtime allows it.


### REQ-MODELS-001: Model compatibility metadata must be represented and enforced
- **Domain:** models
- **Guide location:** Multiple entries across updates/news; recurring “incompatible with UVR | use MSST” notes
- **Requirement (MUST):**
  - MUST represent whether a model is UVR/audio-separator compatible vs MSST-only / incompatible.
  - MUST enforce routing/guardrails so incompatible models cannot run through the wrong engine.
  - MUST preserve special installation requirements (custom inference scripts, replaced files) as first-class metadata or documented constraints in-app/docs.
- **Current implementation (observed):**
  - HyperACE is special-cased to avoid audio-separator path (forced legacy path).
  - No confirmed first-class compatibility metadata in registry at report time.
- **Status:** ⚠️ Partial (E1)
- **Impact:** High (wrong engine → wrong outputs or runtime failures)
- **Proposed fix direction (not yet implemented):**
  - Add explicit compatibility flags to model registry (e.g. `runtime`, `uvr_compatible`, `requires_custom_code`).
  - Enforce routing in `SeparationEngine.separate` and/or higher-level job creation.
- **Verification plan:**
  - Unit tests: selecting MSST-only model must be blocked/routed deterministically.
  - Integration tests: backend preflight must detect missing/invalid runtime.

---

### REQ-PHASEFIX-001: Phase Fix workflow must match guide exactly
- **Domain:** phase_fix
- **Guide location:** Phase Fix section (explicit), plus multiple model notes referencing phase fix requirements
- **Requirement (MUST):**
  - MUST implement Phase Fix / Phase Swapper exactly:
    - reference model selection logic
    - frequency band defaults and overrides (lowHz/highHz)
    - weight semantics (highFreqWeight)
    - final combination algorithm (e.g. Max Spec vs Average) as specified
  - MUST produce deterministic, inspectable outputs (optionally intermediate artifacts if guide implies it).
- **Current implementation (observed):**
  - Pipeline supports explicit `action == "phase_fix"` step:
    - separate reference model → `_phase_fix_blend(target, ref_vocal)` → `max_spec` with ref_inst
  - Ensemble supports phase fix as a pre-step when enabled.
- **Status:** ⚠️ Partial (E1) — requires exact spec extraction for semantics validation
- **Impact:** Very high (core quality feature; subtle semantic mismatch yields wrong cleanup)
- **Verification plan:**
  - Deterministic test tracks: measure phase alignment improvement in band (before/after).
  - Ensure weight semantics match guide (see deviation D-001).

---

### REQ-MODELS-002: MSST-only models: special install requirements must be representable and enforceable
- **Domain:** models
- **Guide location:** MSST-only model notes and install instructions (e.g. HyperACE/FNO guidance; Torch version constraints; additional libs)
- **Requirement (MUST):**
  - MUST represent “MSST-only / incompatible with UVR/audio-separator” as a first-class constraint.
  - MUST represent *special installation requirements* for certain MSST-only models, including but not limited to:
    - Replacing or customizing `bs_roformer.py` in MSST
    - Adding a new `model_type` so older models remain functional
    - Pinning dependencies (example: `neuraloperator==1.0.2` for FNO1d compatibility)
    - PyTorch version constraints and security loading changes (`torch.serialization.safe_globals`, `strict=False` guidance)
  - MUST prevent users from unknowingly running a model without these requirements satisfied (hard error with actionable remediation).
- **Current implementation (observed):**
  - Some behaviors exist as ad-hoc special-casing (e.g. HyperACE forcing non-audio-separator path), but no confirmed structured metadata or install-check enforcement.
- **Status:** ❓ Unknown (E0) — requires full extraction + code tracing.
- **Impact:** Very high (silent wrong inference path or broken installs degrade quality or fail at runtime).
- **Proposed fix direction (not yet implemented):**
  - Extend the model registry schema with:
    - `compatibility`: `{ runtime: "audio-separator" | "msst" | "either", uvr_compatible: bool, notes?: string }`
    - `requirements`: `{ python?: {torch_min?, torch_max?, extra_pip?: [...]}, files?: {...}, manual_steps?: [...] }`
  - Add preflight checks that validate these requirements before job start.
- **Verification plan:**
  - Unit tests: registry entries parse and produce expected “blocked/routed” decisions.
  - Integration tests: selecting an MSST-only model without required deps yields deterministic, actionable error.

---

### REQ-OUTPUT-001: Output duration/length invariants must hold (validate/retry deterministically)
- **Domain:** output
- **Guide location:** (pending extraction of explicit statements; treated as MUST due to strict-spec quality policy)
- **Requirement (MUST):**
  - MUST ensure output stems match input duration within defined tolerance.
  - MUST fail fast or retry deterministically if invariants fail.
- **Current implementation (observed):**
  - `scripts/inference.py` validates output duration via `soundfile.info` and retries once.
- **Status:** ⚠️ Partial (E1) — mechanism exists; full spec acceptance criteria pending
- **Verification plan:**
  - Unit test for duration validation logic.
  - Integration test: intentionally induce mismatch, assert retry and/or explicit error.

---

### REQ-PHASEFIX-002: Phase Fix reference model constraints (only specific models work as references)
- **Domain:** phase_fix
- **Guide location:** Statements indicating that only certain models work for phase fixer/swapper references (e.g. “none models work for phase fixer/swapper besides 1296/1297 … and unwa BS Large V1”).
- **Requirement (MUST):**
  - MUST encode and enforce which models are valid as Phase Fix reference/source models.
  - MUST prevent invalid phase-fix combinations (or auto-select a valid reference model if the guide implies a default).
  - MUST expose the reference selection logic deterministically (no “mystery behavior”).
- **Current implementation (observed):**
  - Phase Fix exists in ensemble and pipeline flows, but reference-model constraints are not confirmed as enforceable rules; current behavior may allow any reference model.
- **Status:** ❓ Unknown (E0) — requires full extraction + code trace.
- **Impact:** High (wrong reference model can remove instruments or fail to reduce artifacts, violating spec).
- **Proposed fix direction (not yet implemented):**
  - Add `phase_fix: { is_valid_reference: bool, recommended_reference_for?: [...] }` to model registry.
  - In UI/backend preflight: validate `(target_model, reference_model)` pair.
- **Verification plan:**
  - Deterministic test matrix: known-good reference models improve band-limited phase metric; invalid references are blocked.

---

### REQ-OUTPUT-002: Float output / clipping rules must match guide (and MVSEP behavior where relied upon)
- **Domain:** output
- **Guide location:** Notes describing MVSEP output behavior:
  - 32-bit float WAV used only when gain exceeds 1.0 to prevent clipping; otherwise 16-bit PCM
  - “If you really need it anyway, 32-bit float output unconditionally is paid”
  - FLAC using 16-bit and normalization constraints; “if stems don’t invert correctly, use WAV output format - it's 32-bit float”
- **Requirement (MUST):**
  - MUST implement output format/bit-depth rules consistent with guide expectations:
    - If the guide workflow depends on nulling/inversion, StemSep MUST provide a path that preserves invertibility (typically WAV float).
    - MUST avoid unintended normalization/clipping that breaks null tests, unless explicitly enabled and surfaced.
  - MUST make this behavior explicit in UI/config (no silent format changes that degrade invertibility).
- **Current implementation (observed):**
  - Python separation path uses WAV float in several places, but exact “conditional float vs PCM” logic and normalization defaults are not confirmed against guide.
- **Status:** ❓ Unknown (E0) — requires extraction of full output section + code tracing.
- **Impact:** High for pro workflows (Atmos, DAW reintegration, nulling).
- **Proposed fix direction (not yet implemented):**
  - Define canonical output policy:
    - “Pro/nulling mode” outputs float WAV unconditionally
    - “Consumer mode” may use conditional float/PCM with explicit toggles
  - Ensure normalization is opt-in and documented.
- **Verification plan:**
  - Golden test: take mixture and stems, verify inversion/null within tolerance when configured for nulling.

---

## 7) Deviations / issues list (tracked)

| ID | Domain | Summary | Severity | Status |
|----|--------|---------|----------|--------|
| D-001 | phase_fix | `highFreqWeight > 1.0` semantics likely lost due to clamping | High | ⚠️ |

### D-001: Phase Fix “highFreqWeight” semantics likely incorrect for values > 1.0
- **Observation:** Implementation clamps blend factor to max 1.0 (“full replace”), making 2.0 not more aggressive than 1.0.
- **Why it matters:** Guide/presets describe values like 2.0 as meaningful; strict-spec mode treats those recommendations as MUST.
- **Next action:** Extract the guide’s Phase Fix section fully; define intended semantics; update implementation and tests.

---

## 8) Verification strategy (how we prove compliance)

### Test layers
1. **Unit tests** (fast, deterministic)
   - parameter normalization
   - output invariants checks
   - phase-fix math properties (band-limited phase alignment)
2. **Integration tests**
   - end-to-end separation job: backend → python → outputs
   - ensure correct event sequence and job_id coherence
3. **Golden audio tests (where feasible)**
   - fixed short clips with known expected behaviors
   - stored as small fixtures; compare via:
     - duration, sample rate
     - energy distribution
     - band-limited phase metrics
     - optionally perceptual metrics if stable

---

## 9) Change plan (placeholder)

This section will be filled after full extraction of the guide and mapping is complete.

- Patch 1: Model compatibility metadata + routing enforcement
- Patch 2: Phase Fix semantics alignment + deterministic tests
- Patch 3: Overlap/chunk semantics harmonization per engine
- Patch 4: Post-processing workflows parity (demudder variants, etc.)
- Patch 5: Output format/bit depth/metadata rules alignment

---

## 10) Open questions (to resolve during extraction)

- What exact semantics does the guide intend for phase-fix “weight” > 1.0?
- What is the canonical overlap definition per architecture (divisor vs ratio vs seconds), and how should UI represent it without ambiguity?
- Which guide “recommendations” are “global MUST” vs “architecture-conditional MUST”?

---