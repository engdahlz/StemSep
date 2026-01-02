# deton24 guide → StemSep spec-compliance index

## Spec policy (strict mode)
**Rule:** Everything presented in the guide as a recommendation is treated as **MUST**.

Interpretation notes:
- If the guide says something “works best”, “recommended”, “use X”, “you should”, or provides a default/preset as the recommended choice → treat as **MUST**.
- Community quotes are treated as **supporting evidence**, not normative by themselves, unless the guide text explicitly elevates them into a recommendation.
- Requirements may be **architecture-conditional MUST** (e.g. “incompatible with UVR, use MSST”) and must be enforced as constraints, not ignored.

Guide canonical location in this repo:
- `docs/vendor/deton24_guide/deton24_guide.txt` (source-of-truth copy used for spec extraction)
- Optional reference formats:
  - `docs/vendor/deton24_guide/deton24_guide.pdf`
  - `docs/vendor/deton24_guide/deton24_guide.docx`

---

## 10) Initial extraction (updates/news section)

Source slice reviewed (partial): guide “Last updates and news” + early updates feed.

### REQ-MODELS-001: Model compatibility metadata must be represented and enforced
- **Domain:** models
- **Guide lines:** (early “Last updates and news”; multiple entries)
- **Requirement:**
  - MUST represent whether a model is UVR/audio-separator compatible vs **MSST-only** / “incompatible with UVR”.
  - MUST surface these constraints in the user-facing selection flow (prevent/guard invalid execution paths).
  - MUST preserve special installation instructions for MSST-only models (custom inference scripts / replaced files) in internal docs.
- **Input contract:**
  - Model registry must carry a compatibility flag(s), e.g.:
    - `runtime: ["audio-separator"]` vs `runtime: ["msst"]`
    - or `uvr_compatible: bool`
    - and optional `special_install: {...}` for cases like HyperACE/FNO.
- **Expected behavior:**
  - Selecting MSST-only model in a UVR/audio-separator path MUST be blocked or auto-routed to the MSST-capable runtime (if available).
- **Code mapping (current):**
  - `StemSepApp/src/audio/separation_engine.py` has a special-case forcing legacy path for HyperACE:
    - `if "hyperace" in arch or "hyperace" in model_id.lower(): use_zfturbo_engine = False`
  - Model compatibility is not yet clearly modeled as first-class metadata in `assets/models.json.bak` (needs verification).
- **Compliance:** ⚠️ (E1) — partial (some ad-hoc special casing exists; spec requires systematic metadata + enforcement).
- **Verification plan:** Add deterministic tests that attempt to run an MSST-only model through audio-separator path and assert it is blocked/routed.

### REQ-PHASEFIX-001: Phase Fix workflow must match the guide exactly (not “close enough”)
- **Domain:** phase_fix
- **Guide lines:** (update mentions “phase fixer section rewritten”; later sections likely contain normative workflow)
- **Requirement:**
  - MUST implement the guide’s Phase Fix / Phase Swapper workflow exactly:
    - reference/source model selection rules
    - band limits (lowHz/highHz)
    - weight semantics (highFreqWeight)
    - final combination method (e.g. Max Spec vs Avg) and target stem selection
  - MUST support workflows described as “process instrumental, then use as source in phase fix tool” equivalents.
- **Code mapping (current):**
  - `StemSepApp/src/core/separation_manager.py` implements pipeline `action == 'phase_fix'` by:
    - running `source_model` separation
    - `_phase_fix_blend(target, ref_vocal, low/high/weight)`
    - `max_spec` blend with `ref_inst`
  - `StemSepApp/src/audio/ensemble_blender.py` implements `_phase_fix_blend` + `phase_fix_workflow`.
- **Compliance:** ⚠️ (E1) — implemented, but must be validated against the guide’s exact algorithm and parameter semantics.
- **Verification plan:** Golden test song segment where phase fix effect is measurable; assert phase alignment improvement within band.

### REQ-OUTPUT-001: Output duration/length invariants must hold (or must be validated)
- **Domain:** output
- **Guide lines:** (not yet extracted; but required by “minsta lilla fel kan förstöra allt” constraints)
- **Requirement:**
  - MUST ensure stem outputs match input duration (within tolerance), even across resample paths.
  - MUST fail fast or retry deterministically if outputs are invalid.
- **Code mapping (current):**
  - `scripts/inference.py` validates output duration via `soundfile.info` and retries once.
- **Compliance:** ✅/⚠️ (E1) — mechanism exists; exact spec requirements still pending full extraction.
- **Verification plan:** Add unit test for validation logic (already present in repo: output length validation tests).

---

## 11) Deviations captured so far (from code vs likely spec)

### D-001: Phase Fix “highFreqWeight” semantics likely incorrect for values > 1.0
- **Status:** ⚠️ (E2 pending)
- **What we see in code:** `_phase_fix_blend` clamps weight to max 1.0 (full replace).
- **Why it matters for spec:** guide/presets mention values like 2.0 as meaningful; if spec expects 2.0 to be “more aggressive than 1.0”, current behavior cannot match.
- **Next action:** When the guide’s Phase Fix section is extracted, define exact semantics and update implementation accordingly.

---

## 12) Next extraction targets (order for full report)

1) Phase Fix / Phase Swapper (full section in guide)  
2) Overlap/chunk/segment semantics per architecture (UVR vs MSST vs Demucs)  
3) Ensemble algorithm rules per stem (avg/max/min + model count limits)  
4) Post-processing workflows (demudder variants, denoise/dereverb, de-crowd)  
5) Output format/bit depth/metadata rules  


**Status:** IN PROGRESS  
**Guide source:** `Copy of Instrumental, vocal & other stems separation & mix_master guide - UVR_MDX_Demucs_GSEP & others.txt`  
**Policy:** Treat guide as **SPEC**. Code must match guide exactly.  
**Repo:** StemSep (`electron-poc/` UI, `stemsep-backend/` Rust orchestration, `StemSepApp/` Python inference)

---

## 0) What this document is

This file is the control center for the “guide as spec” effort:

- Extract **normative requirements** from the guide (rules, defaults, workflows).
- Map each requirement to **exact code locations** (file + function/class).
- Mark compliance state: ✅ compliant / ⚠️ partial / ❌ non-compliant / ❓ unknown.
- Track decisions, risks, and verification strategy.
- Produce a final **Spec Compliance Report** and a concrete change list.

---

## 1) Compliance status legend

- ✅ **Compliant**: Code matches guide requirement exactly (including parameter semantics).
- ⚠️ **Partial**: Implemented but not exact (different defaults, different semantics, missing edge cases).
- ❌ **Non-compliant**: Guide requires behavior not present or code contradicts it.
- ❓ **Unknown**: Not yet verified / not yet traced.

Evidence levels:
- **E0**: Assumption (no code trace / no guide excerpt)
- **E1**: Code location identified
- **E2**: Code excerpt verified against guide requirement
- **E3**: Tested via deterministic reproduction (golden input / expected output)

---

## 2) System architecture (high-level map)

### UI → Backend → Inference (authoritative flow)
- Electron UI (renderer) → Electron main → Rust backend via stdio JSON
- Rust backend spawns Python `scripts/inference.py` per job (argv JSON config)
- Python emits NDJSON events:
  - `separation_started`
  - `separation_progress`
  - `separation_complete` with `output_files`
  - `separation_error` / `separation_cancelled`

Key code anchors (initial):
- Electron: `electron-poc/electron/main.ts`
- Rust backend: `stemsep-backend/src/main.rs`
- Python runner: `scripts/inference.py`
- Core orchestration: `StemSepApp/src/core/separation_manager.py`
- Separation engines:
  - `StemSepApp/src/audio/separation_engine.py`
  - `StemSepApp/src/audio/simple_separator.py` (vendored audio-separator path)
- Blending & post-processing:
  - `StemSepApp/src/audio/ensemble_blender.py`

---

## 3) Known critical spec areas (to be fully extracted from guide)

This is the checklist of guide domains. Each item will become its own section with:
- Guide requirement(s)
- Code mapping
- Compliance state
- Tests / verification

### A) Model selection & compatibility matrix
- UVR / audio-separator compatibility
- MSST-only models (incompatible with UVR)
- Special cases (HyperACE inference script differences)

Status: ❓ (pending full extraction)

### B) Overlap / segment_size / chunk_size semantics
- Different semantics per engine:
  - audio-separator (“segment_size=256” + overlap as seconds/divisor)
  - legacy ZFTurbo demix (“chunk_size/segment_size” based, overlap ratio)
- Default overlap guidance, safe overlap vs dim_t, speed/quality tradeoffs

Status: ❓ (pending full extraction)

### C) Phase Fix / Phase Swapper workflow (instrumental noise / buzzing)
- Exact workflow steps (reference model selection, phase copy band, final blend algorithm)
- Default params:
  - lowHz / highHz
  - highFreqWeight
- Which algorithm to use after phase copy (e.g. Max Spec vs Average)
- Constraints: requires torch/STFT, sample-rate assumptions

Status: ⚠️ (implemented, but likely semantic mismatch; see “Known deviations”)

### D) Ensemble algorithms & per-stem algorithm selection
- Average vs Max Spec vs Min Spec by target stem
- Limit on number of models in ensemble (diminishing returns)
- Weighting behavior & volume compensation (dB per extra model)

Status: ❓ (pending full extraction)

### E) Post-processing workflows (Demudder, high-pass, denoise, dereverb, crowd removal)
- When to apply
- Exact steps (phase rotate/remix/combine)
- Dependencies (librosa/scipy/torch)
- Safety (OOM risks on low VRAM)

Status: ❓ (pending full extraction)

### F) Output correctness invariants
- Output length/duration matching
- Sample rate expectations (44.1k vs 48k handling)
- Bit depth / float export rules
- Metadata preservation requirements

Status: ❓ (pending full extraction)

---

## 4) Known deviations / risks already identified (to verify vs guide)

### D-001: Phase Fix “highFreqWeight > 1.0” semantics may be lost
- **Symptom:** Code clamps phase blend factor to max 1.0 in `_phase_fix_blend`.
- **Risk:** Guide/presets mention values like 2.0 (aggressive). If guide intends >1 to increase effect beyond “full replace”, current code does not honor it.
- **Likely impacted code:** `StemSepApp/src/audio/ensemble_blender.py` (`_phase_fix_blend`)
- **Status:** ⚠️ pending full guide extraction for exact intended semantics

---

## 5) Extraction plan (guide → requirements)

We will process the guide in slices and produce “requirements” objects:

Each requirement becomes:

- **REQ-ID:** `REQ-<domain>-<nnn>`
- **Guide excerpt / location:** line range(s) in the `.txt`
- **Normative statement:** “MUST / MUST NOT / SHOULD (if guide uses strong language)”
- **Inputs:** config fields, model types, stems
- **Algorithm:** step-by-step
- **Outputs:** expected stem keys, files, meta
- **Edge cases:** short audio, mono, missing models, incompatible architectures
- **Verification:** how to test deterministically

---

## 6) Mapping template (copy/paste for each requirement)

### REQ-XXX-000: <title>
- **Domain:** <phase_fix | overlap | ensemble | output | models | postproc>
- **Guide lines:** L?-L?
- **Requirement:**  
  - MUST:  
  - MUST NOT:  
  - Notes:
- **Input contract:** (fields, types, defaults)
- **Expected behavior:** (step-by-step)
- **Output contract:** (stem keys, files, metadata, events)
- **Code mapping:**  
  - Files/classes/functions:
- **Compliance:** ✅/⚠️/❌/❓ (Evidence E0-E3)
- **Verification plan:** (tests/golden files)
- **Notes / open questions:**

---

## 7) Tracking table (initial)

| ID | Domain | Title | Status | Evidence | Owner |
|----|--------|-------|--------|----------|-------|
| D-001 | phase_fix | highFreqWeight semantics | ⚠️ | E1 | AI |
| REQ-MODELS-003 | models | First-class registry schema for runtime/compatibility/requirements | ❌ | E0 | AI |
| REQ-MODELS-002 | models | MSST-only models: special install requirements | ❓ | E0 | AI |
| REQ-PHASEFIX-002 | phase_fix | Phase Fix reference model constraints | ❓ | E0 | AI |
| REQ-OUTPUT-002 | output | Float/bit-depth/nulling output rules | ❓ | E0 | AI |
| TBD | overlap | overlap/segment mapping per engine | ❓ | E0 | AI |
| TBD | ensemble | per-stem algorithm presets | ❓ | E0 | AI |
| TBD | output | duration/length invariants | ❓ | E0 | AI |
| TBD | postproc | demudder workflows | ❓ | E0 | AI |

---

## 8) Final deliverables (what “done” means)

1. **Spec Compliance Report**
   - Full requirement list + mappings
   - Deviations list with severity and impact
2. **Change Plan**
   - Ordered patch set (small safe diffs)
   - Migration notes
3. **Verification Suite**
   - Deterministic tests for the critical requirements
   - Golden outputs where feasible

---

## 9) Notes

- This repo mixes multiple separation engines with different parameter semantics. The guide-as-spec approach requires that UI/backends expose consistent meaning, or that meaning is explicitly architecture-specific and documented in the UI.
- Any “silent fallback” that changes model choice or algorithm violates spec unless explicitly stated in guide.
