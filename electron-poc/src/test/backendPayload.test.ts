import { describe, expect, it } from "vitest"
import { buildSeparationBackendPayload, toBackendOverlap, toBackendSegmentSize } from "@/lib/separation/backendPayload"

describe("backendPayload", () => {
  it("converts overlap divisor to backend ratio", () => {
    expect(toBackendOverlap(4)).toBeCloseTo(0.75)
    expect(toBackendOverlap(0.5)).toBeCloseTo(0.5)
    expect(toBackendOverlap(0)).toBeUndefined()
  })

  it("normalizes segment size", () => {
    expect(toBackendSegmentSize("352800")).toBe(352800)
    expect(toBackendSegmentSize(0)).toBeUndefined()
  })

  it("builds payload from resolved plan", () => {
    const payload = buildSeparationBackendPayload({
      inputFile: "C:/audio.wav",
      outputDir: "C:/out",
      config: {
        mode: "simple",
        presetId: "workflow_phase_fix_instrumental",
        device: "cuda:0",
        outputFormat: "wav",
        advancedParams: {
          overlap: 4,
          segmentSize: 352800,
          shifts: 2,
          bitrate: "320k",
          tta: true,
        },
      },
      plan: {
        effectiveModelId: "workflow_phase_fix_instrumental",
        effectiveStems: ["instrumental", "vocals"],
        effectiveEnsembleConfig: undefined,
        effectivePostProcessingSteps: undefined,
        effectiveWorkflow: {
          version: 1,
          id: "workflow_phase_fix_instrumental",
          name: "Best Instrumental",
          kind: "pipeline",
          surface: "workflow",
          stems: ["instrumental", "vocals"],
          models: [
            { model_id: "unwa-inst-v1e-plus", role: "primary" },
            { model_id: "becruily-vocal", role: "phase_reference" },
          ],
          steps: [
            { id: "separate", action: "separate", model_id: "unwa-inst-v1e-plus" },
            {
              id: "phase_fix",
              action: "phase_fix",
              source_model: "becruily-vocal",
              apply_to: "instrumental",
            },
          ],
        },
        effectiveGlobalPhaseParams: undefined,
        missingModels: [],
        canProceed: true,
        isExplicitPipelinePhaseFix: false,
        debug: {
          mode: "simple",
          presetId: "workflow_phase_fix_instrumental",
          usedPreset: true,
          usedEnsemble: false,
          usedRecipePreset: true,
          usedGlobalPhaseParams: false,
        },
      },
    })

    expect(payload.modelId).toBe("workflow_phase_fix_instrumental")
    expect(payload.device).toBe("cuda:0")
    expect(payload.overlap).toBeCloseTo(0.75)
    expect(payload.segmentSize).toBe(352800)
    expect(payload.outputFormat).toBe("wav")
    expect(payload.workflow?.kind).toBe("pipeline")
    expect(payload.workflow?.steps?.[1]?.action).toBe("phase_fix")
  })
})
