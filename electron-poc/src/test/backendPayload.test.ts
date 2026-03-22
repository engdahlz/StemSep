import { describe, expect, it, vi } from "vitest"
import {
  buildSeparationBackendPayload,
  executeSeparation,
  executeSeparationPreflight,
  toBackendOverlap,
  toBackendSegmentSize,
} from "@/lib/separation/backendPayload"

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
        selectionEnvelope: {
          catalogTier: "verified",
          sourceKind: "guide",
          installPolicy: "direct",
        },
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
        effectiveAdvancedParams: {
          overlap: 4,
          segmentSize: 352800,
          shifts: 2,
          tta: true,
        },
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
        blockingIssues: [],
        warnings: [],
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
    expect(payload.selectionType).toBe("workflow")
    expect(payload.selectionId).toBe("workflow_phase_fix_instrumental")
    expect(payload.selectionEnvelope?.catalogTier).toBe("verified")
    expect(payload.pipelineConfig).toEqual(payload.workflow?.steps)
    expect(payload.workflow?.kind).toBe("pipeline")
    expect(payload.workflow?.steps?.[1]?.action).toBe("phase_fix")
  })

  it("forwards explicit pipelineConfig for preflight transport", async () => {
    const separationPreflight = vi.fn(async () => ({ ok: true }))
    const payload = {
      inputFile: "C:/audio.wav",
      modelId: "workflow_phase_fix_instrumental",
      outputDir: "C:/out",
      outputFormat: "wav" as const,
      pipelineConfig: [{ step_name: "explicit", action: "separate", model_id: "A" }],
      workflow: {
        version: 1 as const,
        id: "workflow_phase_fix_instrumental",
        name: "Workflow",
        kind: "pipeline" as const,
        steps: [{ id: "workflow", action: "separate", model_id: "B" }],
      },
    } as any

    await executeSeparationPreflight(
      { separationPreflight } as unknown as Window["electronAPI"],
      payload,
    )

    expect(separationPreflight).toHaveBeenCalledTimes(1)
    const args = separationPreflight.mock.calls[0] as any[]
    expect(args[3]).toBe("workflow")
    expect(args[4]).toBe("workflow_phase_fix_instrumental")
    expect(args[20]).toEqual(payload.pipelineConfig)
    expect(args[21]).toEqual(payload.workflow)
    expect(args[24]).toEqual({
      selectionType: "workflow",
      selectionId: "workflow_phase_fix_instrumental",
    })
  })

  it("forwards explicit pipelineConfig for execution transport", async () => {
    const separateAudio = vi.fn(async () => ({ success: true, outputFiles: {} }))
    const payload = {
      inputFile: "C:/audio.wav",
      modelId: "workflow_phase_fix_instrumental",
      outputDir: "C:/out",
      outputFormat: "wav" as const,
      pipelineConfig: [{ step_name: "explicit", action: "separate", model_id: "A" }],
      workflow: {
        version: 1 as const,
        id: "workflow_phase_fix_instrumental",
        name: "Workflow",
        kind: "pipeline" as const,
        steps: [{ id: "workflow", action: "separate", model_id: "B" }],
      },
    } as any

    await executeSeparation(
      { separateAudio } as unknown as Window["electronAPI"],
      payload,
    )

    expect(separateAudio).toHaveBeenCalledTimes(1)
    const args = separateAudio.mock.calls[0] as any[]
    expect(args[3]).toBe("workflow")
    expect(args[4]).toBe("workflow_phase_fix_instrumental")
    expect(args[20]).toEqual(payload.pipelineConfig)
    expect(args[21]).toEqual(payload.workflow)
    expect(args[24]).toEqual({
      selectionType: "workflow",
      selectionId: "workflow_phase_fix_instrumental",
    })
  })
})
