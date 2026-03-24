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
          batchSize: 1,
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
          batchSize: 1,
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
    expect(payload.batchSize).toBe(1)
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
    expect(args[21]).toEqual(payload.pipelineConfig)
    expect(args[22]).toEqual(payload.workflow)
    expect(args[25]).toEqual({
      selectionType: "workflow",
      selectionId: "workflow_phase_fix_instrumental",
    })
  })

  it("forwards explicit pipelineConfig for execution transport", async () => {
    const runSelectionJob = vi.fn(async (_payload: any) => ({
      success: true,
      job_id: "job-123",
      status: "started",
      job: {
        job_id: "job-123",
        status: "completed",
        output_files: { vocals: "C:/out/vocals.wav" },
      },
    }))
    const getSelectionJob = vi.fn(async () => ({
      job_id: "job-123",
      status: "completed",
      output_files: { vocals: "C:/out/vocals.wav" },
    }))
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
      { runSelectionJob, getSelectionJob } as unknown as Window["electronAPI"],
      payload,
    )

    expect(runSelectionJob).toHaveBeenCalledTimes(1)
    expect(getSelectionJob).not.toHaveBeenCalled()
    const request = runSelectionJob.mock.calls[0]?.[0] as any
    expect(request.selectionType).toBe("workflow")
    expect(request.selectionId).toBe("workflow_phase_fix_instrumental")
    expect(request.pipelineConfig).toEqual(payload.pipelineConfig)
    expect(request.workflow).toEqual(payload.workflow)
    expect(request.selectionEnvelope).toEqual({
      selectionType: "workflow",
      selectionId: "workflow_phase_fix_instrumental",
    })
  })

  it("forwards profile-derived batch size to transport", () => {
    const payload = buildSeparationBackendPayload({
      inputFile: "C:/audio.wav",
      outputDir: "C:/out",
      config: {
        mode: "advanced",
        modelId: "model-1",
        device: "auto",
        outputFormat: "wav",
        qualityProfile: "long_file_safe",
        advancedParams: {
          batchSize: 2,
        },
      },
      plan: {
        effectiveModelId: "model-1",
        effectiveStems: ["vocals", "instrumental"],
        effectiveEnsembleConfig: undefined,
        effectivePostProcessingSteps: undefined,
        effectiveAdvancedParams: {
          overlap: 2,
          segmentSize: 112455,
          batchSize: 1,
          shifts: 1,
          tta: false,
        },
        effectiveWorkflow: undefined,
        effectiveGlobalPhaseParams: undefined,
        missingModels: [],
        blockingIssues: [],
        warnings: [],
        canProceed: true,
        isExplicitPipelinePhaseFix: false,
        debug: {
          mode: "advanced",
          usedPreset: false,
          usedEnsemble: false,
          usedRecipePreset: false,
          usedGlobalPhaseParams: false,
        },
      },
    })

    expect(payload.batchSize).toBe(1)
    expect(payload.segmentSize).toBe(112455)
  })
})
