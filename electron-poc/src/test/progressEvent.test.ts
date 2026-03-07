import { describe, expect, it } from "vitest"
import { calculateQueueEta, getQueueStepLabel, queueUpdatesFromProgressEvent } from "@/lib/separation/progressEvent"

describe("progressEvent helpers", () => {
  it("maps structured progress events onto queue item updates", () => {
    const updates = queueUpdatesFromProgressEvent(
      {
        id: "q1",
        file: "C:/audio.wav",
        status: "processing",
        progress: 12,
      },
      {
        jobId: "job-1",
        kind: "progress",
        progress: 44,
        message: "Processing chunk 3/8",
        phase: "pipeline",
        stepId: "pipeline:0:foo",
        stepLabel: "Vocals pass (foo)",
        stepIndex: 0,
        stepCount: 3,
        modelId: "foo",
        chunksDone: 3,
        chunksTotal: 8,
      },
      1000,
    )

    expect(updates.progress).toBe(44)
    expect(updates.message).toBe("Processing chunk 3/8")
    expect(updates.activeStepLabel).toBe("Vocals pass (foo)")
    expect(updates.activeStepCount).toBe(3)
    expect(updates.chunksDone).toBe(3)
    expect(updates.lastProgressTime).toBe(1000)
  })

  it("formats step labels with step counters", () => {
    expect(
      getQueueStepLabel({
        id: "q1",
        file: "x",
        status: "processing",
        activeStepLabel: "Phase Fix",
        activeStepIndex: 1,
        activeStepCount: 3,
      }),
    ).toBe("Phase Fix (2/3)")
  })

  it("uses step progress as an ETA stabilizer", () => {
    const eta = calculateQueueEta(
      {
        id: "q1",
        file: "x",
        status: "processing",
        progress: 20,
        startTime: 0,
        activeStepIndex: 1,
        activeStepCount: 4,
      },
      60_000,
    )

    expect(eta).toBeTruthy()
  })
})
