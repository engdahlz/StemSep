import { describe, expect, it } from "vitest";

import { getResultAvailabilityStatus } from "@/lib/results/availability";

describe("results availability", () => {
  it("treats exported sessions as exported", () => {
    expect(
      getResultAvailabilityStatus({
        status: "completed",
        outputFiles: { vocals: "C:/preview/vocals.wav" },
        playback: { sourceKind: "preview_cache", previewDir: "C:/preview" },
        exportSummary: {
          status: "exported",
          exportedAt: "2026-03-15T10:00:00.000Z",
          exportDir: "C:/exports",
          format: "wav",
          exportedFiles: { vocals: "C:/exports/vocals.wav" },
        },
      }),
    ).toBe("exported");
  });

  it("keeps old history items without export metadata backward compatible", () => {
    expect(
      getResultAvailabilityStatus({
        status: "completed",
        outputFiles: { instrumental: "C:/preview/instrumental.wav" },
        playback: { sourceKind: "preview_cache", previewDir: "C:/preview" },
      }),
    ).toBe("preview_ready");
  });

  it("detects missing sources and failed sessions", () => {
    expect(
      getResultAvailabilityStatus({
        status: "completed",
        outputFiles: { vocals: "C:/missing/vocals.wav" },
        playback: { sourceKind: "missing_source" },
      }),
    ).toBe("missing_source");

    expect(
      getResultAvailabilityStatus({
        status: "failed",
        outputFiles: {},
      }),
    ).toBe("failed");
  });
});
