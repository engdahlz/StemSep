import { describe, expect, it } from "vitest";

import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";
import type { SeparationConfig } from "@/types/separation";

const baseConfig: SeparationConfig = {
  mode: "simple",
  presetId: "preset_quality",
  device: "cuda:0",
  outputFormat: "wav",
  advancedParams: {
    overlap: 12,
    segmentSize: 999999,
    shifts: 7,
    tta: false,
  },
};

describe("resolveSeparationPlan", () => {
  it("uses preset defaults in simple mode instead of raw advanced overrides", () => {
    const plan = resolveSeparationPlan({
      config: baseConfig,
      presets: [
        {
          id: "preset_quality",
          name: "Quality Preset",
          modelId: "model-balanced",
          recipe: {
            defaults: {
              overlap: 4,
              segment_size: 352800,
              shifts: 2,
              tta: true,
            },
          },
        },
      ],
      models: [{ id: "model-balanced", installed: true }],
      runtimeSupport: { fnoSupported: true },
    });

    expect(plan.canProceed).toBe(true);
    expect(plan.effectiveAdvancedParams).toEqual({
      overlap: 4,
      segmentSize: 352800,
      shifts: 2,
      tta: true,
    });
  });

  it("applies quality profiles on top of guided preset defaults", () => {
    const plan = resolveSeparationPlan({
      config: {
        ...baseConfig,
        qualityProfile: "maximum_quality",
      },
      presets: [
        {
          id: "preset_quality",
          name: "Quality Preset",
          modelId: "model-balanced",
          simpleGoal: "instrumental",
          recipe: {
            defaults: {
              overlap: 4,
              segment_size: 352800,
              shifts: 2,
              tta: false,
            },
          },
        },
      ],
      models: [{ id: "model-balanced", installed: true }],
      runtimeSupport: { fnoSupported: true },
    });

    expect(plan.canProceed).toBe(true);
    expect(plan.effectiveAdvancedParams).toEqual({
      overlap: 5,
      segmentSize: 485100,
      batchSize: 1,
      shifts: 3,
      tta: true,
    });
  });

  it("blocks FNO models in simple mode when runtime support is missing", () => {
    const plan = resolveSeparationPlan({
      config: baseConfig,
      presets: [
        {
          id: "preset_quality",
          name: "FNO Preset",
          modelId: "fno-model",
        },
      ],
      models: [
        {
          id: "fno-model",
          name: "FNO Model",
          installed: true,
          runtime: { variant: "fno1d" },
        },
      ],
      runtimeSupport: { fnoSupported: false },
    });

    expect(plan.canProceed).toBe(false);
    expect(plan.blockingIssues).toEqual([
      "FNO Model requires FNO/neuralop runtime support.",
    ]);
  });

  it("warns but still proceeds in advanced mode for runtime-blocked models", () => {
    const plan = resolveSeparationPlan({
      config: {
        ...baseConfig,
        mode: "advanced",
        presetId: undefined,
        modelId: "fno-model",
      },
      models: [
        {
          id: "fno-model",
          name: "FNO Model",
          installed: true,
          runtime: { variant: "fno1d" },
          status: { readiness: "experimental", simple_allowed: true },
        },
      ],
      runtimeSupport: { fnoSupported: false },
    });

    expect(plan.canProceed).toBe(true);
    expect(plan.blockingIssues).toEqual([]);
    expect(plan.warnings).toContain(
      "FNO Model requires FNO/neuralop runtime support.",
    );
  });
});
