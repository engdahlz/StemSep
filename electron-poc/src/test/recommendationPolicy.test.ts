import { describe, expect, it } from "vitest";
import {
  recommendModelChain,
  recommendWorkflowPreset,
} from "@/lib/policy/recommendationPolicy";

describe("recommendModelChain", () => {
  it("prefers target-aligned installed models deterministically", () => {
    const models: any[] = [
      {
        id: "model-b",
        installed: true,
        vram_required: 6,
        quality_profile: {
          target_roles: ["instrumental"],
          quality_tier: "quality",
          deterministic_priority: 2,
        },
      },
      {
        id: "model-a",
        installed: true,
        vram_required: 4,
        quality_profile: {
          target_roles: ["instrumental"],
          quality_tier: "quality",
          deterministic_priority: 2,
        },
      },
    ];

    const rec = recommendModelChain(
      "instrumental",
      models,
      { device: "cuda:0", vramGb: 8 },
    );
    expect(rec.blocked).toBe(false);
    // Equal scores => deterministic lexical fallback.
    expect(rec.chain[0]).toBe("model-a");
    expect(rec.guardrails.allowSilentFallback).toBe(false);
    expect(rec.guardrails.failFastMissingModels).toBe(true);
  });

  it("returns conservative guardrails for low-vram/cpu profiles", () => {
    const models: any[] = [
      {
        id: "voc-1",
        installed: true,
        quality_profile: { target_roles: ["vocals"], quality_tier: "fast" },
      },
    ];

    const rec = recommendModelChain(
      "vocals",
      models,
      { device: "cpu", vramGb: 0 },
    );
    expect(rec.hardwareTier).toBe("cpu_only");
    expect(rec.guardrails.overlap).toBe(2);
    expect(rec.guardrails.segmentSize).toBe(112455);
  });

  it("prefers curated simple-surface workflows over generic presets", () => {
    const models: any[] = [
      { id: "m1", installed: true, status: { simple_allowed: true } },
      { id: "m2", installed: true, status: { simple_allowed: true } },
    ];
    const presets: any[] = [
      {
        id: "generic",
        name: "Generic",
        qualityLevel: "quality",
        estimatedVram: 8,
        tags: [],
        recommended: true,
        simpleGoal: "instrumental",
        modelId: "m1",
      },
      {
        id: "guided",
        name: "Guided",
        qualityLevel: "quality",
        estimatedVram: 8,
        tags: [],
        recommended: true,
        simpleVisible: true,
        guideRank: 1,
        simpleGoal: "instrumental",
        modelId: "m2",
      },
    ];

    const rec = recommendWorkflowPreset(
      "instrumental",
      presets,
      models,
      { device: "cuda:0", vramGb: 8 },
    );
    expect(rec.blocked).toBe(false);
    expect(rec.recommendedPresetId).toBe("guided");
  });

  it("avoids FNO workflows when runtime support is missing", () => {
    const models: any[] = [
      {
        id: "fno",
        name: "FNO",
        installed: true,
        runtime: { variant: "fno" },
        status: { simple_allowed: true },
      },
      {
        id: "stable",
        name: "Stable",
        installed: true,
        status: { simple_allowed: true },
      },
    ];
    const presets: any[] = [
      {
        id: "fno-workflow",
        name: "FNO Workflow",
        qualityLevel: "ultra",
        estimatedVram: 8,
        tags: [],
        recommended: true,
        simpleVisible: true,
        guideRank: 1,
        simpleGoal: "instrumental",
        modelId: "fno",
      },
      {
        id: "stable-workflow",
        name: "Stable Workflow",
        qualityLevel: "quality",
        estimatedVram: 8,
        tags: [],
        recommended: true,
        simpleVisible: true,
        guideRank: 2,
        simpleGoal: "instrumental",
        modelId: "stable",
      },
    ];

    const rec = recommendWorkflowPreset(
      "instrumental",
      presets,
      models,
      { device: "cuda:0", vramGb: 8 },
      { fnoSupported: false },
    );
    expect(rec.recommendedPresetId).toBe("stable-workflow");
  });
});

