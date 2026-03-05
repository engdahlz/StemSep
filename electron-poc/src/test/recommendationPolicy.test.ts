import { describe, expect, it } from "vitest";
import { recommendModelChain } from "@/lib/policy/recommendationPolicy";

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
});

