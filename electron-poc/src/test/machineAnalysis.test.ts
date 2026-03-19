import { describe, expect, it } from "vitest";

import type { Preset } from "../presets";
import type { Model } from "../types/store";
import { buildMachineAnalysis } from "../lib/systemRuntime/machineAnalysis";

const makeModel = (overrides: Partial<Model>): Model =>
  ({
    id: "model",
    name: "Model",
    architecture: "BS-Roformer",
    version: "1.0",
    category: "instrumental",
    description: "",
    sdr: 0,
    fullness: 0,
    bleedless: 0,
    vram_required: 6,
    speed: "medium",
    stems: ["vocals", "instrumental"],
    file_size: 100,
    installed: false,
    downloading: false,
    downloadProgress: 0,
    recommended: false,
    status: {
      readiness: "verified",
      simple_allowed: true,
    },
    ...overrides,
  }) as Model;

const makePreset = (overrides: Partial<Preset>): Preset =>
  ({
    id: "preset",
    name: "Preset",
    description: "",
    stems: ["vocals", "instrumental"],
    recommended: true,
    category: "instrumental",
    qualityLevel: "quality",
    estimatedVram: 6,
    tags: [],
    simpleVisible: true,
    ...overrides,
  }) as Preset;

describe("buildMachineAnalysis", () => {
  it("classifies a CUDA-ready 10 GB machine as high-vram and supports heavy workflows", () => {
    const models: Model[] = [
      makeModel({
        id: "inst-high",
        name: "Inst High",
        quality_profile: { target_roles: ["instrumental"], quality_tier: "ultra" },
        hardware_tiers: [{ tier: "high_vram", min_vram_gb: 10 }],
      }),
      makeModel({
        id: "voc-mid",
        name: "Voc Mid",
        category: "vocals",
        vram_required: 6,
        quality_profile: { target_roles: ["vocals"], quality_tier: "quality" },
        hardware_tiers: [{ tier: "mid_vram", min_vram_gb: 6 }],
      }),
      makeModel({
        id: "karaoke-high",
        name: "Karaoke High",
        category: "utility",
        vram_required: 10,
        quality_profile: { target_roles: ["karaoke"], quality_tier: "quality" },
        hardware_tiers: [{ tier: "high_vram", min_vram_gb: 10 }],
      }),
      makeModel({
        id: "cleanup-mid",
        name: "Cleanup Mid",
        category: "utility",
        vram_required: 6,
        tags: ["cleanup"],
        quality_profile: { target_roles: ["restoration"], quality_tier: "quality" },
        hardware_tiers: [{ tier: "mid_vram", min_vram_gb: 6 }],
      }),
    ];

    const presets: Preset[] = [
      makePreset({
        id: "inst",
        name: "Best Instrumental",
        simpleGoal: "instrumental",
        modelId: "inst-high",
        qualityLevel: "ultra",
        estimatedVram: 10,
      }),
      makePreset({
        id: "voc",
        name: "Best Vocals",
        category: "vocals",
        simpleGoal: "vocals",
        modelId: "voc-mid",
      }),
      makePreset({
        id: "karaoke",
        name: "Best Karaoke",
        category: "utility",
        simpleGoal: "karaoke",
        modelId: "karaoke-high",
        estimatedVram: 10,
      }),
      makePreset({
        id: "cleanup",
        name: "Cleanup",
        category: "utility",
        simpleGoal: "cleanup",
        modelId: "cleanup-mid",
      }),
    ];

    const runtimeInfo: SystemRuntimeInfo = {
      fetchedAt: new Date().toISOString(),
      gpu: {
        has_cuda: true,
        gpus: [{ recommended: true, memory_gb: 10, name: "RTX 3080" }],
        system_info: { cpu_count: 16, memory_total_gb: 32, platform: "Windows" },
      },
      runtimeFingerprint: {
        version: "3.11",
        torch: { version: "2.6", cuda_available: true },
        neuralop: { fno1d_import_ok: true },
      },
      runtimeFingerprintError: null,
      previewCachePolicy: { ephemeral: true },
    };

    const result = buildMachineAnalysis(runtimeInfo, models, presets);

    expect(result.tier).toBe("high_vram");
    expect(result.overallScore).toBeGreaterThanOrEqual(80);
    expect(result.workflows.every((workflow) => workflow.status === "supported")).toBe(true);
  });

  it("flags a detected GPU when the current torch runtime cannot use CUDA", () => {
    const models: Model[] = [
      makeModel({
        id: "fast-preview",
        name: "Fast Preview",
        vram_required: 3,
        architecture: "MDX23C",
        hardware_tiers: [{ tier: "cpu_only", max_vram_gb: 0 }],
      }),
    ];

    const presets: Preset[] = [
      makePreset({
        id: "preview",
        name: "Fast Preview",
        simpleGoal: "instrumental",
        modelId: "fast-preview",
        qualityLevel: "fast",
        estimatedVram: 3,
      }),
    ];

    const runtimeInfo: SystemRuntimeInfo = {
      fetchedAt: new Date().toISOString(),
      gpu: {
        has_cuda: true,
        gpus: [{ recommended: true, memory_gb: 8, name: "RTX 3060" }],
        system_info: { cpu_count: 12, memory_total_gb: 16, platform: "Windows" },
      },
      runtimeFingerprint: {
        version: "3.11",
        torch: { version: "2.6", cuda_available: false },
        neuralop: { fno1d_import_ok: true },
      },
      runtimeFingerprintError: null,
      previewCachePolicy: { ephemeral: true },
    };

    const result = buildMachineAnalysis(runtimeInfo, models, presets);

    expect(result.accelerationLabel).toContain("runtime fallback");
    expect(result.issues.some((issue) => issue.includes("CUDA"))).toBe(true);
    expect(result.verdictTitle).toContain("GPU present");
  });

  it("blocks FNO-dependent workflows when neuralop support is missing", () => {
    const models: Model[] = [
      makeModel({
        id: "fno-inst",
        name: "FNO Instrumental",
        runtime: { variant: "fno" },
        hardware_tiers: [{ tier: "high_vram", min_vram_gb: 10 }],
      }),
    ];

    const presets: Preset[] = [
      makePreset({
        id: "fno",
        name: "FNO Instrumental",
        simpleGoal: "instrumental",
        modelId: "fno-inst",
        qualityLevel: "ultra",
        estimatedVram: 10,
      }),
    ];

    const runtimeInfo: SystemRuntimeInfo = {
      fetchedAt: new Date().toISOString(),
      gpu: {
        has_cuda: true,
        gpus: [{ recommended: true, memory_gb: 12, name: "RTX 4070 Ti" }],
        system_info: { cpu_count: 16, memory_total_gb: 32, platform: "Windows" },
      },
      runtimeFingerprint: {
        version: "3.11",
        torch: { version: "2.6", cuda_available: true },
        neuralop: { fno1d_import_ok: false, fno1d_import_error: "Import failed" },
      },
      runtimeFingerprintError: null,
      previewCachePolicy: { ephemeral: true },
    };

    const result = buildMachineAnalysis(runtimeInfo, models, presets);
    const instrumental = result.workflows.find((workflow) => workflow.target === "instrumental");

    expect(instrumental?.status).toBe("blocked");
    expect(result.issues.some((issue) => issue.includes("FNO"))).toBe(true);
  });
});
