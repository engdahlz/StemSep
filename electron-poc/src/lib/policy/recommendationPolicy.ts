import type { Preset } from "@/presets";
import { getRequiredModels } from "@/presets";
import type { Model } from "@/types/store";

export type PolicyTarget =
  | "vocals"
  | "instrumental"
  | "karaoke"
  | "restoration"
  | "drums"
  | "bass";

export type HardwareTier = "cpu_only" | "low_vram" | "mid_vram" | "high_vram";

export interface HardwareProfile {
  device: string;
  vramGb: number;
}

export interface PolicyGuardrails {
  overlap: number;
  segmentSize: number;
  failFastMissingModels: true;
  allowSilentFallback: false;
}

export interface PolicyRecommendation {
  target: PolicyTarget;
  hardwareTier: HardwareTier;
  chain: string[];
  guardrails: PolicyGuardrails;
  blocked: boolean;
  reason?: string;
}

export interface WorkflowRecommendation {
  target: PolicyTarget;
  hardwareTier: HardwareTier;
  recommendedPresetId?: string;
  rankedPresetIds: string[];
  blocked: boolean;
  reason?: string;
  notes: string[];
}

type RuntimeHints = {
  fnoSupported?: boolean;
};

export function inferHardwareTier(profile: HardwareProfile): HardwareTier {
  const vram = Number.isFinite(profile.vramGb) ? profile.vramGb : 0;
  const d = (profile.device || "").toLowerCase();
  if (d.includes("cpu") || vram <= 0) return "cpu_only";
  if (vram < 6) return "low_vram";
  if (vram < 10) return "mid_vram";
  return "high_vram";
}

function guardrailsForTier(tier: HardwareTier): PolicyGuardrails {
  if (tier === "cpu_only") {
    return {
      overlap: 2,
      segmentSize: 112455,
      failFastMissingModels: true,
      allowSilentFallback: false,
    };
  }
  if (tier === "low_vram") {
    return {
      overlap: 2,
      segmentSize: 112455,
      failFastMissingModels: true,
      allowSilentFallback: false,
    };
  }
  if (tier === "mid_vram") {
    return {
      overlap: 3,
      segmentSize: 352800,
      failFastMissingModels: true,
      allowSilentFallback: false,
    };
  }
  return {
    overlap: 4,
    segmentSize: 485100,
    failFastMissingModels: true,
    allowSilentFallback: false,
  };
}

function supportsTier(model: Model, tier: HardwareTier): boolean {
  const tiers = model.hardware_tiers;
  if (!Array.isArray(tiers) || tiers.length === 0) return true;
  return tiers.some((t) => {
    if (t?.tier && t.tier !== tier) return false;
    const min = typeof t?.min_vram_gb === "number" ? t.min_vram_gb : null;
    const max = typeof t?.max_vram_gb === "number" ? t.max_vram_gb : null;
    if (tier === "cpu_only") return min == null || min <= 0;
    if (min != null && tier === "low_vram" && min > 6) return false;
    if (max != null && tier === "high_vram" && max < 10) return false;
    return true;
  });
}

function supportsTarget(model: Model, target: PolicyTarget): boolean {
  const roles = model.quality_profile?.target_roles || [];
  if (roles.length > 0) return roles.includes(target as any);

  const tags = (model.tags || []).map((t) => (t || "").toLowerCase());
  if (target === "restoration") {
    return tags.some((t) =>
      ["dereverb", "denoise", "debleed", "restoration", "decrowd"].includes(t),
    );
  }
  if (target === "karaoke") return tags.includes("karaoke");
  if (target === "drums") return tags.includes("drumsep");
  if (target === "bass") return tags.includes("bass");
  return true;
}

function tierScore(model: Model): number {
  const qt = model.quality_profile?.quality_tier || "";
  switch (qt) {
    case "ultra":
      return 30;
    case "quality":
      return 20;
    case "balanced":
      return 10;
    case "fast":
      return 5;
    default:
      return 0;
  }
}

function deterministicPriorityScore(model: Model): number {
  const p = model.quality_profile?.deterministic_priority;
  if (typeof p !== "number" || !Number.isFinite(p) || p < 1) return 0;
  return Math.max(0, 50 - Math.round(p));
}

function modelScore(
  model: Model,
  target: PolicyTarget,
  tier: HardwareTier,
  vramGb: number,
): number {
  let score = 0;
  if (supportsTarget(model, target)) score += 40;
  if (supportsTier(model, tier)) score += 20;
  if (model.installed) score += 15;
  if (model.status?.simple_allowed !== false) score += 10;
  if (model.status?.readiness === "verified") score += 10;
  score += tierScore(model);
  score += deterministicPriorityScore(model);
  if (typeof model.guide_rank === "number") {
    score += Math.max(0, 40 - model.guide_rank * 8);
  }
  const required = Number(model.vram_required || 0);
  if (required > 0 && vramGb > 0 && required > vramGb + 0.5) score -= 30;
  if (model.status?.readiness === "blocked" || model.status?.readiness === "manual") {
    score -= 100;
  }
  return score;
}

function presetGoalMatches(preset: Preset, target: PolicyTarget): boolean {
  const explicit = preset.simpleGoal;
  if (explicit === "cleanup") return target === "restoration";
  if (explicit === "instruments") return target === "drums" || target === "bass";
  if (explicit && explicit === target) return true;

  const recipeTarget = preset.recipe?.target?.toLowerCase();
  if (recipeTarget === target) return true;
  if (target === "restoration" && recipeTarget === "cleanup") return true;
  return false;
}

function presetVramFits(preset: Preset, tier: HardwareTier): boolean {
  const expected = preset.expectedVramTier;
  if (expected) {
    if (tier === "cpu_only") return expected === "cpu_only" || expected === "low_vram";
    if (tier === "low_vram") return expected === "cpu_only" || expected === "low_vram";
    if (tier === "mid_vram") return expected !== "high_vram";
    return true;
  }

  const vram = typeof preset.estimatedVram === "number" ? preset.estimatedVram : 0;
  if (tier === "cpu_only") return vram <= 4;
  if (tier === "low_vram") return vram <= 6;
  if (tier === "mid_vram") return vram <= 10;
  return true;
}

function modelRequiresFnoRuntime(model: Model | undefined): boolean {
  const variant = (model?.runtime as any)?.variant;
  return typeof variant === "string" && variant.toLowerCase() === "fno";
}

function workflowScore(
  preset: Preset,
  models: Model[],
  tier: HardwareTier,
  runtimeHints?: RuntimeHints,
): { score: number; notes: string[] } {
  const notes: string[] = [];
  let score = 0;

  if (preset.simpleVisible) score += 200;
  if (preset.recommended) score += 60;
  if (typeof preset.guideRank === "number") {
    score += Math.max(0, 120 - preset.guideRank * 15);
  }
  if (presetVramFits(preset, tier)) score += 40;
  else score -= 80;

  const requiredModels = getRequiredModels(preset);
  const required = requiredModels.map((id) => models.find((m) => m.id === id)).filter(Boolean) as Model[];
  const installedCount = required.filter((m) => m.installed).length;
  score += installedCount * 20;
  score -= (required.length - installedCount) * 15;

  for (const model of required) {
    if (model.status?.simple_allowed === false) {
      score -= 30;
      if (model.status.blocking_reason) notes.push(model.status.blocking_reason);
    }
    if (model.status?.readiness === "verified") score += 10;
    if (typeof model.guide_rank === "number") {
      score += Math.max(0, 20 - model.guide_rank * 4);
    }
    if (runtimeHints?.fnoSupported === false && modelRequiresFnoRuntime(model)) {
      score -= 200;
      notes.push(`${model.name} requires neuralop/FNO1d.`);
    }
  }

  if (preset.qualityLevel === "ultra") score += 20;
  if (preset.qualityLevel === "quality") score += 10;
  if (tier === "cpu_only" && preset.qualityLevel === "fast") score += 20;

  return { score, notes: Array.from(new Set(notes)) };
}

export function recommendModelChain(
  target: PolicyTarget,
  models: Model[],
  hardware: HardwareProfile,
): PolicyRecommendation {
  const tier = inferHardwareTier(hardware);
  const guardrails = guardrailsForTier(tier);
  const filtered = models.filter((m) => supportsTarget(m, target));
  const ranked = [...filtered].sort((a, b) => {
    const sa = modelScore(a, target, tier, hardware.vramGb);
    const sb = modelScore(b, target, tier, hardware.vramGb);
    if (sa !== sb) return sb - sa;
    return a.id.localeCompare(b.id);
  });

  const desiredChainLen =
    target === "instrumental" || target === "karaoke" ? 2 : 1;
  const chain = ranked.slice(0, desiredChainLen).map((m) => m.id);

  if (chain.length === 0) {
    return {
      target,
      hardwareTier: tier,
      chain: [],
      guardrails,
      blocked: true,
      reason: "No policy-eligible models found for target.",
    };
  }

  return {
    target,
    hardwareTier: tier,
    chain,
    guardrails,
    blocked: false,
  };
}

export function recommendWorkflowPreset(
  target: PolicyTarget,
  presets: Preset[],
  models: Model[],
  hardware: HardwareProfile,
  runtimeHints?: RuntimeHints,
): WorkflowRecommendation {
  const hardwareTier = inferHardwareTier(hardware);
  const simpleSurface = presets.some((preset) => preset.simpleVisible);
  const filtered = presets.filter((preset) => {
    if (simpleSurface && !preset.simpleVisible) return false;
    return presetGoalMatches(preset, target);
  });

  const ranked = [...filtered]
    .map((preset) => {
      const { score, notes } = workflowScore(
        preset,
        models,
        hardwareTier,
        runtimeHints,
      );
      return { preset, score, notes };
    })
    .sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score;
      return a.preset.id.localeCompare(b.preset.id);
    });

  if (ranked.length === 0) {
    return {
      target,
      hardwareTier,
      blocked: true,
      reason: "No workflow presets match the requested target.",
      rankedPresetIds: [],
      notes: [],
    };
  }

  return {
    target,
    hardwareTier,
    blocked: false,
    recommendedPresetId: ranked[0]?.preset.id,
    rankedPresetIds: ranked.map((entry) => entry.preset.id),
    notes: Array.from(new Set(ranked.flatMap((entry) => entry.notes))).slice(0, 5),
  };
}
