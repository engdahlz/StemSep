import type { Model } from "@/types/store";

export type PolicyTarget =
  | "vocals"
  | "instrumental"
  | "karaoke"
  | "restoration";

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

function inferHardwareTier(profile: HardwareProfile): HardwareTier {
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
  const tiers = (model as any).hardware_tiers as
    | Array<{
        tier?: string;
        min_vram_gb?: number | null;
        max_vram_gb?: number | null;
      }>
    | undefined;

  if (!Array.isArray(tiers) || tiers.length === 0) return true;
  return tiers.some((t) => (t?.tier || "").toLowerCase() === tier);
}

function supportsTarget(model: Model, target: PolicyTarget): boolean {
  const roles = ((model as any).quality_profile?.target_roles || []) as string[];
  if (Array.isArray(roles) && roles.length > 0) {
    return roles.includes(target);
  }
  const tags = model.tags || [];
  if (target === "restoration") {
    return tags.some((t) =>
      ["dereverb", "denoise", "debleed", "restoration"].includes(
        (t || "").toLowerCase(),
      ),
    );
  }
  if (target === "karaoke") return tags.includes("karaoke");
  return true;
}

function tierScore(model: Model): number {
  const qt = ((model as any).quality_profile?.quality_tier || "") as string;
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
  const p = (model as any).quality_profile?.deterministic_priority;
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
  score += tierScore(model);
  score += deterministicPriorityScore(model);
  const required = Number(model.vram_required || 0);
  if (required > 0 && vramGb > 0 && required > vramGb + 0.5) score -= 30;
  return score;
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

  const desiredChainLen = target === "instrumental" || target === "karaoke" ? 2 : 1;
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

