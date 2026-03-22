import { getRequiredModels, type Preset } from "@/presets";
import type { Model } from "@/types/store";
import { getModelCatalogTier, isManualCatalogModel } from "@/lib/models/catalog";
import {
  inferHardwareTier,
  type HardwareTier,
  type PolicyTarget,
} from "@/lib/policy/recommendationPolicy";
import { modelRequiresFnoRuntime } from "./modelRuntime";

export type MachineWorkflowStatus = "supported" | "limited" | "blocked";
export type ModelMachineFitStatus =
  | "fits_this_machine"
  | "heavy_for_this_machine"
  | "runtime_blocked"
  | "manual_setup";

export interface MachineAnalysisMetric {
  label: string;
  value: string;
  score: number;
  hint: string;
}

export interface MachineWorkflowAssessment {
  target: PolicyTarget | "cleanup";
  label: string;
  status: MachineWorkflowStatus;
  presetName: string | null;
  reason: string;
}

export interface MachineAnalysisSummary {
  overallScore: number;
  tier: HardwareTier;
  tierLabel: string;
  accelerationLabel: string;
  verdictTitle: string;
  verdictText: string;
  metrics: MachineAnalysisMetric[];
  recommendations: string[];
  issues: string[];
  workflows: MachineWorkflowAssessment[];
}

export interface ModelMachineFit {
  status: ModelMachineFitStatus;
  label: string;
  reason: string;
}

type RuntimeFacts = {
  hasDetectedGpu: boolean;
  rawVramGb: number;
  hasCudaRuntime: boolean;
  hasTorch: boolean;
  hasPython: boolean;
  hasFnoSupport: boolean;
  runtimeError: string | null;
  systemRamGb: number;
  cpuCount: number;
  platform: string;
  gpuName: string;
};

const tierRank: Record<HardwareTier, number> = {
  cpu_only: 0,
  low_vram: 1,
  mid_vram: 2,
  high_vram: 3,
};

const targetLabels: Array<{
  target: PolicyTarget | "cleanup";
  label: string;
}> = [
  { target: "instrumental", label: "Instrumental" },
  { target: "vocals", label: "Vocals" },
  { target: "karaoke", label: "Karaoke" },
  { target: "cleanup", label: "Cleanup" },
];

function qualityRank(level: Preset["qualityLevel"]): number {
  switch (level) {
    case "ultra":
      return 4;
    case "quality":
      return 3;
    case "balanced":
      return 2;
    case "fast":
      return 1;
    default:
      return 0;
  }
}

function tierLabel(tier: HardwareTier): string {
  switch (tier) {
    case "high_vram":
      return "High-VRAM GPU";
    case "mid_vram":
      return "Mid-VRAM GPU";
    case "low_vram":
      return "Low-VRAM GPU";
    default:
      return "CPU-safe";
  }
}

function operationalTierFromRuntime(facts: RuntimeFacts): HardwareTier {
  return inferHardwareTier({
    device: facts.hasCudaRuntime ? "cuda:0" : "cpu",
    vramGb: facts.hasCudaRuntime ? facts.rawVramGb : 0,
  });
}

function getRuntimeFacts(runtimeInfo: SystemRuntimeInfo | null): RuntimeFacts {
  const gpu = runtimeInfo?.gpu;
  const runtime = runtimeInfo?.runtimeFingerprint;
  const system = gpu?.system_info || {};
  const activeGpu =
    gpu?.gpus?.find((entry: any) => entry?.recommended) ||
    gpu?.gpus?.[0] ||
    null;

  const rawVramGb = Number(
    activeGpu?.memory_gb ?? gpu?.recommended_profile?.vram_gb ?? 0,
  );
  const hasDetectedGpu = Boolean(gpu?.has_cuda || activeGpu);
  const hasTorch = Boolean(runtime?.torch?.version);
  const hasPython = Boolean(runtime?.version);
  const hasCudaRuntime = Boolean(
    hasDetectedGpu && runtime?.torch?.cuda_available === true,
  );

  return {
    hasDetectedGpu,
    rawVramGb: Number.isFinite(rawVramGb) ? rawVramGb : 0,
    hasCudaRuntime,
    hasTorch,
    hasPython,
    hasFnoSupport: runtime?.neuralop?.fno1d_import_ok !== false,
    runtimeError:
      runtimeInfo?.runtimeFingerprintError ||
      (!runtime && !gpu ? "Runtime information unavailable." : null),
    systemRamGb: Number(system?.memory_total_gb || 0),
    cpuCount: Number(system?.cpu_count || 0),
    platform: String(system?.platform || runtime?.platform || "Unknown"),
    gpuName: String(activeGpu?.name || (hasDetectedGpu ? "Detected GPU" : "CPU only")),
  };
}

function presetMatchesTarget(
  preset: Preset,
  target: PolicyTarget | "cleanup",
): boolean {
  if (preset.simpleGoal === target) return true;
  if (target === "cleanup" && preset.simpleGoal === "cleanup") return true;

  const recipeTarget = preset.recipe?.target?.toLowerCase();
  if (recipeTarget === target) return true;
  if (target === "cleanup" && recipeTarget === "restoration") return true;

  if (target === "instrumental") {
    return preset.category === "instrumental";
  }
  if (target === "vocals") {
    return preset.category === "vocals";
  }
  if (target === "karaoke") {
    return preset.tags.some((tag) =>
      ["karaoke", "no-vocals", "backing-track"].includes(tag),
    );
  }
  return (
    preset.category === "utility" ||
    preset.category === "smart" ||
    preset.tags.some((tag) =>
      ["cleanup", "debleed", "dereverb", "denoise"].includes(tag),
    )
  );
}

function presetExpectedTierFits(preset: Preset, tier: HardwareTier): boolean {
  const expected = preset.expectedVramTier;
  if (!expected) {
    const estimated = Number(preset.estimatedVram || 0);
    if (tier === "cpu_only") return estimated <= 4;
    if (tier === "low_vram") return estimated <= 6;
    if (tier === "mid_vram") return estimated <= 10;
    return true;
  }
  return tierRank[tier] >= tierRank[expected];
}

function modelFitsTier(model: Model, tier: HardwareTier, vramGb: number): boolean {
  const tiers = model.hardware_tiers;
  if (Array.isArray(tiers) && tiers.length > 0) {
    return tiers.some((entry) => {
      const entryTier = entry?.tier;
      const min = typeof entry?.min_vram_gb === "number" ? entry.min_vram_gb : null;
      const max = typeof entry?.max_vram_gb === "number" ? entry.max_vram_gb : null;

      if (tier === "cpu_only") {
        return entryTier === "cpu_only" || (max !== null && max <= 0);
      }
      if (entryTier && tierRank[tier] < tierRank[entryTier]) return false;
      if (min !== null && vramGb < min) return false;
      if (max !== null && vramGb > max) return false;
      return true;
    });
  }

  const required = Number(model.vram_required || 0);
  if (tier === "cpu_only") return required <= 4;
  if (required <= 0) return true;
  return required <= vramGb + 0.5;
}

function modelFitsCurrentRuntime(
  model: Model,
  tier: HardwareTier,
  vramGb: number,
  hasFnoSupport: boolean,
): { ok: boolean; reason?: string } {
  if (model.status?.readiness === "blocked" || getModelCatalogTier(model) === "blocked") {
    return {
      ok: false,
      reason: model.status?.blocking_reason || `${model.name} is blocked in the registry.`,
    };
  }

  if (model.status?.readiness === "manual" || isManualCatalogModel(model)) {
    return {
      ok: false,
      reason:
        model.status?.blocking_reason || `${model.name} requires manual setup.`,
    };
  }

  if (modelRequiresFnoRuntime(model) && !hasFnoSupport) {
    return {
      ok: false,
      reason: `${model.name} needs neuralop/FNO support in the runtime.`,
    };
  }

  if (!modelFitsTier(model, tier, vramGb)) {
    return {
      ok: false,
      reason:
        tier === "cpu_only"
          ? `${model.name} is not marked as CPU-safe in the guide registry.`
          : `${model.name} expects more VRAM than this runtime tier provides.`,
    };
  }

  return { ok: true };
}

function pickBestPresetForTarget(
  target: PolicyTarget | "cleanup",
  presets: Preset[],
  models: Model[],
  tier: HardwareTier,
  vramGb: number,
  hasFnoSupport: boolean,
): MachineWorkflowAssessment {
  const modelsById = new Map(models.map((model) => [model.id, model]));
  const candidates = presets.filter(
    (preset) => preset.simpleVisible !== false && presetMatchesTarget(preset, target),
  );

  let bestSupported: { preset: Preset; status: MachineWorkflowStatus } | null = null;
  let firstFailureReason =
    "No workflow in the current catalog cleanly matches this machine.";

  for (const preset of candidates) {
    if (!presetExpectedTierFits(preset, tier)) {
      firstFailureReason =
        firstFailureReason ===
        "No workflow in the current catalog cleanly matches this machine."
          ? `${preset.name} expects a higher VRAM tier.`
          : firstFailureReason;
      continue;
    }

    const required = getRequiredModels(preset)
      .map((id) => modelsById.get(id))
      .filter(Boolean) as Model[];
    if (required.length === 0) continue;

    const failedModel = required
      .map((model) => modelFitsCurrentRuntime(model, tier, vramGb, hasFnoSupport))
      .find((result) => !result.ok);

    if (failedModel) {
      firstFailureReason = failedModel.reason || firstFailureReason;
      continue;
    }

    const status: MachineWorkflowStatus =
      tier === "cpu_only" ||
      (tier === "low_vram" && qualityRank(preset.qualityLevel) >= 3)
        ? "limited"
        : "supported";

    const current = {
      preset,
      status,
      score:
        (preset.recommended ? 80 : 0) +
        (preset.simpleVisible ? 50 : 0) +
        qualityRank(preset.qualityLevel) * 20 +
        (typeof preset.guideRank === "number"
          ? Math.max(0, 40 - preset.guideRank * 4)
          : 0),
    };

    if (
      !bestSupported ||
      current.score >
        ((bestSupported.preset.recommended ? 80 : 0) +
          (bestSupported.preset.simpleVisible ? 50 : 0) +
          qualityRank(bestSupported.preset.qualityLevel) * 20 +
          (typeof bestSupported.preset.guideRank === "number"
            ? Math.max(0, 40 - bestSupported.preset.guideRank * 4)
            : 0))
    ) {
      bestSupported = { preset, status };
    }
  }

  const label = targetLabels.find((entry) => entry.target === target)?.label || target;

  if (!bestSupported) {
    return {
      target,
      label,
      status: "blocked",
      presetName: null,
      reason: firstFailureReason,
    };
  }

  return {
    target,
    label,
    status: bestSupported.status,
    presetName: bestSupported.preset.name,
    reason:
      bestSupported.status === "supported"
        ? `Recommended workflow: ${bestSupported.preset.name}.`
        : `Usable, but best treated as a lighter run: ${bestSupported.preset.name}.`,
  };
}

function buildRecommendations(
  facts: RuntimeFacts,
  tier: HardwareTier,
  workflows: MachineWorkflowAssessment[],
): string[] {
  const items: string[] = [];

  if (!facts.hasPython || !facts.hasTorch) {
    items.push("Fix the Python/Torch runtime first before trusting heavy workflow recommendations.");
  } else if (facts.hasDetectedGpu && !facts.hasCudaRuntime) {
    items.push("A CUDA-capable GPU is present, but the current Torch runtime is not using it. StemSep will behave like a CPU-only setup until that is fixed.");
  } else if (tier === "high_vram") {
    items.push("This machine is ready for heavier guide-style GPU workflows and larger context settings.");
  } else if (tier === "mid_vram") {
    items.push("This machine is a good fit for most curated quality workflows without pushing into the heaviest chain setups.");
  } else if (tier === "low_vram") {
    items.push("Prefer lighter GPU runs or balanced presets to keep runtime stable and preview responsive.");
  } else {
    items.push("Treat this machine as CPU-safe only and start with lighter presets or one file at a time.");
  }

  if (!facts.hasFnoSupport) {
    items.push("FNO-based models should stay disabled until neuralop/FNO1d support is available in the runtime.");
  }

  if (facts.systemRamGb > 0 && facts.systemRamGb < 16) {
    items.push("System RAM is on the tighter side for longer files or larger batch queues, so keep jobs smaller.");
  }

  const blocked = workflows.filter((workflow) => workflow.status === "blocked").length;
  if (blocked >= 2) {
    items.push("Several guided workflows are unavailable on this setup, so the safest path is to stay close to curated fallback presets.");
  }

  return Array.from(new Set(items)).slice(0, 4);
}

export function buildMachineAnalysis(
  runtimeInfo: SystemRuntimeInfo | null,
  models: Model[],
  presets: Preset[],
): MachineAnalysisSummary {
  const facts = getRuntimeFacts(runtimeInfo);
  const tier = operationalTierFromRuntime(facts);
  const tierDisplay = tierLabel(tier);

  const workflows = targetLabels.map(({ target }) =>
    pickBestPresetForTarget(
      target,
      presets,
      models,
      tier,
      facts.hasCudaRuntime ? facts.rawVramGb : 0,
      facts.hasFnoSupport,
    ),
  );

  const verifiedModels = models.filter(
    (model) =>
      model.status?.readiness === "verified" &&
      modelFitsCurrentRuntime(
        model,
        tier,
        facts.hasCudaRuntime ? facts.rawVramGb : 0,
        facts.hasFnoSupport,
      ).ok,
  );

  const runtimeScore = (() => {
    if (!facts.hasPython || !facts.hasTorch) return 25;
    if (facts.hasDetectedGpu && !facts.hasCudaRuntime) return 45;
    let score = facts.hasCudaRuntime ? 92 : 58;
    if (!facts.hasFnoSupport) score -= 12;
    return Math.max(20, score);
  })();

  const hardwareScore = (() => {
    if (facts.hasCudaRuntime) {
      if (facts.rawVramGb >= 10) return 94;
      if (facts.rawVramGb >= 6) return 80;
      if (facts.rawVramGb >= 4) return 64;
    }
    if (facts.hasDetectedGpu) return 52;
    return 40;
  })();

  const workflowScore = (() => {
    const supported = workflows.filter((entry) => entry.status === "supported").length;
    const limited = workflows.filter((entry) => entry.status === "limited").length;
    return Math.round(
      20 + ((supported + limited * 0.5) / Math.max(1, workflows.length)) * 80,
    );
  })();

  const coverageScore = (() => {
    const curatedRelevant = models.filter(
      (model) =>
        model.status?.readiness === "verified" &&
        model.status?.simple_allowed !== false,
    ).length;
    if (curatedRelevant === 0) return 40;
    return Math.round(25 + (verifiedModels.length / curatedRelevant) * 75);
  })();

  const overallScore = Math.round(
    runtimeScore * 0.32 +
      hardwareScore * 0.24 +
      workflowScore * 0.24 +
      coverageScore * 0.2,
  );

  const verdict = (() => {
    if (!facts.hasPython || !facts.hasTorch) {
      return {
        title: "Runtime needs attention",
        text: "StemSep cannot make a trustworthy hardware recommendation until the Python and Torch runtime are detected correctly.",
      };
    }
    if (facts.hasDetectedGpu && !facts.hasCudaRuntime) {
      return {
        title: "GPU present, but not active in runtime",
        text: "The hardware looks stronger than the current runtime path. Separation will behave like a CPU setup until CUDA is available in Torch.",
      };
    }
    if (tier === "high_vram") {
      return {
        title: "Ready for heavier guide workflows",
        text: "This machine aligns well with the app's higher-end GPU guidance and should handle premium separation chains comfortably.",
      };
    }
    if (tier === "mid_vram") {
      return {
        title: "Ready for most curated workflows",
        text: "This machine is in the sweet spot for standard quality runs and most guided presets without pushing into the heaviest chains.",
      };
    }
    if (tier === "low_vram") {
      return {
        title: "Best with lighter GPU workflows",
        text: "The setup is usable for separation, but you should stay closer to balanced runs and lighter model combinations.",
      };
    }
    return {
      title: "CPU-safe fallback only",
      text: "This setup can still separate audio, but it should be treated as a lighter, slower path rather than a premium GPU workflow machine.",
    };
  })();

  const issues = [
    facts.runtimeError,
    !facts.hasFnoSupport
      ? "FNO-based models are blocked because neuralop/FNO1d support is missing."
      : null,
    facts.hasDetectedGpu && !facts.hasCudaRuntime
      ? "A GPU was detected, but the current Torch runtime reports CUDA as unavailable."
      : null,
  ].filter(Boolean) as string[];

  const metrics: MachineAnalysisMetric[] = [
    {
      label: "Runtime path",
      value: facts.hasCudaRuntime
        ? "CUDA ready"
        : facts.hasDetectedGpu
          ? "CPU fallback"
          : "CPU only",
      score: runtimeScore,
      hint: facts.hasCudaRuntime
        ? "Torch and the active runtime are ready to use the detected GPU."
        : facts.hasDetectedGpu
          ? "The machine has GPU hardware, but the current runtime is not using it."
          : "StemSep will rely on CPU-safe behavior on this machine.",
    },
    {
      label: "Hardware tier",
      value: facts.hasDetectedGpu
        ? `${facts.gpuName} · ${facts.rawVramGb || "?"} GB VRAM`
        : `${facts.cpuCount || "?"} CPU cores · ${facts.systemRamGb || "?"} GB RAM`,
      score: hardwareScore,
      hint:
        "Thresholds follow the app's own guardrails: roughly 4 GB starts low-VRAM GPU use, 6 GB is the standard mid tier, and 10 GB+ unlocks heavier guide-oriented runs.",
    },
    {
      label: "Workflow coverage",
      value: `${workflows.filter((entry) => entry.status !== "blocked").length}/${workflows.length} guided paths fit`,
      score: workflowScore,
      hint:
        "This is based on curated preset compatibility, runtime blockers and the registry's guide-derived hardware tiers.",
    },
    {
      label: "Guide coverage",
      value: `${verifiedModels.length} verified models fit`,
      score: coverageScore,
      hint:
        "Counts guide-curated verified models that the current runtime tier can actually support cleanly.",
    },
  ];

  return {
    overallScore,
    tier,
    tierLabel: tierDisplay,
    accelerationLabel: facts.hasCudaRuntime
      ? "GPU acceleration active"
      : facts.hasDetectedGpu
        ? "GPU hardware detected, runtime fallback active"
        : "CPU-only path",
    verdictTitle: verdict.title,
    verdictText: verdict.text,
    metrics,
    recommendations: buildRecommendations(facts, tier, workflows),
    issues,
    workflows,
  };
}

export function getModelMachineFit(
  model: Model,
  runtimeInfo: SystemRuntimeInfo | null,
): ModelMachineFit {
  const facts = getRuntimeFacts(runtimeInfo);
  const tier = operationalTierFromRuntime(facts);
  const vramGb = facts.hasCudaRuntime ? facts.rawVramGb : 0;

  if (model.status?.readiness === "manual" || isManualCatalogModel(model)) {
    return {
      status: "manual_setup",
      label: "Manual Setup",
      reason:
        model.status?.blocking_reason ||
        `${model.name} requires manual assets or setup before it can run.`,
    };
  }

  const runtimeCheck = modelFitsCurrentRuntime(
    model,
    tier,
    vramGb,
    facts.hasFnoSupport,
  );
  if (!runtimeCheck.ok) {
    return {
      status: "runtime_blocked",
      label: "Runtime Blocked",
      reason: runtimeCheck.reason || `${model.name} is blocked on this machine.`,
    };
  }

  const preferredTier = Array.isArray(model.hardware_tiers)
    ? model.hardware_tiers.reduce<HardwareTier | null>((current, entry) => {
        const nextTier = entry?.tier;
        if (!nextTier) return current;
        if (!current) return nextTier;
        return tierRank[nextTier] > tierRank[current] ? nextTier : current;
      }, null)
    : null;

  if (preferredTier && tierRank[tier] < tierRank[preferredTier]) {
    return {
      status: "heavy_for_this_machine",
      label: "Heavy",
      reason: `${model.name} is available, but its preferred hardware tier is above this machine's current runtime tier.`,
    };
  }

  const requiredVram = Number(model.vram_required || 0);
  if (facts.hasCudaRuntime && requiredVram > 0 && vramGb > 0 && requiredVram >= vramGb) {
    return {
      status: "heavy_for_this_machine",
      label: "Heavy",
      reason: `${model.name} should run, but it is likely to push this machine close to its VRAM ceiling.`,
    };
  }

  return {
    status: "fits_this_machine",
    label: "Fits This Machine",
    reason: `${model.name} fits the current runtime and guide-derived hardware tier.`,
  };
}
