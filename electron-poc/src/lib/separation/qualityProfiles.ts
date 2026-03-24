import type { Preset } from "@/presets";
import type { QualityProfile, SeparationWorkflow } from "@/types/separation";

export type QualityProfilePreset = {
  id: QualityProfile;
  label: string;
  shortLabel: string;
  description: string;
};

export type WorkflowTier = "verified" | "advanced";
export type RuntimeBurden = "light" | "moderate" | "heavy" | "extreme";
export type HardwareTierHint =
  | "CPU / Low VRAM"
  | "Mid VRAM"
  | "High VRAM"
  | "High VRAM+";

type QualityProfilePresetContext = {
  advancedDefaults?: Preset["advancedDefaults"];
  recipe?: {
    type?: string;
    target?: string;
    defaults?: {
      overlap?: number;
      segment_size?: number;
      chunk_size?: number;
      shifts?: number;
      tta?: boolean;
      batch_size?: number;
    };
  };
  simpleGoal?: string;
  workflowFamily?: string;
  ensembleConfig?: Preset["ensembleConfig"];
};

export type QualityProfileAdvancedParams = {
  overlap?: number;
  segmentSize?: number;
  batchSize?: number;
  shifts?: number;
  tta?: boolean;
};

const QUALITY_PROFILES: QualityProfilePreset[] = [
  {
    id: "fast",
    label: "Fast",
    shortLabel: "Fast",
    description: "Shortest turnaround for previews and quick comparison passes.",
  },
  {
    id: "balanced",
    label: "Balanced",
    shortLabel: "Balanced",
    description: "Default guide-style settings tuned for quality without excessive runtime.",
  },
  {
    id: "maximum_quality",
    label: "Maximum Quality",
    shortLabel: "Max",
    description: "Higher overlap, longer segments and heavier passes when the machine can handle it.",
  },
  {
    id: "long_file_safe",
    label: "Long File Safe",
    shortLabel: "Long",
    description: "Safer chunking for long files, weaker machines or memory-sensitive runs.",
  },
];

export function getQualityProfiles(): QualityProfilePreset[] {
  return QUALITY_PROFILES;
}

export function deriveWorkflowTier(preset: Preset): WorkflowTier {
  if (
    preset.isRecipe &&
    preset.recipe?.qa_status === "verified" &&
    preset.recipe?.promotion_status === "curated" &&
    (!preset.recipe.surface_blockers || preset.recipe.surface_blockers.length === 0)
  ) {
    return "verified";
  }
  return "advanced";
}

export function deriveRuntimeBurden(preset: Preset): RuntimeBurden {
  const estimatedVram = Number.isFinite(preset.estimatedVram)
    ? preset.estimatedVram
    : 0;
  const runtimeTier = preset.expectedRuntimeTier || preset.recipe?.expectedRuntimeTier;
  const isEnsemble = !!preset.ensembleConfig || preset.recipe?.type === "ensemble";
  const isRestoration =
    preset.simpleGoal === "cleanup" ||
    preset.recipe?.target === "restoration" ||
    preset.workflowFamily?.toLowerCase().includes("restoration");

  if (runtimeTier === "advanced" || estimatedVram >= 12 || (isEnsemble && estimatedVram >= 10)) {
    return "extreme";
  }
  if (runtimeTier === "standard" || estimatedVram >= 8 || isRestoration || isEnsemble) {
    return "heavy";
  }
  if (runtimeTier === "fast" || estimatedVram >= 5) {
    return "moderate";
  }
  return "light";
}

export function deriveRecommendedHardwareTier(preset: Preset): HardwareTierHint {
  const estimatedVram = Number.isFinite(preset.estimatedVram)
    ? preset.estimatedVram
    : 0;
  if (estimatedVram >= 12) return "High VRAM+";
  if (estimatedVram >= 8) return "High VRAM";
  if (estimatedVram >= 5) return "Mid VRAM";
  return "CPU / Low VRAM";
}

export function deriveTargetUseCase(preset: Preset): string {
  if (preset.recommendedFor?.length) return preset.recommendedFor[0];
  if (preset.simpleGoal === "vocals") return "Lead vocal extraction";
  if (preset.simpleGoal === "instrumental") return "Instrumental / backing track";
  if (preset.simpleGoal === "karaoke") return "Karaoke / backing vocal removal";
  if (preset.simpleGoal === "cleanup") return "Restoration and cleanup";
  if (preset.simpleGoal === "instruments") return "Focused stem recovery";
  return preset.workflowSummary || preset.description || "General-purpose separation";
}

function clampPositive(value: number | undefined, fallback: number) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return fallback;
  return value;
}

function getBaseDefaults(args: {
  preset?: QualityProfilePresetContext | null;
  current?: QualityProfileAdvancedParams | null;
}): Required<QualityProfileAdvancedParams> {
  const presetDefaults = args.preset?.advancedDefaults;
  const recipeDefaults = args.preset?.recipe?.defaults;
  const current = args.current || {};
  return {
    overlap: clampPositive(
      current.overlap ??
        presetDefaults?.overlap ??
        recipeDefaults?.overlap,
      4,
    ),
    segmentSize: clampPositive(
      current.segmentSize ??
        presetDefaults?.segmentSize ??
        recipeDefaults?.segment_size ??
        recipeDefaults?.chunk_size,
      352800,
    ),
    batchSize: clampPositive(current.batchSize, 1),
    shifts: clampPositive(current.shifts ?? presetDefaults?.shifts ?? recipeDefaults?.shifts, 2),
    tta: current.tta ?? presetDefaults?.tta ?? recipeDefaults?.tta ?? false,
  };
}

function isRestorationPreset(
  preset?: QualityProfilePresetContext | null,
  workflow?: SeparationWorkflow | null,
) {
  const family = preset?.workflowFamily?.toLowerCase() || workflow?.family?.toLowerCase() || "";
  return (
    preset?.simpleGoal === "cleanup" ||
    preset?.recipe?.target === "restoration" ||
    workflow?.surface === "restoration" ||
    family.includes("restoration")
  );
}

function isEnsemblePreset(
  preset?: QualityProfilePresetContext | null,
  workflow?: SeparationWorkflow | null,
) {
  return !!preset?.ensembleConfig || workflow?.kind === "ensemble" || preset?.recipe?.type === "ensemble";
}

export function resolveQualityProfileAdvancedParams(args: {
  profile?: QualityProfile | null;
  preset?: QualityProfilePresetContext | null;
  workflow?: SeparationWorkflow | null;
  current?: QualityProfileAdvancedParams | null;
}): QualityProfileAdvancedParams | undefined {
  if (!args.profile) return args.current || undefined;

  const base = getBaseDefaults({ preset: args.preset, current: args.current });
  const restoration = isRestorationPreset(args.preset, args.workflow);
  const ensemble = isEnsemblePreset(args.preset, args.workflow);

  const maxQualityOverlap = restoration ? 8 : ensemble ? 6 : 5;
  const maxQualitySegment = restoration ? 352800 : ensemble ? 485100 : 485100;
  const longFileSegment = restoration ? 176400 : 112455;

  switch (args.profile) {
    case "fast":
      return {
        overlap: Math.min(base.overlap, 2),
        segmentSize: Math.min(base.segmentSize, restoration ? 176400 : 112455),
        batchSize: ensemble ? 1 : 2,
        shifts: 1,
        tta: false,
      };
    case "balanced":
      return {
        overlap: Math.max(3, Math.min(base.overlap, restoration ? 6 : 4)),
        segmentSize: Math.max(base.segmentSize, restoration ? 176400 : 352800),
        batchSize: 1,
        shifts: Math.max(2, base.shifts),
        tta: base.tta,
      };
    case "maximum_quality":
      return {
        overlap: Math.max(base.overlap, maxQualityOverlap),
        segmentSize: Math.max(base.segmentSize, maxQualitySegment),
        batchSize: 1,
        shifts: Math.max(base.shifts, 3),
        tta: true,
      };
    case "long_file_safe":
      return {
        overlap: Math.min(base.overlap, restoration ? 3 : 2),
        segmentSize: Math.min(base.segmentSize, longFileSegment),
        batchSize: 1,
        shifts: 1,
        tta: false,
      };
    default:
      return base;
  }
}
