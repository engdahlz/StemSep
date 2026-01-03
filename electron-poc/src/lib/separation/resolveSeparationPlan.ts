import type { SeparationConfig } from "@/types/separation";

/**
 * Canonical separation plan resolver.
 *
 * Goal:
 * - Provide one place where the UI decides the "effective" model/pipeline to run,
 *   regardless of whether the user used Simple presets, Advanced manual config,
 *   ensemble/recipe presets, or optional phase correction.
 * - Make downstream callers (SeparatePage/useSeparation) consume one normalized plan.
 *
 * Notes on phase behavior (recommended):
 * - "Pipeline Phase Fix" (preset/advanced) is explicit and takes priority.
 * - Global settings.phaseParams (legacy) is treated as an optional enhancement
 *   and only applies if no explicit pipeline phase fix is configured.
 *
 * This file intentionally does not import UI store; you pass in what you need.
 */

export type PhaseFixParams = {
  lowHz: number;
  highHz: number;
  highFreqWeight: number;
};

export type GlobalPhaseParams = {
  enabled: boolean;
  lowHz: number;
  highHz: number;
  highFreqWeight: number;
};

export type ModelLike = {
  id: string;
  installed?: boolean;
  phase_fix?: {
    recommended_params?: Partial<PhaseFixParams>;
    is_valid_reference?: boolean;
    reference_model_id?: string;
  };
};

export type PresetLike = {
  id: string;
  name?: string;
  modelId?: string;
  stems?: string[];
  // Ensemble presets or "pipeline defined" presets
  ensembleConfig?: {
    models: { model_id: string; weight?: number }[];
    algorithm:
      | "average"
      | "avg_wave"
      | "max_spec"
      | "min_spec"
      | "phase_fix"
      | "frequency_split";
    stemAlgorithms?: {
      vocals?: "average" | "max_spec" | "min_spec";
      instrumental?: "average" | "max_spec" | "min_spec";
    };
    phaseFixEnabled?: boolean;
    phaseFixParams?: PhaseFixParams;
  };
  // Optional post-processing pipeline steps (used by some presets/recipes)
  postProcessingSteps?: any[];
  // Some presets/recipes attach metadata in a "recipe" field; we don't need it here
  // but we keep the shape open.
  [k: string]: any;
};

export type ResolveInputs = {
  config: SeparationConfig;

  /**
   * All presets that can be used for Simple mode. Should include recipe-derived presets too.
   */
  presets?: PresetLike[];

  /**
   * Optional preset->modelId mapping (legacy / compatibility layer).
   */
  modelMap?: Record<string, string | undefined>;

  /**
   * Available models from the store (used for install checks and metadata).
   */
  models?: ModelLike[];

  /**
   * Global phase params from settings (legacy enhancement).
   * Applies only if no explicit pipeline phase fix is configured.
   */
  globalPhaseParams?: GlobalPhaseParams;
};

export type MissingModel = {
  modelId: string;
  reason: "not_installed";
};

export type SeparationPlan = {
  /**
   * The effective model id to send to backend.
   * Can be "ensemble" (virtual) if an ensembleConfig is present.
   */
  effectiveModelId: string;

  /**
   * Effective stems to request; may come from config, preset, or ensemble default.
   */
  effectiveStems?: string[];

  /**
   * Effective ensemble config (only when needed).
   */
  effectiveEnsembleConfig?: SeparationConfig["ensembleConfig"];

  /**
   * Effective post-processing steps (from preset or config).
   */
  effectivePostProcessingSteps?: SeparationConfig["postProcessingSteps"];

  /**
   * Effective phase params to send as legacy global phase_params.
   * This is only set when globalPhaseParams is enabled and no pipeline phase fix is used.
   */
  effectiveGlobalPhaseParams?: GlobalPhaseParams;

  /**
   * Missing dependency models (not installed).
   * Use this to block starting a separation and prompt for downloads.
   */
  missingModels: MissingModel[];

  /**
   * Convenience flags for callers.
   */
  canProceed: boolean;
  isExplicitPipelinePhaseFix: boolean;

  /**
   * Optional debug metadata for logging/UI display.
   */
  debug: {
    mode: SeparationConfig["mode"];
    presetId?: string;
    usedPreset?: boolean;
    usedEnsemble?: boolean;
    usedRecipePreset?: boolean;
    usedGlobalPhaseParams?: boolean;
  };
};

function findPreset(
  presets: PresetLike[] | undefined,
  presetId: string | undefined,
): PresetLike | undefined {
  if (!presetId || !Array.isArray(presets)) return undefined;
  return presets.find((p) => p.id === presetId);
}

function findModel(
  models: ModelLike[] | undefined,
  modelId: string | undefined,
) {
  if (!modelId || !Array.isArray(models)) return undefined;
  return models.find((m) => m.id === modelId);
}

function isInstalled(model: ModelLike | undefined): boolean {
  return !!model?.installed;
}

function addMissing(
  acc: MissingModel[],
  modelId: string,
  missing: MissingModel,
): void {
  if (!modelId) return;
  if (acc.some((m) => m.modelId === modelId)) return;
  acc.push(missing);
}

function getDefaultEnsembleStems(stems?: string[]): string[] {
  if (Array.isArray(stems) && stems.length > 0) return stems;
  // Convention in current UI: ensemble defaults to vocals+instrumental
  return ["vocals", "instrumental"];
}

function hasExplicitPipelinePhaseFix(
  ensemble: SeparationConfig["ensembleConfig"] | undefined,
): boolean {
  if (!ensemble) return false;
  if (ensemble.algorithm === "phase_fix") return true;
  if (ensemble.phaseFixEnabled) return true;
  return false;
}

export function resolveSeparationPlan(inputs: ResolveInputs): SeparationPlan {
  const {
    config,
    presets = [],
    modelMap = {},
    models = [],
    globalPhaseParams,
  } = inputs;

  const missingModels: MissingModel[] = [];

  // --- Step 1: Start from config defaults ---
  let effectiveModelId = config.modelId || "htdemucs";
  let effectiveStems = config.stems;
  let effectiveEnsembleConfig: SeparationConfig["ensembleConfig"] | undefined =
    config.ensembleConfig;
  let effectivePostProcessingSteps = config.postProcessingSteps;

  const preset =
    config.mode === "simple" ? findPreset(presets, config.presetId) : undefined;
  const usedPreset = !!preset;

  // Simple mode is preset-authoritative: ignore any UI-provided overrides.
  // The caller should pass only presetId in Simple mode, but we defensively clamp here.
  if (config.mode === "simple") {
    effectiveModelId = "htdemucs";
    effectiveEnsembleConfig = undefined;
    effectivePostProcessingSteps = undefined;
    // Stems in Simple mode come from preset unless explicitly supported later.
    // We still allow a preset to define stems.
    effectiveStems = undefined;
  }

  // --- Step 2: Apply Simple preset selection rules (if any) ---
  if (config.mode === "simple" && preset) {
    // If preset defines an ensemble config, that is authoritative (pipeline defined)
    if (preset.ensembleConfig && preset.ensembleConfig.models?.length > 0) {
      effectiveModelId = "ensemble";
      effectiveEnsembleConfig = {
        ...preset.ensembleConfig,
      } as any;
      effectiveStems = getDefaultEnsembleStems(config.stems || preset.stems);
    } else {
      // Single model preset
      const mappedModelId = preset.modelId || modelMap[preset.id];
      if (mappedModelId) {
        effectiveModelId = mappedModelId;
      }
      // Stems from preset (unless user explicitly set stems in config)
      if (!config.stems && Array.isArray(preset.stems)) {
        effectiveStems = preset.stems;
      }
    }

    // Post-processing pipeline from preset (if the preset defines it)
    if (
      Array.isArray(preset.postProcessingSteps) &&
      preset.postProcessingSteps.length > 0
    ) {
      effectivePostProcessingSteps = preset.postProcessingSteps;
    }
  }

  // --- Step 3: If config contains a custom ensemble, it takes priority (Advanced only) ---
  if (
    config.mode !== "simple" &&
    config.ensembleConfig &&
    Array.isArray(config.ensembleConfig.models) &&
    config.ensembleConfig.models.length > 0
  ) {
    effectiveModelId = "ensemble";
    effectiveEnsembleConfig = config.ensembleConfig;
    effectiveStems = getDefaultEnsembleStems(config.stems);
  }

  // --- Step 4: Determine phase behavior ---
  const explicitPipelinePhaseFix = hasExplicitPipelinePhaseFix(
    effectiveEnsembleConfig,
  );

  // Global phase params (legacy) are an Advanced-only enhancement.
  // Simple mode should be driven exclusively by preset/recipe-defined pipelines.
  const shouldApplyGlobalPhaseParams = false;

  const effectiveGlobalPhaseParams = shouldApplyGlobalPhaseParams
    ? globalPhaseParams
    : undefined;

  // --- Step 5: Validate dependencies (installed) ---
  // Model(s) to validate:
  // - if ensemble: validate each referenced model_id
  // - else: validate effectiveModelId (unless it's a recipe id - we can't know here reliably)
  //
  // NOTE: Backend preflight also resolves recipe step dependencies, but we want the UI
  // to block early whenever possible for better UX.
  const validateModelId = (mid: string) => {
    const m = findModel(models, mid);
    if (!isInstalled(m)) {
      addMissing(missingModels, mid, {
        modelId: mid,
        reason: "not_installed",
      });
    }
  };

  if (
    effectiveEnsembleConfig &&
    Array.isArray(effectiveEnsembleConfig.models) &&
    effectiveEnsembleConfig.models.length > 0
  ) {
    for (const mm of effectiveEnsembleConfig.models) {
      if (mm?.model_id) validateModelId(mm.model_id);
    }
  } else {
    // If this is a recipe id, it might not be a downloadable artifact. We don't try to
    // validate in that case here; backend preflight will report missing step models.
    // Heuristic: if model id matches a known model in store, validate it.
    const known = !!findModel(models, effectiveModelId);
    if (known) validateModelId(effectiveModelId);
  }

  // Also validate post-processing steps if they reference models (best effort)
  //
  // Note: PostProcessingStep is typed and does not guarantee these legacy keys exist,
  // so we probe via a safe Record<string, unknown> to avoid TS errors.
  if (Array.isArray(effectivePostProcessingSteps)) {
    for (const step of effectivePostProcessingSteps) {
      const stepAny = step as unknown as Record<string, unknown>;
      const mid =
        (typeof stepAny?.modelId === "string"
          ? (stepAny.modelId as string)
          : undefined) ||
        (typeof stepAny?.model_id === "string"
          ? (stepAny.model_id as string)
          : undefined) ||
        (typeof stepAny?.model === "string"
          ? (stepAny.model as string)
          : undefined) ||
        (typeof stepAny?.id === "string" ? (stepAny.id as string) : undefined);

      if (typeof mid === "string" && mid.trim()) {
        // Only validate if this looks like a real model id we know about
        const known = !!findModel(models, mid);
        if (known) validateModelId(mid);
      }
    }
  }

  const canProceed = missingModels.length === 0;

  return {
    effectiveModelId,
    effectiveStems,
    effectiveEnsembleConfig,
    effectivePostProcessingSteps,
    effectiveGlobalPhaseParams,
    missingModels,
    canProceed,
    isExplicitPipelinePhaseFix: explicitPipelinePhaseFix,
    debug: {
      mode: config.mode,
      presetId: config.presetId,
      usedPreset,
      usedEnsemble: !!effectiveEnsembleConfig,
      usedRecipePreset: !!(preset && (preset as any).isRecipe),
      usedGlobalPhaseParams: !!effectiveGlobalPhaseParams,
    },
  };
}
