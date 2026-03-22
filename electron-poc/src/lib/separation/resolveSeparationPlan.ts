import type {
  SeparationConfig,
  SeparationWorkflow,
  WorkflowBlendConfig,
  WorkflowModelRef,
  WorkflowPhaseFixParams,
  WorkflowStep,
  WorkflowSurface,
} from "@/types/separation";
import type { ModelSelectionEnvelope, ModelVerificationMetadata } from "@/types/modelCatalog";
import { getModelCatalogTier, isManualCatalogModel } from "@/lib/models/catalog";
import { modelRequiresFnoRuntime } from "@/lib/systemRuntime/modelRuntime";

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

export type PhaseFixParams = WorkflowPhaseFixParams;

export type GlobalPhaseParams = {
  enabled: boolean;
  lowHz: number;
  highHz: number;
  highFreqWeight: number;
};

export type ModelLike = {
  id: string;
  installed?: boolean;
  name?: string;
  architecture?: string;
  catalog?: {
    tier?: string;
    sourceKind?: string;
    installPolicy?: string;
    verification?: ModelVerificationMetadata;
  };
  catalog_tier?: string;
  source_kind?: string;
  install_policy?: string;
  verification?: ModelVerificationMetadata;
  selection_envelope?: ModelSelectionEnvelope;
  catalog_status?: string;
  download?: {
    mode?: string;
    sources?: Array<{
      host?: string;
      url?: string;
      manual?: boolean;
    }>;
  };
  availability?: {
    class?: string;
  };
  install?: {
    mode?: string;
  };
  runtime?: {
    variant?: string;
    install_mode?: string;
  };
  status?: {
    readiness?: "verified" | "experimental" | "manual" | "blocked";
    simple_allowed?: boolean;
    blocking_reason?: string;
  };
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
  description?: string;
  simpleGoal?: string;
  workflowSummary?: string;
  difficulty?: string;
  expectedRuntimeTier?: string;
  expectedVramTier?: string;
  recipe?: {
    type?: string;
    target?: string;
    selection_envelope?: ModelSelectionEnvelope;
    defaults?: {
      overlap?: number;
      segment_size?: number;
      chunk_size?: number;
      shifts?: number;
      tta?: boolean;
    };
    steps?: any[];
  };
  advancedDefaults?: {
    overlap?: number;
    segment_size?: number;
    chunk_size?: number;
    shifts?: number;
    tta?: boolean;
  };
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
  selectionEnvelope?: ModelSelectionEnvelope;
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
  runtimeSupport?: {
    fnoSupported?: boolean;
  };
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
  effectiveWorkflow?: SeparationWorkflow;
  effectiveAdvancedParams?: {
    overlap?: number;
    segmentSize?: number;
    shifts?: number;
    tta?: boolean;
  };

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
  blockingIssues: string[];
  warnings: string[];

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

function normalizeSurface(value: string | undefined): WorkflowSurface {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  if (
    normalized === "single" ||
    normalized === "ensemble" ||
    normalized === "workflow" ||
    normalized === "restoration" ||
    normalized === "special_stem"
  ) {
    return normalized;
  }
  if (normalized === "karaoke" || normalized === "cleanup") {
    return "workflow";
  }
  if (normalized === "drums" || normalized === "bass" || normalized === "guitar") {
    return "special_stem";
  }
  return "workflow";
}

function recipeStepToWorkflowStep(step: Record<string, unknown>, index: number): WorkflowStep {
  const output = step.output;
  return {
    id:
      (typeof step.step_name === "string" && step.step_name) ||
      (typeof step.name === "string" && step.name) ||
      `step_${index + 1}`,
    name:
      (typeof step.step_name === "string" && step.step_name) ||
      (typeof step.name === "string" && step.name) ||
      `Step ${index + 1}`,
    action:
      typeof step.action === "string" && step.action.trim()
        ? step.action
        : typeof step.source_model === "string" && step.source_model
          ? "phase_fix"
          : "separate",
    model_id:
      typeof step.model_id === "string" && step.model_id.trim()
        ? step.model_id
        : undefined,
    source_model:
      typeof step.source_model === "string" && step.source_model.trim()
        ? step.source_model
        : undefined,
    input_source:
      (typeof step.input_source === "string" && step.input_source.trim()
        ? step.input_source
        : undefined) ||
      (typeof step.input_from === "string" && step.input_from.trim()
        ? step.input_from
        : undefined),
    output:
      typeof output === "string" || Array.isArray(output)
        ? (output as string | string[])
        : undefined,
    apply_to:
      typeof step.apply_to === "string" && step.apply_to.trim()
        ? step.apply_to
        : undefined,
    role:
      typeof step.role === "string" && step.role.trim() ? step.role : undefined,
    weight:
      typeof step.weight === "number" && Number.isFinite(step.weight)
        ? step.weight
        : undefined,
    optional: step.optional === true,
    params: Object.fromEntries(
      Object.entries(step).filter(([key, value]) => {
        if (value == null) return false;
        return ![
          "step_name",
          "name",
          "action",
          "model_id",
          "source_model",
          "input_source",
          "input_from",
          "output",
          "apply_to",
          "role",
          "weight",
          "optional",
        ].includes(key);
      }),
    ),
  };
}

function buildRecipeWorkflow(
  preset: PresetLike,
  config: SeparationConfig,
): SeparationWorkflow | undefined {
  const recipe = preset.recipe;
  if (!recipe) return undefined;
  const selectionEnvelope =
    config.selectionEnvelope ||
    preset.selectionEnvelope ||
    recipe.selection_envelope ||
    (recipe as any).selectionEnvelope;

  const steps = Array.isArray(recipe.steps)
    ? recipe.steps.map((step, index) =>
        recipeStepToWorkflowStep((step || {}) as Record<string, unknown>, index),
      )
    : [];

  const models = new Map<string, WorkflowModelRef>();
  for (const step of steps) {
    if (step.model_id) {
      models.set(step.model_id, {
        model_id: step.model_id,
        weight: step.weight,
        role: (step.role as WorkflowModelRef["role"]) || "primary",
      });
    }
    if (step.source_model) {
      models.set(step.source_model, {
        model_id: step.source_model,
        role: "phase_reference",
        required: !step.optional,
      });
    }
  }

  const kind =
    recipe.type === "pipeline" || recipe.type === "chained"
      ? "pipeline"
      : recipe.type === "ensemble"
        ? "ensemble"
        : "single";

  const blend: WorkflowBlendConfig | undefined =
    kind === "ensemble"
      ? {
          algorithm:
            preset.ensembleConfig?.algorithm ||
            (typeof (recipe as Record<string, unknown>).algorithm === "string"
              ? ((recipe as Record<string, unknown>).algorithm as WorkflowBlendConfig["algorithm"])
              : "average"),
          stemAlgorithms: preset.ensembleConfig?.stemAlgorithms,
          splitFreq:
            typeof config.splitFreq === "number"
              ? config.splitFreq
              : typeof (recipe as Record<string, unknown>).algorithm_config === "object" &&
                  (recipe as Record<string, unknown>).algorithm_config &&
                  typeof ((recipe as Record<string, unknown>).algorithm_config as Record<string, unknown>).split_freq ===
                    "number"
                ? (((recipe as Record<string, unknown>).algorithm_config as Record<string, unknown>)
                    .split_freq as number)
                : undefined,
          phaseFixEnabled: preset.ensembleConfig?.phaseFixEnabled,
          phaseFixParams: preset.ensembleConfig?.phaseFixParams,
        }
      : undefined;

  return {
    version: 1,
    id: preset.id,
    name: preset.name,
    kind,
    selectionEnvelope,
    family:
      typeof (preset.recipe as any)?.family === "string"
        ? (preset.recipe as any).family
        : undefined,
    surface: normalizeSurface(
      preset.recipe?.target || preset.simpleGoal || config.mode,
    ),
    description: preset.workflowSummary || preset.description,
    stems: config.stems || preset.stems,
    models: Array.from(models.values()),
    steps,
    blend,
    postprocess: config.postProcessingSteps,
    intermediateOutputs: Array.isArray((preset.recipe as any)?.intermediate_outputs)
      ? (preset.recipe as any).intermediate_outputs
      : undefined,
    fallbackPolicy:
      typeof (preset.recipe as any)?.fallback_policy === "object" &&
      (preset.recipe as any)?.fallback_policy
        ? {
            mode: (preset.recipe as any).fallback_policy.mode,
            reason: (preset.recipe as any).fallback_policy.reason,
            runtimeOrder: (preset.recipe as any).fallback_policy.runtime_order,
            fallbackWorkflowId:
              (preset.recipe as any).fallback_policy.fallback_workflow_id,
            fallbackOperatingProfile:
              (preset.recipe as any).fallback_policy.fallback_operating_profile,
          }
        : undefined,
    operatingProfile:
      typeof (preset.recipe as any)?.operating_profile === "string"
        ? (preset.recipe as any).operating_profile
        : undefined,
    runtimePolicy:
      config.runtimePolicy || (preset.recipe as any)?.runtime_policy
        ? {
            required:
              config.runtimePolicy?.required ||
              (preset.recipe as any)?.runtime_policy?.required,
            fallbacks:
              config.runtimePolicy?.fallbacks ||
              (preset.recipe as any)?.runtime_policy?.fallbacks,
            allowManualModels:
              config.runtimePolicy?.allowManualModels ??
              (preset.recipe as any)?.runtime_policy?.allow_manual_models,
            preferredRuntime:
              config.runtimePolicy?.preferredRuntime ||
              (preset.recipe as any)?.runtime_policy?.preferred_runtime,
          }
        : undefined,
    exportPolicy:
      config.exportPolicy || (preset.recipe as any)?.export_policy
        ? {
            stems:
              config.exportPolicy?.stems ||
              (preset.recipe as any)?.export_policy?.stems ||
              config.stems ||
              preset.stems,
            outputFormat:
              config.exportPolicy?.outputFormat ||
              (preset.recipe as any)?.export_policy?.output_format ||
              config.outputFormat,
            intermediateOutputs:
              config.exportPolicy?.intermediateOutputs ||
              (preset.recipe as any)?.export_policy?.intermediate_outputs ||
              (preset.recipe as any)?.intermediate_outputs,
          }
        : undefined,
  };
}

function buildEnsembleWorkflow(
  args: {
    id?: string;
    name?: string;
    description?: string;
    surface?: string;
    stems?: string[];
    selectionEnvelope?: ModelSelectionEnvelope;
    ensembleConfig: NonNullable<SeparationConfig["ensembleConfig"]>;
    postprocess?: SeparationConfig["postProcessingSteps"];
    runtimePolicy?: SeparationConfig["runtimePolicy"];
    exportPolicy?: SeparationConfig["exportPolicy"];
    splitFreq?: number;
    outputFormat?: SeparationConfig["outputFormat"];
  },
): SeparationWorkflow {
  const {
    id,
    name,
    description,
    surface,
    stems,
    selectionEnvelope,
    ensembleConfig,
    postprocess,
    runtimePolicy,
    exportPolicy,
    splitFreq,
    outputFormat,
  } = args;

  return {
    version: 1,
    id,
    name,
    kind: "ensemble",
    selectionEnvelope,
    surface: normalizeSurface(surface),
    description,
    stems,
    models: ensembleConfig.models.map((model, index) => ({
      model_id: model.model_id,
      weight: model.weight,
      role:
        ensembleConfig.phaseFixEnabled && index === 1
          ? "phase_reference"
          : "ensemble_partner",
      required: true,
    })),
    blend: {
      algorithm: ensembleConfig.algorithm,
      stemAlgorithms: ensembleConfig.stemAlgorithms,
      splitFreq,
      phaseFixEnabled: ensembleConfig.phaseFixEnabled,
      phaseFixParams: ensembleConfig.phaseFixParams,
    },
    postprocess,
    runtimePolicy,
    exportPolicy: exportPolicy || {
      stems,
      outputFormat,
    },
  };
}

function buildSingleWorkflow(args: {
  id?: string;
  name?: string;
  description?: string;
  surface?: string;
  modelId: string;
  stems?: string[];
  selectionEnvelope?: ModelSelectionEnvelope;
  postprocess?: SeparationConfig["postProcessingSteps"];
  runtimePolicy?: SeparationConfig["runtimePolicy"];
  exportPolicy?: SeparationConfig["exportPolicy"];
  outputFormat?: SeparationConfig["outputFormat"];
}): SeparationWorkflow {
  return {
    version: 1,
    id: args.id,
    name: args.name,
    kind: "single",
    selectionEnvelope: args.selectionEnvelope,
    surface: normalizeSurface(args.surface),
    description: args.description,
    stems: args.stems,
    models: [{ model_id: args.modelId, role: "primary", required: true }],
    postprocess: args.postprocess,
    runtimePolicy: args.runtimePolicy,
    exportPolicy: args.exportPolicy || {
      stems: args.stems,
      outputFormat: args.outputFormat,
    },
  };
}

function referencedWorkflowModelIds(
  workflow: SeparationWorkflow | undefined,
): string[] {
  if (!workflow) return [];
  const ids = new Set<string>();
  for (const model of workflow.models || []) {
    if (model?.model_id) ids.add(model.model_id);
  }
  for (const step of workflow.steps || []) {
    if (step?.model_id) ids.add(step.model_id);
    if (step?.source_model) ids.add(step.source_model);
  }
  return Array.from(ids);
}

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

function getPresetAdvancedDefaults(preset?: PresetLike) {
  const direct = preset?.advancedDefaults;
  const recipeDefaults = preset?.recipe?.defaults;
  return {
    overlap:
      typeof direct?.overlap === "number"
        ? direct.overlap
        : typeof recipeDefaults?.overlap === "number"
          ? recipeDefaults.overlap
          : undefined,
    segmentSize:
      typeof direct?.segment_size === "number"
        ? direct.segment_size
        : typeof direct?.chunk_size === "number"
          ? direct.chunk_size
          : typeof recipeDefaults?.segment_size === "number"
            ? recipeDefaults.segment_size
            : typeof recipeDefaults?.chunk_size === "number"
              ? recipeDefaults.chunk_size
              : undefined,
    shifts:
      typeof direct?.shifts === "number"
        ? direct.shifts
        : typeof recipeDefaults?.shifts === "number"
          ? recipeDefaults.shifts
          : undefined,
    tta:
      typeof direct?.tta === "boolean"
        ? direct.tta
        : typeof recipeDefaults?.tta === "boolean"
          ? recipeDefaults.tta
          : undefined,
  };
}

export function resolveSeparationPlan(inputs: ResolveInputs): SeparationPlan {
  const {
    config,
    presets = [],
    modelMap = {},
    models = [],
    globalPhaseParams,
    runtimeSupport,
  } = inputs;

  const missingModels: MissingModel[] = [];
  const blockingIssues: string[] = [];
  const warnings: string[] = [];

  // --- Step 1: Start from config defaults ---
  let effectiveModelId = config.modelId || "htdemucs";
  let effectiveStems = config.stems;
  let effectiveEnsembleConfig: SeparationConfig["ensembleConfig"] | undefined =
    config.ensembleConfig;
  let effectivePostProcessingSteps = config.postProcessingSteps;
  let effectiveWorkflow = config.workflow;
  let effectiveAdvancedParams =
    config.mode === "advanced"
      ? {
          overlap: config.advancedParams?.overlap,
          segmentSize: config.advancedParams?.segmentSize,
          shifts: config.advancedParams?.shifts,
          tta: config.advancedParams?.tta,
        }
      : undefined;

  const preset =
    config.mode === "simple" ? findPreset(presets, config.presetId) : undefined;
  const usedPreset = !!preset;

  // Simple mode is preset-authoritative: ignore any UI-provided overrides.
  // The caller should pass only presetId in Simple mode, but we defensively clamp here.
  if (config.mode === "simple") {
    effectiveModelId = config.workflowId || "htdemucs";
    effectiveEnsembleConfig = undefined;
    effectivePostProcessingSteps = undefined;
    // Stems in Simple mode come from preset unless explicitly supported later.
    // We still allow a preset to define stems.
    effectiveStems = undefined;
    effectiveAdvancedParams = getPresetAdvancedDefaults(preset);
  }

  // --- Step 2: Apply Simple preset selection rules (if any) ---
  if (config.mode === "simple" && preset) {
    const presetWorkflow = buildRecipeWorkflow(preset, config);
    if (presetWorkflow) {
      effectiveWorkflow = presetWorkflow;
      effectiveModelId = presetWorkflow.id || preset.id;
      effectiveStems = presetWorkflow.stems || effectiveStems;
    }

    // If preset defines an ensemble config, that is authoritative (pipeline defined)
    if (preset.ensembleConfig && preset.ensembleConfig.models?.length > 0) {
      effectiveModelId = "ensemble";
      effectiveEnsembleConfig = {
        ...preset.ensembleConfig,
      } as any;
      effectiveStems = getDefaultEnsembleStems(config.stems || preset.stems);
      effectiveWorkflow = buildEnsembleWorkflow({
        id: preset.id,
        name: preset.name,
        description: preset.workflowSummary || preset.description,
        surface: preset.simpleGoal || preset.recipe?.target || "ensemble",
        stems: effectiveStems,
        selectionEnvelope: preset.selectionEnvelope || preset.recipe?.selection_envelope,
        ensembleConfig: effectiveEnsembleConfig!,
        postprocess: effectivePostProcessingSteps,
        runtimePolicy: config.runtimePolicy,
        exportPolicy: config.exportPolicy,
        splitFreq: config.splitFreq,
        outputFormat: config.outputFormat,
      });
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
      if (!effectiveWorkflow && mappedModelId) {
        effectiveWorkflow = buildSingleWorkflow({
          id: preset.id,
          name: preset.name,
          description: preset.description,
          surface: preset.simpleGoal || "single",
          modelId: mappedModelId,
          stems: effectiveStems,
          selectionEnvelope: preset.selectionEnvelope || preset.recipe?.selection_envelope,
          postprocess: effectivePostProcessingSteps,
          runtimePolicy: config.runtimePolicy,
          exportPolicy: config.exportPolicy,
          outputFormat: config.outputFormat,
        });
      }
    }

    // Post-processing pipeline from preset (if the preset defines it)
  if (
    Array.isArray(preset.postProcessingSteps) &&
    preset.postProcessingSteps.length > 0
  ) {
    effectivePostProcessingSteps = preset.postProcessingSteps;
    if (effectiveWorkflow) {
      effectiveWorkflow = {
        ...effectiveWorkflow,
        postprocess: preset.postProcessingSteps,
      };
    }
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
    effectiveWorkflow = buildEnsembleWorkflow({
      id: config.workflowId,
      name: config.workflow?.name || "Custom Ensemble",
      description: config.workflow?.description,
      surface: config.workflow?.surface || "ensemble",
      stems: effectiveStems,
      selectionEnvelope: config.selectionEnvelope,
      ensembleConfig: config.ensembleConfig,
      postprocess: effectivePostProcessingSteps,
      runtimePolicy: config.runtimePolicy,
      exportPolicy: config.exportPolicy,
      splitFreq: config.splitFreq,
      outputFormat: config.outputFormat,
    });
  }

  if (!effectiveWorkflow && config.workflow) {
    effectiveWorkflow = config.workflow;
    effectiveModelId =
      config.workflowId ||
      config.workflow.id ||
      config.modelId ||
      (config.workflow.kind === "ensemble" ? "ensemble" : "pipeline");
    effectiveStems = config.workflow.stems || effectiveStems;
  }

  if (!effectiveWorkflow && config.mode !== "simple" && effectiveModelId) {
    effectiveWorkflow = buildSingleWorkflow({
      id: config.workflowId,
      name: config.workflow?.name,
      description: config.workflow?.description,
      surface: config.workflow?.surface || "single",
      modelId: effectiveModelId,
      stems: effectiveStems,
      selectionEnvelope: config.selectionEnvelope,
      postprocess: effectivePostProcessingSteps,
      runtimePolicy: config.runtimePolicy,
      exportPolicy: config.exportPolicy,
      outputFormat: config.outputFormat,
    });
  }

  if (effectiveWorkflow) {
    effectiveWorkflow = {
      ...effectiveWorkflow,
      stems: effectiveWorkflow.stems || effectiveStems,
      postprocess:
        effectiveWorkflow.postprocess || effectivePostProcessingSteps,
      runtimePolicy:
        effectiveWorkflow.runtimePolicy || config.runtimePolicy,
      exportPolicy:
        effectiveWorkflow.exportPolicy ||
        config.exportPolicy || {
          stems: effectiveWorkflow.stems || effectiveStems,
          outputFormat: config.outputFormat,
        },
    };
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

  const pushBlocking = (message: string) => {
    if (!blockingIssues.includes(message)) blockingIssues.push(message);
  };

  const pushWarning = (message: string) => {
    if (!warnings.includes(message)) warnings.push(message);
  };

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

  const workflowModelIds = referencedWorkflowModelIds(effectiveWorkflow);
  const effectiveReferencedModelIds =
    workflowModelIds.length > 0
      ? workflowModelIds
      : effectiveEnsembleConfig?.models?.map((entry) => entry.model_id).filter(Boolean) ||
        (effectiveModelId && effectiveModelId !== "ensemble" ? [effectiveModelId] : []);

  if (workflowModelIds.length > 0) {
    for (const workflowModelId of workflowModelIds) {
      validateModelId(workflowModelId);
    }
  } else if (
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

  for (const modelId of effectiveReferencedModelIds) {
    const model = findModel(models, modelId);
    if (!model) continue;

    if (config.mode === "simple" && model.status?.simple_allowed === false) {
      pushBlocking(
        `${model.name || model.id} is not allowed in Simple mode and must be run from Advanced mode.`,
      );
    } else if (model.status?.simple_allowed === false) {
      pushWarning(
        `${model.name || model.id} is marked as Advanced-only in the guide metadata.`,
      );
    }

    if (model.status?.readiness === "blocked" || getModelCatalogTier(model as any) === "blocked") {
      const message =
        model.status?.blocking_reason ||
        `${model.name || model.id} is blocked by the registry/runtime policy.`;
      if (config.mode === "simple") pushBlocking(message);
      else pushWarning(message);
    }

    if (model.status?.readiness === "manual" || isManualCatalogModel(model as any)) {
      const message =
        model.status?.blocking_reason ||
        `${model.name || model.id} requires manual setup before it can run cleanly.`;
      if (config.mode === "simple") pushBlocking(message);
      else pushWarning(message);
    }

    if (runtimeSupport?.fnoSupported === false && modelRequiresFnoRuntime(model)) {
      const message = `${model.name || model.id} requires FNO/neuralop runtime support.`;
      if (config.mode === "simple") pushBlocking(message);
      else pushWarning(message);
    }
  }

  if (effectiveWorkflow?.steps?.some((step) => step.action === "phase_fix")) {
    const invalidPhaseReference = effectiveWorkflow.steps.find((step) => {
      if (step.action !== "phase_fix" || !step.source_model) return false;
      const referenceModel = findModel(models, step.source_model);
      return referenceModel?.phase_fix?.is_valid_reference === false;
    });
    if (invalidPhaseReference?.source_model) {
      const message = `${invalidPhaseReference.source_model} is not marked as a valid phase-fix reference in the registry.`;
      if (config.mode === "simple") pushBlocking(message);
      else pushWarning(message);
    }
  }

  const canProceed = missingModels.length === 0 && blockingIssues.length === 0;

  return {
    effectiveModelId,
    effectiveStems,
    effectiveEnsembleConfig,
    effectivePostProcessingSteps,
    effectiveWorkflow,
    effectiveAdvancedParams,
    effectiveGlobalPhaseParams,
    missingModels,
    blockingIssues,
    warnings,
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
