import { useState, useEffect, useMemo, useRef } from "react";
import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";
import {
  ArrowLeft,
  Settings2,
  Layers,
  Zap,
  Music,
  AlertTriangle,
} from "lucide-react";
import MissingModelsDialog from "./dialogs/MissingModelsDialog";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { useStore } from "../stores/useStore";
import { toast } from "sonner";
import { Preset } from "../presets";
import type { SeparationConfig } from "../types/separation";
import { EnsembleBuilder } from "./EnsembleBuilder";
import { ModelSelector } from "./ModelSelector";
import { VRAMUsageMeter, estimateVRAMUsage } from "./ui/vram-meter";
import { CPUOnlyWarning, LowVRAMWarning, EnsembleTip } from "./ui/warning-tip";
import { bestVolumeCompensation } from "../utils/volumeCompensation";
import { SimplePresetPicker } from "./simple/SimplePresetPicker";
import { PageShell } from "./PageShell";
import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";
import { modelRequiresFnoRuntime } from "../lib/systemRuntime/modelRuntime";
import { recommendWorkflowPreset } from "@/lib/policy/recommendationPolicy";
import { SeparationPlanCard } from "./SeparationPlanCard";
import { CollapsibleSection } from "./ui/collapsible-section";
import type { SeparationPreflightReport } from "@/types/preflight";
import {
  buildSeparationBackendPayload,
  executeSeparationPreflight,
} from "@/lib/separation/backendPayload";

type AdvancedParams = {
  overlap: number;
  segmentSize: number;
  shifts: number;
  tta: boolean;
  bitrate: string;
};

const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));

const getRecommendedPhaseFixParams = (
  model: any,
): { lowHz: number; highHz: number; highFreqWeight: number } | null => {
  const p = model?.phase_fix?.recommended_params;
  if (!p || typeof p !== "object") return null;

  const lowHz =
    typeof p.lowHz === "number" && Number.isFinite(p.lowHz) ? p.lowHz : null;
  const highHz =
    typeof p.highHz === "number" && Number.isFinite(p.highHz) ? p.highHz : null;
  const highFreqWeight =
    typeof p.highFreqWeight === "number" && Number.isFinite(p.highFreqWeight)
      ? p.highFreqWeight
      : null;

  if (lowHz === null || highHz === null || highFreqWeight === null) return null;
  return { lowHz, highHz, highFreqWeight };
};

const computeAutoTunedAdvancedParams = (args: {
  mode: "simple" | "advanced";
  device: string;
  availableVRAM: number;
  models: any[];
  selectedModelId: string;
  isEnsembleMode: boolean;
  ensembleConfig: { model_id: string; weight: number }[];
  separationPlan: any;
  globalAdvancedSettings?: any;
  current: AdvancedParams;
}): AdvancedParams => {
  const {
    mode,
    device,
    availableVRAM,
    models,
    selectedModelId,
    isEnsembleMode,
    ensembleConfig,
    separationPlan,
    globalAdvancedSettings,
    current,
  } = args;

  // Base: start from current so we keep user/global defaults for fields we don't tune.
  let next: AdvancedParams = { ...current };

  // If CPU-only or unknown VRAM, use safe-ish defaults and avoid playing games.
  const isCuda = device.startsWith("cuda");
  const vram = Number.isFinite(availableVRAM) ? availableVRAM : 0;

  // Determine "effective model type" for heuristics.
  const effectiveModelId =
    separationPlan?.resolved?.model_id ||
    separationPlan?.effectiveModelId ||
    (isEnsembleMode ? "ensemble" : selectedModelId) ||
    "";

  const modelForDefaults =
    models.find((m) => m.id === effectiveModelId) ||
    models.find((m) => m.id === selectedModelId) ||
    null;

  const arch = String(
    modelForDefaults?.architecture ||
      modelForDefaults?.name ||
      effectiveModelId ||
      "",
  ).toLowerCase();

  const isRoformer =
    arch.includes("roformer") ||
    String(effectiveModelId).toLowerCase().includes("roformer");

  // How many models will be active (approx).
  const activeModelsCount =
    separationPlan?.resolved?.ensemble_config?.length ||
    separationPlan?.effectiveEnsembleConfig?.models?.length ||
    (Array.isArray(ensembleConfig) && ensembleConfig.length > 0
      ? ensembleConfig.length
      : 1);
  const HQ = 485100;
  const MID = 352800;
  const LOW = 112455;
  const MID_LOW = 200000;

  let desiredSegment = MID;

  if (isRoformer && isCuda && vram >= 8) {
    desiredSegment = HQ;
  } else if (!isCuda || vram < 4) {
    desiredSegment = LOW;
  } else if (vram < 5) {
    desiredSegment = MID_LOW;
  }

  if (isCuda && activeModelsCount >= 2) {
    if (vram > 0 && vram < 8) desiredSegment = Math.min(desiredSegment, LOW);
    else if (vram > 0 && vram < 10)
      desiredSegment = Math.min(desiredSegment, MID);
  }
  if (mode === "advanced" && !isEnsembleMode && modelForDefaults) {
    const modelRec =
      modelForDefaults.recommended_settings?.segment_size ??
      modelForDefaults.recommended_settings?.chunk_size ??
      modelForDefaults.chunk_size;
    if (
      typeof modelRec === "number" &&
      Number.isFinite(modelRec) &&
      modelRec > 0
    ) {
      desiredSegment = Math.max(desiredSegment, modelRec);
    }
  }

  next.segmentSize = desiredSegment;

  let desiredOverlap = current.overlap ?? 4;
  if (mode !== "simple") {
    if (isRoformer) {
      if (isCuda && vram >= 10) desiredOverlap = 8;
      else desiredOverlap = 4;
    } else {
      desiredOverlap = 4;
    }

    if (activeModelsCount >= 2) desiredOverlap = Math.max(desiredOverlap, 4);

    if (mode === "advanced" && !isEnsembleMode && modelForDefaults) {
      const modelOverlap = modelForDefaults.recommended_settings?.overlap;
      if (typeof modelOverlap === "number" && Number.isFinite(modelOverlap)) {
        desiredOverlap = modelOverlap;
      }
    }
  }

  next.overlap = clamp(desiredOverlap, 2, 50);

  next.shifts = Number.isFinite(current.shifts) ? current.shifts : 1;
  next.tta = !!current.tta;

  next.bitrate = current.bitrate || globalAdvancedSettings?.bitrate || "320k";

  return next;
};

const computeBestSegmentSizeFromVRAM = (availableVRAM: number) => {
  const vram = Number.isFinite(availableVRAM) ? availableVRAM : 0;
  if (vram <= 0) return 0;
  if (vram >= 8.0) return 485100;
  if (vram >= 5.0) return 352800;
  if (vram >= 4.0) return 112455;
  return 56227;
};

export type { SeparationConfig } from "../types/separation";

export interface ConfigurePageProps {
  fileName: string;
  filePath: string;
  onBack: () => void;
  onConfirm: (config: SeparationConfig) => void;
  initialPresetId?: string;
  presets: Preset[];
  availability?: Record<string, any>;
  modelMap?: Record<string, string>;
  models?: any[];
  onNavigateToModels?: (modelId?: string) => void;
}

export function ConfigurePage({
  fileName,
  filePath,
  onBack,
  onConfirm,
  initialPresetId,
  presets = [],
  availability,
  modelMap = {},
  models = [],
  onNavigateToModels,
}: ConfigurePageProps) {
  const globalAdvancedSettings = useStore(
    (state) => state.settings.advancedSettings,
  );

  const normalizeOverlap = (value: unknown) => {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return 4;
    if (n < 1) {
      const denom = Math.max(1e-6, 1 - n);
      return Math.max(2, Math.min(50, Math.round(1 / denom)));
    }
    return Math.max(2, Math.min(50, Math.round(n)));
  };

  const setAdvancedSettings = useStore((state) => state.setAdvancedSettings);
  const phaseParams = useStore((state) => state.settings.phaseParams);
  const startDownload = useStore((state) => state.startDownload);
  const setDownloadError = useStore((state) => state.setDownloadError);
  const [mode, setMode] = useState<"simple" | "advanced">("simple");
  const [selectedPresetId, setSelectedPresetId] = useState<string>(
    initialPresetId || (presets.length > 0 ? presets[0].id : ""),
  );
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [device, setDevice] = useState<string>(() => {
    const stored = globalAdvancedSettings?.device;
    const preferred = globalAdvancedSettings?.preferredCudaDevice;

    // Normalize legacy "cuda" to a concrete index, if we have one.
    if (stored === "cuda") return preferred || "cuda:0";

    // If the user saved a concrete cuda:<idx>, keep it.
    if (typeof stored === "string" && stored.startsWith("cuda:")) return stored;

    return stored || "auto";
  });
  const [invert, _setInvert] = useState(false);
  const [normalize, _setNormalize] = useState(false);
  const [volumeCompEnabled, setVolumeCompEnabled] = useState(false);
  const [bitDepth, _setBitDepth] = useState("16");
  const [_splitFreq, _setSplitFreq] = useState(750);
  const [advancedParams, setAdvancedParams] = useState<AdvancedParams>({
    overlap: normalizeOverlap(globalAdvancedSettings?.overlap ?? 4),
    segmentSize: globalAdvancedSettings?.segmentSize ?? 0,
    shifts: globalAdvancedSettings?.shifts || 1,
    tta: false,
    bitrate: globalAdvancedSettings?.bitrate || "320k",
  });

  const [advancedParamsDirty, setAdvancedParamsDirty] = useState(false);

  const [missingDialogOpen, setMissingDialogOpen] = useState(false);
  const [pendingPresetMissingPromptId, setPendingPresetMissingPromptId] =
    useState<string | null>(null);
  const [backendPreflight, setBackendPreflight] =
    useState<SeparationPreflightReport | null>(null);
  const [isPreflightLoading, setIsPreflightLoading] = useState(false);
  const [sourceFileExists, setSourceFileExists] = useState(true);
  const preflightRequestRef = useRef(0);

  const handleQuickDownload = async (modelId: string) => {
    try {
      startDownload(modelId);
      if (window.electronAPI?.installSelection)
        await window.electronAPI.installSelection("model", modelId);
    } catch (e) {
      setDownloadError(modelId, e instanceof Error ? e.message : String(e));
    }
  };

  // Auto Post-Processing Pipeline State
  const [enableAutoPipeline] = useState(true);

  // Ensemble State
  const [isEnsembleMode, setIsEnsembleMode] = useState(false);
  const [ensembleConfig, setEnsembleConfig] = useState<
    { model_id: string; weight: number }[]
  >([]);
  const [ensembleAlgorithm, setEnsembleAlgorithm] = useState<
    "average" | "max_spec" | "min_spec" | "frequency_split"
  >("average");
  const [phaseFixEnabled, setPhaseFixEnabled] = useState(false);
  const [stemAlgorithms, setStemAlgorithms] = useState<
    | {
        vocals?: "average" | "max_spec" | "min_spec";
        instrumental?: "average" | "max_spec" | "min_spec";
      }
    | undefined
  >(undefined);
  const [phaseFixParams, setPhaseFixParams] = useState<{
    lowHz: number;
    highHz: number;
    highFreqWeight: number;
  }>({ lowHz: 500, highHz: 5000, highFreqWeight: 2.0 });

  const { info: runtimeInfo } = useSystemRuntimeInfo();
  const gpuInfo = runtimeInfo?.gpu ?? null;

  // Update selected preset when initialPresetId changes
  useEffect(() => {
    if (initialPresetId) {
      setSelectedPresetId(initialPresetId);
    }
  }, [initialPresetId]);

  // Update selected model when preset changes
  useEffect(() => {
    if (mode === "simple" && selectedPresetId) {
      const preset = presets.find((p) => p.id === selectedPresetId);
      if (preset) {
        if (preset.ensembleConfig) {
          // Ensemble preset - handled in render
        } else if ((preset as any).isRecipe) {
          // Recipe/workflow presets execute via presetId->modelId mapping in SeparatePage.
          // They are not selectable as a single installed model.
          setSelectedModelId("");
        } else {
          const modelId = preset.modelId || modelMap[preset.id];
          if (modelId) setSelectedModelId(modelId);
        }
      }
    }
  }, [selectedPresetId, mode, presets, modelMap]);

  // Update processing parameters when model changes in Advanced Mode
  // NOTE: This only runs when the user hasn't manually adjusted advanced params (dirty=false).
  useEffect(() => {
    if (mode === "advanced" && selectedModelId && !advancedParamsDirty) {
      const model = models.find((m) => m.id === selectedModelId);
      if (model) {
        const defaultOverlap = 4;
        const defaultSegmentSize = 352800; // Safe default for most models

        // Use model's recommended settings if available
        const recommendedOverlap =
          model.recommended_settings?.overlap ?? defaultOverlap;
        const recommendedSegmentSize =
          model.recommended_settings?.segment_size ??
          model.recommended_settings?.chunk_size ??
          model.chunk_size ??
          defaultSegmentSize;

        setAdvancedParams((prev) => ({
          ...prev,
          overlap: recommendedOverlap,
          // Preserve Auto (0) so the backend can apply model/machine recommended defaults.
          segmentSize: prev.segmentSize > 0 ? recommendedSegmentSize : prev.segmentSize,
        }));
      }
    }
  }, [selectedModelId, mode, models, advancedParamsDirty]);

  // Auto-suggest phase-fix params for the selected model (when user enables phase correction)
  // - We only auto-apply when enabling (to avoid overwriting manual tweaks).
  // - Works for both Simple and Advanced model selection, but only when a single model is selected.
  useEffect(() => {
    if (!phaseFixEnabled) return;
    if (!selectedModelId) return;

    const model = models.find((m) => m.id === selectedModelId);
    if (!model) return;

    const recommended = getRecommendedPhaseFixParams(model);
    if (!recommended) return;

    setPhaseFixParams((prev) => {
      // If user already changed away from defaults, don't stomp their values.
      const isDefault =
        prev.lowHz === 500 &&
        prev.highHz === 5000 &&
        prev.highFreqWeight === 2.0;

      if (!isDefault) return prev;
      return recommended;
    });
  }, [phaseFixEnabled, selectedModelId, models]);

  const separationPlan = useMemo(() => {
    // Construct a minimal config that reflects what ConfigurePage would run if confirmed.
    // Note: we intentionally do NOT set outputFormat, etc. here; the resolver only cares
    // about model/preset/ensemble/post-processing/phase params.
    const usePresetId = mode === "simple" ? selectedPresetId : undefined;
    const preset =
      mode === "simple"
        ? presets.find((p) => p.id === selectedPresetId)
        : undefined;

    const config: any = {
      mode,
      presetId: usePresetId,
      modelId: isEnsembleMode ? undefined : selectedModelId,
      stems: isEnsembleMode ? ["vocals", "instrumental"] : undefined,
      ensembleConfig:
        mode === "advanced" && isEnsembleMode
          ? {
              models: ensembleConfig,
              algorithm: ensembleAlgorithm as any,
              stemAlgorithms: stemAlgorithms,
              phaseFixEnabled: phaseFixEnabled,
              phaseFixParams: phaseFixEnabled ? phaseFixParams : undefined,
            }
          : undefined,
    };

    if (mode === "simple") {
      if (preset?.ensembleConfig) {
        config.ensembleConfig = preset.ensembleConfig as any;
        config.modelId = undefined;
      }

      if (
        enableAutoPipeline &&
        preset?.postProcessingSteps &&
        preset.postProcessingSteps.length > 0
      ) {
        config.postProcessingSteps = preset.postProcessingSteps;
      }
    }

    return resolveSeparationPlan({
      config,
      presets: presets as any,
      models: models as any,
      globalPhaseParams: phaseParams,
      runtimeSupport: {
        fnoSupported:
          runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok !== false,
      },
    });
  }, [
    mode,
    selectedPresetId,
    presets,
    models,
    selectedModelId,
    isEnsembleMode,
    ensembleConfig,
    ensembleAlgorithm,
    stemAlgorithms,
    phaseFixEnabled,
    phaseFixParams,
    enableAutoPipeline,
    phaseParams,
    runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok,
  ]);

  // Canonical missing dependencies from the separation plan
  const missingModels = useMemo(() => {
    return separationPlan.missingModels.map((m) => m.modelId);
  }, [separationPlan.missingModels]);

  const validEnsembleModels = useMemo(() => {
    if (!Array.isArray(ensembleConfig)) return [];
    return ensembleConfig.filter(
      (m) => typeof m?.model_id === "string" && m.model_id.trim().length > 0,
    );
  }, [ensembleConfig]);

  useEffect(() => {
    if (missingModels.length === 0) {
      setMissingDialogOpen(false);
    }
  }, [missingModels.length]);

  useEffect(() => {
    if (mode !== "simple" || !pendingPresetMissingPromptId) return;
    if (selectedPresetId !== pendingPresetMissingPromptId) return;

    if (missingModels.length > 0) {
      setMissingDialogOpen(true);
    }

    setPendingPresetMissingPromptId(null);
  }, [missingModels.length, mode, pendingPresetMissingPromptId, selectedPresetId]);

  const plannedModelIds = useMemo(() => {
    const ids = new Set<string>();

    const planEnsemble = separationPlan.effectiveEnsembleConfig;
    if (planEnsemble?.models && Array.isArray(planEnsemble.models)) {
      for (const entry of planEnsemble.models) {
        if (typeof entry?.model_id === "string" && entry.model_id.trim()) {
          ids.add(entry.model_id.trim());
        }
      }
    }

    if (
      typeof separationPlan.effectiveModelId === "string" &&
      separationPlan.effectiveModelId.trim() &&
      separationPlan.effectiveModelId !== "ensemble"
    ) {
      ids.add(separationPlan.effectiveModelId.trim());
    }

    if (mode === "simple") {
      const preset = presets.find((p) => p.id === selectedPresetId);
      if (preset?.ensembleConfig?.models?.length) {
        for (const m of preset.ensembleConfig.models) {
          if (typeof m?.model_id === "string" && m.model_id.trim()) {
            ids.add(m.model_id.trim());
          }
        }
      } else if (typeof preset?.modelId === "string" && preset.modelId.trim()) {
        ids.add(preset.modelId.trim());
      }
    } else if (isEnsembleMode) {
      for (const m of validEnsembleModels) {
        if (typeof m?.model_id === "string" && m.model_id.trim()) {
          ids.add(m.model_id.trim());
        }
      }
    } else if (selectedModelId) {
      ids.add(selectedModelId);
    }

    return Array.from(ids);
  }, [
    isEnsembleMode,
    mode,
    presets,
    selectedModelId,
    selectedPresetId,
    separationPlan.effectiveEnsembleConfig,
    separationPlan.effectiveModelId,
    validEnsembleModels,
  ]);

  const blockedFnoModels = useMemo(() => {
    const fnoSupported =
      runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok !== false;
    if (fnoSupported) return [] as string[];
    return plannedModelIds.filter((modelId) => {
      const model = models.find((candidate) => candidate.id === modelId);
      return modelRequiresFnoRuntime(model);
    });
  }, [models, plannedModelIds, runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok]);

  const hasFnoRuntimeBlock = blockedFnoModels.length > 0;

  const selectedPreset = presets.find((p) => p.id === selectedPresetId);

  const cudaGpus = useMemo(() => {
    const gpus = gpuInfo?.gpus;
    if (!Array.isArray(gpus)) return [];
    return gpus.filter((g: any) => {
      const type = typeof g?.type === "string" ? g.type.toLowerCase() : "";
      if (type === "cuda" || type === "nvidia") return true;
      const id = typeof g?.id === "string" ? g.id.toLowerCase() : "";
      if (id.startsWith("cuda:")) return true;
      // Fallbacks for older payloads.
      if (Number.isFinite(g?.index) && id.includes("cuda")) return true;
      const name = typeof g?.name === "string" ? g.name.toLowerCase() : "";
      if (name.includes("nvidia") || name.includes("geforce") || name.includes("rtx")) {
        return true;
      }
      return false;
    });
  }, [gpuInfo]);

  const hasCuda = !!gpuInfo?.has_cuda || cudaGpus.length > 0;
  const primaryCudaGpu = cudaGpus[0];
  const primaryCudaDevice = primaryCudaGpu
    ? `cuda:${Number.isFinite(primaryCudaGpu.index) ? primaryCudaGpu.index : 0}`
    : "cuda:0";

  useEffect(() => {
    if (!hasCuda && device.startsWith("cuda")) {
      setDevice("cpu");
    }

    // If state ever contains legacy "cuda", normalize to a concrete device.
    if (hasCuda && device === "cuda") {
      setDevice(
        globalAdvancedSettings?.preferredCudaDevice || primaryCudaDevice,
      );
    }
  }, [
    hasCuda,
    device,
    primaryCudaDevice,
    globalAdvancedSettings?.preferredCudaDevice,
  ]);

  // Get VRAM info
  const selectedCudaIndex = device.startsWith("cuda:")
    ? parseInt(device.split(":")[1] || "0", 10)
    : null;
  const selectedGpuForMeter =
    selectedCudaIndex !== null
      ? cudaGpus.find(
          (g: any, idx: number) =>
            (Number.isFinite(g?.index) ? g.index : idx) === selectedCudaIndex,
        ) || primaryCudaGpu
      : primaryCudaGpu;
  const availableVRAM = selectedGpuForMeter?.memory_gb || 0;
  const fnoFallbackRecommendation = useMemo(() => {
    if (!hasFnoRuntimeBlock) return null;
    const target = (() => {
      const explicit = selectedPreset?.simpleGoal;
      if (explicit === "cleanup") return "restoration" as const;
      if (explicit === "karaoke") return "karaoke" as const;
      if (explicit === "vocals") return "vocals" as const;
      if (explicit === "instruments") return "instrumental" as const;
      return "instrumental" as const;
    })();
    return recommendWorkflowPreset(
      target,
      presets,
      models as any,
      {
        device,
        vramGb: availableVRAM,
      },
      { fnoSupported: false },
    );
  }, [availableVRAM, device, hasFnoRuntimeBlock, models, presets, selectedPreset?.simpleGoal]);
  const modelType = selectedPreset?.name || selectedModelId || "unknown";
  const estimatedVRAM = estimateVRAMUsage(
    modelType,
    advancedParams.segmentSize,
    advancedParams.overlap,
    advancedParams.tta,
  );
  const isCPUOnly = !hasCuda;
  const maxSafeSegmentSize = computeBestSegmentSizeFromVRAM(availableVRAM);

  const effectiveConfig = useMemo<SeparationConfig>(() => {
    const preset =
      mode === "simple" ? presets.find((p) => p.id === selectedPresetId) : undefined;
    const usePresetId = mode === "simple" ? selectedPresetId : undefined;
    const presetOverlap = (() => {
      const explicit =
        preset?.advancedDefaults?.overlap ?? preset?.recipe?.defaults?.overlap;
      if (typeof explicit === "number" && Number.isFinite(explicit)) return explicit;

      const byQuality = (() => {
        const q = preset?.qualityLevel;
        if (q === "fast") return 2;
        if (q === "balanced") return 4;
        if (q === "quality") return 4;
        if (q === "ultra") return 8;
        return 4;
      })();

      const withEnsembleFloor = preset?.ensembleConfig ? Math.max(byQuality, 4) : byQuality;
      return withEnsembleFloor;
    })();

    const config: SeparationConfig = {
      mode,
      presetId: usePresetId,
      modelId: isEnsembleMode ? undefined : selectedModelId,
      device,
      outputFormat: "wav",
      volumeCompensation: volumeCompEnabled
        ? bestVolumeCompensation()
        : undefined,
      stems: isEnsembleMode ? ["vocals", "instrumental"] : undefined,
      invert,
      normalize,
      bitDepth,
      splitFreq:
        mode === "advanced" &&
        isEnsembleMode &&
        ensembleAlgorithm === "frequency_split"
          ? _splitFreq
          : undefined,
      advancedParams:
        mode === "advanced"
          ? advancedParams
          : {
              segmentSize: (() => {
                if (maxSafeSegmentSize > 0) return maxSafeSegmentSize;
                const fallback =
                  (globalAdvancedSettings?.segmentSize ?? advancedParams.segmentSize) ||
                  0;
                return typeof fallback === "number" && Number.isFinite(fallback)
                  ? fallback
                  : 0;
              })(),
              shifts:
                preset?.qualityLevel === "fast"
                  ? 1
                  : preset?.qualityLevel === "balanced"
                    ? 2
                    : preset?.qualityLevel === "quality"
                      ? 3
                      : 4,
              overlap: presetOverlap,
            },
      ensembleConfig:
        mode === "advanced" && isEnsembleMode
          ? {
              models: validEnsembleModels,
              algorithm: ensembleAlgorithm as any,
              stemAlgorithms: stemAlgorithms,
              phaseFixEnabled: phaseFixEnabled,
              phaseFixParams: phaseFixEnabled ? phaseFixParams : undefined,
            }
          : undefined,
    };

    if (mode === "simple") {
      if (preset?.ensembleConfig) {
        config.ensembleConfig = preset.ensembleConfig as any;
        config.modelId = undefined;
      }

      if (
        enableAutoPipeline &&
        preset?.postProcessingSteps &&
        preset.postProcessingSteps.length > 0
      ) {
        config.postProcessingSteps = preset.postProcessingSteps;
      }
    }

    return config;
  }, [
    _splitFreq,
    advancedParams,
    bitDepth,
    device,
    enableAutoPipeline,
    ensembleAlgorithm,
    globalAdvancedSettings?.segmentSize,
    invert,
    isEnsembleMode,
    maxSafeSegmentSize,
    mode,
    normalize,
    phaseFixEnabled,
    phaseFixParams,
    presets,
    selectedModelId,
    selectedPresetId,
    stemAlgorithms,
    validEnsembleModels,
    volumeCompEnabled,
  ]);

  useEffect(() => {
    if (!window.electronAPI?.checkFileExists || !filePath) {
      setSourceFileExists(true);
      return;
    }

    let cancelled = false;

    const verifySourceFile = async () => {
      const exists = await window.electronAPI?.checkFileExists?.(filePath);
      if (!cancelled) {
        setSourceFileExists(!!exists);
      }
    };

    void verifySourceFile();

    return () => {
      cancelled = true;
    };
  }, [filePath]);

  useEffect(() => {
    if (!window.electronAPI?.separationPreflight || !filePath) {
      setBackendPreflight(null);
      setIsPreflightLoading(false);
      return;
    }

    if (!sourceFileExists) {
      setBackendPreflight({
        can_proceed: false,
        errors: [
          "The selected source file no longer exists on disk. Go back and import or capture it again.",
        ],
        warnings: [],
        missing_models: [],
      });
      setIsPreflightLoading(false);
      return;
    }

    const hasMinimumSelection =
      (mode === "simple" && !!selectedPresetId) ||
      (mode === "advanced" && !isEnsembleMode && !!selectedModelId) ||
      (mode === "advanced" && isEnsembleMode && validEnsembleModels.length >= 2);

    if (!hasMinimumSelection || !separationPlan.effectiveModelId) {
      setBackendPreflight(null);
      setIsPreflightLoading(false);
      return;
    }

    const requestId = ++preflightRequestRef.current;
    setIsPreflightLoading(true);

    const timer = window.setTimeout(async () => {
      try {
        const stillExists = await window.electronAPI?.checkFileExists?.(filePath);
        if (!stillExists) {
          if (preflightRequestRef.current !== requestId) return;
          setSourceFileExists(false);
          setBackendPreflight({
            can_proceed: false,
            errors: [
              "The selected source file no longer exists on disk. Go back and import or capture it again.",
            ],
            warnings: [],
            missing_models: [],
          });
          return;
        }

        const payload = buildSeparationBackendPayload({
          inputFile: filePath,
          outputDir: "",
          config: effectiveConfig,
          plan: separationPlan,
        });
        const result = await executeSeparationPreflight(
          window.electronAPI,
          payload,
        );

        if (preflightRequestRef.current !== requestId) return;
        setBackendPreflight(result as SeparationPreflightReport);
      } catch (error) {
        if (preflightRequestRef.current !== requestId) return;
        const message =
          error instanceof Error ? error.message : String(error ?? "Preflight failed");
        setBackendPreflight({
          can_proceed: false,
          errors: [message],
          warnings: [],
          missing_models: separationPlan.missingModels.map((item) => item.modelId),
        });
      } finally {
        if (preflightRequestRef.current === requestId) {
          setIsPreflightLoading(false);
        }
      }
    }, 250);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    effectiveConfig,
    filePath,
    isEnsembleMode,
    mode,
    selectedModelId,
    selectedPresetId,
    separationPlan,
    sourceFileExists,
    validEnsembleModels.length,
  ]);

  const handleConfirm = () => {
    const continueConfirm = () => {
      if (mode === "simple" && hasFnoRuntimeBlock) {
        toast.error("Selected model is blocked by environment", {
          description:
            "FNO-based model requires neuraloperator/neuralop with FNO1d support. " +
            `Blocked: ${blockedFnoModels.join(", ")}.`,
        });
        return;
      }

      if (mode === "advanced" && isEnsembleMode) {
        if (validEnsembleModels.length < 2) {
          toast.error("Ensemble requires at least 2 models.", {
            description:
              "Add another model in Ensemble mode (or install more models) and try again.",
          });
          return;
        }
      }

      if (missingModels.length > 0) {
        setMissingDialogOpen(true);
        toast.error("Install the required model before saving this configuration.");
        return;
      }

      if (separationPlan.blockingIssues.length > 0) {
        toast.error(separationPlan.blockingIssues[0]);
        return;
      }

      onConfirm(effectiveConfig);
    };

    if (!window.electronAPI?.checkFileExists || !filePath) {
      if (!sourceFileExists) {
        toast.error("Source file is missing", {
          description:
            "Go back and re-import or re-capture the audio before saving this configuration.",
        });
        return;
      }
      continueConfirm();
      return;
    }

    void (async () => {
      const exists = await window.electronAPI?.checkFileExists?.(filePath);
      if (!exists) {
        setSourceFileExists(false);
        setBackendPreflight({
          can_proceed: false,
          errors: [
            "The selected source file no longer exists on disk. Go back and import or capture it again.",
          ],
          warnings: [],
          missing_models: [],
        });
        toast.error("Source file is missing", {
          description:
            "Go back and re-import or re-capture the audio before saving this configuration.",
        });
        return;
      }
      continueConfirm();
    })();
  };

  const handleSelectPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    setPendingPresetMissingPromptId(presetId);
  };

  useEffect(() => {
    if (advancedParamsDirty) return;

    setAdvancedParams((curr) =>
      computeAutoTunedAdvancedParams({
        mode,
        device,
        availableVRAM,
        models,
        selectedModelId,
        isEnsembleMode,
        ensembleConfig,
        separationPlan,
        globalAdvancedSettings,
        current: curr,
      }),
    );
  }, [
    advancedParamsDirty,
    mode,
    device,
    availableVRAM,
    models,
    selectedModelId,
    isEnsembleMode,
    ensembleConfig,
    separationPlan,
    globalAdvancedSettings,
  ]);

  return (
    <PageShell scroll={false}>
    <div className="flex h-full items-center justify-center px-6 py-6">
    <div className="relative flex h-[min(980px,calc(100vh-2rem))] w-full max-w-[1360px] flex-col overflow-hidden rounded-[2rem] border border-white/70 bg-[rgba(250,248,252,0.78)] text-slate-800 shadow-[0_40px_120px_rgba(141,150,179,0.22)] backdrop-blur-2xl">
      {/* Header */}
      <div className="border-b border-white/60 px-7 py-5">
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={onBack}
            aria-label="Back"
            className="inline-flex h-11 w-11 items-center justify-center rounded-[14px] border border-white/60 bg-white/68 text-slate-600 transition-all hover:bg-white/84 hover:text-slate-900"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="stemsep-config-chip">
                Separation Config
              </span>
              <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
                {mode === "simple" ? "Simple" : "Advanced"}
              </span>
            </div>
            <h1 className="text-[30px] font-normal tracking-[-1.2px] text-slate-800">
              Configure Separation
            </h1>
            <div className="mt-1 flex min-w-0 items-center gap-2 text-slate-500">
              <Music className="h-4 w-4 shrink-0" />
              <span className="truncate text-sm">{fileName}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="stemsep-config-scroll flex-1 min-h-0 overflow-y-auto px-7 py-6 pr-4">
        <div className="mx-auto max-w-[1240px] space-y-6">
          {/* Mode Tabs */}
          <div className="stemsep-config-segmented grid w-full grid-cols-2 rounded-[1.3rem] border border-white/70 bg-white/54 p-1.5">
            <button
              onClick={() => setMode("simple")}
              className={`rounded-[1.05rem] px-4 py-2.5 text-sm font-medium tracking-[-0.2px] transition-all ${
                mode === "simple"
                  ? "stemsep-config-segmented-active text-[#111111]"
                  : "text-slate-500 hover:bg-white/48 hover:text-slate-800"
              }`}
            >
              Simple Mode
            </button>
            <button
              onClick={() => setMode("advanced")}
              className={`rounded-[1.05rem] px-4 py-2.5 text-sm font-medium tracking-[-0.2px] transition-all ${
                mode === "advanced"
                  ? "stemsep-config-segmented-active text-[#111111]"
                  : "text-slate-500 hover:bg-white/48 hover:text-slate-800"
              }`}
            >
              Advanced Mode
            </button>
          </div>

          {mode === "advanced" && (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(280px,360px)]">
              <div className="rounded-[1.7rem] border border-white/65 bg-[linear-gradient(135deg,rgba(255,255,255,0.64),rgba(255,255,255,0.34))] px-6 py-5 shadow-[0_28px_80px_rgba(141,150,179,0.16)] backdrop-blur-2xl">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <span className="stemsep-config-chip">Configuration</span>
                  <span className="stemsep-config-chip stemsep-config-chip-subtle">
                    Machine-aware
                  </span>
                </div>
                <h2 className="text-[24px] font-normal tracking-[-0.7px] text-slate-800">
                  Build a custom separation run with full control
                </h2>
                <p className="mt-2 max-w-3xl text-[14px] leading-[1.55] text-slate-500">
                  Advanced mode is the workshop. Choose device, strategy, chunking and ensemble behavior explicitly when you need to push quality or solve edge cases.
                </p>
              </div>
              <div className="rounded-[1.7rem] border border-white/65 bg-[rgba(255,255,255,0.48)] px-5 py-5 shadow-[0_22px_60px_rgba(141,150,179,0.14)] backdrop-blur-2xl">
                <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-400">
                  Input
                </div>
                <div className="truncate text-[15px] font-medium text-slate-800">
                  {fileName}
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Device
                    </div>
                    <div className="mt-1 text-[14px] text-slate-700">
                      {device === "auto"
                        ? hasCuda
                          ? "Auto / GPU preferred"
                          : "Auto / CPU"
                        : device.startsWith("cuda")
                          ? selectedGpuForMeter?.name || "CUDA"
                          : "CPU"}
                    </div>
                  </div>
                  <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Runtime
                    </div>
                    <div className="mt-1 text-[14px] text-slate-700">
                      {hasCuda ? `${availableVRAM || "GPU"} GB VRAM` : "CPU only"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {!sourceFileExists && (
            <Card className="rounded-[1.5rem] border-rose-300/55 bg-rose-50/82 p-4 text-slate-800 backdrop-blur-xl">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-5 w-5 text-destructive" />
                <div className="space-y-1.5">
                  <div className="font-medium text-destructive">
                    Source file missing
                  </div>
                  <div className="text-sm text-rose-900/80">
                    The selected audio file no longer exists on disk. Go back and import or capture it again before saving this configuration.
                  </div>
                </div>
              </div>
            </Card>
          )}

          {hasFnoRuntimeBlock && (
            <Card className="rounded-[1.5rem] border-rose-300/55 bg-rose-50/82 p-4 text-slate-800 backdrop-blur-xl">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
                <div className="space-y-1">
                  <div className="font-medium text-destructive">
                    {mode === "simple"
                      ? "FNO model blocked in Simple mode"
                      : "FNO runtime warning"}
                  </div>
                  <div className="text-sm text-rose-900/75">
                    {mode === "simple"
                      ? "Install a neuraloperator/neuralop build that exposes "
                      : "This model can still be sent from Advanced mode, but the current runtime does not expose "}
                    <code>neuralop.models.FNO1d</code>
                    {mode === "simple" ? " and restart StemSep." : "."}
                  </div>
                  <div className="text-xs text-rose-700/75">
                    Blocked model(s): {blockedFnoModels.join(", ")}
                  </div>
                  {fnoFallbackRecommendation?.recommendedPresetId && (
                    <div className="text-xs text-rose-700/75">
                      Recommended fallback workflow:{" "}
                      {
                        presets.find(
                          (preset) =>
                            preset.id ===
                            fnoFallbackRecommendation.recommendedPresetId,
                        )?.name
                      }
                    </div>
                  )}
                </div>
              </div>
            </Card>
          )}

          {mode === "advanced" && separationPlan.warnings.length > 0 && (
            <Card className="rounded-[1.5rem] border-amber-300/55 bg-amber-50/82 p-4 text-slate-800 backdrop-blur-xl">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-700" />
                <div className="space-y-1.5">
                  <div className="font-medium text-amber-800">
                    Configuration warnings
                  </div>
                  <div className="space-y-1 text-sm text-amber-900/80">
                    {separationPlan.warnings.map((warning) => (
                      <div key={warning}>{warning}</div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          )}

          {mode === "simple" ? (
            <div>
              <SimplePresetPicker
                presets={presets}
                selectedPresetId={selectedPresetId}
                onSelectPreset={handleSelectPreset}
                availability={availability}
              />
            </div>
          ) : (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.18fr)_380px]">
              <div className="space-y-6">
              <div className="rounded-[1.4rem] border border-amber-300/60 bg-amber-50/82 p-5 text-amber-900/80">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-amber-300/55 bg-amber-50/88 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-amber-700">
                    Advanced Mode
                  </span>
                  <span className="rounded-full border border-white/60 bg-white/65 px-3 py-1 text-[11px] text-slate-500">
                    Manual control
                  </span>
                </div>
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <Settings2 className="h-4 w-4" />
                  Advanced Configuration
                </div>
                <p className="text-sm text-amber-900/76">
                  Fine-tune device, model strategy and processing parameters.
                  Use this when presets are not enough.
                </p>
              </div>

              {/* Processing Device */}
              <Card className="space-y-4 rounded-[1.5rem] border-white/55 bg-[rgba(255,255,255,0.5)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium text-slate-800">
                      Processing Device
                    </label>
                    <p className="mt-1 text-xs text-slate-500">
                      Choose how this run should use your machine.
                    </p>
                  </div>
                  <span
                    className={`rounded-full border px-2.5 py-1 text-[11px] ${
                      hasCuda
                        ? "border-emerald-300/55 bg-emerald-50/88 text-emerald-700"
                        : "border-white/60 bg-white/65 text-slate-500"
                    }`}
                  >
                    {hasCuda ? "CUDA available" : "CPU only"}
                  </span>
                </div>
                <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
                  <div className="space-y-3">
                    <select
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                      className="flex h-11 w-full items-center justify-between rounded-[1rem] border border-white/60 bg-white/70 px-3 py-2 text-sm text-slate-800 outline-none disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <option value="auto">Auto (GPU if available)</option>
                      <option value="cpu">CPU</option>
                      {cudaGpus.length > 0 ? (
                        cudaGpus.map((g: any, idx: number) => {
                          const index = Number.isFinite(g?.index) ? g.index : idx;
                          const label = g?.name
                            ? `GPU (CUDA): ${g.name}`
                            : `GPU (CUDA) #${index}`;
                          return (
                            <option key={`cuda:${index}`} value={`cuda:${index}`}>
                              {label}
                            </option>
                          );
                        })
                      ) : (
                        <option value={primaryCudaDevice} disabled={!hasCuda}>
                          GPU (CUDA)
                        </option>
                      )}
                    </select>
                    <p className="text-xs text-slate-500">
                      Choose Auto for best defaults. Use GPU (CUDA) for faster
                      separation when available.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="stemsep-config-secondary rounded-[999px] border border-white/70 bg-white/72 px-4 text-[13px] font-normal tracking-[-0.2px] text-slate-700 hover:bg-white/88 hover:text-slate-900"
                      onClick={() => {
                        const isCuda = device.startsWith("cuda");
                        const normalizedDevice =
                          device === "cuda"
                            ? globalAdvancedSettings?.preferredCudaDevice ||
                              primaryCudaDevice
                            : device;

                        setAdvancedSettings({
                          device: normalizedDevice,
                          preferredCudaDevice: isCuda ? device : undefined,
                        });
                      }}
                    >
                      Set as default
                    </Button>
                    <p className="max-w-[220px] text-xs text-slate-500">
                      Saves your current CPU/GPU choice as the default for new
                      runs.
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="space-y-4 rounded-[1.5rem] border-white/55 bg-[rgba(255,255,255,0.5)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium text-slate-800">
                      Separation Strategy
                    </label>
                    <p className="mt-1 text-xs text-slate-500">
                      Pick one model for a direct run, or build an ensemble for
                      more control.
                    </p>
                  </div>
                  <span className="rounded-full border border-white/60 bg-white/65 px-3 py-1 text-[11px] text-slate-500">
                    {isEnsembleMode ? "Ensemble" : "Single model"}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={() => setIsEnsembleMode(false)}
                    className={`stemsep-config-option px-4 py-2 text-sm font-medium transition-all ${!isEnsembleMode ? "stemsep-config-option-active" : ""}`}
                  >
                    Single Model
                  </button>
                  <button
                    onClick={() => setIsEnsembleMode(true)}
                    className={`stemsep-config-option px-4 py-2 text-sm font-medium transition-all ${isEnsembleMode ? "stemsep-config-option-active" : ""}`}
                  >
                    <Layers className="mr-1 inline h-4 w-4" />
                    Ensemble
                  </button>
                </div>

                {isEnsembleMode ? (
                  <EnsembleBuilder
                    models={models}
                    config={ensembleConfig}
                    algorithm={ensembleAlgorithm as any}
                    phaseFixEnabled={phaseFixEnabled}
                    volumeCompEnabled={volumeCompEnabled}
                    onVolumeCompEnabledChange={setVolumeCompEnabled}
                    stemAlgorithms={stemAlgorithms}
                    phaseFixParams={{
                      ...phaseFixParams,
                      enabled: phaseFixEnabled,
                    }}
                    onChange={(
                      config,
                      alg,
                      newStemAlgos,
                      newPhaseFixParams,
                      newPhaseFixEnabled,
                    ) => {
                      setEnsembleConfig(config);
                      setEnsembleAlgorithm(alg as any);
                      if (newStemAlgos) setStemAlgorithms(newStemAlgos);
                      if (newPhaseFixParams)
                        setPhaseFixParams({
                          lowHz: newPhaseFixParams.lowHz,
                          highHz: newPhaseFixParams.highHz,
                          highFreqWeight: newPhaseFixParams.highFreqWeight,
                        });
                      if (newPhaseFixEnabled !== undefined)
                        setPhaseFixEnabled(newPhaseFixEnabled);
                    }}
                  />
                ) : (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-800">
                      Select Model
                    </label>
                    <ModelSelector
                      selectedModelId={selectedModelId}
                      onSelectModel={setSelectedModelId}
                      models={models}
                    />
                  </div>
                )}
              </Card>

              {isEnsembleMode && ensembleAlgorithm === "frequency_split" && (
                <Card className="space-y-3 rounded-[1.5rem] border-white/55 bg-[rgba(255,255,255,0.5)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <label className="text-sm font-medium text-slate-800">
                        Crossover Frequency
                      </label>
                      <p className="mt-1 text-xs text-slate-500">
                        Split lows and highs between the first two ensemble
                        models.
                      </p>
                    </div>
                    <span className="rounded border border-white/60 bg-white/70 px-2.5 py-1 text-sm font-mono text-slate-700">
                      {_splitFreq} Hz
                    </span>
                  </div>
                  <input
                    type="range"
                    min="200"
                    max="5000"
                    step="50"
                    value={_splitFreq}
                    onChange={(e) => _setSplitFreq(parseInt(e.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-900/10 accent-slate-800"
                  />
                  <p className="text-xs text-slate-500">
                    Frequencies below {_splitFreq}Hz use the first model (Low).
                    Above use the second (High).
                  </p>
                </Card>
              )}

              {!isEnsembleMode && (
                <CollapsibleSection
                  title="Enhancements (Optional)"
                  buttonLabel="Enhancements"
                  icon={<Zap className="h-4 w-4" />}
                  className="border-white/55 bg-[rgba(255,255,255,0.5)]"
                >
                  <div className="space-y-3">
                    <div className="text-xs leading-[1.5] text-slate-500">
                      Optional helpers for export polish and safer gain staging.
                    </div>
                    <div className="flex items-center gap-3 rounded-[1.1rem] border border-white/60 bg-white/62 p-3">
                      <input
                        type="checkbox"
                        id="configure-volume-comp-enabled"
                        checked={volumeCompEnabled}
                        onChange={(e) => setVolumeCompEnabled(e.target.checked)}
                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <div>
                        <label
                          htmlFor="configure-volume-comp-enabled"
                          className="cursor-pointer text-sm font-medium text-slate-800"
                        >
                          Enable VC
                        </label>
                        <p className="text-xs text-slate-500">
                          Applies the best default volume compensation profile
                          to reduce clipping risk when exporting stems.
                        </p>
                      </div>
                    </div>
                  </div>
                </CollapsibleSection>
              )}

              {/* Advanced Parameters */}
              <Card className="space-y-4 rounded-[1.5rem] border-white/55 bg-[rgba(255,255,255,0.5)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl">
                <div>
                  <h4 className="font-medium text-slate-800">Processing Parameters</h4>
                  <p className="mt-1 text-xs text-slate-500">
                    Tune chunking, augmentation and runtime behavior for this
                    run.
                  </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-xs text-slate-500">
                      Overlap
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="2"
                      max="50"
                      className="h-11 w-full rounded-xl border border-white/60 bg-white/70 p-3 text-slate-800"
                      value={advancedParams.overlap}
                      onChange={(e) => {
                        const next = parseInt(e.target.value);
                        if (!Number.isFinite(next)) return;
                        setAdvancedParamsDirty(true);
                        setAdvancedParams((p) => ({
                          ...p,
                          overlap: clamp(next, 2, 50),
                        }));
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-slate-500">
                      Shifts
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      max="20"
                      className="h-11 w-full rounded-xl border border-white/60 bg-white/70 p-3 text-slate-800"
                      value={advancedParams.shifts}
                      onChange={(e) => {
                        const next = parseInt(e.target.value, 10)
                        if (!Number.isFinite(next)) return
                        setAdvancedParamsDirty(true)
                        setAdvancedParams((p) => ({
                          ...p,
                          shifts: clamp(next, 1, 20),
                        }))
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-slate-500">
                      TTA
                    </label>
                    <button
                      type="button"
                      onClick={() => {
                        setAdvancedParamsDirty(true)
                        setAdvancedParams((p) => ({ ...p, tta: !p.tta }))
                      }}
                      className={`flex h-11 w-full items-center justify-between rounded-xl border px-3 transition-colors ${
                        advancedParams.tta
                          ? "bg-emerald-50/82 border-emerald-300/55"
                          : "bg-white/65 border-white/60"
                      }`}
                    >
                      <span className="text-sm text-slate-800">{advancedParams.tta ? "ON" : "OFF"}</span>
                      <span
                        className={`text-xs ${advancedParams.tta ? "text-emerald-700" : "text-slate-500"}`}
                      >
                        Test-time augmentation
                      </span>
                    </button>
                  </div>

                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <label className="text-xs text-slate-500">
                        Segment Size ({advancedParams.segmentSize === 0 ? "Auto" : advancedParams.segmentSize})
                      </label>
                      <div className="flex gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="stemsep-config-secondary rounded-[999px] border border-white/70 bg-white/72 px-4 text-[13px] font-normal tracking-[-0.2px] text-slate-700 hover:bg-white/88 hover:text-slate-900"
                          onClick={() => {
                            setAdvancedParamsDirty(true);
                            setAdvancedParams((p) => ({ ...p, segmentSize: 0 }));
                          }}
                        >
                          Auto
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="stemsep-config-secondary rounded-[999px] border border-white/70 bg-white/72 px-4 text-[13px] font-normal tracking-[-0.2px] text-slate-700 hover:bg-white/88 hover:text-slate-900"
                          onClick={() => {
                            const best = maxSafeSegmentSize > 0 ? maxSafeSegmentSize : null;
                            if (!best) {
                              toast.message(
                                "Could not detect GPU VRAM reliably. Leaving segment size unchanged.",
                              );
                              return;
                            }
                            setAdvancedParamsDirty(true);
                            setAdvancedParams((p) => ({ ...p, segmentSize: best }));
                            toast.success(
                              `Applied best segment size for this machine: ${best}`,
                            );
                          }}
                        >
                          Best for my machine
                        </Button>
                      </div>
                    </div>
                    <input
                      type="number"
                      className="h-11 w-full rounded-xl border border-white/60 bg-white/70 p-3 text-slate-800 placeholder:text-slate-400"
                      value={advancedParams.segmentSize === 0 ? "" : advancedParams.segmentSize}
                      placeholder="Auto"
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw.trim() === "") {
                          setAdvancedParamsDirty(true);
                          setAdvancedParams((p) => ({
                            ...p,
                            segmentSize: 0,
                          }));
                          return;
                        }

                        const next = parseInt(raw, 10);
                        if (!Number.isFinite(next)) return;
                        setAdvancedParamsDirty(true);
                        setAdvancedParams((p) => ({
                          ...p,
                          segmentSize: next,
                        }));
                      }}
                    />
                  </div>
                </div>
              </Card>

              </div>

              <div className="space-y-4 xl:sticky xl:top-0 xl:self-start">
                <SeparationPlanCard
                  report={backendPreflight}
                  loading={isPreflightLoading}
                  mode={mode}
                />

                <Card className="rounded-[1.6rem] border-white/60 bg-[rgba(255,255,255,0.48)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.12)] backdrop-blur-xl">
                  <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-400">
                    Run Summary
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                    <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                        Strategy
                      </div>
                      <div className="mt-1 text-[14px] text-slate-700">
                        {isEnsembleMode
                          ? `${validEnsembleModels.length || 0}-model ensemble`
                          : selectedModelId
                            ? "Single model"
                            : "Choose a strategy"}
                      </div>
                    </div>
                    <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                        Device
                      </div>
                      <div className="mt-1 text-[14px] text-slate-700">
                        {device === "auto"
                          ? hasCuda
                            ? "Auto / GPU preferred"
                            : "Auto / CPU"
                          : device.startsWith("cuda")
                            ? selectedGpuForMeter?.name || "CUDA"
                            : "CPU"}
                      </div>
                    </div>
                    <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                        Segment
                      </div>
                      <div className="mt-1 text-[14px] text-slate-700">
                        {advancedParams.segmentSize === 0
                          ? "Auto"
                          : advancedParams.segmentSize}
                      </div>
                    </div>
                    <div className="rounded-[1.1rem] border border-white/60 bg-white/62 px-3 py-3">
                      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
                        Overlap / Shifts
                      </div>
                      <div className="mt-1 text-[14px] text-slate-700">
                        {advancedParams.overlap} / {advancedParams.shifts}
                      </div>
                    </div>
                  </div>
                </Card>

                <Card className="rounded-[1.6rem] border-white/60 bg-[rgba(255,255,255,0.48)] p-5 text-slate-800 shadow-[0_20px_60px_rgba(141,150,179,0.12)] backdrop-blur-xl">
                  <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-400">
                    Machine Guidance
                  </div>
                  <div className="space-y-2 text-[13px] leading-[1.55] text-slate-500">
                    <p>
                      Advanced mode exposes raw controls for edge cases, heavier
                      quality passes and custom ensembles.
                    </p>
                    <p>
                      {hasCuda
                        ? `Current machine profile: ${availableVRAM || "GPU"} GB VRAM.`
                        : "Current machine profile: CPU only."}{" "}
                      {maxSafeSegmentSize > 0
                        ? `Recommended segment ceiling for this machine is ${maxSafeSegmentSize}.`
                        : "Use Auto segment size when VRAM telemetry is unavailable."}
                    </p>
                  </div>
                </Card>

                {availableVRAM > 0 && (
                  <VRAMUsageMeter
                    availableVRAM={availableVRAM}
                    estimatedVRAM={estimatedVRAM}
                  />
                )}

                <div className="space-y-2">
                  {isCPUOnly && <CPUOnlyWarning />}
                  {availableVRAM > 0 && estimatedVRAM > availableVRAM * 0.8 && (
                    <LowVRAMWarning
                      available={availableVRAM}
                      required={estimatedVRAM}
                    />
                  )}
                  {isEnsembleMode && <EnsembleTip />}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Missing Models Modal */}
      <MissingModelsDialog
        open={missingModels.length > 0 && missingDialogOpen}
        missing={separationPlan.missingModels}
        models={models as any}
        onClose={() => setMissingDialogOpen(false)}
        onQuickDownload={handleQuickDownload}
        onNavigateToModels={(modelId) => onNavigateToModels?.(modelId)}
      />

      {/* Footer */}
      <div className="border-t border-white/60 bg-white/16 px-7 py-5 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="rounded-full border border-white/40 bg-white/28 px-4 py-2 text-[12px] text-slate-500 shadow-[inset_0_1px_0_rgba(255,255,255,0.45)]">
            {mode === "simple"
              ? "Choose a preset and save. StemSep will handle the technical defaults behind the scenes."
              : "Tune models and machine limits, then save this configuration."}
          </div>
          <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={onBack}
            className="stemsep-config-secondary rounded-[999px] border border-white/75 bg-white/78 px-6 py-3 text-[15px] font-medium tracking-[-0.25px] text-slate-700 shadow-[0_16px_32px_rgba(141,150,179,0.12)] hover:-translate-y-[1px] hover:bg-white hover:text-slate-900"
          >
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={
              !sourceFileExists ||
              (!selectedPresetId && mode === "simple") ||
              (mode === "advanced" && !isEnsembleMode && !selectedModelId) ||
              (mode === "advanced" &&
                isEnsembleMode &&
                validEnsembleModels.length < 2)
            }
            className="stemsep-config-action relative overflow-hidden rounded-[999px] border border-white/80 bg-white/88 px-7 py-3 text-[16px] font-medium tracking-[-0.35px] text-[#23324c] shadow-[0_20px_40px_rgba(141,150,179,0.18)] transition-all duration-300 hover:-translate-y-[1px] hover:bg-white disabled:border-white/45 disabled:bg-white/45 disabled:text-slate-400 disabled:shadow-none"
          >
            <Zap className="w-4 h-4 mr-2" />
            Save Configuration
          </Button>
        </div>
        </div>
      </div>
    </div>
    </div>
    </PageShell>
  );
}
