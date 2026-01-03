import { useState, useEffect, useMemo } from "react";
import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";
import {
  ArrowLeft,
  Info,
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
import { Preset } from "../presets";
import type { SeparationConfig } from "../types/separation";
import { EnsembleBuilder } from "./EnsembleBuilder";
import { ModelSelector } from "./ModelSelector";
import { VRAMUsageMeter, estimateVRAMUsage } from "./ui/vram-meter";
import { CPUOnlyWarning, LowVRAMWarning, EnsembleTip } from "./ui/warning-tip";
import { bestVolumeCompensation } from "../utils/volumeCompensation";
import { SimplePresetPicker } from "./simple/SimplePresetPicker";
import { CollapsibleSection } from "./ui/collapsible-section";

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
  filePath: _filePath,
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
  const resumeDownload = useStore((state) => state.resumeDownload);
  const setDownloadError = useStore((state) => state.setDownloadError);
  const [mode, setMode] = useState<"simple" | "advanced">("simple");
  const [simpleSpeed, setSimpleSpeed] = useState<number>(70);
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

  const handleQuickDownload = async (modelId: string) => {
    const model = models.find((m) => m.id === modelId);

    try {
      if (model?.downloadPaused) {
        resumeDownload(modelId);
        if (window.electronAPI?.resumeDownload)
          await window.electronAPI.resumeDownload(modelId);
        return;
      }

      startDownload(modelId);
      if (window.electronAPI?.downloadModel)
        await window.electronAPI.downloadModel(modelId);
    } catch (e) {
      setDownloadError(modelId, e instanceof Error ? e.message : String(e));
    }
  };

  // Phase Correction State
  const [usePhaseCorrection, setUsePhaseCorrection] = useState(false);

  // Auto Post-Processing Pipeline State
  const [enableAutoPipeline, setEnableAutoPipeline] = useState(true);

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

  // GPU Info for VRAM estimation
  const [gpuInfo, setGpuInfo] = useState<any>(null);
  useEffect(() => {
    const loadGpuInfo = async () => {
      if (window.electronAPI?.getGpuDevices) {
        try {
          const info = await window.electronAPI.getGpuDevices();
          setGpuInfo(info);
        } catch (e) {
          console.error("Failed to load GPU info:", e);
        }
      }
    };
    loadGpuInfo();
  }, []);

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
    if (!usePhaseCorrection) return;
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
  }, [usePhaseCorrection, selectedModelId, models]);

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
    usePhaseCorrection,
    enableAutoPipeline,
    phaseParams,
  ]);

  // Canonical missing dependencies from the separation plan
  const missingModels = useMemo(() => {
    return separationPlan.missingModels.map((m) => m.modelId);
  }, [separationPlan.missingModels]);

  useEffect(() => {
    if (missingModels.length > 0) {
      setMissingDialogOpen(true);
    } else {
      setMissingDialogOpen(false);
    }
  }, [missingModels.length]);

  // Check if separation can proceed (no missing dependencies)
  const canStartSeparation = separationPlan.canProceed;

  const handleConfirm = () => {
    const preset = mode === "simple" ? presets.find((p) => p.id === selectedPresetId) : undefined;
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
      // Preview/playback is always WAV. Export format is chosen later in Results.
      outputFormat: "wav",
      volumeCompensation: volumeCompEnabled
        ? bestVolumeCompensation()
        : undefined,
      // For ensemble mode, default to vocals+instrumental (2 stems)
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
                const maxSafe = computeBestSegmentSizeFromVRAM(availableVRAM);
                const segmentSize = maxSafe > 0 ? maxSafe : 0;
                return segmentSize;
              })(),
              shifts: (() => {
                const speed = clamp(simpleSpeed, 0, 100);
                return speed >= 85 ? 1 : speed >= 65 ? 2 : speed >= 35 ? 3 : 4;
              })(),
              overlap: presetOverlap,
            },
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

    // Simple mode is locked to preset definitions.
    if (mode === "simple") {
      const preset = presets.find((p) => p.id === selectedPresetId);

      if (preset?.ensembleConfig) {
        config.ensembleConfig = preset.ensembleConfig as any;
        config.modelId = undefined;
      }

      // Add post-processing steps if enabled and preset has them
      if (
        enableAutoPipeline &&
        preset?.postProcessingSteps &&
        preset.postProcessingSteps.length > 0
      ) {
        config.postProcessingSteps = preset.postProcessingSteps;
      }
    }

    onConfirm(config);
  };

  const selectedPreset = presets.find((p) => p.id === selectedPresetId);

  const cudaGpus = useMemo(() => {
    const gpus = gpuInfo?.gpus;
    if (!Array.isArray(gpus)) return [];
    return gpus.filter((g: any) => g?.type === "cuda");
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
  const modelType = selectedPreset?.name || selectedModelId || "unknown";
  const estimatedVRAM = estimateVRAMUsage(
    modelType,
    advancedParams.segmentSize,
    advancedParams.overlap,
    advancedParams.tta,
  );
  const isCPUOnly = !hasCuda;
  const maxSafeSegmentSize = computeBestSegmentSizeFromVRAM(availableVRAM);

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
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b bg-card/50 px-6 py-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="flex-1">
            <h1 className="text-2xl font-bold">Configure Separation</h1>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Music className="h-4 w-4" />
              <span className="text-sm truncate max-w-md">{fileName}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Mode Tabs */}
          <div className="grid w-full grid-cols-2 bg-muted p-1 rounded-lg">
            <button
              onClick={() => setMode("simple")}
              className={`py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === "simple" ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
            >
              Simple Mode
            </button>
            <button
              onClick={() => setMode("advanced")}
              className={`py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === "advanced" ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
            >
              Advanced Mode
            </button>
          </div>

          {mode === "simple" ? (
            <div className="space-y-6">
              <SimplePresetPicker
                presets={presets}
                selectedPresetId={selectedPresetId}
                onSelectPreset={setSelectedPresetId}
                availability={availability}
              />

              <Card className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Speed</div>
                    <div className="text-xs text-muted-foreground">
                      {maxSafeSegmentSize > 0
                        ? `Machine limit: ${maxSafeSegmentSize}`
                        : "Machine limit: Auto"}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSimpleSpeed(70)}
                  >
                    Best for my machine
                  </Button>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={simpleSpeed}
                  onChange={(e) => setSimpleSpeed(parseInt(e.target.value, 10))}
                  className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Quality</span>
                  <span>Speed</span>
                </div>
              </Card>

              {/* Selected preset info */}
              {selectedPreset && (
                <Card className="p-4 space-y-3">
                  {/* Header with quality badge */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm text-muted-foreground">
                        {selectedPreset.description}
                      </p>
                    </div>
                    {"qualityLevel" in selectedPreset && (
                      <span
                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ml-3 shrink-0 ${
                          selectedPreset.qualityLevel === "ultra"
                            ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                            : selectedPreset.qualityLevel === "quality"
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              : selectedPreset.qualityLevel === "balanced"
                                ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                : "bg-orange-500/20 text-orange-400 border border-orange-500/30"
                        }`}
                      >
                        {selectedPreset.qualityLevel === "ultra"
                          ? "⭐ Ultra"
                          : selectedPreset.qualityLevel === "quality"
                            ? "✓ Quality"
                            : selectedPreset.qualityLevel === "balanced"
                              ? "◎ Balanced"
                              : "⚡ Fast"}
                      </span>
                    )}
                  </div>

                  {/* VRAM and Stems Row */}
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                      {"estimatedVram" in selectedPreset && (
                        <div className="flex items-center gap-1.5 text-sm">
                          <span className="text-muted-foreground">VRAM:</span>
                          <span
                            className={`font-medium ${
                              selectedPreset.estimatedVram >= 12
                                ? "text-orange-400"
                                : selectedPreset.estimatedVram >= 8
                                  ? "text-yellow-400"
                                  : "text-emerald-400"
                            }`}
                          >
                            {selectedPreset.estimatedVram}GB
                          </span>
                        </div>
                      )}
                      <span className="text-sm font-medium text-muted-foreground">
                        Stems:
                      </span>
                    </div>
                    <div className="flex gap-1 flex-wrap">
                      {selectedPreset.stems.map((stem) => (
                        <span
                          key={stem}
                          className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-transparent bg-secondary text-secondary-foreground"
                        >
                          {stem}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Tags */}
                  {"tags" in selectedPreset &&
                    selectedPreset.tags &&
                    selectedPreset.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {selectedPreset.tags.map((tag: string) => (
                          <span
                            key={tag}
                            className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    )}

                  {/* Ensemble Info */}
                  {selectedPreset.ensembleConfig && (
                    <div className="pt-2 border-t">
                      <div className="flex items-center gap-2">
                        <Layers className="w-4 h-4 text-primary" />
                        <span className="text-sm font-medium">
                          Ensemble:{" "}
                          {selectedPreset.ensembleConfig.models.length} models
                        </span>
                        <span className="text-xs text-muted-foreground">
                          (
                          {selectedPreset.ensembleConfig.algorithm ===
                          "max_spec"
                            ? "Max Spec"
                            : selectedPreset.ensembleConfig.algorithm ===
                                "min_spec"
                              ? "Min Spec"
                              : "Average"}
                          )
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Workflow (Recipe) Details */}
                  {(selectedPreset as any).isRecipe &&
                    (selectedPreset as any).recipe && (
                      <div className="pt-2 border-t space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4 text-primary" />
                            <span className="text-sm font-medium">
                              Multi-step preset
                            </span>
                            <span className="text-xs text-muted-foreground">
                              ({(selectedPreset as any).recipe.type})
                            </span>
                          </div>
                        </div>

                        {(selectedPreset as any).recipe.warning && (
                          <div className="flex items-start gap-2 text-xs text-orange-400 bg-orange-500/10 rounded p-2 border border-orange-500/20">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                            <span>
                              {(selectedPreset as any).recipe.warning}
                            </span>
                          </div>
                        )}

                        <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                          {(selectedPreset as any).recipe.source && (
                            <div className="bg-muted/50 rounded p-2">
                              <span className="font-medium text-foreground/80">
                                Source:
                              </span>{" "}
                              {(selectedPreset as any).recipe.source}
                            </div>
                          )}
                          {(selectedPreset as any).recipe.defaults?.overlap !=
                            null && (
                            <div className="bg-muted/50 rounded p-2">
                              <span className="font-medium text-foreground/80">
                                Default overlap:
                              </span>{" "}
                              {(selectedPreset as any).recipe.defaults.overlap}
                            </div>
                          )}
                        </div>

                        {(selectedPreset as any).recipe.requiredModels?.length >
                          0 && (
                          <div className="space-y-1">
                            <div className="text-xs text-muted-foreground">
                              Required models:
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {(
                                selectedPreset as any
                              ).recipe.requiredModels.map((id: string) => (
                                <span
                                  key={id}
                                  className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] ${
                                    missingModels.includes(id)
                                      ? "border-destructive/40 bg-destructive/10 text-destructive"
                                      : "border-transparent bg-secondary text-secondary-foreground"
                                  }`}
                                >
                                  {id}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {(selectedPreset as any).recipe.steps?.length > 0 && (
                          <CollapsibleSection
                            title={`What happens? (${(selectedPreset as any).recipe.steps.length} steps)`}
                            defaultOpen={false}
                          >
                            <div className="space-y-2">
                              <div className="text-xs text-muted-foreground">
                                Runs these steps automatically in one job.
                              </div>
                              <div className="space-y-1">
                                {(selectedPreset as any).recipe.steps.map(
                                  (step: any, idx: number) => {
                                    const name =
                                      step.step_name ||
                                      step.name ||
                                      step.action ||
                                      `step_${idx + 1}`;
                                    const optional = !!step.optional;
                                    return (
                                      <div
                                        key={idx}
                                        className="flex items-start justify-between gap-3 bg-muted/30 rounded px-2 py-1"
                                      >
                                        <div className="min-w-0">
                                          <div className="text-xs font-medium text-foreground/90 truncate">
                                            {idx + 1}. {name}
                                            {optional ? " (optional)" : ""}
                                          </div>
                                          {step.note && (
                                            <div className="text-[11px] text-muted-foreground">
                                              {step.note}
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  },
                                )}
                              </div>
                            </div>
                          </CollapsibleSection>
                        )}
                      </div>
                    )}

                  {/* Post-Processing Pipeline Toggle */}
                  {"postProcessingSteps" in selectedPreset &&
                  selectedPreset.postProcessingSteps &&
                  selectedPreset.postProcessingSteps.length > 0 ? (
                    <div className="pt-2 border-t">
                      <button
                        onClick={() =>
                          setEnableAutoPipeline(!enableAutoPipeline)
                        }
                        className={`w-full flex items-start gap-3 text-left p-3 rounded-lg transition-all ${
                          enableAutoPipeline
                            ? "bg-emerald-500/10 border border-emerald-500/30"
                            : "bg-muted/50 border border-transparent hover:border-muted-foreground/20"
                        }`}
                      >
                        <div
                          className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${
                            enableAutoPipeline
                              ? "border-emerald-500 bg-emerald-500"
                              : "border-muted-foreground/40"
                          }`}
                        >
                          {enableAutoPipeline && (
                            <svg
                              className="w-3 h-3 text-white"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={3}
                                d="M5 13l4 4L19 7"
                              />
                            </svg>
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span
                              className={`text-sm font-medium ${enableAutoPipeline ? "text-emerald-400" : "text-muted-foreground"}`}
                            >
                              Auto Post-Processing
                            </span>
                            {enableAutoPipeline && (
                              <span className="text-xs bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">
                                ON
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {selectedPreset.postProcessingSteps.map(
                              (step: any, i: number) => (
                                <span key={i}>
                                  {i > 0 && " → "}
                                  {step.description}
                                </span>
                              ),
                            )}
                          </div>
                        </div>
                      </button>
                    </div>
                  ) : (
                    // Show regular pipeline note if no postProcessingSteps
                    "pipelineNote" in selectedPreset &&
                    selectedPreset.pipelineNote && (
                      <div className="pt-2 border-t">
                        <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 rounded p-2">
                          <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                          <span>{selectedPreset.pipelineNote}</span>
                        </div>
                      </div>
                    )
                  )}

                  {/* Missing models are shown as a modal dialog below */}
                </Card>
              )}

              <CollapsibleSection
                title="Output & Device"
                icon={<Settings2 className="h-4 w-4" />}
                defaultOpen={false}
              >
                {/* Device Selection (GPU/CPU) */}
                <div className="space-y-2">
                  <label className="text-sm font-medium block">
                    Processing Device
                  </label>
                  <div className="flex gap-2 flex-wrap">
                    <Button
                      variant={device === "auto" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setDevice("auto")}
                    >
                      Auto
                    </Button>
                    {gpuInfo?.has_cuda && (
                      <Button
                        variant={
                          device.startsWith("cuda") ? "default" : "outline"
                        }
                        size="sm"
                        onClick={() => setDevice("cuda")}
                      >
                        GPU (CUDA)
                      </Button>
                    )}
                    <Button
                      variant={device === "cpu" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setDevice("cpu")}
                    >
                      CPU {!gpuInfo?.has_cuda && "(No GPU detected)"}
                    </Button>
                  </div>
                  {device === "cpu" && gpuInfo?.has_cuda && (
                    <p className="text-xs text-muted-foreground">
                      CPU processing is slower but uses less memory
                    </p>
                  )}
                </div>

                <div className="text-xs text-muted-foreground">
                  Preview stems are saved as WAV for lossless playback. Use
                  Export in Results to create MP3/FLAC.
                </div>
              </CollapsibleSection>

              <CollapsibleSection
                title="Enhancements (Optional)"
                icon={<Zap className="h-4 w-4" />}
                defaultOpen={false}
              >
                {/* Phase Correction Toggle */}
                {selectedPreset && !selectedPreset.ensembleConfig && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium block">
                      Phase Correction
                    </label>
                    <button
                      onClick={() => setUsePhaseCorrection(!usePhaseCorrection)}
                      className={`w-full flex items-start gap-3 text-left p-3 rounded-lg transition-all ${
                        usePhaseCorrection
                          ? "bg-blue-500/10 border border-blue-500/30"
                          : "bg-muted/50 border border-transparent hover:border-muted-foreground/20"
                      }`}
                    >
                      <div
                        className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${
                          usePhaseCorrection
                            ? "border-blue-500 bg-blue-500"
                            : "border-muted-foreground/40"
                        }`}
                      >
                        {usePhaseCorrection && (
                          <svg
                            className="w-3 h-3 text-white"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                              clipRule="evenodd"
                            />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1">
                        <div
                          className={`font-medium ${usePhaseCorrection ? "text-blue-500" : "text-foreground"}`}
                        >
                          Phase Swap Enhancement
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Uses a reference model to improve phase accuracy.
                          Reduces artifacts in high frequencies.
                        </p>
                      </div>
                    </button>
                  </div>
                )}

                {/* Volume Compensation */}
                <div className="space-y-2">
                  <label className="text-sm font-medium block">
                    Volume Compensation (VC)
                  </label>
                  <button
                    onClick={() => setVolumeCompEnabled(!volumeCompEnabled)}
                    className={`w-full flex items-start gap-3 text-left p-3 rounded-lg transition-all ${
                      volumeCompEnabled
                        ? "bg-blue-500/10 border border-blue-500/30"
                        : "bg-muted/50 border border-transparent hover:border-muted-foreground/20"
                    }`}
                  >
                    <div
                      className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${
                        volumeCompEnabled
                          ? "border-blue-500 bg-blue-500"
                          : "border-muted-foreground/40"
                      }`}
                    >
                      {volumeCompEnabled && (
                        <svg
                          className="w-3 h-3 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1">
                      <div
                        className={`font-medium ${volumeCompEnabled ? "text-blue-500" : "text-foreground"}`}
                      >
                        Enable VC
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Adds headroom when combining multiple models (reduces
                        clipping risk). When enabled, uses best defaults.
                      </p>
                    </div>
                  </button>
                </div>
              </CollapsibleSection>

              {/* Warnings */}
              <div className="space-y-2">
                {isCPUOnly && <CPUOnlyWarning />}
                {availableVRAM > 0 && estimatedVRAM > availableVRAM * 0.8 && (
                  <LowVRAMWarning
                    available={availableVRAM}
                    required={estimatedVRAM}
                  />
                )}
                {selectedPreset?.ensembleConfig && <EnsembleTip />}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 text-sm text-yellow-600 dark:text-yellow-400">
                <div className="flex items-center gap-2 font-semibold mb-1">
                  <Settings2 className="w-4 h-4" />
                  Advanced Configuration
                </div>
                Manual control over model parameters. Use with caution.
              </div>

              {/* Processing Device */}
              <Card className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">
                    Processing Device
                  </label>
                  <span
                    className={`text-xs ${hasCuda ? "text-green-600 dark:text-green-400" : "text-muted-foreground"}`}
                  >
                    {hasCuda ? "CUDA available" : "CPU only"}
                  </span>
                </div>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
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
                <p className="text-xs text-muted-foreground">
                  Choose Auto for best defaults. Use GPU (CUDA) for faster
                  separation when available.
                </p>

                <div className="pt-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      // Persist the current selection as the global default.
                      // Also persist a preferred CUDA index when relevant.
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
                  <p className="text-xs text-muted-foreground mt-1">
                    Saves your current CPU/GPU choice as the default for new
                    runs.
                  </p>
                </div>
              </Card>

              {/* Single vs Ensemble Toggle */}
              <div className="flex items-center space-x-4 mb-4">
                <button
                  onClick={() => setIsEnsembleMode(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${!isEnsembleMode ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"}`}
                >
                  Single Model
                </button>
                <button
                  onClick={() => setIsEnsembleMode(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${isEnsembleMode ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"}`}
                >
                  <Layers className="w-4 h-4 inline mr-1" />
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
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Select Model</label>
                    <ModelSelector
                      selectedModelId={selectedModelId}
                      onSelectModel={setSelectedModelId}
                      models={models}
                    />
                  </div>
                </div>
              )}

              {isEnsembleMode && ensembleAlgorithm === "frequency_split" && (
                <Card className="p-4 space-y-3">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">
                      Crossover Frequency
                    </label>
                    <span className="text-sm font-mono bg-background px-2 rounded border">
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
                    className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                  <p className="text-xs text-muted-foreground">
                    Frequencies below {_splitFreq}Hz use the first model (Low).
                    Above use the second (High).
                  </p>
                </Card>
              )}

              {/* Advanced Parameters */}
              <Card className="p-4 space-y-4">
                <h4 className="font-medium">Processing Parameters</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-muted-foreground">
                      Overlap
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      max="50"
                      className="w-full p-2 rounded border bg-background"
                      value={advancedParams.overlap}
                      onChange={(e) => {
                        const next = parseInt(e.target.value);
                        if (!Number.isFinite(next)) return;
                        setAdvancedParamsDirty(true);
                        setAdvancedParams((p) => ({ ...p, overlap: next }));
                      }}
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-muted-foreground">
                        Segment Size ({advancedParams.segmentSize === 0 ? "Auto" : advancedParams.segmentSize})
                      </label>
                      <div className="flex gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
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
                          onClick={() => {
                            const best = maxSafeSegmentSize > 0 ? maxSafeSegmentSize : 0;
                            setAdvancedParamsDirty(true);
                            setAdvancedParams((p) => ({ ...p, segmentSize: best }));
                          }}
                        >
                          Best for my machine
                        </Button>
                      </div>
                    </div>
                    <input
                      type="number"
                      className="w-full p-2 rounded border bg-background"
                      value={advancedParams.segmentSize}
                      onChange={(e) => {
                        const next = parseInt(e.target.value, 10);
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

              {/* VRAM Meter */}
              {availableVRAM > 0 && (
                <VRAMUsageMeter
                  availableVRAM={availableVRAM}
                  estimatedVRAM={estimatedVRAM}
                />
              )}
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
      <div className="border-t bg-card/50 px-6 py-4">
        <div className="max-w-2xl mx-auto flex justify-between">
          <Button variant="outline" onClick={onBack}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={
              !canStartSeparation ||
              (!selectedPresetId && mode === "simple") ||
              (mode === "advanced" && !isEnsembleMode && !selectedModelId) ||
              (mode === "advanced" &&
                isEnsembleMode &&
                ensembleConfig.length === 0)
            }
          >
            <Zap className="w-4 h-4 mr-2" />
            Start Separation
          </Button>
        </div>
      </div>
    </div>
  );
}
