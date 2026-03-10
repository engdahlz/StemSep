import { CSSProperties, useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  Disc3,
  Headphones,
  Mic2,
  Music2,
  Radio,
  Upload,
} from "lucide-react";

import { PresetBrowser } from "./PresetBrowser";
import SettingsDialog from "./SettingsDialog";
import HistoryDialog from "./HistoryDialog";
import { useStore } from "../stores/useStore";
import { SeparationConfig } from "./ConfigurePage";
import { QueueItem } from "../types/store";
import { toast } from "sonner";
import { ModelDetails } from "./ModelDetails";
import { ALL_PRESETS, Preset } from "../presets";
import { recipesToPresets } from "@/lib/recipePresets";
import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";
import {
  recommendModelChain,
  recommendWorkflowPreset,
} from "@/lib/policy/recommendationPolicy";
import MissingModelsDialog from "./dialogs/MissingModelsDialog";
import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";
import { modelRequiresFnoRuntime } from "../lib/systemRuntime/modelRuntime";
import {
  buildSeparationBackendPayload,
  executeSeparation,
  executeSeparationPreflight,
} from "@/lib/separation/backendPayload";
import type { SeparationProgressEvent } from "@/types/media";
import { queueUpdatesFromProgressEvent } from "@/lib/separation/progressEvent";

// --- Constants & Types ---

interface PendingSeparationConfig {
  config: SeparationConfig;
  file: { name: string; path: string; presetId?: string };
}

interface SeparatePageProps {
  onNavigateToModels?: () => void;
  onNavigateToConfigure?: (
    fileName: string,
    filePath: string,
    presetId?: string,
  ) => void;
  onNavigateToResults?: () => void;
  pendingSeparationConfig?: PendingSeparationConfig | null;
  onClearPendingConfig?: () => void;
}

export default function SeparatePage({
  onNavigateToModels,
  onNavigateToConfigure,
  onNavigateToResults,
  pendingSeparationConfig,
  onClearPendingConfig,
}: SeparatePageProps) {
  // --- State ---
  const [isDragging, setIsDragging] = useState(false);

  // Missing models dialog (for single-file runs started from this page)
  const [missingDialogOpen, setMissingDialogOpen] = useState(false);
  const [missingDialogItems, setMissingDialogItems] = useState<
    {
      modelId: string;
      reason: "not_installed";
    }[]
  >([]);

  const [selectedPreset, setSelectedPreset] =
    useState<string>("workflow_phase_fix_instrumental");
  const [presetUserLocked, setPresetUserLocked] = useState(false);
  const [showPresetBrowser, setShowPresetBrowser] = useState(false);
  const [favoritePresetIds, setFavoritePresetIds] = useState<string[]>(() => {
    const saved = localStorage.getItem("favoritePresets");
    return saved
      ? JSON.parse(saved)
      : ["best_instrumental", "best_vocals", "best_karaoke"];
  });
  const [presetAvailability, setPresetAvailability] = useState<
    Record<
      string,
      {
        available: boolean;
        model_id: string;
        model_name: string | null;
        model_exists: boolean;
        installed: boolean;
        file_size: number;
      }
    >
  >({});

  // Dialogs State
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Model Details Overlay State
  const [detailsModelId, setDetailsModelId] = useState<string | null>(null);

  const models = useStore((state) => state.models);
  const recipes = useStore((state) => state.recipes);
  const setRecipes = useStore((state) => state.setRecipes);
  const startDownload = useStore((state) => state.startDownload);
  const setDownloadError = useStore((state) => state.setDownloadError);
  const setModelInstalled = useStore((state) => state.setModelInstalled);
  const { info: runtimeInfo } = useSystemRuntimeInfo();

  // GPU info for VRAM filtering
  const gpuVRAM = useMemo(() => {
    const gpu = runtimeInfo?.gpu;
    return gpu?.recommended_profile?.vram_gb || gpu?.gpus?.[0]?.memory_gb || 0;
  }, [runtimeInfo]);

  // Wait for bridge to be ready before fetching recipes
  // This eliminates the race condition that caused timeout on first attempt
  useEffect(() => {
    let cleanup: (() => void) | null = null;

    // Set up listener for bridge ready event
    if (window.electronAPI?.onBridgeReady) {
      cleanup = window.electronAPI.onBridgeReady((data) => {
        console.log(
          `Bridge ready! Models: ${data.modelsCount}, Recipes: ${data.recipesCount}`,
        );
        // Now safe to fetch recipes
        if (window.electronAPI?.getRecipes) {
          window.electronAPI.getRecipes().then(setRecipes).catch(console.error);
        }
      });
    }

    // Fallback: also try immediately in case bridge is already ready
    // (e.g., after hot-reload or if event was missed)
    const fallbackTimer = setTimeout(() => {
      if (window.electronAPI?.getRecipes) {
        window.electronAPI.getRecipes().then(setRecipes).catch(console.error);
      }
    }, 5000); // 5s fallback

    return () => {
      cleanup?.();
      clearTimeout(fallbackTimer);
    };
  }, [setRecipes]);

  const combinedPresets: Preset[] = useMemo(() => {
    // Safety check for recipes
    if (!Array.isArray(recipes)) {
      console.warn("Recipes is not an array:", recipes);
      return ALL_PRESETS;
    }

    return [...ALL_PRESETS, ...recipesToPresets(recipes)];
  }, [recipes]);

  useEffect(() => {
    if (presetUserLocked) return;
    if (!Array.isArray(models) || models.length === 0) return;
    if (!Array.isArray(combinedPresets) || combinedPresets.length === 0) return;

    const workflowRec = recommendWorkflowPreset(
      "instrumental",
      combinedPresets,
      models as any,
      {
        device: gpuVRAM > 0 ? "cuda:0" : "cpu",
        vramGb: gpuVRAM,
      },
      {
        fnoSupported:
          runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok !== false,
      },
    );

    if (!workflowRec.blocked && workflowRec.recommendedPresetId) {
      if (workflowRec.recommendedPresetId !== selectedPreset) {
        setSelectedPreset(workflowRec.recommendedPresetId);
      }
      return;
    }

    const rec = recommendModelChain("instrumental", models as any, {
      device: gpuVRAM > 0 ? "cuda:0" : "cpu",
      vramGb: gpuVRAM,
    });
    if (rec.blocked || rec.chain.length === 0) return;

    const installed = new Set(
      models.filter((m) => m.installed).map((m) => m.id),
    );

    const ensemblePick = combinedPresets.find((p) => {
      if (!p.ensembleConfig?.models?.length) return false;
      const mids = p.ensembleConfig.models.map((m) => m.model_id);
      if (!mids.every((mid) => rec.chain.includes(mid))) return false;
      return mids.every((mid) => installed.has(mid));
    });

    const singlePick =
      combinedPresets.find(
        (p) => p.modelId && p.modelId === rec.chain[0] && installed.has(p.modelId),
      ) ||
      combinedPresets.find((p) => p.id === "best_instrumental");

    const pick = ensemblePick || singlePick;
    if (pick && pick.id !== selectedPreset) {
      setSelectedPreset(pick.id);
    }
  }, [combinedPresets, gpuVRAM, models, presetUserLocked, runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok, selectedPreset]);

  const activeDetailsModel = useMemo(() => {
    return models.find((m) => m.id === detailsModelId) || null;
  }, [models, detailsModelId]);

  // Download handlers for the overlay
  const handleDownloadModel = async (modelId: string) => {
    startDownload(modelId);
    try {
      if (window.electronAPI?.downloadModel)
        await window.electronAPI.downloadModel(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handleRemoveModel = async (modelId: string) => {
    if (!confirm("Remove this model?")) return;
    if (window.electronAPI?.removeModel)
      await window.electronAPI.removeModel(modelId);
    setModelInstalled(modelId, false);
  };

  // Actions
  const addToQueueStore = useStore((state) => state.addToQueue);
  const removeFromQueue = useStore((state) => state.removeFromQueue);
  const updateQueueItem = useStore((state) => state.updateQueueItem);
  const setSeparationStatus = useStore((state) => state.startSeparation);
  const setSeparationProgress = useStore(
    (state) => state.setSeparationProgress,
  );
  const completeSeparation = useStore((state) => state.completeSeparation);
  const addLog = useStore((state) => state.addLog);
  const addToHistory = useStore((state) => state.addToHistory);
  const outputDirectory = useStore((state) => state.settings.defaultOutputDir);
  const cancelSeparation = useStore((state) => state.cancelSeparation);
  const phaseParams = useStore((state) => state.settings.phaseParams);

  // Global State
  const queue = useStore((state) => state.separation.queue);
  const isProcessing = useStore((state) => state.separation.isProcessing);
  const progressMessage = useStore((state) => state.separation.message);
  const separationStartTime = useStore((state) => state.separation.startTime);

  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!isProcessing) return;
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [isProcessing]);

  const processingQueueItem = useMemo(() => {
    return queue.find((q) => q.status === "processing");
  }, [queue]);

  const formatDuration = useCallback((totalSeconds: number) => {
    const s = Math.max(0, Math.floor(totalSeconds));
    const mm = Math.floor(s / 60);
    const ss = s % 60;
    return `${mm.toString().padStart(2, "0")}:${ss.toString().padStart(2, "0")}`;
  }, []);

  const processingTimingText = useMemo(() => {
    if (!isProcessing) return "";

    const startedAt = processingQueueItem?.startTime ?? separationStartTime;
    if (!startedAt) return "";

    const lastUpdateAt = processingQueueItem?.lastProgressTime ?? startedAt;
    const elapsedSec = (nowMs - startedAt) / 1000;
    const sinceUpdateSec = (nowMs - lastUpdateAt) / 1000;

    const parts: string[] = [`Elapsed ${formatDuration(elapsedSec)}`];
    if (sinceUpdateSec >= 15) {
      parts.push(`No updates ${formatDuration(sinceUpdateSec)}`);
    }
    return parts.join(" • ");
  }, [
    formatDuration,
    isProcessing,
    nowMs,
    processingQueueItem,
    separationStartTime,
  ]);

  // Prepared run created in ConfigurePage. This should stay idle until the user
  // explicitly presses the primary CTA on the home screen.
  const [preparedSeparation, setPreparedSeparation] =
    useState<PendingSeparationConfig | null>(null);

  // Handle pending separation config from ConfigurePage.
  // We store it locally instead of auto-starting so the entry flow becomes:
  // add audio -> configure -> explicit start.
  const processingPendingConfigRef = useRef(false);
  const heroIcons = useMemo(() => {
    const orbitIcons = [Music2, Headphones, Mic2, Radio, Disc3, Upload];
    const circles = [80, 120, 160];
    const iconsPerCircle = [6, 8, 10];

    return circles.flatMap((radius, circleIndex) => {
      const count = iconsPerCircle[circleIndex];
      return Array.from({ length: count }).flatMap((_, iconIndex) => {
        const angle = (iconIndex / count) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        const rotation = Math.atan2(y, x) * (180 / Math.PI);
        const Icon = orbitIcons[(circleIndex + iconIndex) % orbitIcons.length];

        const buildStyle = (
          opacity: number,
          scale: number,
          delayOffset: number,
        ): CSSProperties => ({
          "--stemsep-icon-x": `${x}px`,
          "--stemsep-icon-y": `${y}px`,
          "--stemsep-icon-rotation": `${rotation}deg`,
          "--stemsep-icon-opacity": opacity,
          "--stemsep-icon-scale": scale,
          animationDelay: `${circleIndex * 0.01 + iconIndex * 0.002 + delayOffset}s`,
        }) as CSSProperties;

        return [
          {
            key: `orbit-${circleIndex}-${iconIndex}`,
            Icon,
            style: buildStyle(1, 1, 0),
          },
          {
            key: `orbit-${circleIndex}-${iconIndex}-trail`,
            Icon,
            style: buildStyle(0.5, 0.7, 0.008),
          },
        ];
      });
    });
  }, []);

  useEffect(() => {
    // Guard: Prevent double execution
    if (processingPendingConfigRef.current) return;

    if (pendingSeparationConfig && onClearPendingConfig) {
      processingPendingConfigRef.current = true;

      const { config, file } = pendingSeparationConfig;

      // Check if file is already in queue
      const existingItem = queue.find((q) => q.file === file.path);

      // If file is completed, remove it first (user must re-upload to reprocess)
      if (existingItem && existingItem.status === "completed") {
        toast.info("Previous separation completed. Re-processing...");
        removeFromQueue(existingItem.id);
      }

      // Only add if not currently processing
      if (!existingItem || existingItem.status === "completed") {
        // Add file to queue with proper QueueItem format
        const queueItem: QueueItem = {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          file: file.path,
          status: "pending",
        };
        addToQueueStore([queueItem]);
      } else if (existingItem.status === "processing") {
        toast.warning("File is already being processed");
        onClearPendingConfig();
        processingPendingConfigRef.current = false;
        return;
      }

      // Store the prepared run locally for the primary CTA.
      setPreparedSeparation({ config, file });

      // Clear the app-level handoff after we have copied it locally.
      onClearPendingConfig();
      toast.success(
        "Configuration ready. Press Start Separation when you are ready.",
      );
      processingPendingConfigRef.current = false;
    }
  }, [pendingSeparationConfig, onClearPendingConfig, addToQueueStore]);
  // NOTE: Removed 'queue' from dependencies to prevent double-execution
  // The queue check uses a snapshot, which is acceptable for this use case

  // --- Effects ---

  useEffect(() => {
    const removeStructuredListener = window.electronAPI?.onSeparationEvent?.(
      (data: SeparationProgressEvent) => {
        if (typeof data.progress === "number" || data.message) {
          setSeparationProgress(
            typeof data.progress === "number"
              ? data.progress
              : useStore.getState().separation.progress,
            data.message || useStore.getState().separation.message || "",
          );
        }

        const byBackendId = data.jobId
          ? queue.find((q) => q.backendJobId === data.jobId)
          : undefined;
        const processingItem =
          byBackendId || queue.find((q) => q.status === "processing");
        if (!processingItem) return;

        const updates = queueUpdatesFromProgressEvent(processingItem, data);
        if (data.jobId && !processingItem.backendJobId) {
          updates.backendJobId = data.jobId;
        }
        if (data.kind === "error" && data.message) {
          updates.error = data.message;
        }
        updateQueueItem(processingItem.id, updates);
      },
    );

    const removeCompleteListener = window.electronAPI?.onSeparationComplete(
      (data) => {
        // Only handle global state completion here. History is handled in the main loop.
        completeSeparation(data.outputFiles);
        addLog(`Separation process finished.`);
      },
    );

    const removeErrorListener = window.electronAPI?.onSeparationError(
      (data) => {
        // Only log here. The main loop handles the queue item status.
        addLog(`Separation error: ${data.error}`);
      },
    );

    return () => {
      removeStructuredListener?.();
      removeCompleteListener?.();
      removeErrorListener?.();
    };
  }, [
    setSeparationProgress,
    completeSeparation,
    addLog,
    queue,
    updateQueueItem,
  ]);

  // Check model availability
  useEffect(() => {
    const availability: Record<string, any> = {};

    // Populate availability for all known models
    models.forEach((model) => {
      availability[model.id] = {
        available: model.installed || false,
        model_id: model.id,
        model_name: model.name,
        model_exists: true,
        installed: model.installed || false,
        file_size: model.file_size || 0,
      };
    });
    setPresetAvailability(availability);
  }, [models]);

  // Save favorites
  useEffect(() => {
    localStorage.setItem("favoritePresets", JSON.stringify(favoritePresetIds));
  }, [favoritePresetIds]);

  // --- Handlers ---

  const toggleFavorite = useCallback((presetId: string) => {
    setFavoritePresetIds((prev) =>
      prev.includes(presetId)
        ? prev.filter((id) => id !== presetId)
        : [...prev, presetId],
    );
  }, []);

  const handleSelectPreset = useCallback((presetId: string) => {
    setPresetUserLocked(true);
    setSelectedPreset(presetId);
  }, []);

  const handleFileSelect = async () => {
    if (!window.electronAPI) {
      toast.error("Electron API not available");
      return;
    }

    try {
      const filePaths = await window.electronAPI.openAudioFileDialog();

      if (filePaths && filePaths.length > 0) {
        await addToQueue(filePaths);
        // Navigate to configure page with first file
        const fileName = filePaths[0].split(/[\\/]/).pop() || "Unknown";
        onNavigateToConfigure?.(fileName, filePaths[0], selectedPreset);
      }
    } catch (error) {
      console.error("Error in handleFileSelect:", error);
      toast.error("Failed to select files");
    }
  };

  const leadQueueItem = useMemo(() => {
    return (
      queue.find(
        (item) =>
          item.status === "pending" ||
          item.status === "queued" ||
          item.status === "failed",
      ) ||
      queue.find((item) => item.status !== "completed") ||
      queue[0] ||
      null
    );
  }, [queue]);

  const hasPreparedConfig =
    !!preparedSeparation &&
    !!preparedSeparation.file?.path &&
    queue.some((item) => item.file === preparedSeparation.file.path);

  const handlePrimaryAction = useCallback(async () => {
    if (isProcessing) return;

    if (queue.length === 0) {
      await handleFileSelect();
      return;
    }

    if (hasPreparedConfig && preparedSeparation) {
      await startSeparationWithFilePath(
        preparedSeparation.config,
        preparedSeparation.file.path,
      );
      setPreparedSeparation(null);
      return;
    }

    const item = leadQueueItem;
    if (!item) return;
    const fileName = item.file.split(/[\\/]/).pop() || "Unknown";
    onNavigateToConfigure?.(fileName, item.file, selectedPreset);
  }, [
    hasPreparedConfig,
    isProcessing,
    leadQueueItem,
    onNavigateToConfigure,
    preparedSeparation,
    queue.length,
    selectedPreset,
  ]);

  const primaryActionLabel = isProcessing
    ? "Processing"
    : queue.length === 0
      ? "Add Audio"
      : hasPreparedConfig
        ? "Start Separation"
        : "Configure";

  const queueStatusText = isDragging
    ? "Drop files to start a new separation"
    : queue.length > 0
      ? isProcessing
        ? `Processing ${queue.filter((item) => item.status === "completed").length}/${queue.length}${progressMessage ? ` · ${progressMessage}` : ""}${processingTimingText ? ` · ${processingTimingText}` : ""}`
        : hasPreparedConfig
          ? "Configuration ready · press Start Separation"
          : `Queue ready · ${queue.length} file${queue.length > 1 ? "s" : ""} pending · configuration required`
      : "Add a track to begin";

  const addToQueue = useCallback(
    async (paths: string[]) => {
      // First, remove any completed items from queue to prevent confusion
      const completedItems = queue.filter((q) => q.status === "completed");
      if (completedItems.length > 0) {
        completedItems.forEach((item) => removeFromQueue(item.id));
      }

      const newItems = paths.map((path) => ({
        id: Math.random().toString(36).substring(7),
        file: path,
        status: "pending" as const,
        progress: 0,
        modelId: selectedPreset, // Default to currently selected preset
      }));
      addToQueueStore(newItems);
      toast.success(
        `Added ${paths.length} file${paths.length > 1 ? "s" : ""} to queue`,
      );
    },
    [addToQueueStore, selectedPreset, queue, removeFromQueue],
  );

  const handleCardDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const paths = files.map((f: any) => f.path || f.name);

      if (paths.length > 0) {
        await addToQueue(paths);
        // Navigate to configure page with first file
        const fileName = paths[0].split(/[\\/]/).pop() || "Unknown";
        onNavigateToConfigure?.(fileName, paths[0], selectedPreset);
      }
    },
    [addToQueue, selectedPreset, onNavigateToConfigure],
  );

  // Start separation for a single file with the given config
  const startSeparationWithFilePath = async (
    config: SeparationConfig,
    filePath: string,
  ) => {
    // Use output directory if set, otherwise backend will use temp directory
    // User can export from Results Studio after previewing
    const targetOutputDir = outputDirectory || "";

    const plan = resolveSeparationPlan({
      config: config as any,
      presets: combinedPresets as any,
      models: models as any,
      globalPhaseParams: phaseParams,
    });

    if (!plan.canProceed) {
      setMissingDialogItems(plan.missingModels);
      setMissingDialogOpen(true);

      // Make sure the queue item isn't left in "processing" if we stop before preflight
      const queueItem = queue.find((q) => q.file === filePath);
      if (queueItem) {
        updateQueueItem(queueItem.id, {
          status: "failed",
          error: "Missing models (see dialog)",
        });
      }

      return;
    }

    const fnoSupported =
      runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok !== false;
    if (!fnoSupported) {
      const candidateModelIds =
        plan.effectiveModelId === "ensemble"
          ? (plan.effectiveEnsembleConfig?.models || []).map((m) => m.model_id)
          : [plan.effectiveModelId];
      const blockedFnoModels = candidateModelIds.filter((id) => {
        const model = models.find((candidate) => candidate.id === id);
        return modelRequiresFnoRuntime(model);
      });
      if (blockedFnoModels.length > 0) {
        toast.error("Separation blocked: model requires FNO1d support", {
          description:
            "Install a neuraloperator/neuralop build with neuralop.models.FNO1d and restart StemSep.",
        });

        const queueItem = queue.find((q) => q.file === filePath);
        if (queueItem) {
          updateQueueItem(queueItem.id, {
            status: "failed",
            error: `Blocked by runtime doctor: ${blockedFnoModels.join(", ")}`,
          });
        }
        return;
      }
    }

    // Update queue item to processing
    const queueItem = queue.find((q) => q.file === filePath);
    if (queueItem) {
      updateQueueItem(queueItem.id, {
        status: "processing",
        progress: 0,
        startTime: Date.now(),
        lastProgressTime: Date.now(),
        message: "Pre-flight...",
      });
    }

    setSeparationStatus();
    addLog(`Starting separation of ${filePath.split(/[\\/]/).pop()}...`);

    try {
      const backendPayload = buildSeparationBackendPayload({
        inputFile: filePath,
        outputDir: targetOutputDir,
        config,
        plan,
      });

      const preflight = await executeSeparationPreflight(
        window.electronAPI,
        backendPayload,
      );

      const warnings = (preflight as any)?.warnings as string[] | undefined;
      if (warnings && warnings.length > 0) {
        toast.warning(warnings[0]);
        addLog(`Pre-flight warning: ${warnings[0]}`);
      }

      const canProceed = (preflight as any)?.can_proceed;
      if (canProceed === false) {
        const errors = (preflight as any)?.errors as string[] | undefined;
        throw new Error(errors?.[0] || "Pre-flight failed");
      }

      // Call backend - if targetOutputDir is empty, backend uses temp directory
      const result = await executeSeparation(window.electronAPI, backendPayload);

      if (queueItem) {
        updateQueueItem(queueItem.id, {
          status: "completed",
          backendJobId: result?.jobId,
          progress: 100,
          message: "Complete!",
        });
      }

      const fileName = filePath.split(/[\\/]/).pop() || "file";

      // Add to history
      // Get preset info for display.
      // Important: a stale presetId can hang around when switching to Advanced mode,
      // so only treat this as a preset run when mode === 'simple'.
      const usedPreset =
        config.mode === "simple" &&
        !!config.presetId &&
        backendPayload.modelId !== "ensemble";
      const presetInfo = usedPreset
        ? combinedPresets.find((p) => p.id === config.presetId)
        : undefined;

      addToHistory({
        inputFile: filePath,
        outputDir: targetOutputDir || "temp",
        modelId: backendPayload.modelId,
        modelName:
          presetInfo?.name ||
          models.find((m) => m.id === backendPayload.modelId)?.name ||
          backendPayload.modelId,
        preset: presetInfo
          ? { id: presetInfo.id, name: presetInfo.name }
          : undefined,
        status: "completed",
        outputFiles: result?.outputFiles || {},
        backendJobId: result?.jobId,
        sourceAudioProfile: result?.sourceAudioProfile,
        stagingDecision: result?.stagingDecision,
        playback: {
          sourceKind: "preview_cache",
          previewDir: result?.outputDir,
        },
        settings: {
          stems: backendPayload.stems || [],
          overlap: config.advancedParams?.overlap,
          segmentSize: config.advancedParams?.segmentSize,
        },
      });

      // Finalize separation
      completeSeparation(result?.outputFiles || {});

      toast.success(`Separated: ${fileName}`);
      addLog(`Completed: ${fileName}`);
    } catch (error) {
      const rawMessage =
        error instanceof Error
          ? error.message
          : String(error ?? "Unknown error");

      const prettyDeviceError = (() => {
        // Expect structured device errors to be embedded in the message as:
        // "STEMSEP_DEVICE_ERROR {json}"
        // (Emitted by Python when CUDA/device preflight/usage fails)
        const prefix = "STEMSEP_DEVICE_ERROR ";
        if (!rawMessage.startsWith(prefix)) return null;

        const jsonPart = rawMessage.slice(prefix.length).trim();
        if (!jsonPart) return null;

        try {
          const payload = JSON.parse(jsonPart) as any;
          if (!payload || typeof payload !== "object") return null;

          const code =
            typeof payload.code === "string" ? payload.code : "DEVICE_ERROR";
          const msg =
            typeof payload.message === "string"
              ? payload.message
              : "Device error";
          const device =
            typeof payload.device === "string" ? payload.device : undefined;

          const title =
            code === "CUDA_OOM"
              ? "GPU out of memory"
              : code === "CUDA_NOT_AVAILABLE"
                ? "GPU not available"
                : code === "CUDA_DEVICE_INVALID"
                  ? "Invalid GPU device"
                  : code === "CUDA_INIT_FAILED"
                    ? "GPU initialization failed"
                    : "Device error";

          const description = device ? `${msg} (requested: ${device})` : msg;

          // Keep details out of the toast (can be long); do log it.
          const details =
            typeof payload.details === "string" ? payload.details : undefined;

          return { title, description, details, code, device };
        } catch {
          return null;
        }
      })();

      const errorMessage = prettyDeviceError
        ? prettyDeviceError.description
        : rawMessage;

      if (queueItem) {
        updateQueueItem(queueItem.id, {
          status: "failed",
          error: errorMessage,
        });
      }

      if (prettyDeviceError) {
        toast.error(prettyDeviceError.title, {
          description: prettyDeviceError.description,
          duration: 12000,
        });
        addLog(
          `Error: ${prettyDeviceError.title}: ${prettyDeviceError.description}${
            prettyDeviceError.details
              ? ` | details: ${prettyDeviceError.details}`
              : ""
          }`,
        );
      } else {
        toast.error(`Failed: ${errorMessage}`);
        addLog(`Error: ${errorMessage}`);
      }
    }
  };

  const handleWindowDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer?.types.includes("Files")) {
      setIsDragging(true);
    }
  }, []);

  return (
    <div
      className="relative h-full overflow-hidden text-white selection:bg-white/20"
      onDragEnter={handleWindowDragEnter}
    >
      <div className="absolute inset-0 flex items-end justify-center px-6 pb-[18%]">
        <div className="flex flex-col items-center">
          <div className="group relative">
            <div className="stemsep-orbit-shell pointer-events-none absolute left-1/2 top-1/2 z-10 h-[360px] w-[360px] -translate-x-1/2 -translate-y-1/2 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              <div className="stemsep-orbit-rotor relative h-full w-full">
                {heroIcons.map(({ key, Icon, style }) => (
                  <div
                    key={key}
                    className="stemsep-orbit-node absolute left-1/2 top-1/2 flex h-11 w-11 items-center justify-center text-white"
                    style={style}
                  >
                    <Icon className="stemsep-orbit-icon h-4 w-4 drop-shadow-[0_0_6px_rgba(255,255,255,0.6)]" />
                  </div>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                void handlePrimaryAction();
              }}
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(false);
              }}
              onDrop={handleCardDrop}
              className={`stemsep-home-button relative z-20 flex flex-row items-center justify-center gap-0.5 overflow-visible rounded-[38px] border px-[22px] py-3 text-center text-[20px] font-normal tracking-[-0.7px] text-white transition-all duration-300 ${
                isDragging
                  ? "scale-105 border-white/50 bg-white/30 shadow-2xl shadow-black/30"
                  : "border-white/30 bg-white/20 shadow-xl shadow-black/20 hover:scale-105 hover:border-white/50 hover:bg-white/30 hover:shadow-2xl hover:shadow-black/30 active:scale-95"
              }`}
            >
              <span className="relative z-20 leading-[1.4] drop-shadow-lg">
                {primaryActionLabel}
              </span>
            </button>
          </div>

          <div className="mt-4 min-h-5 text-center text-[13px] tracking-[-0.2px] text-white/58 drop-shadow">
            {queueStatusText}
          </div>

          {queue.length > 0 && (
            <div className="mt-4 flex items-center gap-2">
              {!isProcessing && (
                <button
                  type="button"
                  onClick={() => {
                    void handleFileSelect();
                  }}
                  className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
                >
                  Add Audio
                </button>
              )}
              {!isProcessing && hasPreparedConfig && (
                <button
                  type="button"
                  onClick={() => {
                    const item = leadQueueItem;
                    if (!item) return;
                    const fileName = item.file.split(/[\\/]/).pop() || "Unknown";
                    onNavigateToConfigure?.(fileName, item.file, selectedPreset);
                  }}
                  className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
                >
                  Edit Config
                </button>
              )}
              {isProcessing && (
                <button
                  type="button"
                  onClick={cancelSeparation}
                  className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
                >
                  Cancel
                </button>
              )}
              <button
                type="button"
                onClick={() => onNavigateToResults?.()}
                className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
              >
                Result Studio
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Dialogs */}
      <PresetBrowser
        open={showPresetBrowser}
        onOpenChange={setShowPresetBrowser}
        presets={combinedPresets}
        favoriteIds={favoritePresetIds}
        onToggleFavorite={toggleFavorite}
        onSelectPreset={handleSelectPreset}
        availability={presetAvailability}
        onNavigateToModels={() => onNavigateToModels?.()}
        onShowModelDetails={setDetailsModelId}
        gpuVRAM={gpuVRAM}
      />

      <SettingsDialog
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
      <HistoryDialog
        isOpen={showHistory}
        onClose={() => setShowHistory(false)}
      />

      <MissingModelsDialog
        open={missingDialogOpen && missingDialogItems.length > 0}
        missing={missingDialogItems}
        models={models as any}
        onClose={() => setMissingDialogOpen(false)}
        onQuickDownload={(modelId) => {
          // Reuse existing download handler used by the model overlay
          handleDownloadModel(modelId);
        }}
        onNavigateToModels={() => onNavigateToModels?.()}
      />

      {/* Model Details Overlay */}
      {activeDetailsModel && (
        <ModelDetails
          model={activeDetailsModel}
          onClose={() => setDetailsModelId(null)}
          onDownload={handleDownloadModel}
          onRemove={handleRemoveModel}
        />
      )}
    </div>
  );
}
