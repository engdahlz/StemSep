import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { Upload, Loader2, Folder } from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Input } from "./ui/input";

import { PresetBrowser } from "./PresetBrowser";
import SettingsDialog from "./SettingsDialog";
import HistoryDialog from "./HistoryDialog";
import { useStore } from "../stores/useStore";
import { SeparationConfig } from "./ConfigurePage";
import { BatchQueueList } from "./BatchQueueList";
import { QueueItem } from "../types/store";
import { toast } from "sonner";
import { Badge } from "./ui/badge";
import { ModelDetails } from "./ModelDetails";
import { ALL_PRESETS, Preset } from "../presets";
import { recipesToPresets } from "@/lib/recipePresets";
import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";
import MissingModelsDialog from "./dialogs/MissingModelsDialog";

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

  // Missing/blocked models dialog (for single-file runs started from this page)
  const [missingDialogOpen, setMissingDialogOpen] = useState(false);
  const [missingDialogItems, setMissingDialogItems] = useState<
    {
      modelId: string;
      reason: "not_installed" | "runtime_blocked";
      details?: string;
    }[]
  >([]);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeStatus, setYoutubeStatus] = useState<string>("");
  const [youtubePercent, setYoutubePercent] = useState<string>("");
  const [isResolvingYoutube, setIsResolvingYoutube] = useState(false);

  const [selectedPreset, setSelectedPreset] =
    useState<string>("best_instrumental");
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

  // GPU info for VRAM filtering
  const [gpuVRAM, setGpuVRAM] = useState<number>(0);
  useEffect(() => {
    const loadGpuInfo = async () => {
      if (window.electronAPI?.getGpuDevices) {
        try {
          const info = await window.electronAPI.getGpuDevices();
          const vram =
            info?.recommended_profile?.vram_gb ||
            info?.gpus?.[0]?.memory_gb ||
            0;
          setGpuVRAM(vram);
        } catch (e) {
          console.error("Failed to load GPU info:", e);
        }
      }
    };
    loadGpuInfo();
  }, []);

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
  const clearQueue = useStore((state) => state.clearQueue);
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

  // New hooks needed for navigation
  const loadSession = useStore((state) => state.loadSession);

  // Global State
  const queue = useStore((state) => state.separation.queue);
  const isProcessing = useStore((state) => state.separation.isProcessing);
  const isPaused = useStore((state) => state.separation.isPaused);
  const progressMessage = useStore((state) => state.separation.message);
  const separationStartTime = useStore((state) => state.separation.startTime);
  const pauseQueue = useStore((state) => state.pauseQueue);
  const resumeQueue = useStore((state) => state.resumeQueue);

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

  // Stored separation config from ConfigurePage (prepared for future use)
  const [_storedConfig, setStoredConfig] = useState<SeparationConfig | null>(
    null,
  );

  // Handle pending separation config from ConfigurePage
  // Use a ref to prevent double-execution when queue updates
  const processingPendingConfigRef = useRef(false);

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

      // Store the config for reference
      setStoredConfig(config);

      // Clear the pending config FIRST to prevent re-triggering
      onClearPendingConfig();

      // Auto-start separation with a small delay to let queue update
      toast.info("Starting separation...");
      setTimeout(() => {
        startSeparationWithFilePath(config, file.path);
        // Reset the guard after starting (allow future configs)
        processingPendingConfigRef.current = false;
      }, 200);
    }
  }, [pendingSeparationConfig, onClearPendingConfig, addToQueueStore]);
  // NOTE: Removed 'queue' from dependencies to prevent double-execution
  // The queue check uses a snapshot, which is acceptable for this use case

  // --- Effects ---

  useEffect(() => {
    const removeStartedListener = window.electronAPI?.onSeparationStarted?.(
      (data) => {
        const processingItem = queue.find((q) => q.status === "processing");
        if (processingItem && data?.jobId && !processingItem.backendJobId) {
          updateQueueItem(processingItem.id, { backendJobId: data.jobId });
        }
      },
    );

    const removeProgressListener = window.electronAPI?.onSeparationProgress(
      (data) => {
        setSeparationProgress(data.progress, data.message);
        // Prefer matching by backend job id (more robust with multiple jobs).
        const byBackendId = data.jobId
          ? queue.find((q) => q.backendJobId === data.jobId)
          : undefined;
        const processingItem =
          byBackendId || queue.find((q) => q.status === "processing");
        if (processingItem) {
          // If we didn't know the backend job id yet, capture it from the first progress event.
          if (data.jobId && !processingItem.backendJobId) {
            updateQueueItem(processingItem.id, { backendJobId: data.jobId });
          }
          updateQueueItem(processingItem.id, {
            progress: data.progress,
            message: data.message,
            lastProgressTime: Date.now(),
          });
        }
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
      removeStartedListener?.();
      removeProgressListener?.();
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

  // YouTube download progress listener
  useEffect(() => {
    if (!window.electronAPI?.onYouTubeProgress) return;
    const cleanup = window.electronAPI.onYouTubeProgress((data) => {
      setYoutubeStatus(data.status || "");
      setYoutubePercent(data.percent || "");
    });
    return () => {
      cleanup?.();
    };
  }, []);

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
          error: "Missing/blocked models (see dialog)",
        });
      }

      return;
    }

    const modelId = plan.effectiveModelId;
    const stems = plan.effectiveStems;

    const effectiveEnsembleConfig =
      plan.effectiveModelId === "ensemble"
        ? plan.effectiveEnsembleConfig
        : undefined;

    const effectiveGlobalPhaseParams = plan.effectiveGlobalPhaseParams;

    const effectivePostProcessingSteps = plan.effectivePostProcessingSteps;

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
      const preflight = window.electronAPI.separationPreflight
        ? await window.electronAPI.separationPreflight(
            filePath,
            modelId,
            targetOutputDir,
            stems,
            config.device && config.device !== "auto"
              ? config.device
              : undefined,
            config.advancedParams?.overlap,
            config.advancedParams?.segmentSize,
            config.advancedParams?.shifts,
            "wav",
            config.advancedParams?.bitrate,
            config.advancedParams?.tta,
            effectiveEnsembleConfig,
            effectiveEnsembleConfig?.algorithm,
            config.invert,
            config.splitFreq,
            effectiveGlobalPhaseParams,
            effectivePostProcessingSteps,
            config.volumeCompensation,
          )
        : null;

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
      const result = await window.electronAPI.separateAudio(
        filePath,
        modelId,
        targetOutputDir,
        stems,
        config.device && config.device !== "auto" ? config.device : undefined,
        config.advancedParams?.overlap,
        config.advancedParams?.segmentSize,
        config.advancedParams?.shifts,
        "wav",
        config.advancedParams?.bitrate,
        config.advancedParams?.tta,
        effectiveEnsembleConfig,
        effectiveEnsembleConfig?.algorithm,
        config.invert,
        config.splitFreq,
        effectiveGlobalPhaseParams,
        effectivePostProcessingSteps,
        config.volumeCompensation,
      );

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
        config.mode === "simple" && !!config.presetId && modelId !== "ensemble";
      const presetInfo = usedPreset
        ? combinedPresets.find((p) => p.id === config.presetId)
        : undefined;

      addToHistory({
        inputFile: filePath,
        outputDir: targetOutputDir || "temp",
        modelId: modelId,
        modelName:
          presetInfo?.name ||
          models.find((m) => m.id === modelId)?.name ||
          modelId,
        preset: presetInfo
          ? { id: presetInfo.id, name: presetInfo.name }
          : undefined,
        status: "completed",
        outputFiles: result?.outputFiles || {},
        backendJobId: result?.jobId,
        settings: {
          stems: stems || [],
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

  const handleResolveYouTube = useCallback(async () => {
    if (!window.electronAPI?.resolveYouTubeUrl) {
      toast.error("YouTube resolution is not available");
      return;
    }
    const url = youtubeUrl.trim();
    if (!url) {
      toast.error("Paste a YouTube link first");
      return;
    }

    try {
      setIsResolvingYoutube(true);
      setYoutubeStatus("starting");
      setYoutubePercent("");

      const result = await window.electronAPI.resolveYouTubeUrl(url);
      const resolvedPath = (result as any)?.file_path;
      const title = (result as any)?.title;

      if (!resolvedPath) {
        throw new Error("No file path returned from backend");
      }

      await addToQueue([resolvedPath]);

      // Navigate to configure using YouTube title as display name
      const displayName =
        title || resolvedPath.split(/[\\/]/).pop() || "YouTube Audio";
      onNavigateToConfigure?.(displayName, resolvedPath, selectedPreset);
      toast.success("YouTube audio ready");
      setYoutubeUrl("");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast.error(`Failed to fetch YouTube audio: ${msg}`);
    } finally {
      setIsResolvingYoutube(false);
    }
  }, [youtubeUrl, addToQueue, onNavigateToConfigure, selectedPreset]);

  const handlePreviewItem = (item: QueueItem) => {
    // Convert QueueItem to HistoryItem format for loader
    const historyItem: any = {
      id: item.id,
      backendJobId: item.backendJobId, // Ensure backend ID is passed!
      inputFile: item.file,
      status: item.status,
      outputFiles: item.outputFiles,
      modelName: item.modelId,
      date: new Date().toISOString(),
      settings: {},
    };
    loadSession(historyItem);
    onNavigateToResults?.();
  };

  return (
    <div
      className="h-full flex flex-col bg-background text-foreground selection:bg-primary/30"
      onDragEnter={handleWindowDragEnter}
    >
      <div className="flex-1 overflow-y-auto relative scroll-smooth">
        {/* Background Ambient Glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

        <div className="flex flex-col max-w-4xl mx-auto w-full p-6 gap-8 relative z-10 animate-in fade-in duration-700">
          {/* 1. Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                New Separation
              </h1>
              <p className="text-muted-foreground mt-1">
                Drop your audio files to begin.
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowHistory(true)}
              >
                <Folder className="w-4 h-4 mr-2" />
                History
              </Button>
            </div>
          </div>

          {/* 2. Upload Area - Expanded */}
          <div className="flex-1 min-h-[400px] flex flex-col">
            <Card
              className={`flex-1 border-2 border-dashed transition-all duration-500 overflow-hidden group relative flex flex-col
                ${isDragging ? "border-primary bg-primary/10 scale-[1.01] shadow-2xl shadow-primary/20" : "border-border/50 hover:border-primary/50 hover:bg-accent/5"}
                `}
            >
              {/* Interactive Overlay - Handles all events */}
              <div
                className="absolute inset-0 z-50 cursor-pointer"
                onClick={() => {
                  handleFileSelect();
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
              />

              {/* Background Glow */}
              <div
                className={`absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5 opacity-0 transition-opacity duration-500 pointer-events-none ${isDragging ? "opacity-100" : "group-hover:opacity-50"}`}
              />

              <CardContent className="flex-1 flex flex-col items-center justify-center text-center space-y-8 relative z-10 pointer-events-none py-20">
                <div
                  className={`p-8 rounded-full bg-background/80 backdrop-blur-sm shadow-xl transition-all duration-500 group-hover:scale-110 group-hover:shadow-2xl group-hover:shadow-primary/20 ${isDragging ? "scale-125 bg-primary/20 text-primary" : "text-muted-foreground"}`}
                >
                  <Upload
                    className={`w-16 h-16 transition-all duration-500 ${isDragging ? "animate-bounce" : ""}`}
                  />
                </div>
                <div className="space-y-4 max-w-md mx-auto">
                  <h3 className="text-3xl font-bold tracking-tight">
                    {isDragging
                      ? "Drop files to separate"
                      : "Drop audio files here"}
                  </h3>
                  <p className="text-muted-foreground text-lg">
                    or click anywhere to browse
                  </p>
                  <div className="flex flex-wrap justify-center gap-2 pt-4">
                    <Badge
                      variant="secondary"
                      className="bg-background/50 backdrop-blur-sm border-border/50 text-xs px-3 py-1"
                    >
                      MP3
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="bg-background/50 backdrop-blur-sm border-border/50 text-xs px-3 py-1"
                    >
                      WAV
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="bg-background/50 backdrop-blur-sm border-border/50 text-xs px-3 py-1"
                    >
                      FLAC
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="bg-background/50 backdrop-blur-sm border-border/50 text-xs px-3 py-1"
                    >
                      M4A
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 2b. YouTube Input */}
          <Card className="p-4">
            <div className="flex flex-col gap-3">
              <div className="text-sm font-medium">YouTube link</div>
              <div className="flex gap-2">
                <Input
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="Paste a YouTube URL (youtube.com / youtu.be)"
                  disabled={isResolvingYoutube}
                />
                <Button
                  variant="outline"
                  onClick={handleResolveYouTube}
                  disabled={isResolvingYoutube || !youtubeUrl.trim()}
                >
                  {isResolvingYoutube ? "Fetching…" : "Fetch"}
                </Button>
              </div>
              {(youtubeStatus || youtubePercent) && (
                <div className="text-xs text-muted-foreground">
                  {youtubeStatus}
                  {youtubePercent ? ` - ${youtubePercent}` : ""}
                </div>
              )}
              <div className="text-xs text-muted-foreground">
                Tip: The audio is downloaded to a temporary WAV file and then
                treated like a normal local file.
              </div>
            </div>
          </Card>

          {/* 3. Queue & Actions */}
          {queue.length > 0 && (
            <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
              <BatchQueueList
                queue={queue}
                onRemoveItem={removeFromQueue}
                onClearQueue={clearQueue}
                onPreviewItem={handlePreviewItem}
                isProcessing={isProcessing}
                isPaused={isPaused}
                onPause={pauseQueue}
                onResume={resumeQueue}
              />

              {/* Action Bar */}
              <div className="sticky bottom-6 bg-background/80 backdrop-blur-md p-4 rounded-2xl border shadow-lg flex items-center justify-between gap-4 z-[60]">
                <div className="text-sm text-muted-foreground pl-2">
                  {isProcessing ? (
                    <span className="flex items-center gap-2 text-primary font-medium">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="flex flex-col">
                        <span>
                          Processing{" "}
                          {queue.filter((i) => i.status === "completed").length}{" "}
                          / {queue.length} files...
                        </span>
                        {progressMessage && (
                          <span className="text-xs text-muted-foreground">
                            {progressMessage}
                          </span>
                        )}
                        {processingTimingText && (
                          <span className="text-xs text-muted-foreground">
                            {processingTimingText}
                          </span>
                        )}
                      </span>
                    </span>
                  ) : (
                    <span>
                      Ready to configure <b>{queue.length}</b> files
                    </span>
                  )}
                </div>
                {isProcessing ? (
                  <Button
                    size="lg"
                    variant="destructive"
                    className="px-8 shadow-lg shadow-destructive/20 text-base font-semibold"
                    onClick={cancelSeparation}
                  >
                    Cancel
                  </Button>
                ) : (
                  <Button
                    size="lg"
                    className="px-8 shadow-lg shadow-primary/20 text-base font-semibold"
                    disabled={queue.length === 0}
                    onClick={() => {
                      const item = queue[0];
                      if (item) {
                        const fileName =
                          item.file.split(/[\\/]/).pop() || "Unknown";
                        onNavigateToConfigure?.(
                          fileName,
                          item.file,
                          selectedPreset,
                        );
                      }
                    }}
                  >
                    Configure Separation
                  </Button>
                )}
              </div>
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
        onSelectPreset={setSelectedPreset}
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
