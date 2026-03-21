import { CSSProperties, useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  Check,
  ChevronDown,
  CloudDownload,
  Disc3,
  Headphones,
  Link2,
  Loader2,
  Mic2,
  Music2,
  Radio,
  RefreshCw,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

import { PresetBrowser } from "./PresetBrowser";
import SettingsDialog from "./SettingsDialog";
import HistoryDialog from "./HistoryDialog";
import { useStore } from "../stores/useStore";
import { SeparationConfig } from "./ConfigurePage";
import { QueueItem } from "../types/store";
import type { QueueSourceMeta, SourceType } from "../types/store";
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
import {
  buildSeparationBackendPayload,
  executeSeparation,
  executeSeparationPreflight,
} from "@/lib/separation/backendPayload";
import type { SeparationProgressEvent } from "@/types/media";
import { queueUpdatesFromProgressEvent } from "@/lib/separation/progressEvent";
import type {
  ActiveLibraryProvider,
  CaptureEnvironmentStatus,
  PlaybackCaptureProgressPayload,
  PlaybackDevice,
  RemoteCatalogItem,
} from "@/types/remote";

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

type QueueInput =
  | string
  | {
      path: string;
      displayName?: string;
      sourceUrl?: string;
      sourceType?: SourceType;
      sourceMeta?: QueueSourceMeta;
    };

type ProviderLibraryState = {
  items: RemoteCatalogItem[];
  authenticated: boolean;
  loaded: boolean;
  error?: string | null;
  message?: string | null;
};

const REMOTE_PROVIDERS: ActiveLibraryProvider[] = ["qobuz"];

const createInitialProviderState = (): Record<
  ActiveLibraryProvider,
  ProviderLibraryState
> => ({
  spotify: {
    items: [],
    authenticated: false,
    loaded: false,
    error: null,
    message: "Sign in, then load your library or search for a track.",
  },
  qobuz: {
    items: [],
    authenticated: false,
    loaded: false,
    error: null,
    message: "Sign in, then load your library or search for a track.",
  },
});

const getDisplayNameForPath = (path: string, displayName?: string) =>
  displayName || path.split(/[\\/]/).pop() || path;

const getProviderLabel = (provider: ActiveLibraryProvider) =>
  provider === "qobuz" ? "Qobuz" : "Spotify";

const getProviderEmptyStateText = (provider: ActiveLibraryProvider) =>
  provider === "qobuz"
    ? "No Qobuz tracks were found in the current view. Load favorites or run a search, then try again."
    : "No tracks were found on the current page. Open your library or search results, then refresh.";

const isLikelyYouTubeUrl = (value: string) => {
  try {
    const parsed = new URL(value);
    const host = parsed.hostname.toLowerCase();
    return (
      host === "youtu.be" ||
      host.endsWith("youtube.com") ||
      host.endsWith("youtube-nocookie.com")
    );
  } catch {
    return false;
  }
};

export default function SeparatePage({
  onNavigateToModels,
  onNavigateToConfigure,
  onNavigateToResults,
  pendingSeparationConfig,
  onClearPendingConfig,
}: SeparatePageProps) {
  // --- State ---
  const [isDragging, setIsDragging] = useState(false);
  const [isOrbitDragActive, setIsOrbitDragActive] = useState(false);

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
  const [activeImportTab, setActiveImportTab] = useState<
    "youtube" | ActiveLibraryProvider
  >("youtube");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isResolvingYouTube, setIsResolvingYouTube] = useState(false);
  const [youtubeProgress, setYoutubeProgress] = useState<{
    status: string;
    percent?: string;
    speed?: string;
    eta?: string;
    error?: string;
  } | null>(null);
  const [providerLibraries, setProviderLibraries] = useState<
    Record<ActiveLibraryProvider, ProviderLibraryState>
  >(createInitialProviderState);
  const [providerQueries, setProviderQueries] = useState<
    Record<ActiveLibraryProvider, string>
  >({
    spotify: "",
    qobuz: "",
  });
  const [loadingProvider, setLoadingProvider] =
    useState<ActiveLibraryProvider | null>(null);
  const [capturingProvider, setCapturingProvider] =
    useState<ActiveLibraryProvider | null>(null);
  const [capturingTrackId, setCapturingTrackId] = useState<string | null>(null);
  const [captureProgress, setCaptureProgress] =
    useState<PlaybackCaptureProgressPayload | null>(null);
  const [isCancellingCapture, setIsCancellingCapture] = useState(false);
  const [captureEnvironment, setCaptureEnvironment] =
    useState<CaptureEnvironmentStatus | null>(null);
  const [playbackDevices, setPlaybackDevices] = useState<PlaybackDevice[]>([]);
  const [selectedPlaybackDeviceId, setSelectedPlaybackDeviceId] = useState("");
  const [isSavingPlaybackDevice, setIsSavingPlaybackDevice] = useState(false);
  const [isPlaybackDeviceMenuOpen, setIsPlaybackDeviceMenuOpen] =
    useState(false);

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
  const history = useStore((state) => state.history);
  const loadSession = useStore((state) => state.loadSession);
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

  const completedQueueItem = useMemo(() => {
    const completedItems = queue.filter((q) => q.status === "completed");
    return completedItems[completedItems.length - 1] || null;
  }, [queue]);

  const latestCompletedSession = useMemo(() => {
    const sortedCompleted = [...history]
      .filter((item) => item.status === "completed")
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    if (!completedQueueItem) {
      return sortedCompleted[0] || null;
    }

    return (
      sortedCompleted.find((item) => item.inputFile === completedQueueItem.file) ||
      sortedCompleted[0] ||
      null
    );
  }, [completedQueueItem, history]);

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

  const isProcessingPossiblyStalled = useMemo(() => {
    if (!isProcessing || !processingQueueItem?.lastProgressTime) return false;
    return nowMs - processingQueueItem.lastProgressTime >= 30_000;
  }, [isProcessing, nowMs, processingQueueItem?.lastProgressTime]);

  const verifyPlayableOutputs = useCallback(
    async (
      outputFiles: Record<string, string> | undefined,
      playback?: {
        sourceKind: "preview_cache" | "saved_output" | "missing_source";
        previewDir?: string;
        savedDir?: string;
      },
    ) => {
      if (!outputFiles || Object.keys(outputFiles).length === 0) {
        return {
          resolved: {} as Record<string, string>,
          issues: {} as Record<string, string>,
        };
      }

      let resolved = outputFiles;
      let issuesFromResolver: Record<
        string,
        { code?: string; hint?: string; originalPath?: string }
      > = {};

      if (window.electronAPI?.resolvePlaybackStems) {
        const playbackResolution = await window.electronAPI.resolvePlaybackStems(
          outputFiles,
          playback,
        );
        if (playbackResolution.success) {
          resolved = playbackResolution.stems || {};
          issuesFromResolver = playbackResolution.issues || {};
        }
      }

      const existenceChecks = await Promise.all(
        Object.entries(resolved).map(async ([stem, filePath]) => {
          const exists = await window.electronAPI?.checkFileExists?.(filePath);
          return [stem, filePath, !!exists] as const;
        }),
      );

      const verifiedOutputs: Record<string, string> = {};
      const issues: Record<string, string> = {};

      for (const [stem, filePath, exists] of existenceChecks) {
        if (exists) {
          verifiedOutputs[stem] = filePath;
          continue;
        }

        const resolverIssue = issuesFromResolver[stem];
        issues[stem] =
          resolverIssue?.hint ||
          `Resolved playback file is missing: ${filePath}`;
      }

      for (const [stem, issue] of Object.entries(issuesFromResolver)) {
        if (!verifiedOutputs[stem] && !issues[stem]) {
          issues[stem] =
            issue?.hint || "Playback source could not be resolved.";
        }
      }

      return { resolved: verifiedOutputs, issues };
    },
    [],
  );

  // Prepared run created in ConfigurePage. This should stay idle until the user
  // explicitly presses the primary CTA on the home screen.
  const [preparedSeparation, setPreparedSeparation] =
    useState<PendingSeparationConfig | null>(null);

  // Handle pending separation config from ConfigurePage.
  // We store it locally instead of auto-starting so the entry flow becomes:
  // add audio -> configure -> explicit start.
  const processingPendingConfigRef = useRef(false);
  const dragDepthRef = useRef(0);
  const playbackDeviceMenuRef = useRef<HTMLDivElement | null>(null);
  const authPollTimeoutRef = useRef<number | null>(null);
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
          displayName: file.name,
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

  useEffect(() => {
    return (
      window.electronAPI?.onYouTubeProgress?.((data) => {
        setYoutubeProgress(data);
      }) || undefined
    );
  }, []);

  useEffect(() => {
    return (
      window.electronAPI?.onPlaybackCaptureProgress?.((data) => {
        setCaptureProgress(data);
      }) || undefined
    );
  }, []);
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
        const fileName = getDisplayNameForPath(filePaths[0]);
        onNavigateToConfigure?.(fileName, filePaths[0], selectedPreset);
      }
    } catch (error) {
      console.error("Error in handleFileSelect:", error);
      toast.error("Failed to select files");
    }
  };

  const handleResolveYouTube = async () => {
    const trimmedUrl = youtubeUrl.trim();
    if (!trimmedUrl) {
      toast.error("Paste a YouTube link first");
      return;
    }
    if (!isLikelyYouTubeUrl(trimmedUrl)) {
      toast.error("That does not look like a valid YouTube link");
      return;
    }
    if (!window.electronAPI?.resolveYouTubeUrl) {
      toast.error("YouTube import is not available in this build");
      return;
    }

    setIsResolvingYouTube(true);
    setYoutubeProgress({ status: "starting" });

    try {
      const result = await window.electronAPI.resolveYouTubeUrl(trimmedUrl);
      if (!result.success) {
        toast.error(result.error, {
          description: result.hint,
        });
        return;
      }

      const displayName = result.title?.trim() || "YouTube Audio";
      await addToQueue([
        {
          path: result.file_path,
          displayName,
          sourceUrl: result.canonical_url || result.source_url || trimmedUrl,
          sourceType: "youtube",
          sourceMeta: {
            provider: "youtube",
            title: displayName,
            channel: result.channel,
            durationSec:
              typeof result.duration_sec === "number"
                ? result.duration_sec
                : undefined,
            artworkUrl: result.thumbnail_url,
            thumbnailUrl: result.thumbnail_url,
            canonicalUrl:
              result.canonical_url || result.source_url || trimmedUrl,
            qualityLabel: "Best available",
            isLossless: false,
            downloadOrigin: "remote_resolve",
          },
        },
      ]);
      setYoutubeUrl("");
      toast.success(`Imported: ${displayName}`);
      onNavigateToConfigure?.(displayName, result.file_path, selectedPreset);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to import YouTube link",
      );
    } finally {
      setIsResolvingYouTube(false);
    }
  };

  const updateProviderLibrary = useCallback(
    (
      provider: ActiveLibraryProvider,
      patch: Partial<ProviderLibraryState>,
    ) => {
      setProviderLibraries((current) => ({
        ...current,
        [provider]: {
          ...current[provider],
          ...patch,
        },
      }));
    },
    [],
  );

  const refreshCaptureEnvironment = useCallback(async () => {
    if (capturingProvider || captureProgress) {
      return;
    }
    try {
      const environment =
        (await window.electronAPI?.getCaptureEnvironmentStatus?.()) ||
        null;
      const devices =
        (await window.electronAPI?.detectPlaybackDevices?.()) || [];

      const resolvedDevices = Array.isArray(devices) ? devices : [];
      setPlaybackDevices(resolvedDevices);

      if (environment) {
        setCaptureEnvironment(environment);
        if (environment.selectedDeviceId) {
          setSelectedPlaybackDeviceId(environment.selectedDeviceId);
          return;
        }
      }

      const defaultDevice =
        resolvedDevices.find((device) => device.isDefault) || resolvedDevices[0];
      setSelectedPlaybackDeviceId(defaultDevice?.id || "");
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to refresh the Qobuz capture environment",
      );
    }
  }, [captureProgress, capturingProvider]);

  useEffect(() => {
    void refreshCaptureEnvironment();
  }, [refreshCaptureEnvironment]);

  const recoverPlaybackCaptureState = useCallback(async () => {
    if (captureProgress) return;
    if (!window.electronAPI?.getPlaybackCaptureStatus) return;

    try {
      const status = await window.electronAPI.getPlaybackCaptureStatus();
      const sessions = Array.isArray(status?.sessions) ? status.sessions : [];
      const activeSession = sessions.find((session: any) => {
        const currentStatus = String(
          session?.progress?.status || session?.backend?.status || "",
        ).toLowerCase();
        return !/completed|cancelled|error|failed/.test(currentStatus);
      });

      if (!activeSession) return;

      if (activeSession.provider === "qobuz" || activeSession.provider === "spotify") {
        setCapturingProvider(activeSession.provider);
      }
      if (typeof activeSession.trackId === "string") {
        setCapturingTrackId(activeSession.trackId);
      }
      if (activeSession.progress) {
        setCaptureProgress(activeSession.progress);
      }
    } catch {
      // Ignore state recovery failures; normal progress events will still drive the UI.
    }
  }, [captureProgress]);

  useEffect(() => {
    if (activeImportTab !== "qobuz") return;
    void recoverPlaybackCaptureState();
  }, [activeImportTab, recoverPlaybackCaptureState]);

  useEffect(() => {
    if (!captureProgress) return;
    const terminal = /completed|cancelled|error|failed/i.test(
      String(captureProgress.status || ""),
    );
    if (!terminal) return;
    setIsCancellingCapture(false);
    setCapturingProvider(null);
    setCapturingTrackId(null);
  }, [captureProgress]);

  useEffect(() => {
    if (!isPlaybackDeviceMenuOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (
        playbackDeviceMenuRef.current &&
        event.target instanceof Node &&
        !playbackDeviceMenuRef.current.contains(event.target)
      ) {
        setIsPlaybackDeviceMenuOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [isPlaybackDeviceMenuOpen]);

  useEffect(() => {
    return () => {
      if (authPollTimeoutRef.current) {
        window.clearTimeout(authPollTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (activeImportTab === "youtube") return;
    if (!window.electronAPI?.getLibraryAuthStatus) return;

    let cancelled = false;
    const provider = activeImportTab;

    const refreshAuthStatus = async () => {
      try {
        const result = await window.electronAPI.getLibraryAuthStatus(provider);
        if (cancelled) return;
        if (result.success) {
          updateProviderLibrary(provider, {
            authenticated: !!result.authenticated,
            error: result.authenticated ? null : undefined,
            message: result.authenticated
              ? `${getProviderLabel(provider)} connected.`
              : undefined,
          });
        }
      } catch {
        // Ignore passive auth refresh failures.
      }
    };

    void refreshAuthStatus();
    void refreshCaptureEnvironment();
    const handleFocus = () => {
      void refreshAuthStatus();
      void refreshCaptureEnvironment();
    };
    window.addEventListener("focus", handleFocus);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", handleFocus);
    };
  }, [activeImportTab, refreshCaptureEnvironment, updateProviderLibrary]);

  const handleAuthLibraryProvider = useCallback(
    async (provider: ActiveLibraryProvider) => {
      if (!window.electronAPI?.authLibraryProvider) {
        toast.error("Library auth is unavailable in this build");
        return;
      }

      const result = await window.electronAPI.authLibraryProvider(provider);
      if (!result.success) {
        updateProviderLibrary(provider, {
          error: result.error,
          message: result.hint || null,
        });
        toast.error(result.error, {
          description: result.hint,
        });
        return;
      }

      updateProviderLibrary(provider, {
        authenticated: !!result.authenticated,
        error: null,
        message:
          result.message || "Sign in, then refresh the library.",
      });
      toast.success(result.message || `Opened ${getProviderLabel(provider)} sign-in window`);

      if (authPollTimeoutRef.current) {
        window.clearTimeout(authPollTimeoutRef.current);
        authPollTimeoutRef.current = null;
      }

      if (!result.authenticated && window.electronAPI?.getLibraryAuthStatus) {
        let attemptsRemaining = 30;
        const poll = async () => {
          try {
            const status = await window.electronAPI?.getLibraryAuthStatus?.(provider);
            if (status?.success) {
              updateProviderLibrary(provider, {
                authenticated: !!status.authenticated,
                error: status.authenticated ? null : undefined,
                message: status.authenticated
                  ? `${getProviderLabel(provider)} connected.`
                  : result.message || "Finish the sign-in in the provider window.",
              });
              await refreshCaptureEnvironment();
              if (status.authenticated) {
                authPollTimeoutRef.current = null;
                return;
              }
            }
          } catch {
            // Ignore transient auth polling failures.
          }

          attemptsRemaining -= 1;
          if (attemptsRemaining <= 0) {
            authPollTimeoutRef.current = null;
            return;
          }
          authPollTimeoutRef.current = window.setTimeout(() => {
            void poll();
          }, 2000);
        };

        authPollTimeoutRef.current = window.setTimeout(() => {
          void poll();
        }, 1200);
      } else {
        window.setTimeout(() => {
          void refreshCaptureEnvironment();
        }, 800);
      }
    },
    [refreshCaptureEnvironment, updateProviderLibrary],
  );

  const handleLoadLibrary = useCallback(
    async (provider: ActiveLibraryProvider, mode: "library" | "search") => {
      if (
        !window.electronAPI?.listLibraryCollection ||
        !window.electronAPI?.searchLibrary
      ) {
        toast.error("Library browsing is unavailable in this build");
        return;
      }

      const query = providerQueries[provider].trim();
      setLoadingProvider(provider);
      updateProviderLibrary(provider, {
        error: null,
        message:
          mode === "search" && query
            ? "Searching tracks..."
            : "Loading library...",
      });

      try {
        const result =
          mode === "search" && query
            ? await window.electronAPI.searchLibrary(provider, query)
            : await window.electronAPI.listLibraryCollection(provider, "library");

        if (!result.success) {
          updateProviderLibrary(provider, {
            authenticated: !!result.authenticated,
            loaded: true,
            error: result.error,
            message: result.hint || null,
          });
          toast.error(result.error, {
            description: result.hint,
          });
          return;
        }

        updateProviderLibrary(provider, {
          authenticated: !!result.authenticated,
          items: result.items,
          loaded: true,
          error: null,
          message:
            result.items.length > 0
              ? `Found ${result.items.length} track${result.items.length === 1 ? "" : "s"} ready for capture.`
              : "No tracks were found in the current view.",
        });
      } finally {
        setLoadingProvider((current) => (current === provider ? null : current));
      }
    },
    [providerQueries, updateProviderLibrary],
  );

  const handleSavePlaybackDevice = useCallback(async () => {
    if (!selectedPlaybackDeviceId) {
      toast.error("Select a silent output device first");
      return;
    }
    if (!window.electronAPI?.setCaptureOutputDevice) {
      toast.error("Capture setup is unavailable in this build");
      return;
    }

    setIsSavingPlaybackDevice(true);
    try {
      const result = await window.electronAPI.setCaptureOutputDevice(
        selectedPlaybackDeviceId,
      );
      if (!result.success) {
        toast.error(result.error || "Failed to save the silent output device");
        return;
      }
      toast.success(
        result.label
          ? `Silent output saved: ${result.label}`
          : "Silent output saved",
      );
      await refreshCaptureEnvironment();
    } finally {
      setIsSavingPlaybackDevice(false);
    }
  }, [refreshCaptureEnvironment, selectedPlaybackDeviceId]);

  const handleCaptureLibraryItem = useCallback(
    async (item: RemoteCatalogItem) => {
      if (
        !window.electronAPI?.preparePlaybackCapture ||
        !window.electronAPI?.startPlaybackCapture
      ) {
        toast.error("Playback capture is unavailable in this build");
        return;
      }
      if (!selectedPlaybackDeviceId) {
        toast.error("Select a playback device first");
        return;
      }
      if (!captureEnvironment?.selectedDeviceReady) {
        toast.error("Save and validate a silent output device first");
        return;
      }

      setCapturingProvider(item.provider as ActiveLibraryProvider);
      setCapturingTrackId(item.trackId);
      setIsCancellingCapture(false);
      setCaptureProgress({
        provider: item.provider as ActiveLibraryProvider,
        status: "starting",
        detail: `Preparing ${item.title} for capture...`,
      });

      try {
        const prepare = await window.electronAPI.preparePlaybackCapture(
          item.provider as ActiveLibraryProvider,
          item.trackId,
        );
        if (!prepare.success) {
          toast.error(prepare.error, {
            description: prepare.hint,
          });
          return;
        }

        const started = await window.electronAPI.startPlaybackCapture(
          item.provider as ActiveLibraryProvider,
          item.trackId,
          selectedPlaybackDeviceId,
        );
        if (!started.success) {
          if (
            started.code === "CAPTURE_CANCELLED" ||
            /cancelled/i.test(started.error || "")
          ) {
            toast("Capture cancelled");
          } else {
            toast.error(started.error, {
              description: started.hint,
            });
          }
          return;
        }

        const displayName =
          started.display_name || prepare.displayName || item.title;
        await addToQueue([
          {
            path: started.file_path,
            displayName,
            sourceUrl:
              started.canonical_url ||
              started.source_url ||
              item.canonicalUrl ||
              item.playbackUrl,
            sourceType: item.provider,
            sourceMeta: {
              provider: item.provider,
              providerTrackId: started.provider_track_id || item.trackId,
              title: item.title,
              artist: started.artist || item.artist,
              album: started.album || item.album,
              artworkUrl: started.artwork_url || item.artworkUrl,
              durationSec:
                typeof started.duration_sec === "number"
                  ? started.duration_sec
                  : item.durationSec,
              canonicalUrl:
                started.canonical_url ||
                item.canonicalUrl ||
                item.playbackUrl ||
                item.pageUrl,
              qualityLabel: started.quality_label || item.qualityLabel,
              isLossless: started.is_lossless,
              ingestMode: started.ingest_mode,
              playbackSurface: started.playback_surface,
              qualityMode: started.quality_mode,
              verifiedLossless: started.verified_lossless,
              captureDeviceId: started.capture_device_id,
              captureSampleRate: started.capture_sample_rate,
              captureChannels: started.capture_channels,
              captureStartAt: started.capture_start_at,
              captureEndAt: started.capture_end_at,
            },
          },
        ]);
        toast.success(`Captured: ${displayName}`);
        onNavigateToConfigure?.(displayName, started.file_path, selectedPreset);
      } catch (error) {
        await window.electronAPI?.cancelPlaybackCapture?.();
        const message =
          error instanceof Error ? error.message : "Playback capture failed";
        if (/cancelled/i.test(message)) {
          toast("Capture cancelled");
        } else {
          toast.error(message);
        }
      } finally {
        setCapturingProvider((current) =>
          current === item.provider ? null : current,
        );
        setCapturingTrackId((current) =>
          current === item.trackId ? null : current,
        );
        setIsCancellingCapture(false);
        setCaptureProgress(null);
      }
    },
    [
      addToQueue,
      captureEnvironment?.selectedDeviceReady,
      onNavigateToConfigure,
      selectedPlaybackDeviceId,
      selectedPreset,
    ],
  );

  const handleCancelCapture = useCallback(async () => {
    if (!window.electronAPI?.cancelPlaybackCapture || !captureProgress) return;

    try {
      setIsCancellingCapture(true);
      setCaptureProgress((current) =>
        current
          ? {
              ...current,
              status: "cancelling",
              detail: "Cancelling capture...",
            }
          : current,
      );
      await window.electronAPI.cancelPlaybackCapture(captureProgress.captureId);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to cancel capture",
      );
      setIsCancellingCapture(false);
    }
  }, [captureProgress]);

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
    const fileName = getDisplayNameForPath(item.file, item.displayName);
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
      ? "Upload Audio"
      : hasPreparedConfig
        ? "Start Separation"
        : "Configure";

  const queueStatusText = isDragging
    ? "Drop files to start a new separation"
    : isResolvingYouTube
      ? `Importing from YouTube${youtubeProgress?.percent ? ` · ${youtubeProgress.percent}` : ""}${youtubeProgress?.speed ? ` · ${youtubeProgress.speed}` : ""}${youtubeProgress?.eta ? ` · ETA ${youtubeProgress.eta}` : ""}`
      : loadingProvider
        ? `${getProviderLabel(loadingProvider)} · ${providerLibraries[loadingProvider].message || "Syncing library..."}`
        : capturingProvider
          ? `${getProviderLabel(capturingProvider)} · ${captureProgress?.detail || captureProgress?.status || "Preparing capture..."}${captureProgress?.percent ? ` · ${captureProgress.percent}` : ""}`
      : queue.length > 0
        ? isProcessing
          ? `Processing ${queue.filter((item) => item.status === "completed").length}/${queue.length}${processingQueueItem?.activeStepLabel ? ` · ${processingQueueItem.activeStepLabel}` : processingQueueItem?.activeModelId ? ` · ${processingQueueItem.activeModelId}` : ""}${progressMessage ? ` · ${progressMessage}` : ""}${processingTimingText ? ` · ${processingTimingText}` : ""}${isProcessingPossiblyStalled ? " · Waiting for next update…" : ""}`
        : hasPreparedConfig
          ? "Configuration ready · press Start Separation"
          : `Queue ready · ${queue.length} file${queue.length > 1 ? "s" : ""} pending · configuration required`
      : "Upload audio to begin";

  const handleClearQueuedFile = useCallback(() => {
    const targetPath = preparedSeparation?.file?.path || leadQueueItem?.file;
    if (!targetPath) return;

    queue
      .filter((item) => item.file === targetPath)
      .forEach((item) => removeFromQueue(item.id));

    setPreparedSeparation((current) =>
      current?.file?.path === targetPath ? null : current,
    );
  }, [leadQueueItem?.file, preparedSeparation?.file?.path, queue, removeFromQueue]);

  const handleOpenLatestResults = useCallback(() => {
    if (!latestCompletedSession) return;
    loadSession(latestCompletedSession);
    onNavigateToResults?.();
  }, [latestCompletedSession, loadSession, onNavigateToResults]);

  async function addToQueue(entries: QueueInput[]) {
    // First, remove any completed items from queue to prevent confusion
    const completedItems = queue.filter((q) => q.status === "completed");
    if (completedItems.length > 0) {
      completedItems.forEach((item) => removeFromQueue(item.id));
    }

    const normalizedEntries = entries.map((entry) =>
      typeof entry === "string"
        ? { path: entry, sourceType: "local_file" as const }
        : entry,
    );

    const newItems = normalizedEntries.map((entry) => ({
      id: Math.random().toString(36).substring(7),
      file: entry.path,
      displayName: entry.displayName,
      sourceUrl: entry.sourceUrl,
      sourceType: entry.sourceType || "local_file",
      sourceMeta: entry.sourceMeta,
      status: "pending" as const,
      progress: 0,
      modelId: selectedPreset, // Default to currently selected preset
    }));
    addToQueueStore(newItems);
    toast.success(
      `Added ${normalizedEntries.length} source${normalizedEntries.length > 1 ? "s" : ""} to queue`,
    );
  }

  const supportedAudioExtensions = useMemo(
    () => new Set(["wav", "flac", "mp3", "m4a", "aac", "ogg", "opus", "aif", "aiff", "wma"]),
    [],
  );

  const filterAudioPaths = useCallback(
    (paths: string[]) =>
      paths.filter((path) => {
        const match = path.toLowerCase().match(/\.([a-z0-9]+)$/);
        return !!match && supportedAudioExtensions.has(match[1]);
      }),
    [supportedAudioExtensions],
  );

  const handleCardDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragDepthRef.current = 0;
      setIsDragging(false);
      setIsOrbitDragActive(false);

      const files = Array.from(e.dataTransfer.files);
      const droppedPaths = files.map((f: any) => f.path || f.name);
      const audioPaths = filterAudioPaths(droppedPaths);

      if (audioPaths.length === 0) {
        toast.error("Only audio files can be dropped here");
        return;
      }

      if (audioPaths.length !== droppedPaths.length) {
        toast.warning("Some dropped files were ignored because they are not audio");
      }

      if (audioPaths.length > 0) {
        await addToQueue(audioPaths);
        // Navigate to configure page with first file
        const fileName = audioPaths[0].split(/[\\/]/).pop() || "Unknown";
        onNavigateToConfigure?.(fileName, audioPaths[0], selectedPreset);
      }
    },
    [addToQueue, filterAudioPaths, onNavigateToConfigure, selectedPreset],
  );

  // Start separation for a single file with the given config
  const startSeparationWithFilePath = async (
    config: SeparationConfig,
    filePath: string,
  ) => {
    // Use output directory if set, otherwise backend will use temp directory
    // User can export from Results after previewing
    const targetOutputDir = outputDirectory || "";

    const plan = resolveSeparationPlan({
      config: config as any,
      presets: combinedPresets as any,
      models: models as any,
      globalPhaseParams: phaseParams,
      runtimeSupport: {
        fnoSupported:
          runtimeInfo?.runtimeFingerprint?.neuralop?.fno1d_import_ok !== false,
      },
    });

    if (!plan.canProceed) {
      if (plan.missingModels.length > 0) {
        setMissingDialogItems(plan.missingModels);
        setMissingDialogOpen(true);
      } else if (plan.blockingIssues.length > 0) {
        toast.error(plan.blockingIssues[0]);
      }

      // Make sure the queue item isn't left in "processing" if we stop before preflight
      const queueItem = queue.find((q) => q.file === filePath);
      if (queueItem) {
        updateQueueItem(queueItem.id, {
          status: "failed",
          error:
            plan.blockingIssues[0] || "Missing models or runtime requirements",
        });
      }

      return;
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
    addLog(
      `Starting separation of ${getDisplayNameForPath(filePath, queueItem?.displayName)}...`,
    );

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

      const verifiedOutputsResult = await verifyPlayableOutputs(
        result?.outputFiles,
        {
          sourceKind: "preview_cache",
          previewDir: result?.outputDir,
        },
      );

      if (Object.keys(verifiedOutputsResult.resolved).length === 0) {
        const firstIssue =
          Object.values(verifiedOutputsResult.issues)[0] ||
          "Separation completed but no playable output files were verified.";
        throw new Error(firstIssue);
      }

      if (queueItem) {
        updateQueueItem(queueItem.id, {
          status: "completed",
          backendJobId: result?.jobId,
          progress: 100,
          message: "Complete!",
          outputFiles: verifiedOutputsResult.resolved,
        });
      }

      const fileName = getDisplayNameForPath(filePath, queueItem?.displayName);

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
        displayName: queueItem?.displayName,
        sourceUrl: queueItem?.sourceUrl,
        sourceType: queueItem?.sourceType,
        sourceMeta: queueItem?.sourceMeta,
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
        outputFiles: verifiedOutputsResult.resolved,
        backendJobId: result?.jobId,
        sourceAudioProfile: result?.sourceAudioProfile,
        stagingDecision: result?.stagingDecision,
        playback: {
          sourceKind: "preview_cache",
          previewDir: result?.outputDir,
        },
        settings: {
          stems: backendPayload.stems || [],
          overlap: plan.effectiveAdvancedParams?.overlap,
          segmentSize: plan.effectiveAdvancedParams?.segmentSize,
        },
      });

      // Finalize separation
      completeSeparation(verifiedOutputsResult.resolved);

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
      dragDepthRef.current += 1;
      setIsDragging(true);
    }
  }, []);

  const processingIcons = useMemo(() => {
    const orbitIcons = [Music2, Headphones, Mic2, Radio, Disc3];
    const circles = [44, 74];
    const iconsPerCircle = [5, 8];

    return circles.flatMap((radius, circleIndex) => {
      const count = iconsPerCircle[circleIndex];
      return Array.from({ length: count }).map((_, iconIndex) => {
        const angle = (iconIndex / count) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        const rotation = Math.atan2(y, x) * (180 / Math.PI);
        const Icon = orbitIcons[(circleIndex + iconIndex) % orbitIcons.length];

        return {
          key: `processing-${circleIndex}-${iconIndex}`,
          Icon,
          style: {
            "--stemsep-processing-x": `${x}px`,
            "--stemsep-processing-y": `${y}px`,
            "--stemsep-processing-rotation": `${rotation}deg`,
            animationDelay: `${circleIndex * 0.18 + iconIndex * 0.09}s`,
          } as CSSProperties,
        };
      });
    });
  }, []);

  const handleWindowDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer?.types.includes("Files")) {
      setIsDragging(true);
    }
  }, []);

  const handleWindowDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!e.dataTransfer?.types.includes("Files")) return;
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragging(false);
      setIsOrbitDragActive(false);
    }
  }, []);

  const handleWindowDropReset = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragging(false);
    setIsOrbitDragActive(false);
  }, []);

  const handleOrbitDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer?.types.includes("Files")) {
      setIsDragging(true);
      setIsOrbitDragActive(true);
    }
  }, []);

  const handleOrbitDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const nextTarget = e.relatedTarget;
    if (nextTarget instanceof Node && e.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsOrbitDragActive(false);
  }, []);

  const activeProvider =
    activeImportTab === "youtube" ? null : activeImportTab;
  const activeProviderLibrary = activeProvider
    ? providerLibraries[activeProvider]
    : null;
  const activeProviderQuery = activeProvider
    ? providerQueries[activeProvider]
    : "";
  const remoteItems = activeProviderLibrary?.items || [];
  const selectedPlaybackDevice = useMemo(
    () =>
      playbackDevices.find((device) => device.id === selectedPlaybackDeviceId) ||
      null,
    [playbackDevices, selectedPlaybackDeviceId],
  );

  return (
    <div
      className="relative h-full overflow-hidden text-white selection:bg-white/20"
      onDragEnter={handleWindowDragEnter}
      onDragOver={handleWindowDragOver}
      onDragLeave={handleWindowDragLeave}
      onDrop={handleWindowDropReset}
    >
      <div className="absolute inset-0 flex items-center justify-center px-6">
        <div className="flex w-full max-w-[760px] flex-col items-center">
          <div
            className="group relative"
            onDragOver={handleOrbitDragOver}
            onDragLeave={handleOrbitDragLeave}
            onDrop={handleCardDrop}
          >
            <div
              className={`stemsep-orbit-shell pointer-events-none absolute left-1/2 top-1/2 z-10 h-[360px] w-[360px] -translate-x-1/2 -translate-y-1/2 ${
                isOrbitDragActive
                  ? "opacity-100"
                  : "opacity-0 group-hover:opacity-100"
              }`}
              style={
                {
                  "--stemsep-orbit-scale": isOrbitDragActive ? 1.12 : 1,
                } as CSSProperties
              }
            >
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
              className={`stemsep-home-button relative z-20 flex flex-row items-center justify-center gap-0.5 overflow-visible rounded-[36px] border px-[20px] py-2.5 text-center text-[18px] font-normal tracking-[-0.6px] text-white transition-all duration-300 ${
                isOrbitDragActive || isDragging
                  ? "stemsep-home-button-drag-active border-white/50 bg-white/30 shadow-2xl shadow-black/30"
                  : "border-white/30 bg-white/20 shadow-xl shadow-black/20 hover:border-white/50 hover:bg-white/30 hover:shadow-2xl hover:shadow-black/30"
              }`}
            >
              <span className="relative z-20 leading-[1.4] drop-shadow-lg">
                {primaryActionLabel}
              </span>
            </button>
          </div>

          {isProcessing && (
            <div className="stemsep-processing-shell mt-10 flex h-[144px] w-[220px] items-center justify-center">
              <div className="stemsep-processing-rotor relative h-full w-full">
                {processingIcons.map(({ key, Icon, style }) => (
                  <div
                    key={key}
                    className="stemsep-processing-node absolute left-1/2 top-1/2 flex h-9 w-9 items-center justify-center text-white/78"
                    style={style}
                  >
                    <Icon className="stemsep-processing-icon h-4 w-4 drop-shadow-[0_0_10px_rgba(255,255,255,0.42)]" />
                  </div>
                ))}
                <div className="stemsep-processing-core absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-end gap-1">
                  {Array.from({ length: 5 }).map((_, index) => (
                    <span
                      key={index}
                      className="stemsep-processing-bar"
                      style={{ animationDelay: `${index * 0.14}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div
            className={`min-h-5 text-center text-[13px] tracking-[-0.2px] text-white/58 drop-shadow ${
              isProcessing ? "mt-6" : "mt-16"
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <span>{queueStatusText}</span>
              {queue.length > 0 && !isProcessing && (
                <button
                  type="button"
                  onClick={handleClearQueuedFile}
                  className="text-white/62 transition-colors hover:text-white/90 hover:underline"
                >
                  clear
                </button>
              )}
            </div>
            {isProcessing && isProcessingPossiblyStalled && (
              <div className="mt-2 text-[12px] text-amber-100/85">
                The job is still running, but progress updates have paused for a moment.
              </div>
            )}
          </div>

          {queue.length > 0 && (
            <div className="mt-4 flex items-center gap-2">
              {!isProcessing && hasPreparedConfig && (
                <button
                  type="button"
                  onClick={() => {
                    const item = leadQueueItem;
                    if (!item) return;
                    const fileName = getDisplayNameForPath(
                      item.file,
                      item.displayName,
                    );
                    onNavigateToConfigure?.(fileName, item.file, selectedPreset);
                  }}
                  className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
                >
                  Edit Config
                </button>
              )}
              {!isProcessing && latestCompletedSession && completedQueueItem && (
                <button
                  type="button"
                  onClick={handleOpenLatestResults}
                  className="rounded-[18px] border border-white/20 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 backdrop-blur-md transition-all hover:bg-white/18"
                >
                  Open Results
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
            </div>
          )}
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-8 z-30 flex justify-center px-6">
        <div className="w-[min(920px,calc(100vw-3rem))] rounded-[1.7rem] border border-white/18 bg-white/10 p-3.5 shadow-[0_18px_50px_rgba(0,0,0,0.16)] backdrop-blur-xl">
          <div className="mb-3 flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/46">
            <CloudDownload className="h-3.5 w-3.5" />
            Import Sources
          </div>

          <div className="mb-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setActiveImportTab("youtube")}
              className={`rounded-[1.1rem] border px-3 py-2 text-[13px] tracking-[-0.2px] transition-all ${
                activeImportTab === "youtube"
                  ? "border-white/34 bg-white/20 text-white"
                  : "border-white/16 bg-white/8 text-white/62 hover:bg-white/14 hover:text-white/84"
              }`}
            >
              YouTube
            </button>
            {REMOTE_PROVIDERS.map((provider) => (
              <button
                key={provider}
                type="button"
                onClick={() => setActiveImportTab(provider)}
                className={`rounded-[1.1rem] border px-3 py-2 text-[13px] tracking-[-0.2px] transition-all ${
                  activeImportTab === provider
                    ? "border-white/34 bg-white/20 text-white"
                    : "border-white/16 bg-white/8 text-white/62 hover:bg-white/14 hover:text-white/84"
                }`}
              >
                {getProviderLabel(provider)}
              </button>
            ))}
          </div>

          {activeImportTab === "youtube" ? (
            <>
              <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/40">
                <Link2 className="h-3.5 w-3.5" />
                Fast Link Import
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="Paste a YouTube link here..."
                  className="flex-1 rounded-[1.2rem] border border-white/16 bg-white/10 px-4 py-3 text-[14px] tracking-[-0.2px] text-white outline-none transition-all placeholder:text-white/30 focus:border-white/32 focus:bg-white/16"
                />
                <button
                  type="button"
                  onClick={() => {
                    void handleResolveYouTube();
                  }}
                  disabled={isResolvingYouTube || !youtubeUrl.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-[1.2rem] border border-white/24 bg-white/14 px-4 py-3 text-[14px] tracking-[-0.2px] text-white transition-all hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {isResolvingYouTube ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Fetching...
                    </>
                  ) : (
                    <>
                      <Link2 className="h-4 w-4" />
                      Import Link
                    </>
                  )}
                </button>
              </div>
              {(youtubeProgress?.status || youtubeProgress?.error) && (
                <div className="mt-2 text-[12px] text-white/58">
                  {youtubeProgress?.error
                    ? youtubeProgress.error
                    : `Status: ${youtubeProgress?.status || "idle"}${youtubeProgress?.percent ? ` · ${youtubeProgress.percent}` : ""}${youtubeProgress?.speed ? ` · ${youtubeProgress.speed}` : ""}${youtubeProgress?.eta ? ` · ETA ${youtubeProgress.eta}` : ""}`}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="mb-1 flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/40">
                    <ShieldCheck className="h-3.5 w-3.5" />
                    Qobuz Capture
                  </div>
                  <p className="text-[12px] text-white/48">
                    Sign in once, save one silent output, then capture in the
                    background.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    void handleAuthLibraryProvider(activeImportTab);
                  }}
                  className={`inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] tracking-[0.04em] transition-all ${
                    activeProviderLibrary?.authenticated
                      ? "border-emerald-200/45 bg-emerald-400/16 text-emerald-50 hover:bg-emerald-400/20"
                      : "border-white/16 bg-white/8 text-white/70 hover:bg-white/12"
                  }`}
                >
                  <ShieldCheck className="h-3.5 w-3.5" />
                  {activeProviderLibrary?.authenticated
                    ? "Authenticated"
                    : "Not Authenticated"}
                </button>
              </div>
              <div className="mb-3 rounded-[1.3rem] border border-white/12 bg-white/6 p-3">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[13px] tracking-[-0.2px] text-white">
                      Silent Output
                    </div>
                    <div className="text-[11px] text-white/46">
                      {captureEnvironment?.selectedDeviceLabel
                        ? `Saved: ${captureEnvironment.selectedDeviceLabel}`
                        : "Choose where hidden Qobuz playback should go."}
                    </div>
                  </div>
                  <span
                    className={`shrink-0 rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] ${
                      captureEnvironment?.selectedDeviceReady
                        ? "border-emerald-200/40 bg-emerald-400/14 text-emerald-50"
                        : "border-white/14 bg-white/8 text-white/58"
                    }`}
                  >
                    {captureEnvironment?.selectedDeviceReady ? "Ready" : "Not Set"}
                  </span>
                </div>
                <div className="flex flex-col gap-2 lg:flex-row">
                  <div
                    ref={playbackDeviceMenuRef}
                    className="relative min-w-0 flex-1"
                  >
                    <button
                      type="button"
                      onClick={() =>
                        setIsPlaybackDeviceMenuOpen((current) => !current)
                      }
                      className="flex w-full items-center justify-between gap-3 rounded-[1.2rem] border border-white/16 bg-white/10 px-4 py-3 text-left text-[14px] tracking-[-0.2px] text-white transition-all hover:bg-white/14"
                    >
                      <span className="truncate">
                        {selectedPlaybackDevice?.label || "Choose silent output"}
                      </span>
                      <ChevronDown
                        className={`h-4 w-4 shrink-0 text-white/60 transition-transform ${
                          isPlaybackDeviceMenuOpen ? "rotate-180" : ""
                        }`}
                      />
                    </button>
                    {isPlaybackDeviceMenuOpen && (
                      <div className="stemsep-soft-scroll absolute left-0 right-0 top-[calc(100%+0.55rem)] z-30 max-h-56 overflow-y-auto rounded-[1.1rem] border border-slate-200/16 bg-slate-950/96 p-1.5 shadow-2xl shadow-black/35 backdrop-blur-xl">
                        {playbackDevices.length === 0 ? (
                          <div className="rounded-[0.9rem] px-3 py-2.5 text-[13px] text-white/55">
                            No outputs found
                          </div>
                        ) : (
                          playbackDevices.map((device) => {
                            const isSelected =
                              device.id === selectedPlaybackDeviceId;
                            return (
                              <button
                                key={device.id}
                                type="button"
                                onClick={() => {
                                  setSelectedPlaybackDeviceId(device.id);
                                  setIsPlaybackDeviceMenuOpen(false);
                                }}
                                className={`flex w-full items-center justify-between gap-3 rounded-[0.9rem] px-3 py-2.5 text-left text-[13px] transition-all ${
                                  isSelected
                                    ? "bg-white text-slate-950"
                                    : "text-white/82 hover:bg-white/10"
                                }`}
                              >
                                <span className="truncate">{device.label}</span>
                                {isSelected && <Check className="h-4 w-4 shrink-0" />}
                              </button>
                            );
                          })
                        )}
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      void handleSavePlaybackDevice();
                    }}
                    disabled={isSavingPlaybackDevice || !selectedPlaybackDeviceId}
                    className="inline-flex items-center justify-center gap-2 rounded-[1.2rem] border border-white/24 bg-white/14 px-4 py-3 text-[14px] tracking-[-0.2px] text-white transition-all hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-55"
                  >
                    {isSavingPlaybackDevice ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Headphones className="h-4 w-4" />
                    )}
                    Save Output
                  </button>
                </div>
              </div>
              <div className="flex flex-col gap-2 lg:flex-row">
                <button
                  type="button"
                  onClick={() => {
                    void handleLoadLibrary(activeImportTab, "library");
                  }}
                  disabled={
                    loadingProvider === activeImportTab ||
                    !activeProviderLibrary?.authenticated
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-[1.2rem] border border-white/24 bg-white/14 px-4 py-3 text-[14px] tracking-[-0.2px] text-white transition-all hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {loadingProvider === activeImportTab ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  Favorites
                </button>
                <input
                  type="text"
                  value={activeProviderQuery}
                  onChange={(event) =>
                    setProviderQueries((current) => ({
                      ...current,
                      [activeImportTab]: event.target.value,
                    }))
                  }
                  placeholder="Search tracks"
                  disabled={!activeProviderLibrary?.authenticated}
                  className="flex-1 rounded-[1.2rem] border border-white/16 bg-white/10 px-4 py-3 text-[14px] tracking-[-0.2px] text-white outline-none transition-all placeholder:text-white/30 focus:border-white/32 focus:bg-white/16"
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      void handleLoadLibrary(
                        activeImportTab,
                        "search",
                      );
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={() => {
                    void handleLoadLibrary(activeImportTab, "search");
                  }}
                  disabled={
                    loadingProvider === activeImportTab ||
                    !activeProviderLibrary?.authenticated ||
                    !activeProviderQuery.trim()
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-[1.2rem] border border-white/24 bg-white/14 px-4 py-3 text-[14px] tracking-[-0.2px] text-white transition-all hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {loadingProvider === activeImportTab ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Link2 className="h-4 w-4" />
                  )}
                  Search
                </button>
              </div>

              {activeProviderLibrary?.error && (
                <div className="mt-2 text-[12px] text-white/58">
                  {activeProviderLibrary.error}
                </div>
              )}

              {captureProgress &&
                captureProgress.provider === activeImportTab &&
                (captureProgress.detail || captureProgress.error) && (
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <div className="min-w-0 text-[12px] text-white/62">
                      {captureProgress.error
                        ? captureProgress.error
                        : [
                            captureProgress.detail || captureProgress.status,
                            captureProgress.percent,
                            typeof captureProgress.elapsedSec === "number"
                              ? `${Math.round(captureProgress.elapsedSec)}s`
                              : null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                    </div>
                    {!captureProgress.error &&
                      capturingProvider === activeImportTab && (
                        <button
                          type="button"
                          onClick={() => {
                            void handleCancelCapture();
                          }}
                          disabled={isCancellingCapture}
                          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-full border border-white/16 bg-white/8 px-3 py-1.5 text-[11px] tracking-[0.02em] text-white/78 transition-all hover:bg-white/12 disabled:cursor-not-allowed disabled:opacity-55"
                        >
                          {isCancellingCapture ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <X className="h-3.5 w-3.5" />
                          )}
                          Cancel
                        </button>
                      )}
                  </div>
                )}

              <div className="stemsep-soft-scroll mt-3 max-h-[240px] space-y-2 overflow-y-auto pr-1">
                {remoteItems.length === 0 ? (
                  <div className="rounded-[1.2rem] border border-white/12 bg-white/8 px-4 py-4 text-[13px] text-white/52">
                    {!activeProviderLibrary?.authenticated
                      ? "Authenticate to access your Qobuz library."
                      : !captureEnvironment?.selectedDeviceReady
                        ? "Save a silent output before you capture."
                        : activeProviderLibrary?.loaded
                          ? getProviderEmptyStateText(activeImportTab)
                          : "Load Favorites or search for a track."}
                  </div>
                ) : (
                  remoteItems.map((item) => (
                    (() => {
                      const isThisItemCapturing =
                        capturingProvider === item.provider &&
                        capturingTrackId === item.trackId;
                      const anyCaptureActive = !!capturingProvider;

                      return (
                        <div
                          key={`${item.provider}-${item.trackId}`}
                          className="flex items-center gap-3 rounded-[1.2rem] border border-white/12 bg-white/8 px-3 py-3"
                        >
                          {item.artworkUrl ? (
                            <img
                              src={item.artworkUrl}
                              alt=""
                              className="h-12 w-12 shrink-0 rounded-[0.9rem] object-cover"
                              loading="lazy"
                            />
                          ) : (
                            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-[0.9rem] border border-white/10 bg-white/8 text-white/56">
                              <Music2 className="h-4 w-4" />
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[14px] text-white">
                              {item.title}
                            </div>
                            <div className="truncate text-[12px] text-white/54">
                              {[
                                item.artist,
                                item.album,
                                item.qualityLabel,
                                item.playbackSurface === "desktop_app"
                                  ? "Desktop app"
                                  : "Browser player",
                              ]
                                .filter(Boolean)
                                .join(" · ")}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              void handleCaptureLibraryItem(item);
                            }}
                            disabled={
                              anyCaptureActive ||
                              !captureEnvironment?.selectedDeviceReady
                            }
                            className="inline-flex items-center justify-center gap-2 rounded-[1rem] border border-white/18 bg-white/12 px-3 py-2 text-[13px] tracking-[-0.2px] text-white transition-all hover:bg-white/18 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {isThisItemCapturing ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Radio className="h-4 w-4" />
                            )}
                            Capture & Separate
                          </button>
                        </div>
                      );
                    })()
                  ))
                )}
              </div>
            </>
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
