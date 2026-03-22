import type {
  ModelCatalogMetadata,
  ModelInstallPolicy,
  ModelSelectionEnvelope,
  ModelSourceKind,
  ModelVerificationMetadata,
} from "./modelCatalog";

export type ModelAvailabilityClass =
  | "direct"
  | "mirror_fallback"
  | "manual_import"
  | "blocked_non_public"
  | "missing_research"
  | string;

export type ModelDownloadState =
  | "idle"
  | "queued"
  | "preflighting"
  | "downloading"
  | "verifying"
  | "paused"
  | "manual_required"
  | "installed"
  | "failed";

export interface ModelAvailability {
  class?: ModelAvailabilityClass;
  reason?: string;
}

export interface ModelDownloadSource {
  role?: string;
  url: string;
  host?: string;
  manual?: boolean;
  channel?: "upstream" | "mirror" | string;
  priority?: number;
  auth?: "none" | "hf_token" | string;
  verified?: boolean;
}

export interface ModelArtifactStatus {
  kind: string;
  filename: string;
  relativePath: string;
  required: boolean;
  manual: boolean;
  exists?: boolean;
  verified?: boolean;
  state?: ModelDownloadState | "missing" | "present" | string;
  source?: string | null;
  sourceHost?: string | null;
  sourceChannel?: string | null;
  sourceAuth?: string | null;
  sizeBytes?: number | null;
  sha256?: string | null;
}

export interface Model {
  id: string;
  name: string;
  architecture: string;
  version: string;
  category: string;
  description: string;
  sdr: number;
  fullness: number;
  bleedless: number;
  vram_required: number;
  speed: string;
  stems: string[];
  file_size: number;
  installed: boolean;
  downloading: boolean;
  downloadPaused?: boolean;
  downloadProgress: number;
  downloadSpeed?: number;
  downloadEta?: number;
  downloadError?: string;
  downloadState?: ModelDownloadState;
  downloadStage?: string;
  downloadMessage?: string;
  downloadCurrentFile?: string;
  downloadCurrentRelativePath?: string;
  downloadCurrentSource?: string;
  downloadVerified?: boolean;
  recommended: boolean;
  is_custom?: boolean;
  availability?: ModelAvailability;
  artifactStatuses?: ModelArtifactStatus[];
  manualInstructions?: string[];
  catalog?: ModelCatalogMetadata;
  catalog_tier?: ModelCatalogMetadata["tier"];
  source_kind?: ModelSourceKind;
  install_policy?: ModelInstallPolicy;
  verification?: ModelVerificationMetadata;
  selection_type?: import("./modelCatalog").CatalogSelectionType;
  selection_id?: string;
  selection_envelope?: ModelSelectionEnvelope;
  recommended_settings?: {
    overlap?: number;
    segment_size?: number;
    chunk_size?: number;
    shifts?: number;
  };
  repo_id?: string;
  chunk_size?: number;
  dim_f?: number;
  dim_t?: number;
  n_fft?: number;
  hop_length?: number;

  // v2 registry metadata (optional - comes from backend registry)
  tags?: string[];
  compatibility?: {
    stemsep_min_version?: string;
    python_min_version?: string;
    engines?: string[];
    devices?: string[];
    os?: string[];
  };
  runtime?: {
    allowed?: boolean | string[];
    preferred?: string;
    variant?: string;
    engine?: 'native_stemsep' | 'msst_builtin' | 'demucs_native' | 'custom_builtin_variant' | string;
    model_type?: string;
    config_ref?: string;
    checkpoint_ref?: string;
    patch_profile?: string;
    install_mode?: 'direct' | 'manual' | 'custom_runtime' | string;
    required?: string[];
    fallbacks?: string[];
    hosts?: string[];
    install_burden?: "low" | "medium" | "high" | string;
    requires_patch?: boolean;
    requires_manual_assets?: boolean;
    required_files?: string[];
    requires_custom_repo_file?: string[];
    blocking_reason?: string;
    adapter?: string;
    requirements?: {
      manual_steps?: string[];
      python_packages?: string[];
      system?: string[];
    };
  };
  phase_fix?: {
    is_valid_reference?: boolean;
    reference_model_id?: string;
    references?: Record<string, string[]>;
    recommended_params?: {
      lowHz?: number;
      highHz?: number;
      highFreqWeight?: number;
    };
  };
  artifacts?: {
    primary?: {
      url?: string;
      filename?: string;
      sha256?: string;
    };
    extra?: Array<{
      url?: string;
      filename?: string;
      sha256?: string;
    }>;
  };
  guide_rank?: number | null;
  guide_notes?: string[];
  status?: {
    readiness?: "verified" | "experimental" | "manual" | "blocked";
    simple_allowed?: boolean;
    blocking_reason?: string;
    curated?: boolean;
    support_tier?: "curated" | "supported_advanced";
  };
  quality_profile?: {
    target_roles?: Array<
      | "vocals"
      | "instrumental"
      | "karaoke"
      | "restoration"
      | "multi_stem"
      | "drums"
      | "bass"
    >;
    quality_tier?: "fast" | "balanced" | "quality" | "ultra" | null;
    deterministic_priority?: number | null;
  };
  hardware_tiers?: Array<{
    tier?: "low_vram" | "mid_vram" | "high_vram" | "cpu_only";
    min_vram_gb?: number | null;
    max_vram_gb?: number | null;
    recommended_segment_size?: number | null;
    recommended_overlap?: number | null;
    notes?: string;
  }>;
  stability_notes?: string[];
  card_metrics?: {
    kind?: "standard" | "special" | "status";
    primary_target?: string;
    labels?: string[];
    values?: Array<number | string | null>;
    source?: string;
    evidence_note?: string | null;
    last_verified?: string;
  };
  catalog_status?: "verified" | "candidate" | "blocked" | "manual_only" | "online_only";
  metrics_status?: "verified" | "guide_curated" | "vendor_matched" | "missing_evidence";
  install?: {
    mode?: "direct" | "manual" | "custom_runtime";
    notes?: string[];
  };
  quality_role?:
    | "primary"
    | "ensemble_partner"
    | "phase_reference"
    | "post_process"
    | "special_stem"
    | Array<
        | "primary"
        | "ensemble_partner"
        | "phase_reference"
        | "post_process"
        | "special_stem"
      >;
  best_for?: string[];
  artifacts_risk?: string[];
  vram_profile?: "cpu_only" | "low_vram" | "mid_vram" | "high_vram" | string;
  chunk_overlap_policy?: {
    default_segment_size?: number | null;
    default_overlap?: number | null;
    notes?: string[];
  };
  workflow_groups?: string[];
  quality_axes?: Record<string, number | string | null>;
  workflow_roles?: string[];
  operating_profiles?: Record<
    string,
    {
      segment_size?: number | null;
      overlap?: number | null;
      batch_size?: number | null;
      shifts?: number | null;
      notes?: string[];
    }
  >;
  content_fit?: string[];
  download?: {
    mode?: "direct" | "multi_artifact_direct" | "manual" | "unavailable" | string;
    install_mode?: "direct" | "manual" | "custom_runtime" | string;
    strategy?: string;
    family?: string;
    artifact_count?: number;
    downloadable_artifact_count?: number;
    source_policy?: string;
    manual_instructions?: string[];
    sources?: ModelDownloadSource[];
    artifacts?: Array<{
      kind: string;
      filename: string;
      relative_path: string;
      required: boolean;
      manual: boolean;
      exists?: boolean;
      verified?: boolean;
      source?: string | null;
      source_host?: string | null;
      source_channel?: string | null;
      source_auth?: string | null;
      sha256?: string | null;
      size_bytes?: number | null;
      sources?: Array<{
        url: string;
        channel?: "upstream" | "mirror" | string;
        priority?: number;
        auth?: "none" | "hf_token" | string;
        verified?: boolean;
      }>;
    }>;
  };
  installation?: {
    installed?: boolean;
    missing_artifacts?: string[];
    relative_paths?: string[];
    artifacts?: Array<{
      kind?: string;
      filename?: string;
      relative_path?: string;
      required?: boolean;
      exists?: boolean;
      verified?: boolean;
      state?: string;
      source?: string | null;
      source_host?: string | null;
      source_channel?: string | null;
      source_auth?: string | null;
      size_bytes?: number | null;
      sha256?: string | null;
    }>;
  };
}

export type Recipe = import("./recipes").Recipe;
export type SourceAudioProfile = import("./media").SourceAudioProfile;
export type StagingDecision = import("./media").StagingDecision;
export type PlaybackMetadata = import("./media").PlaybackMetadata;
export type SourceType = import("./remote").SourceType;
export type RemoteSourceProvider = import("./remote").RemoteSourceProvider;
export type IngestMode = import("./remote").IngestMode;
export type PlaybackSurface = import("./remote").PlaybackSurface;
export type CaptureQualityMode = import("./remote").CaptureQualityMode;

export interface QueueSourceMeta {
  provider?: RemoteSourceProvider;
  providerTrackId?: string;
  title?: string;
  artist?: string;
  album?: string;
  channel?: string;
  durationSec?: number;
  artworkUrl?: string;
  thumbnailUrl?: string;
  canonicalUrl?: string;
  qualityLabel?: string;
  isLossless?: boolean;
  downloadOrigin?: string;
  ingestMode?: IngestMode;
  playbackSurface?: PlaybackSurface;
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
  captureDeviceId?: string;
  captureSampleRate?: number;
  captureChannels?: number;
  captureStartAt?: string;
  captureEndAt?: string;
}

export interface HistoryExportSummary {
  status: "exported";
  exportedAt: string;
  exportDir: string;
  format: "wav" | "flac" | "mp3";
  exportedFiles: Record<string, string>;
}

export type ResultAvailabilityStatus =
  | "preview_ready"
  | "preview_only"
  | "exported"
  | "playback_issue"
  | "missing_source"
  | "failed";

export interface HistoryItem {
  id: string;
  backendJobId?: string;
  date: string;
  inputFile: string;
  displayName?: string;
  sourceUrl?: string;
  sourceType?: SourceType;
  sourceMeta?: QueueSourceMeta;
  outputDir: string;
  modelId: string;
  modelName: string;
  preset?: { id: string; name: string };
  status: "completed" | "failed";
  duration?: number;
  outputFiles?: Record<string, string>;
  isFavorite?: boolean;
  sourceAudioProfile?: SourceAudioProfile;
  stagingDecision?: StagingDecision;
  playback?: PlaybackMetadata;
  exportSummary?: HistoryExportSummary;
  settings: {
    overlap?: number;
    segmentSize?: number;
    stems?: string[];
  };
}

export interface SeparationState {
  isProcessing: boolean;
  isPaused: boolean;
  progress: number;
  message: string;
  logs: string[];
  outputFiles: Record<string, string> | null;
  error: string | null;
  startTime?: number;
  queue: QueueItem[];
}

export interface QueueItem {
  id: string;
  backendJobId?: string;
  file: string;
  displayName?: string;
  sourceUrl?: string;
  sourceType?: SourceType;
  sourceMeta?: QueueSourceMeta;
  status:
    | "pending"
    | "processing"
    | "completed"
    | "failed"
    | "cancelled"
    | "queued";
  outputFiles?: Record<string, string>;
  error?: string;
  device?: string;
  progress?: number;
  message?: string; // Progress message (Loading model, Processing chunk X/Y, Finalizing...)
  startTime?: number; // For ETR calculation
  lastProgressTime?: number; // Timestamp of the last progress/progress-like update (for UI reassurance)
  modelId?: string;
  activePhase?: string;
  activeStepId?: string;
  activeStepLabel?: string;
  activeStepIndex?: number;
  activeStepCount?: number;
  activeModelId?: string;
  chunksDone?: number;
  chunksTotal?: number;
  lastStepDurationMs?: number;
}

export interface PhaseParams {
  enabled: boolean;
  lowHz: number;
  highHz: number;
  highFreqWeight: number;
}

export interface SettingsState {
  theme: "light" | "dark" | "system";
  defaultOutputDir?: string;
  defaultExportDir?: string;
  modelsDir?: string;
  defaultModelId?: string;
  phaseParams: PhaseParams;
  advancedSettings?: {
    device?: "auto" | "cpu" | "cuda" | string;
    /**
     * Persist a specific CUDA device (e.g. "cuda:0", "cuda:1") so users can set a
     * stable default GPU choice rather than relying on "cuda" or "auto".
     */
    preferredCudaDevice?: string;
    shifts: number;
    overlap: number;
    segmentSize: number;
    outputFormat: "wav" | "mp3" | "flac";
    bitrate: string;
  };
}

// Forward declaration for slices to use
// Note: Slices import types from here. To avoid circular dependency on AppState,
// we define AppState interface parts here or use partials.
// But AppState is composition of slices.
// Let's define a BaseAppState or just not export AppState fully here if not needed by slices.
// Slices usually need AppState for set/get types.
// So we define AppState as `any` or generic here, or define it in a separate file.
// Circular dep: store.ts -> slices -> store.ts
// Solution: Define `AppState` in `store.ts` (this file) without importing slices.
// Just define the shape if possible, or use `any`.
// Or better: Slices shouldn't depend on `AppState`. They depend on `StateCreator`.
// `StateCreator` takes the full state type.
// Let's move `AppState` definition to `useStore.ts` (where slices are combined), and slices import it from there? No, circular.
// Let's define `AppState` here in `types/store.ts` but without importing slices values, just types?
// We can't import types from slices if slices import from here.
// Solution: Put ALL interfaces (including Slice interfaces) in `types/store.ts`.

// Slice interfaces:

export interface ModelSlice {
  models: Model[];
  recipes: Recipe[];
  setModels: (models: Model[]) => void;
  setRecipes: (recipes: Recipe[]) => void;
  startDownload: (modelId: string) => void;
  setDownloadProgress: (data: {
    modelId: string;
    progress: number;
    stage?: string;
    artifactIndex?: number;
    artifactCount?: number;
    currentFile?: string;
    currentRelativePath?: string;
    currentSource?: string;
    verified?: boolean;
    message?: string;
    speed?: number;
    eta?: number;
  }) => void;
  completeDownload: (modelId: string) => void;
  setDownloadError: (modelId: string, error: string) => void;
  pauseDownload: (modelId: string) => void;
  resumeDownload: (modelId: string) => void;
  setModelInstalled: (modelId: string, installed: boolean) => void;
  upsertModel: (model: Model) => void;
  mergeModel: (modelId: string, patch: Partial<Model>) => void;
}

export interface SeparationSlice {
  separation: SeparationState;
  setQueue: (queue: QueueItem[]) => void;
  addToQueue: (items: QueueItem[]) => void;
  removeFromQueue: (id: string) => void;
  updateQueueItem: (id: string, updates: Partial<QueueItem>) => void;
  clearQueue: () => void;
  startSeparation: () => void;
  cancelSeparation: () => void;
  pauseQueue: () => void;
  resumeQueue: () => void;
  reorderQueue: (jobIds: string[]) => void;
  setSeparationProgress: (progress: number, message: string) => void;
  addLog: (message: string) => void;
  completeSeparation: (outputFiles: Record<string, string>) => void;
  failSeparation: (error: string) => void;
  clearSeparation: () => void;
  loadQueue: () => Promise<void>;
}

export interface SettingsSlice {
  history: HistoryItem[];
  settings: SettingsState;
  sessionToLoad: HistoryItem | null;
  watchModeEnabled: boolean;
  watchPath: string;

  addToHistory: (item: Omit<HistoryItem, "id" | "date">) => void;
  updateHistoryItem: (id: string, patch: Partial<HistoryItem>) => void;
  removeFromHistory: (id: string) => void;
  toggleHistoryFavorite: (id: string) => void;
  clearHistory: () => void;
  loadSession: (item: HistoryItem) => void;
  clearLoadedSession: () => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
  setDefaultOutputDir: (path: string) => void;
  setDefaultExportDir: (path: string) => void;
  setModelsDir: (path: string) => void;
  setDefaultModel: (modelId: string) => void;
  setAdvancedSettings: (
    settings: Partial<SettingsState["advancedSettings"]>,
  ) => void;
  setWatchMode: (enabled: boolean) => void;
  setWatchPath: (path: string) => void;
  setPhaseParams: (params: PhaseParams) => void;
}

export type AppState = ModelSlice & SeparationSlice & SettingsSlice;
