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
  recommended: boolean;
  is_custom?: boolean;
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
  guide_revision?: string | null;
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
    evidence_url?: string | null;
    evidence_note?: string | null;
    last_verified?: string;
  };
  catalog_status?: "verified" | "candidate" | "blocked" | "manual_only" | "online_only";
  metrics_status?: "verified" | "guide_curated" | "vendor_matched" | "missing_evidence";
  metrics_evidence?: Array<{
    source: string;
    url?: string | null;
    note: string;
    verified_at: string;
  }>;
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
    family?: string;
    artifact_count?: number;
    downloadable_artifact_count?: number;
    source_policy?: string;
    manual_instructions?: string[];
    sources?: Array<{
      role: string;
      url: string;
      host: string;
      manual?: boolean;
    }>;
    artifacts?: Array<{
      kind: string;
      filename: string;
      relative_path: string;
      required: boolean;
      manual: boolean;
      exists?: boolean;
      source?: string | null;
      source_host?: string | null;
      sha256?: string | null;
    }>;
  };
  installation?: {
    installed?: boolean;
    missing_artifacts?: string[];
    relative_paths?: string[];
  };
}

export type Recipe = import("./recipes").Recipe;
export type SourceAudioProfile = import("./media").SourceAudioProfile;
export type StagingDecision = import("./media").StagingDecision;
export type PlaybackMetadata = import("./media").PlaybackMetadata;

export interface HistoryItem {
  id: string;
  backendJobId?: string;
  date: string;
  inputFile: string;
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
    speed?: number;
    eta?: number;
  }) => void;
  completeDownload: (modelId: string) => void;
  setDownloadError: (modelId: string, error: string) => void;
  pauseDownload: (modelId: string) => void;
  resumeDownload: (modelId: string) => void;
  setModelInstalled: (modelId: string, installed: boolean) => void;
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
