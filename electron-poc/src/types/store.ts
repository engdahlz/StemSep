export interface Model {
  id: string
  name: string
  architecture: string
  version: string
  category: string
  description: string
  sdr: number
  fullness: number
  bleedless: number
  vram_required: number
  speed: string
  stems: string[]
  file_size: number
  installed: boolean
  downloading: boolean
  downloadPaused?: boolean
  downloadProgress: number
  downloadSpeed?: number
  downloadEta?: number
  downloadError?: string
  recommended: boolean
  is_custom?: boolean
  recommended_settings?: {
    overlap?: number
    segment_size?: number
    chunk_size?: number
    shifts?: number
  }
  repo_id?: string
  chunk_size?: number
  dim_f?: number
  dim_t?: number
  n_fft?: number
  hop_length?: number
}

export interface HistoryItem {
  id: string
  backendJobId?: string
  date: string
  inputFile: string
  outputDir: string
  modelId: string
  modelName: string
  preset?: { id: string; name: string }
  status: 'completed' | 'failed'
  duration?: number
  outputFiles?: Record<string, string>
  isFavorite?: boolean
  settings: {
    overlap?: number
    segmentSize?: number
    stems?: string[]
  }
}

export interface SeparationState {
  isProcessing: boolean
  isPaused: boolean
  progress: number
  message: string
  logs: string[]
  outputFiles: Record<string, string> | null
  error: string | null
  startTime?: number
  queue: QueueItem[]
}

export interface QueueItem {
  id: string
  backendJobId?: string
  file: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'queued'
  outputFiles?: Record<string, string>
  error?: string
  device?: string
  progress?: number
  message?: string // Progress message (Loading model, Processing chunk X/Y, Finalizing...)
  startTime?: number // For ETR calculation
  modelId?: string
}

export interface PhaseParams {
  enabled: boolean
  lowHz: number
  highHz: number
  highFreqWeight: number
}

export interface SettingsState {
  theme: 'light' | 'dark' | 'system'
  defaultOutputDir?: string
  defaultExportDir?: string
  modelsDir?: string
  defaultModelId?: string
  phaseParams: PhaseParams
  advancedSettings?: {
    shifts: number
    overlap: number
    segmentSize: number
    outputFormat: 'mp3' | 'wav' | 'flac'
    bitrate: string
  }
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
  models: Model[]
  recipes: any[]
  setModels: (models: Model[]) => void
  setRecipes: (recipes: any[]) => void
  startDownload: (modelId: string) => void
  setDownloadProgress: (data: { modelId: string, progress: number, speed?: number, eta?: number }) => void
  completeDownload: (modelId: string) => void
  setDownloadError: (modelId: string, error: string) => void
  pauseDownload: (modelId: string) => void
  resumeDownload: (modelId: string) => void
  setModelInstalled: (modelId: string, installed: boolean) => void
}

export interface SeparationSlice {
  separation: SeparationState
  setQueue: (queue: QueueItem[]) => void
  addToQueue: (items: QueueItem[]) => void
  removeFromQueue: (id: string) => void
  updateQueueItem: (id: string, updates: Partial<QueueItem>) => void
  clearQueue: () => void
  startSeparation: () => void
  cancelSeparation: () => void
  pauseQueue: () => void
  resumeQueue: () => void
  reorderQueue: (jobIds: string[]) => void
  setSeparationProgress: (progress: number, message: string) => void
  addLog: (message: string) => void
  completeSeparation: (outputFiles: Record<string, string>) => void
  failSeparation: (error: string) => void
  clearSeparation: () => void
  loadQueue: () => Promise<void>
}

export interface SettingsSlice {
  history: HistoryItem[]
  settings: SettingsState
  sessionToLoad: HistoryItem | null
  watchModeEnabled: boolean
  watchPath: string

  addToHistory: (item: Omit<HistoryItem, 'id' | 'date'>) => void
  removeFromHistory: (id: string) => void
  toggleHistoryFavorite: (id: string) => void
  clearHistory: () => void
  loadSession: (item: HistoryItem) => void
  clearLoadedSession: () => void
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  setDefaultOutputDir: (path: string) => void
  setDefaultExportDir: (path: string) => void
  setModelsDir: (path: string) => void
  setDefaultModel: (modelId: string) => void
  setAdvancedSettings: (settings: Partial<SettingsState['advancedSettings']>) => void
  setWatchMode: (enabled: boolean) => void
  setWatchPath: (path: string) => void
  setPhaseParams: (params: PhaseParams) => void
}

export type AppState = ModelSlice & SeparationSlice & SettingsSlice
