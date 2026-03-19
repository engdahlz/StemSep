import { contextBridge, ipcRenderer } from 'electron'

type PreloadWorkflow = Record<string, any>
type PreloadRuntimePolicy = Record<string, any>
type PreloadExportPolicy = Record<string, any>

contextBridge.exposeInMainWorld('electronAPI', {
  // Audio file operations
  openAudioFileDialog: () => ipcRenderer.invoke('open-audio-file-dialog'),
  openModelFileDialog: () => ipcRenderer.invoke('open-model-file-dialog'),
  selectOutputDirectory: () => ipcRenderer.invoke('select-output-directory'),
  selectModelsDirectory: () => ipcRenderer.invoke('select-models-directory'),
  openFolder: (folderPath: string) => ipcRenderer.invoke('open-folder', folderPath),
  checkFileExists: (filePath: string) => ipcRenderer.invoke('check-file-exists', filePath),
  readAudioFile: (filePath: string) => ipcRenderer.invoke('read-audio-file', filePath),
  resolvePlaybackStems: (outputFiles: Record<string, string>, playback?: Record<string, any>) =>
    ipcRenderer.invoke('resolve-playback-stems', { outputFiles, playback }),
  resolveMediaUrl: (filePath: string) => {
    if (/^(blob:|data:|https?:)/i.test(filePath)) return filePath
    const normalized = String(filePath || '').replace(/\\/g, '/')
    if (/^[A-Za-z]:\//.test(normalized)) {
      return `media:///${encodeURI(normalized)}`
    }
    return `media://${encodeURI(normalized)}`
  },

  authRemoteSource: (provider: 'qobuz' | 'bandcamp') =>
    ipcRenderer.invoke('auth-remote-source', { provider }),
  searchRemoteCatalog: (provider: 'qobuz' | 'bandcamp', query: string) =>
    ipcRenderer.invoke('search-remote-catalog', { provider, query }),
  listRemoteCollection: (provider: 'qobuz' | 'bandcamp', scope?: string) =>
    ipcRenderer.invoke('list-remote-collection', { provider, scope }),
  resolveRemoteTrack: (provider: 'qobuz' | 'bandcamp', trackId: string, variantId?: string) =>
    ipcRenderer.invoke('resolve-remote-track', { provider, trackId, variantId }),
  onRemoteResolveProgress: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('remote-resolve-progress', handler)
    return () => ipcRenderer.removeListener('remote-resolve-progress', handler)
  },
  detectPlaybackDevices: () => ipcRenderer.invoke('detect-playback-devices'),
  getCaptureEnvironmentStatus: () => ipcRenderer.invoke('get-capture-environment-status'),
  setCaptureOutputDevice: (deviceId: string) =>
    ipcRenderer.invoke('set-capture-output-device', { deviceId }),
  authLibraryProvider: (provider: 'spotify' | 'qobuz') =>
    ipcRenderer.invoke('auth-library-provider', { provider }),
  getLibraryAuthStatus: (provider: 'spotify' | 'qobuz') =>
    ipcRenderer.invoke('get-library-auth-status', { provider }),
  searchLibrary: (provider: 'spotify' | 'qobuz', query: string) =>
    ipcRenderer.invoke('search-library', { provider, query }),
  listLibraryCollection: (provider: 'spotify' | 'qobuz', scope?: string) =>
    ipcRenderer.invoke('list-library-collection', { provider, scope }),
  preparePlaybackCapture: (provider: 'spotify' | 'qobuz', trackId: string) =>
    ipcRenderer.invoke('prepare-playback-capture', { provider, trackId }),
  startPlaybackCapture: (
    provider: 'spotify' | 'qobuz',
    trackId: string,
    deviceId: string,
  ) => ipcRenderer.invoke('start-playback-capture', { provider, trackId, deviceId }),
  cancelPlaybackCapture: (captureId?: string) =>
    ipcRenderer.invoke('cancel-playback-capture', { captureId }),
  onPlaybackCaptureProgress: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('playback-capture-progress', handler)
    return () => ipcRenderer.removeListener('playback-capture-progress', handler)
  },

  // YouTube URL -> local temp audio file
  resolveYouTubeUrl: (url: string) => ipcRenderer.invoke('resolve-youtube-url', { url }),
  onYouTubeProgress: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('youtube-progress', handler)
    return () => ipcRenderer.removeListener('youtube-progress', handler)
  },

  separationPreflight: (inputFile: string, modelId: string, outputDir: string, stems?: string[], device?: string, overlap?: number, segmentSize?: number, shifts?: number, outputFormat?: string, bitrate?: string, tta?: boolean, ensembleConfig?: any, ensembleAlgorithm?: string, invert?: boolean, splitFreq?: number, phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }, postProcessingSteps?: any[], volumeCompensation?: { enabled: boolean; stage?: 'export' | 'blend' | 'both'; dbPerExtraModel?: number }, pipelineConfig?: any[], workflow?: PreloadWorkflow, runtimePolicy?: PreloadRuntimePolicy, exportPolicy?: PreloadExportPolicy) =>
    ipcRenderer.invoke('separation-preflight', { inputFile, modelId, outputDir, stems, device, overlap, segmentSize, shifts, outputFormat, bitrate, tta, ensembleConfig, ensembleAlgorithm, invert, splitFreq, phaseParams, postProcessingSteps, volumeCompensation, pipelineConfig, workflow, runtimePolicy, exportPolicy }),

  separateAudio: (inputFile: string, modelId: string, outputDir: string, stems?: string[], device?: string, overlap?: number, segmentSize?: number, shifts?: number, outputFormat?: string, bitrate?: string, tta?: boolean, ensembleConfig?: any, ensembleAlgorithm?: string, invert?: boolean, splitFreq?: number, phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }, postProcessingSteps?: any[], volumeCompensation?: { enabled: boolean; stage?: 'export' | 'blend' | 'both'; dbPerExtraModel?: number }, pipelineConfig?: any[], workflow?: PreloadWorkflow, runtimePolicy?: PreloadRuntimePolicy, exportPolicy?: PreloadExportPolicy) =>
    ipcRenderer.invoke('separate-audio', { inputFile, modelId, outputDir, stems, device, overlap, segmentSize, shifts, outputFormat, bitrate, tta, ensembleConfig, ensembleAlgorithm, invert, splitFreq, phaseParams, postProcessingSteps, volumeCompensation, pipelineConfig, workflow, runtimePolicy, exportPolicy }),
  cancelSeparation: (jobId: string) => ipcRenderer.invoke('cancel-separation', jobId),
  saveJobOutput: (jobId: string) => ipcRenderer.invoke('save-job-output', jobId),
  discardJobOutput: (jobId: string) => ipcRenderer.invoke('discard-job-output', jobId),
  exportOutput: (jobId: string, exportPath: string, format: string, bitrate: string, requestId?: string) =>
    ipcRenderer.invoke('export-output', { jobId, exportPath, format, bitrate, requestId }),
  pauseQueue: () => ipcRenderer.invoke('pause-queue'),
  resumeQueue: () => ipcRenderer.invoke('resume-queue'),
  reorderQueue: (jobIds: string[]) => ipcRenderer.invoke('reorder-queue', jobIds),
  exportFiles: (sourceFiles: Record<string, string>, exportPath: string, format: string, bitrate: string, requestId?: string) =>
    ipcRenderer.invoke('export-files', { sourceFiles, exportPath, format, bitrate, requestId }),
  onSeparationProgress: (callback: (data: { progress: number, message: string, jobId?: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-progress', handler)
    return () => ipcRenderer.removeListener('separation-progress', handler)
  },
  onSeparationEvent: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-progress-event', handler)
    return () => ipcRenderer.removeListener('separation-progress-event', handler)
  },
  onSeparationComplete: (callback: (data: { outputFiles: Record<string, string> }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-complete', handler)
    return () => ipcRenderer.removeListener('separation-complete', handler)
  },
  onSeparationStarted: (callback: (data: { jobId: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-started', handler)
    return () => ipcRenderer.removeListener('separation-started', handler)
  },
  onSeparationError: (callback: (data: { error: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-error', handler)
    return () => ipcRenderer.removeListener('separation-error', handler)
  },

  // Model operations
  getModels: () => ipcRenderer.invoke('get-models'),
  getModelTech: (modelId: string) => ipcRenderer.invoke('get-model-tech', modelId),
  resolveModelDownload: (modelId: string) => ipcRenderer.invoke('resolve-model-download', modelId),
  getModelInstallation: (modelId: string) => ipcRenderer.invoke('get-model-installation', modelId),
  getRecipes: () => ipcRenderer.invoke('get-recipes'),
  qualityBaselineCreate: (payload: Record<string, any>) =>
    ipcRenderer.invoke('quality-baseline-create', payload),
  qualityCompare: (payload: Record<string, any>) =>
    ipcRenderer.invoke('quality-compare', payload),
  downloadModel: (modelId: string) => ipcRenderer.invoke('download-model', modelId),
  pauseDownload: (modelId: string) => ipcRenderer.invoke('pause-download', modelId),
  resumeDownload: (modelId: string) => ipcRenderer.invoke('resume-download', modelId),
  importModelFiles: (modelId: string, files: Array<{ kind?: string, path: string }>, allowCopy = true) =>
    ipcRenderer.invoke('import-model-files', { modelId, files, allowCopy }),
  removeModel: (modelId: string) => ipcRenderer.invoke('remove-model', modelId),
  importCustomModel: (filePath: string, modelName: string, architecture?: string) =>
    ipcRenderer.invoke('import-custom-model', { filePath, modelName, architecture }),
  checkPresetModels: (presetMappings: Record<string, string>) =>
    ipcRenderer.invoke('check-preset-models', presetMappings),
  getGpuDevices: () => ipcRenderer.invoke('get-gpu-devices'),
  getSystemRuntimeInfo: () => ipcRenderer.invoke('get-system-runtime-info'),
  getWorkflows: () => ipcRenderer.invoke('get-workflows'),

  // Model download progress
  onDownloadProgress: (callback: (data: { modelId: string, progress: number, stage?: string, artifactIndex?: number, artifactCount?: number, currentFile?: string, currentRelativePath?: string, currentSource?: string, verified?: boolean, message?: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-progress', handler)
    return () => ipcRenderer.removeListener('download-progress', handler)
  },
  onDownloadComplete: (callback: (data: { modelId: string, artifactCount?: number, stage?: string, verified?: boolean }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-complete', handler)
    return () => ipcRenderer.removeListener('download-complete', handler)
  },
  onDownloadError: (callback: (data: { modelId: string, error: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-error', handler)
    return () => ipcRenderer.removeListener('download-error', handler)
  },
  onDownloadPaused: (callback: (data: { modelId: string, artifactIndex?: number, artifactCount?: number, currentFile?: string, currentRelativePath?: string, currentSource?: string, progress?: number, stage?: string, verified?: boolean }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-paused', handler)
    return () => ipcRenderer.removeListener('download-paused', handler)
  },

  // Backend error handling
  onBackendError: (callback: (data: { error: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('backend-error', handler)
    return () => ipcRenderer.removeListener('backend-error', handler)
  },

  // Bridge ready event - signals Python backend is fully initialized
  onBridgeReady: (callback: (data: { capabilities: string[], modelsCount: number, recipesCount: number }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('bridge-ready', handler)
    return () => ipcRenderer.removeListener('bridge-ready', handler)
  },
  onQualityProgress: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('quality-progress', handler)
    return () => ipcRenderer.removeListener('quality-progress', handler)
  },
  onQualityComplete: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('quality-complete', handler)
    return () => ipcRenderer.removeListener('quality-complete', handler)
  },
  onExportProgress: (callback: (data: any) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('export-progress', handler)
    return () => ipcRenderer.removeListener('export-progress', handler)
  },

  // Queue Persistence
  saveQueue: (queueData: any) => ipcRenderer.invoke('save-queue', queueData),
  loadQueue: () => ipcRenderer.invoke('load-queue'),

  // Watch Mode
  startWatchMode: (path: string) => ipcRenderer.invoke('start-watch-mode', path),
  stopWatchMode: () => ipcRenderer.invoke('stop-watch-mode'),
  onWatchFileDetected: (callback: (path: string) => void) => {
    const handler = (_event: any, path: string) => callback(path)
    ipcRenderer.on('watch-file-detected', handler)
    return () => ipcRenderer.removeListener('watch-file-detected', handler)
  },

  // App Config (for settings that main process needs)
  setModelsDir: (modelsDir: string) => ipcRenderer.invoke('set-models-dir', modelsDir),
  saveAppConfig: (config: Record<string, any>) => ipcRenderer.invoke('save-app-config', config),
  getAppConfig: () => ipcRenderer.invoke('get-app-config'),

  // Hugging Face auth (optional)
  setHuggingFaceToken: (token: string) => ipcRenderer.invoke('set-huggingface-token', token),
  clearHuggingFaceToken: () => ipcRenderer.invoke('clear-huggingface-token'),
  getHuggingFaceAuthStatus: () => ipcRenderer.invoke('get-huggingface-auth-status'),

  // External links (safe browser open)
  openExternalUrl: (url: string) => ipcRenderer.invoke('open-external-url', url),
})
