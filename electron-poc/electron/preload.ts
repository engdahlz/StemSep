import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  // Audio file operations
  openAudioFileDialog: () => ipcRenderer.invoke('open-audio-file-dialog'),
  openModelFileDialog: () => ipcRenderer.invoke('open-model-file-dialog'),
  selectOutputDirectory: () => ipcRenderer.invoke('select-output-directory'),
  openFolder: (folderPath: string) => ipcRenderer.invoke('open-folder', folderPath),
  checkFileExists: (filePath: string) => ipcRenderer.invoke('check-file-exists', filePath),
  readAudioFile: (filePath: string) => ipcRenderer.invoke('read-audio-file', filePath),

  // YouTube URL -> local temp audio file
  resolveYouTubeUrl: (url: string) => ipcRenderer.invoke('resolve-youtube-url', { url }),
  onYouTubeProgress: (callback: (data: { status: string, percent?: string, speed?: string, eta?: string, error?: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('youtube-progress', handler)
    return () => ipcRenderer.removeListener('youtube-progress', handler)
  },

  separationPreflight: (inputFile: string, modelId: string, outputDir: string, stems?: string[], device?: string, overlap?: number, segmentSize?: number, shifts?: number, outputFormat?: string, bitrate?: string, tta?: boolean, ensembleConfig?: any, ensembleAlgorithm?: string, invert?: boolean, splitFreq?: number, phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }, postProcessingSteps?: any[], volumeCompensation?: { enabled: boolean; stage?: 'export' | 'blend' | 'both'; dbPerExtraModel?: number }) =>
    ipcRenderer.invoke('separation-preflight', { inputFile, modelId, outputDir, stems, device, overlap, segmentSize, shifts, outputFormat, bitrate, tta, ensembleConfig, ensembleAlgorithm, invert, splitFreq, phaseParams, postProcessingSteps, volumeCompensation }),

  separateAudio: (inputFile: string, modelId: string, outputDir: string, stems?: string[], device?: string, overlap?: number, segmentSize?: number, shifts?: number, outputFormat?: string, bitrate?: string, tta?: boolean, ensembleConfig?: any, ensembleAlgorithm?: string, invert?: boolean, splitFreq?: number, phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }, postProcessingSteps?: any[], volumeCompensation?: { enabled: boolean; stage?: 'export' | 'blend' | 'both'; dbPerExtraModel?: number }) =>
    ipcRenderer.invoke('separate-audio', { inputFile, modelId, outputDir, stems, device, overlap, segmentSize, shifts, outputFormat, bitrate, tta, ensembleConfig, ensembleAlgorithm, invert, splitFreq, phaseParams, postProcessingSteps, volumeCompensation }),
  cancelSeparation: (jobId: string) => ipcRenderer.invoke('cancel-separation', jobId),
  saveJobOutput: (jobId: string) => ipcRenderer.invoke('save-job-output', jobId),
  discardJobOutput: (jobId: string) => ipcRenderer.invoke('discard-job-output', jobId),
  exportOutput: (jobId: string, exportPath: string, format: string, bitrate: string) =>
    ipcRenderer.invoke('export-output', { jobId, exportPath, format, bitrate }),
  pauseQueue: () => ipcRenderer.invoke('pause-queue'),
  resumeQueue: () => ipcRenderer.invoke('resume-queue'),
  reorderQueue: (jobIds: string[]) => ipcRenderer.invoke('reorder-queue', jobIds),
  exportFiles: (sourceFiles: Record<string, string>, exportPath: string, format: string, bitrate: string) =>
    ipcRenderer.invoke('export-files', { sourceFiles, exportPath, format, bitrate }),
  onSeparationProgress: (callback: (data: { progress: number, message: string, jobId?: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('separation-progress', handler)
    return () => ipcRenderer.removeListener('separation-progress', handler)
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
  getRecipes: () => ipcRenderer.invoke('get-recipes'),
  downloadModel: (modelId: string) => ipcRenderer.invoke('download-model', modelId),
  pauseDownload: (modelId: string) => ipcRenderer.invoke('pause-download', modelId),
  resumeDownload: (modelId: string) => ipcRenderer.invoke('resume-download', modelId),
  removeModel: (modelId: string) => ipcRenderer.invoke('remove-model', modelId),
  importCustomModel: (filePath: string, modelName: string, architecture?: string) =>
    ipcRenderer.invoke('import-custom-model', { filePath, modelName, architecture }),
  checkPresetModels: (presetMappings: Record<string, string>) =>
    ipcRenderer.invoke('check-preset-models', presetMappings),
  getGpuDevices: () => ipcRenderer.invoke('get-gpu-devices'),
  getWorkflows: () => ipcRenderer.invoke('get-workflows'),

  // Model download progress
  onDownloadProgress: (callback: (data: { modelId: string, progress: number }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-progress', handler)
    return () => ipcRenderer.removeListener('download-progress', handler)
  },
  onDownloadComplete: (callback: (data: { modelId: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-complete', handler)
    return () => ipcRenderer.removeListener('download-complete', handler)
  },
  onDownloadError: (callback: (data: { modelId: string, error: string }) => void) => {
    const handler = (_event: any, data: any) => callback(data)
    ipcRenderer.on('download-error', handler)
    return () => ipcRenderer.removeListener('download-error', handler)
  },
  onDownloadPaused: (callback: (data: { modelId: string }) => void) => {
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
  saveAppConfig: (config: Record<string, any>) => ipcRenderer.invoke('save-app-config', config),
  getAppConfig: () => ipcRenderer.invoke('get-app-config'),

  // Hugging Face auth (optional)
  setHuggingFaceToken: (token: string) => ipcRenderer.invoke('set-huggingface-token', token),
  clearHuggingFaceToken: () => ipcRenderer.invoke('clear-huggingface-token'),
  getHuggingFaceAuthStatus: () => ipcRenderer.invoke('get-huggingface-auth-status'),

  // External links (safe browser open)
  openExternalUrl: (url: string) => ipcRenderer.invoke('open-external-url', url),
})
