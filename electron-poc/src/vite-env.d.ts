/// <reference types="vite/client" />

type VolumeCompensation = import('./types/separation').VolumeCompensation
type Recipe = import('./types/recipes').Recipe

interface ElectronAPI {
    selectOutputDirectory: () => Promise<string | null>
    openAudioFileDialog: () => Promise<string[] | null>
    openModelFileDialog: () => Promise<string[] | null>
    scanDirectory: (folderPath: string) => Promise<string[]>
    separationPreflight: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        stems?: string[],
        device?: string,
        overlap?: number,
        segmentSize?: number,
        shifts?: number,
        outputFormat?: string,
        bitrate?: string,
        tta?: boolean,
        ensembleConfig?: any,
        ensembleAlgorithm?: string,
        invert?: boolean,
        splitFreq?: number,
        phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number },
        postProcessingSteps?: any[],
        volumeCompensation?: VolumeCompensation
    ) => Promise<any>
    separateAudio: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        stems?: string[],
        device?: string,
        overlap?: number,
        segmentSize?: number,
        shifts?: number,
        outputFormat?: string,
        bitrate?: string,
        tta?: boolean,
        ensembleConfig?: any,
        ensembleAlgorithm?: string,
        invert?: boolean,
        splitFreq?: number,
        phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number },
        postProcessingSteps?: any[],
        volumeCompensation?: VolumeCompensation
    ) => Promise<{ success: boolean; outputFiles: Record<string, string>; jobId?: string; error?: string }>
    cancelSeparation: (jobId: string) => Promise<any>
    saveJobOutput: (jobId: string) => Promise<{ success: boolean; outputFiles?: Record<string, string>; error?: string }>
    discardJobOutput: (jobId: string) => Promise<{ success: boolean; error?: string }>
    pauseQueue: () => Promise<void>
    resumeQueue: () => Promise<void>
    reorderQueue: (jobIds: string[]) => Promise<void>
    exportOutput: (jobId: string, exportPath: string, format: string, bitrate: string) => Promise<{ success: boolean; error?: string }>
    exportFiles: (sourceFiles: Record<string, string>, exportPath: string, format: string, bitrate: string) => Promise<{ status: string; path: string }>
    onSeparationProgress: (callback: (data: { progress: number; message: string; jobId?: string }) => void) => () => void
    onSeparationStarted: (callback: (data: { jobId: string }) => void) => () => void
    onSeparationComplete: (callback: (data: { outputFiles: Record<string, string> }) => void) => () => void
    onSeparationError: (callback: (data: { error: string }) => void) => () => void

    // Model operations
    getModels: () => Promise<any[]>
    getModelTech: (modelId: string) => Promise<any>
    getRecipes: () => Promise<Recipe[]>
    downloadModel: (modelId: string) => Promise<boolean>
    pauseDownload: (modelId: string) => Promise<any>
    resumeDownload: (modelId: string) => Promise<any>
    removeModel: (modelId: string) => Promise<any>
    importCustomModel: (filePath: string, modelName: string, architecture?: string) => Promise<any>
    openFolder: (folderPath: string) => Promise<void>
    readAudioFile: (filePath: string) => Promise<{ success: boolean; data?: string; mimeType?: string; error?: string }>

    // YouTube URL -> local temp audio file
    resolveYouTubeUrl: (url: string) => Promise<{ file_path: string; title: string; source_url?: string }>
    onYouTubeProgress: (callback: (data: { status: string; percent?: string; speed?: string; eta?: string; error?: string }) => void) => () => void
    getGpuDevices: () => Promise<any>
    getWorkflows: () => Promise<{ workflows: Record<string, any> }>
    checkPresetModels: (presetMappings: Record<string, string>) => Promise<Record<string, boolean>>
    onDownloadProgress: (callback: (data: { modelId: string; progress: number }) => void) => () => void
    onDownloadComplete: (callback: (data: { modelId: string }) => void) => () => void
    onDownloadError: (callback: (data: { modelId: string; error: string }) => void) => () => void
    onDownloadPaused: (callback: (data: { modelId: string }) => void) => () => void
    openDirectoryDialog: () => Promise<{ canceled: boolean; filePaths: string[] }>
    onBackendError: (callback: (data: { error: string }) => void) => () => void
    onBridgeReady: (callback: (data: { capabilities: string[]; modelsCount: number; recipesCount: number }) => void) => () => void
    saveQueue: (queue: any[]) => Promise<{ success: boolean; error?: string }>
    loadQueue: () => Promise<any[] | null>
    startWatchMode: (path: string) => Promise<boolean>
    stopWatchMode: () => Promise<boolean>
    onWatchFileDetected: (callback: (path: string) => void) => () => void

    // App Config
    saveAppConfig: (config: Record<string, any>) => Promise<boolean>
    getAppConfig: () => Promise<Record<string, any>>

    // Hugging Face auth (optional)
    setHuggingFaceToken: (token: string) => Promise<{ success: boolean; error?: string }>
    clearHuggingFaceToken: () => Promise<{ success: boolean; error?: string }>
    getHuggingFaceAuthStatus: () => Promise<{ configured: boolean }>

    // External links
    openExternalUrl: (url: string) => Promise<boolean>
}

interface Window {
    electronAPI: ElectronAPI
}
