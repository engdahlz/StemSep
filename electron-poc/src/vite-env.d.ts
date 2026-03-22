/// <reference types="vite/client" />

type VolumeCompensation = import('./types/separation').VolumeCompensation
type SeparationWorkflow = import('./types/separation').SeparationWorkflow
type WorkflowRuntimePolicy = import('./types/separation').WorkflowRuntimePolicy
type WorkflowExportPolicy = import('./types/separation').WorkflowExportPolicy
type Recipe = import('./types/recipes').Recipe
type SourceAudioProfile = import('./types/media').SourceAudioProfile
type StagingDecision = import('./types/media').StagingDecision
type SeparationProgressEvent = import('./types/media').SeparationProgressEvent
type ExportProgressEvent = import('./types/media').ExportProgressEvent
type PlaybackMetadata = import('./types/media').PlaybackMetadata
type RemoteLibraryProvider = import('./types/remote').RemoteLibraryProvider
type ActiveLibraryProvider = import('./types/remote').ActiveLibraryProvider
type RemoteCatalogItem = import('./types/remote').RemoteCatalogItem
type RemoteAuthResult = import('./types/remote').RemoteAuthResult
type RemoteCatalogResult = import('./types/remote').RemoteCatalogResult
type RemoteResolveResult = import('./types/remote').RemoteResolveResult
type RemoteResolveProgressPayload = import('./types/remote').RemoteResolveProgressPayload
type PlaybackDevice = import('./types/remote').PlaybackDevice
type CaptureEnvironmentStatus = import('./types/remote').CaptureEnvironmentStatus
type PlaybackCapturePrepareResult = import('./types/remote').PlaybackCapturePrepareResult
type PlaybackCaptureCompleteResult = import('./types/remote').PlaybackCaptureCompleteResult
type PlaybackCaptureProgressPayload = import('./types/remote').PlaybackCaptureProgressPayload
type CatalogRuntimeManifest = import('./types/modelCatalog').CatalogRuntimeManifest
type CatalogSelectionType = import('./types/modelCatalog').CatalogSelectionType
type SelectionInstallPlan = import('./types/modelCatalog').SelectionInstallPlan
type ModelSelectionEnvelope = import('./types/modelCatalog').ModelSelectionEnvelope
type CatalogRuntimeManifest = import('./types/modelCatalog').CatalogRuntimeManifest
type SelectionInstallPlan = import('./types/modelCatalog').SelectionInstallPlan
type CatalogSelectionType = import('./types/modelCatalog').CatalogSelectionType
type ModelSelectionEnvelope = import('./types/modelCatalog').ModelSelectionEnvelope

type MissingAudioCode = 'MISSING_CACHE_FILE' | 'STALE_SESSION' | 'MISSING_SOURCE_FILE'

type ReadAudioFileResult =
    | { success: true; data: string; mimeType: string; resolvedPath?: string }
    | { success: false; error: string; code?: MissingAudioCode; hint?: string }

type ExportFilesResult =
    | {
        success: true
        exported: Record<string, string>
        requestId?: string
        outputFiles?: Record<string, string>
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }
    | { success: false; error: string; code?: MissingAudioCode; hint?: string; requestId?: string }

interface SystemRuntimeInfo {
    fetchedAt: string
    cache?: {
        ttlMs?: number
        gpuSource?: 'cache' | 'fresh' | string
        runtimeFingerprintSource?: 'cache' | 'fresh' | 'error' | string
    }
    gpu?: any
    runtimeFingerprint?: {
        version?: string
        platform?: string
        torch?: {
            version?: string
            cuda_available?: boolean
            cuda_device_count?: number
            cuda_device_name_0?: string
        }
        neuralop?: {
            version?: string
            fno1d_import_ok?: boolean
            fno1d_import_error?: string
            fno_import_ok?: boolean
            fno_import_error?: string
        }
    } | null
    runtimeFingerprintError?: string | null
    previewCachePolicy?: {
        baseDir?: string
        keepLast?: number
        maxAgeDays?: number
        ephemeral?: boolean
    }
}

interface ElectronAPI {
    selectOutputDirectory: () => Promise<string | null>
    selectModelsDirectory: () => Promise<string | null>
    openAudioFileDialog: () => Promise<string[] | null>
    openModelFileDialog: () => Promise<string[] | null>
    scanDirectory: (folderPath: string) => Promise<string[]>
    separationPreflight: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        selectionType?: CatalogSelectionType,
        selectionId?: string,
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
        volumeCompensation?: VolumeCompensation,
        pipelineConfig?: SeparationWorkflow['steps'],
        workflow?: SeparationWorkflow,
        runtimePolicy?: WorkflowRuntimePolicy,
        exportPolicy?: WorkflowExportPolicy,
        selectionEnvelope?: ModelSelectionEnvelope
    ) => Promise<any & { sourceAudioProfile?: SourceAudioProfile; stagingDecision?: StagingDecision }>
    separateAudio: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        selectionType?: CatalogSelectionType,
        selectionId?: string,
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
        volumeCompensation?: VolumeCompensation,
        pipelineConfig?: SeparationWorkflow['steps'],
        workflow?: SeparationWorkflow,
        runtimePolicy?: WorkflowRuntimePolicy,
        exportPolicy?: WorkflowExportPolicy,
        selectionEnvelope?: ModelSelectionEnvelope
    ) => Promise<{
        success: boolean
        outputFiles: Record<string, string>
        jobId?: string
        error?: string
        outputDir?: string
        playbackSourceKind?: string
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }>
    cancelSeparation: (jobId: string) => Promise<any>
    saveJobOutput: (jobId: string) => Promise<{
        success: boolean
        outputFiles?: Record<string, string>
        error?: string
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }>
    discardJobOutput: (jobId: string) => Promise<{ success: boolean; error?: string }>
    pauseQueue: () => Promise<void>
    resumeQueue: () => Promise<void>
    reorderQueue: (jobIds: string[]) => Promise<void>
    exportOutput: (jobId: string, exportPath: string, format: string, bitrate: string, requestId?: string) => Promise<ExportFilesResult>
    exportFiles: (sourceFiles: Record<string, string>, exportPath: string, format: string, bitrate: string, requestId?: string) => Promise<ExportFilesResult>
    onSeparationProgress: (callback: (data: { progress: number; message: string; jobId?: string; meta?: Record<string, any> }) => void) => () => void
    onSeparationEvent: (callback: (data: SeparationProgressEvent) => void) => () => void
    onSeparationStarted: (callback: (data: { jobId: string }) => void) => () => void
    onSeparationComplete: (callback: (data: { outputFiles: Record<string, string> }) => void) => () => void
    onSeparationError: (callback: (data: { error: string }) => void) => () => void
    onExportProgress: (callback: (data: ExportProgressEvent) => void) => () => void

    // Model operations
    getCatalog: () => Promise<CatalogRuntimeManifest>
    getModels: () => Promise<any[]>
    getModelTech: (modelId: string) => Promise<any>
    resolveModelDownload: (modelId: string) => Promise<any>
    getModelInstallation: (modelId: string) => Promise<any>
    getSelectionInstallation: (
        selectionType: CatalogSelectionType,
        selectionId: string
    ) => Promise<SelectionInstallPlan>
    resolveInstallPlan: (
        selectionType: CatalogSelectionType,
        selectionId: string
    ) => Promise<SelectionInstallPlan>
    installSelection: (
        selectionType: CatalogSelectionType,
        selectionId: string,
        options?: Record<string, any>
    ) => Promise<any>
    importSelectionArtifacts: (
        selectionType: CatalogSelectionType,
        selectionId: string,
        files: Array<{ kind?: string; path: string }>,
        allowCopy?: boolean
    ) => Promise<any>
    verifySelectionArtifacts: (
        selectionType: CatalogSelectionType,
        selectionId: string
    ) => Promise<any>
    resolveExecutionPlan: (
        selectionType: CatalogSelectionType,
        selectionId: string,
        config?: Record<string, any>
    ) => Promise<any>
    runSelectionJob: (payload: Record<string, any>) => Promise<any>
    cancelSelectionJob: (jobId: string) => Promise<any>
    getSelectionJob: (jobId: string) => Promise<any>
    listSelectionJobs: () => Promise<any[]>
    exportSelectionJob: (jobId: string, exportPath: string) => Promise<any>
    discardSelectionJob: (jobId: string) => Promise<any>
    importModelFiles: (modelId: string, files: Array<{ kind?: string; path: string }>, allowCopy?: boolean) => Promise<any>
    getRecipes: () => Promise<Recipe[]>
    qualityBaselineCreate: (payload: Record<string, any>) => Promise<any>
    qualityCompare: (payload: Record<string, any>) => Promise<any>
    downloadModel: (modelId: string) => Promise<boolean>
    pauseDownload: (modelId: string) => Promise<any>
    resumeDownload: (modelId: string) => Promise<any>
    removeModel: (modelId: string) => Promise<any>
    importCustomModel: (filePath: string, modelName: string, architecture?: string) => Promise<any>
    openFolder: (folderPath: string) => Promise<void>
    checkFileExists: (filePath: string) => Promise<boolean>
    readAudioFile: (filePath: string) => Promise<ReadAudioFileResult>
    resolvePlaybackStems: (
        outputFiles: Record<string, string>,
        playback?: PlaybackMetadata
    ) => Promise<{
        success: boolean
        stems: Record<string, string>
        issues: Record<string, { code?: MissingAudioCode; hint?: string; originalPath?: string }>
        error?: string
    }>

    authRemoteSource: (provider: RemoteLibraryProvider) => Promise<RemoteAuthResult>
    searchRemoteCatalog: (provider: RemoteLibraryProvider, query: string) => Promise<RemoteCatalogResult>
    listRemoteCollection: (
        provider: RemoteLibraryProvider,
        scope?: string
    ) => Promise<RemoteCatalogResult>
    resolveRemoteTrack: (
        provider: RemoteLibraryProvider,
        trackId: string,
        variantId?: string
    ) => Promise<RemoteResolveResult>
    onRemoteResolveProgress: (
        callback: (data: RemoteResolveProgressPayload) => void
    ) => () => void
    detectPlaybackDevices: () => Promise<PlaybackDevice[]>
    getCaptureEnvironmentStatus: () => Promise<CaptureEnvironmentStatus>
    setCaptureOutputDevice: (
        deviceId: string
    ) => Promise<{ success: boolean; error?: string; deviceId?: string; label?: string }>
    authLibraryProvider: (provider: ActiveLibraryProvider) => Promise<RemoteAuthResult>
    getLibraryAuthStatus: (provider: ActiveLibraryProvider) => Promise<RemoteAuthResult>
    searchLibrary: (provider: ActiveLibraryProvider, query: string) => Promise<RemoteCatalogResult>
    listLibraryCollection: (
        provider: ActiveLibraryProvider,
        scope?: string
    ) => Promise<RemoteCatalogResult>
    preparePlaybackCapture: (
        provider: ActiveLibraryProvider,
        trackId: string
    ) => Promise<PlaybackCapturePrepareResult>
    startPlaybackCapture: (
        provider: ActiveLibraryProvider,
        trackId: string,
        deviceId: string
    ) => Promise<PlaybackCaptureCompleteResult>
    cancelPlaybackCapture: (captureId?: string) => Promise<{ success: boolean; error?: string }>
    getPlaybackCaptureStatus: (captureId?: string) => Promise<any>
    onPlaybackCaptureProgress: (
        callback: (data: PlaybackCaptureProgressPayload) => void
    ) => () => void

    // YouTube URL -> local temp audio file
    resolveYouTubeUrl: (url: string) => Promise<
        | {
            success: true
            file_path: string
            title: string
            source_url?: string
            channel?: string
            duration_sec?: number
            thumbnail_url?: string
            canonical_url?: string
          }
        | { success: false; code?: string; error: string; hint?: string }
    >
    onYouTubeProgress: (callback: (data: RemoteResolveProgressPayload) => void) => () => void
    getGpuDevices: () => Promise<any>
    getSystemRuntimeInfo: () => Promise<SystemRuntimeInfo>
    getWorkflows: () => Promise<{ workflows: Record<string, any> }>
    checkPresetModels: (presetMappings: Record<string, string>) => Promise<Record<string, boolean>>
    onDownloadProgress: (callback: (data: { modelId: string; progress: number; stage?: string; artifactIndex?: number; artifactCount?: number; currentFile?: string; currentRelativePath?: string; currentSource?: string; verified?: boolean; message?: string }) => void) => () => void
    onDownloadComplete: (callback: (data: { modelId: string; artifactCount?: number; stage?: string; verified?: boolean }) => void) => () => void
    onDownloadError: (callback: (data: { modelId: string; error: string }) => void) => () => void
    onDownloadPaused: (callback: (data: { modelId: string; artifactIndex?: number; artifactCount?: number; currentFile?: string; currentRelativePath?: string; currentSource?: string; progress?: number; stage?: string; verified?: boolean }) => void) => () => void
    openDirectoryDialog: () => Promise<{ canceled: boolean; filePaths: string[] }>
    onBackendError: (callback: (data: { error: string }) => void) => () => void
    onBridgeReady: (callback: (data: { capabilities: string[]; modelsCount: number; recipesCount: number }) => void) => () => void
    onQualityProgress: (callback: (data: any) => void) => () => void
    onQualityComplete: (callback: (data: any) => void) => () => void
    saveQueue: (queue: any[]) => Promise<{ success: boolean; error?: string }>
    loadQueue: () => Promise<any[] | null>
    startWatchMode: (path: string) => Promise<boolean>
    stopWatchMode: () => Promise<boolean>
    onWatchFileDetected: (callback: (path: string) => void) => () => void

    // App Config
    setModelsDir: (modelsDir: string) => Promise<{ success: boolean; modelsDir: string; models?: any[] }>
    saveAppConfig: (config: Record<string, any>) => Promise<boolean>
    getAppConfig: () => Promise<Record<string, any>>

    // Hugging Face auth (optional)
    setHuggingFaceToken: (token: string) => Promise<{ success: boolean; error?: string }>
    clearHuggingFaceToken: () => Promise<{ success: boolean; error?: string }>
    getHuggingFaceAuthStatus: () => Promise<{ configured: boolean }>

    // External links
    openExternalUrl: (url: string) => Promise<boolean>
    resolveMediaUrl?: (filePath: string) => string
}

interface Window {
    electronAPI: ElectronAPI
}
