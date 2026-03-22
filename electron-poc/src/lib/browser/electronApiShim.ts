import { browserDemoModels } from "./demoModels"
import type {
  ExportProgressEvent,
  SeparationProgressEvent,
  SourceAudioProfile,
  StagingDecision,
} from "@/types/media"
import type { RemoteResolveProgressPayload } from "@/types/remote"

type DownloadProgressPayload = {
  modelId: string
  progress: number
  speed?: number
  eta?: number
  artifactIndex?: number
  artifactCount?: number
  currentFile?: string
  currentRelativePath?: string
  message?: string
}

type DownloadSimplePayload = {
  modelId: string
  artifactCount?: number
  artifactIndex?: number
  currentFile?: string
  currentRelativePath?: string
  progress?: number
}
type DownloadErrorPayload = { modelId: string; error: string }
type BackendErrorPayload = { error: string }
type BrowserFileEntry = {
  path: string
  blob: Blob
  objectUrl: string
  name: string
  type: string
  lastModified: number
}

type ShimModel = (typeof browserDemoModels)[number] & {
  installed: boolean
  downloading: boolean
  downloadPaused?: boolean
  downloadProgress: number
  is_custom?: boolean
  installation?: {
    installed?: boolean
    missing_artifacts?: string[]
  }
}

const STORAGE_PREFIX = "stemsep-browser"
const MODELS_KEY = `${STORAGE_PREFIX}:models`
const CONFIG_KEY = `${STORAGE_PREFIX}:app-config`
const QUEUE_KEY = `${STORAGE_PREFIX}:queue`
const CUSTOM_MODELS_KEY = `${STORAGE_PREFIX}:custom-models`

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

function writeJson<T>(key: string, value: T) {
  window.localStorage.setItem(key, JSON.stringify(value))
}

function createEmitter<T>() {
  const listeners = new Set<(payload: T) => void>()
  return {
    emit(payload: T) {
      for (const listener of listeners) listener(payload)
    },
    subscribe(listener: (payload: T) => void) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

async function chooseFiles(options: {
  accept?: string
  multiple?: boolean
}): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = options.accept || ""
    input.multiple = !!options.multiple
    input.style.position = "fixed"
    input.style.left = "-9999px"
    input.onchange = () => {
      const files = Array.from(input.files || [])
      input.remove()
      resolve(files)
    }
    input.oncancel = () => {
      input.remove()
      resolve([])
    }
    document.body.appendChild(input)
    input.click()
  })
}

function extensionForType(type: string, fallbackName: string): string {
  const lower = type.toLowerCase()
  if (lower.includes("wav")) return ".wav"
  if (lower.includes("flac")) return ".flac"
  if (lower.includes("mpeg")) return ".mp3"
  if (lower.includes("mp4") || lower.includes("aac")) return ".m4a"
  const match = fallbackName.match(/\.[a-z0-9]+$/i)
  return match?.[0] || ".wav"
}

function basename(path: string) {
  return path.split(/[\\/]/).pop() || path
}

function stripExt(name: string) {
  return name.replace(/\.[^/.]+$/, "")
}

export function installBrowserElectronApi() {
  if (typeof window === "undefined") return
  if ((window as any).electronAPI) return

  ;(window as any).__STEMSEP_BROWSER_MODE__ = true

  const fileStore = new Map<string, BrowserFileEntry>()
  const jobOutputs = new Map<string, Record<string, string>>()

  const separationEventEmitter = createEmitter<SeparationProgressEvent>()
  const separationStartedEmitter = createEmitter<{ jobId: string }>()
  const separationCompleteEmitter = createEmitter<{ outputFiles: Record<string, string> }>()
  const separationErrorEmitter = createEmitter<{ error: string }>()
  const exportProgressEmitter = createEmitter<ExportProgressEvent>()
  const downloadProgressEmitter = createEmitter<DownloadProgressPayload>()
  const downloadCompleteEmitter = createEmitter<DownloadSimplePayload>()
  const downloadErrorEmitter = createEmitter<DownloadErrorPayload>()
  const downloadPausedEmitter = createEmitter<DownloadSimplePayload>()
  const backendErrorEmitter = createEmitter<BackendErrorPayload>()
  const bridgeReadyEmitter = createEmitter<{ capabilities: string[]; modelsCount: number; recipesCount: number }>()
  const qualityProgressEmitter = createEmitter<any>()
  const qualityCompleteEmitter = createEmitter<any>()
  const youTubeProgressEmitter = createEmitter<RemoteResolveProgressPayload>()
  const watchFileEmitter = createEmitter<string>()

  let shimModels: ShimModel[] = readJson<ShimModel[]>(MODELS_KEY, browserDemoModels.map((model) => ({
    ...model,
    installed: model.installed,
    downloading: false,
    downloadPaused: false,
    downloadProgress: 0,
  } as ShimModel)))
  let customModels = readJson<any[]>(CUSTOM_MODELS_KEY, [])
  let appConfig = readJson<Record<string, any>>(CONFIG_KEY, {
    outputDir: "Browser Downloads",
    modelsDir: "Browser Model Cache",
  })

  const persistModels = () => writeJson(MODELS_KEY, shimModels)
  const persistCustomModels = () => writeJson(CUSTOM_MODELS_KEY, customModels)
  const persistConfig = () => writeJson(CONFIG_KEY, appConfig)

  const hydrateModelList = () => [...shimModels, ...customModels]

  const registerFile = (file: File | Blob, explicitName?: string) => {
    const name =
      explicitName ||
      ("name" in file && typeof file.name === "string" && file.name.trim()
        ? file.name
        : `audio-${Date.now()}${extensionForType(file.type || "", "audio.wav")}`)
    const path = `browser://media/${crypto.randomUUID()}/${name}`
    const objectUrl = URL.createObjectURL(file)
    fileStore.set(path, {
      path,
      blob: file,
      objectUrl,
      name,
      type: file.type || "audio/wav",
      lastModified: "lastModified" in file && typeof file.lastModified === "number" ? file.lastModified : Date.now(),
    })
    return path
  }

  const getFileEntry = (path: string) => fileStore.get(path)

  const sourceProfileFor = (path: string): SourceAudioProfile => {
    const entry = getFileEntry(path)
    const name = entry?.name || basename(path)
    const ext = name.split(".").pop()?.toLowerCase() || null
    const codec =
      ext === "wav"
        ? "pcm_s16le"
        : ext === "flac"
          ? "flac"
          : ext === "mp3"
            ? "mp3"
            : ext === "m4a"
              ? "aac"
              : ext
    const bitDepth = ext === "flac" || ext === "wav" ? 16 : null
    return {
      path,
      container: ext,
      codec,
      codecLongName: codec,
      sampleRate: 44100,
      channels: 2,
      sampleFormat: bitDepth ? `s${bitDepth}` : null,
      bitDepth,
      durationSeconds: null,
      isLossless: ext === "wav" || ext === "flac",
    }
  }

  const stagingDecisionFor = (path: string): StagingDecision => {
    const ext = basename(path).split(".").pop()?.toLowerCase() || "wav"
    const lossless = ext === "wav" || ext === "flac"
    return {
      sourcePath: path,
      workingPath: path,
      sourceExt: ext,
      copiedDirectly: true,
      workingCodec: lossless ? "original" : "pcm_s16le",
      reason: lossless
        ? "Browser preview keeps lossless inputs unchanged."
        : "Browser preview reuses the uploaded source without native transcoding.",
    }
  }

  const stemListFor = (modelId: string, requested?: string[]) => {
    if (requested && requested.length > 0) return requested
    const model = hydrateModelList().find((item) => item.id === modelId)
    if (model?.stems?.length) return model.stems
    return ["vocals", "instrumental"]
  }

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const electronAPI: Window["electronAPI"] & {
    resolveMediaUrl?: (filePath: string) => string
  } = {
    selectOutputDirectory: async () => appConfig.outputDir || "Browser Downloads",
    selectModelsDirectory: async () => appConfig.modelsDir || "Browser Model Cache",
    openDirectoryDialog: async () => ({
      canceled: false,
      filePaths: [appConfig.outputDir || "Browser Downloads"],
    }),
    openAudioFileDialog: async () => {
      const files = await chooseFiles({
        accept: "audio/*,.wav,.flac,.mp3,.m4a",
        multiple: true,
      })
      if (files.length === 0) return null
      return files.map((file) => registerFile(file, file.name))
    },
    openModelFileDialog: async () => {
      const files = await chooseFiles({
        accept: ".ckpt,.pth,.pt,.onnx,.safetensors",
        multiple: false,
      })
      if (files.length === 0) return null
      return files.map((file) => registerFile(file, file.name))
    },
    scanDirectory: async () => [],
    separationPreflight: async (
      inputFile: string,
      modelId: string,
      _outputDir: string,
      _selectionType?: "model" | "recipe" | "workflow",
      _selectionId?: string,
      stems?: string[],
      device?: string,
      _overlap?: number,
      _segmentSize?: number,
      _shifts?: number,
      _outputFormat?: string,
      _bitrate?: string,
      _tta?: boolean,
      _ensembleConfig?: any,
      _ensembleAlgorithm?: string,
      _invert?: boolean,
      _splitFreq?: number,
      _phaseParams?: any,
      _postProcessingSteps?: any[],
      _volumeCompensation?: any,
      _pipelineConfig?: any[],
      _workflow?: Record<string, any>,
      _runtimePolicy?: Record<string, any>,
      _exportPolicy?: Record<string, any>,
      _selectionEnvelope?: Record<string, any>,
    ) => {
      const requiredModels = [modelId].filter(Boolean).map((id) => ({
        model_id: id,
        installed: !!hydrateModelList().find((model) => model.id === id)?.installed,
      }))
      const sourceAudioProfile = sourceProfileFor(inputFile)
      const stagingDecision = stagingDecisionFor(inputFile)
      return {
        can_proceed: true,
        errors: [],
        warnings: ["Browser preview mode uses a simulated separation pipeline."],
        missing_models: [],
        sourceAudioProfile,
        stagingDecision,
        plan: {
          mode: "browser_preview",
          recommendation_surface: "simple",
          selected_workflow: modelId,
          fallback_reason: "Browser preview mode duplicates the uploaded file per output stem.",
          required_models: requiredModels,
          runtime_requirements: ["Browser preview"],
          resolved_device: device || "browser",
          estimated_vram_gb: 0,
          recommended_adjustments: ["Use the desktop app for real inference and native export."],
          resolved_recipe: null,
          output_stems: stemListFor(modelId, stems),
        },
      }
    },
    separateAudio: async (
      inputFile: string,
      modelId: string,
      _outputDir: string,
      _selectionType?: "model" | "recipe" | "workflow",
      _selectionId?: string,
      stems?: string[],
      _device?: string,
      _overlap?: number,
      _segmentSize?: number,
      _shifts?: number,
      _outputFormat?: string,
      _bitrate?: string,
      _tta?: boolean,
      _ensembleConfig?: any,
      _ensembleAlgorithm?: string,
      _invert?: boolean,
      _splitFreq?: number,
      _phaseParams?: any,
      _postProcessingSteps?: any[],
      _volumeCompensation?: any,
      _pipelineConfig?: any[],
      _workflow?: Record<string, any>,
      _runtimePolicy?: Record<string, any>,
      _exportPolicy?: Record<string, any>,
      _selectionEnvelope?: Record<string, any>,
    ) => {
      const jobId = `browser-job-${crypto.randomUUID()}`
      const source = getFileEntry(inputFile)
      if (!source) {
        const error = "Browser preview could not find the selected file."
        separationErrorEmitter.emit({ error })
        separationEventEmitter.emit({ jobId, kind: "error", error, message: error, ts: Date.now() })
        return { success: false, outputFiles: {}, error }
      }

      const outputStems = stemListFor(modelId, stems)
      const startedAt = Date.now()

      separationStartedEmitter.emit({ jobId })
      separationEventEmitter.emit({
        jobId,
        kind: "job_started",
        progress: 0,
        message: "Preparing browser preview",
        stepCount: outputStems.length,
        ts: startedAt,
      })

      await delay(120)

      const outputFiles: Record<string, string> = {}
      for (let index = 0; index < outputStems.length; index += 1) {
        const stem = outputStems[index]
        separationEventEmitter.emit({
          jobId,
          kind: "step_started",
          progress: Math.round((index / outputStems.length) * 100),
          message: `Generating ${stem} preview`,
          stepId: stem,
          stepLabel: `Preview ${stem}`,
          stepIndex: index,
          stepCount: outputStems.length,
          modelId,
          chunksDone: index,
          chunksTotal: outputStems.length,
          ts: Date.now(),
        })
        await delay(180)
        const outputName = `${stripExt(source.name)}-${stem}${extensionForType(source.type, source.name)}`
        outputFiles[stem] = registerFile(source.blob, outputName)
        separationEventEmitter.emit({
          jobId,
          kind: "step_completed",
          progress: Math.round(((index + 1) / outputStems.length) * 100),
          message: `${stem} ready`,
          stepId: stem,
          stepLabel: `Preview ${stem}`,
          stepIndex: index,
          stepCount: outputStems.length,
          modelId,
          chunksDone: index + 1,
          chunksTotal: outputStems.length,
          elapsedMs: Date.now() - startedAt,
          ts: Date.now(),
        })
      }

      jobOutputs.set(jobId, outputFiles)

      separationEventEmitter.emit({
        jobId,
        kind: "completed",
        progress: 100,
        message: "Browser preview ready",
        stepCount: outputStems.length,
        elapsedMs: Date.now() - startedAt,
        ts: Date.now(),
      })
      separationCompleteEmitter.emit({ outputFiles })

      return {
        success: true,
        outputFiles,
        jobId,
        outputDir: _outputDir || "Browser Preview",
        playbackSourceKind: "preview_cache",
        sourceAudioProfile: sourceProfileFor(inputFile),
        stagingDecision: stagingDecisionFor(inputFile),
      }
    },
    cancelSeparation: async () => ({ success: true }),
    saveJobOutput: async (jobId) => {
      const outputFiles = jobOutputs.get(jobId)
      if (!outputFiles) return { success: false, error: "No browser preview outputs found for this job." }
      for (const [stem, path] of Object.entries(outputFiles)) {
        const entry = getFileEntry(path)
        if (entry) downloadBlob(entry.blob, entry.name || `${stem}.wav`)
      }
      return { success: true, outputFiles }
    },
    discardJobOutput: async (jobId) => {
      const outputs = jobOutputs.get(jobId)
      if (outputs) {
        Object.values(outputs).forEach((path) => {
          const entry = fileStore.get(path)
          if (entry) {
            URL.revokeObjectURL(entry.objectUrl)
            fileStore.delete(path)
          }
        })
        jobOutputs.delete(jobId)
      }
      return { success: true }
    },
    pauseQueue: async () => undefined,
    resumeQueue: async () => undefined,
    reorderQueue: async () => undefined,
    exportOutput: async (jobId, exportPath, format, bitrate, requestId) => {
      const outputFiles = jobOutputs.get(jobId)
      if (!outputFiles) {
        return { success: false, error: "No preview outputs available for export.", requestId }
      }
      return electronAPI.exportFiles(outputFiles, exportPath, format, bitrate, requestId)
    },
    exportFiles: async (sourceFiles, exportPath, format, _bitrate, requestId) => {
      const stems = Object.entries(sourceFiles)
      exportProgressEmitter.emit({
        requestId: requestId || crypto.randomUUID(),
        status: "preflight",
        totalProgress: 0,
        detail: `Preparing ${stems.length} browser download(s) to ${exportPath || "Browser Downloads"}.`,
      })
      for (let index = 0; index < stems.length; index += 1) {
        const [stem, path] = stems[index]
        const entry = getFileEntry(path)
        if (!entry) {
          return { success: false, error: `Missing preview output for ${stem}.`, requestId }
        }
        exportProgressEmitter.emit({
          requestId: requestId || "",
          status: "copying",
          stem,
          fileIndex: index + 1,
          fileCount: stems.length,
          fileProgress: 100,
          totalProgress: Math.round((index / stems.length) * 100),
          detail: `Downloading ${stem}`,
          format,
        })
        const filename = `${stripExt(entry.name)}.${format}`
        downloadBlob(entry.blob, filename)
        await delay(50)
      }
      exportProgressEmitter.emit({
        requestId: requestId || "",
        status: "completed",
        totalProgress: 100,
        detail: "Browser export complete.",
        format,
      })
      return { success: true, exported: sourceFiles, outputFiles: sourceFiles, requestId }
    },
    onSeparationProgress: (callback) =>
      separationEventEmitter.subscribe((event) =>
        callback({ progress: event.progress ?? 0, message: event.message || "", jobId: event.jobId, meta: event.meta }),
      ),
    onSeparationEvent: (callback) => separationEventEmitter.subscribe(callback),
    onSeparationStarted: (callback) => separationStartedEmitter.subscribe(callback),
    onSeparationComplete: (callback) => separationCompleteEmitter.subscribe(callback),
    onSeparationError: (callback) => separationErrorEmitter.subscribe(callback),
    onExportProgress: (callback) => exportProgressEmitter.subscribe(callback),
    getCatalog: async () => {
      const models = hydrateModelList()
      const selectionIndex = models.map((model) => ({
        selectionType: "model" as const,
        selectionId: model.id,
        name: model.name,
        catalogTier: model.catalog_tier ?? model.catalog?.tier,
        sourceKind: model.source_kind ?? model.catalog?.sourceKind,
        installPolicy: model.install_policy ?? model.catalog?.installPolicy,
        selectionEnvelope: model.selection_envelope,
        requiredModelIds: [model.id],
      }))
      return {
        version: "browser-preview",
        schema_version: "catalog-runtime-v3",
        models,
        recipes: [],
        workflows: [],
        selection_index: selectionIndex,
      }
    },
    getModels: async () => hydrateModelList(),
    getModelTech: async (modelId) => hydrateModelList().find((model) => model.id === modelId) || null,
    resolveModelDownload: async (modelId) => {
      const model = hydrateModelList().find((entry) => entry.id === modelId)
      return model
        ? {
            modelId,
            download: model.download || null,
            installation:
              model.installation || { installed: !!model.installed, missing_artifacts: [] },
          }
        : null
    },
    getModelInstallation: async (modelId) => {
      const model = hydrateModelList().find((entry) => entry.id === modelId)
      return model?.installation || { installed: !!model?.installed, missing_artifacts: [] }
    },
    getSelectionInstallation: async (selectionType, selectionId) => {
      const model = selectionType === "model"
        ? hydrateModelList().find((entry) => entry.id === selectionId)
        : null
      return {
        selectionType,
        selectionId,
        installed: !!model?.installed,
        requiredModelIds: model ? [model.id] : [],
        required_model_ids: model ? [model.id] : [],
        selectionEnvelope: model?.selection_envelope,
        installation: model?.installation || { installed: !!model?.installed, missing_artifacts: [] },
      }
    },
    resolveInstallPlan: async (selectionType, selectionId) => {
      const model = selectionType === "model"
        ? hydrateModelList().find((entry) => entry.id === selectionId)
        : null
      return {
        success: true,
        selectionType,
        selectionId,
        requiredModelIds: model ? [model.id] : [],
        required_model_ids: model ? [model.id] : [],
        selectionEnvelope: model?.selection_envelope,
        installPolicy: model?.install_policy ?? model?.catalog?.installPolicy ?? "manual",
        catalogTier: model?.catalog_tier ?? model?.catalog?.tier,
        sourceKind: model?.source_kind ?? model?.catalog?.sourceKind,
        installation: model?.installation || { installed: !!model?.installed, missing_artifacts: [] },
        model,
      }
    },
    installSelection: async (selectionType, selectionId) => {
      if (selectionType !== "model") {
        return { success: false, error: "Browser preview only installs model selections." }
      }
      const success = await electronAPI.downloadModel(selectionId)
      return {
        success,
        selectionType,
        selectionId,
        installation: await electronAPI.getSelectionInstallation(selectionType, selectionId),
      }
    },
    importSelectionArtifacts: async (selectionType, selectionId, files, allowCopy = true) => {
      if (selectionType !== "model") {
        return { success: false, error: "Browser preview only imports model selections." }
      }
      return electronAPI.importModelFiles(selectionId, files, allowCopy)
    },
    verifySelectionArtifacts: async (selectionType, selectionId) => {
      return electronAPI.getSelectionInstallation(selectionType, selectionId)
    },
    resolveExecutionPlan: async (selectionType, selectionId) => {
      const model = selectionType === "model"
        ? hydrateModelList().find((entry) => entry.id === selectionId)
        : null
      return {
        selection_type: selectionType,
        selection_id: selectionId,
        selection_envelope: model?.selection_envelope,
        required_model_ids: model ? [model.id] : [],
        execution_constraints: { browser_preview: true },
        resolved_bundle: model
          ? {
              selection_type: selectionType,
              selection_id: selectionId,
              required_model_ids: [model.id],
              selection_envelope: model.selection_envelope,
            }
          : null,
      }
    },
    runSelectionJob: async (payload) => {
      const jobId = crypto.randomUUID()
      return {
        success: true,
        job_id: jobId,
        status: "started",
        job: {
          job_id: jobId,
          status: "starting",
          requested_at: new Date().toISOString(),
          progress: 0,
          selection_type: payload?.selection_type || payload?.selectionType || "model",
          selection_id: payload?.selection_id || payload?.selectionId || payload?.model_id || payload?.modelId || null,
        },
      }
    },
    cancelSelectionJob: async (jobId) => ({
      success: true,
      job_id: jobId,
      status: "cancelled",
    }),
    getSelectionJob: async (jobId) => ({
      job_id: jobId,
      status: "completed",
      requested_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
      progress: 100,
    }),
    listSelectionJobs: async () => [],
    exportSelectionJob: async (jobId, exportPath) => ({
      success: true,
      job_id: jobId,
      export_path: exportPath,
      output_files: {},
    }),
    discardSelectionJob: async (jobId) => ({
      success: true,
      job_id: jobId,
      discarded: true,
    }),
    getRecipes: async () => [],
    qualityBaselineCreate: async (payload) => {
      qualityProgressEmitter.emit({ kind: "progress", message: "Creating browser baseline..." })
      await delay(150)
      const result = {
        success: true,
        manifest_path: `browser://quality/${crypto.randomUUID()}/baseline.json`,
        baseline_name: payload?.baseline_name || "browser-baseline",
      }
      qualityCompleteEmitter.emit(result)
      return result
    },
    qualityCompare: async (payload) => {
      qualityProgressEmitter.emit({ kind: "progress", message: "Comparing browser candidate..." })
      await delay(150)
      const result = {
        success: true,
        compatible: true,
        score_delta: 0,
        differences: [],
        output_mismatches: [],
        outputs: Object.keys(payload?.candidate_manifest?.output_files || {}),
      }
      qualityCompleteEmitter.emit(result)
      return result
    },
    downloadModel: async (modelId) => {
      shimModels = shimModels.map((model) =>
        model.id === modelId ? { ...model, downloading: true, downloadProgress: 10 } : model,
      )
      persistModels()
      for (const progress of [10, 45, 80, 100]) {
        downloadProgressEmitter.emit({ modelId, progress })
        await delay(80)
      }
      shimModels = shimModels.map((model) =>
        model.id === modelId
          ? { ...model, installed: true, downloading: false, downloadProgress: 100, downloadPaused: false }
          : model,
      )
      persistModels()
      downloadCompleteEmitter.emit({ modelId })
      return true
    },
    pauseDownload: async (modelId) => {
      shimModels = shimModels.map((model) =>
        model.id === modelId ? { ...model, downloadPaused: true, downloading: false } : model,
      )
      persistModels()
      downloadPausedEmitter.emit({ modelId })
      return true
    },
    resumeDownload: async (modelId) => {
      return electronAPI.downloadModel(modelId)
    },
    importModelFiles: async (modelId, files) => {
      const installed = Array.isArray(files) && files.length > 0
      shimModels = shimModels.map((model) =>
        model.id === modelId
          ? {
              ...model,
              installed,
              installation: {
                ...(model.installation || {}),
                installed,
                missing_artifacts: installed ? [] : model.installation?.missing_artifacts || [],
              },
            }
          : model,
      )
      persistModels()
      return { model_id: modelId, installation: { installed, missing_artifacts: [] } }
    },
    removeModel: async (modelId) => {
      const isCustom = customModels.some((model) => model.id === modelId)
      if (isCustom) {
        customModels = customModels.filter((model) => model.id !== modelId)
        persistCustomModels()
        return true
      }
      shimModels = shimModels.map((model) =>
        model.id === modelId ? { ...model, installed: false, downloading: false, downloadProgress: 0 } : model,
      )
      persistModels()
      return true
    },
    importCustomModel: async (filePath, modelName, architecture) => {
      const customModel = {
        id: `custom_${crypto.randomUUID().slice(0, 8)}`,
        name: modelName,
        architecture: architecture || "Custom",
        version: "browser",
        category: "custom",
        description: basename(filePath),
        sdr: 0,
        fullness: 0,
        bleedless: 0,
        vram_required: 0,
        speed: "unknown",
        stems: [],
        file_size: getFileEntry(filePath)?.blob.size || 0,
        installed: true,
        downloading: false,
        downloadProgress: 0,
        recommended: false,
        is_custom: true,
        catalog_status: "verified",
        metrics_status: "missing_evidence",
        card_metrics: {
          kind: "status",
          labels: ["METRICS", "DOWNLOAD", "RUNTIME"],
          values: ["Custom", "Imported", "Browser"],
          source: "Browser import",
          evidence_note: basename(filePath),
          last_verified: new Date().toISOString().slice(0, 10),
        },
        runtime: {
          allowed: ["browser"],
          preferred: "browser",
        },
        tags: ["custom"],
      }
      customModels = [...customModels, customModel]
      persistCustomModels()
      return customModel
    },
    openFolder: async (folderPath) => {
      window.alert(`Browser mode cannot open native folders.\n\nTarget: ${folderPath}`)
    },
    checkFileExists: async (filePath) => {
      return fileStore.has(filePath)
    },
    readAudioFile: async (filePath) => {
      const entry = getFileEntry(filePath)
      if (!entry) return { success: false as const, error: "File not found in browser preview." }
      const buffer = await entry.blob.arrayBuffer()
      const bytes = new Uint8Array(buffer)
      let binary = ""
      for (let index = 0; index < bytes.byteLength; index += 1) {
        binary += String.fromCharCode(bytes[index])
      }
      return {
        success: true as const,
        data: btoa(binary),
        mimeType: entry.type || "audio/wav",
        resolvedPath: entry.objectUrl,
      }
    },
    resolvePlaybackStems: async (outputFiles) => ({
      success: true,
      stems: outputFiles,
      issues: {},
    }),
    authRemoteSource: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} auth is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native remote-source support.",
    }),
    searchRemoteCatalog: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} browsing is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native remote-source support.",
    }),
    listRemoteCollection: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} collection sync is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native remote-source support.",
    }),
    resolveRemoteTrack: async (provider) => ({
      success: false as const,
      provider,
      error: `${provider} import is unavailable in browser preview mode.`,
      code: "BROWSER_PREVIEW_UNSUPPORTED",
      hint: "Use the Electron desktop app for native remote-source support.",
    }),
    onRemoteResolveProgress: (callback) => youTubeProgressEmitter.subscribe(callback),
    detectPlaybackDevices: async () => [],
    getCaptureEnvironmentStatus: async () => ({
      windowsSupported: false,
      provider: "qobuz" as const,
      authenticated: false,
      selectedDeviceId: null,
      selectedDeviceLabel: null,
      selectedDeviceReady: false,
      speakerSelectionAvailable: false,
      message: "Hidden Qobuz capture is unavailable in browser preview mode.",
    }),
    setCaptureOutputDevice: async () => ({
      success: false,
      error: "Hidden Qobuz capture is unavailable in browser preview mode.",
    }),
    authLibraryProvider: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} auth is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    getLibraryAuthStatus: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} auth status is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    searchLibrary: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} browsing is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    listLibraryCollection: async (provider) => ({
      success: false as const,
      provider,
      authenticated: false,
      error: `${provider} browsing is unavailable in browser preview mode.`,
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    preparePlaybackCapture: async (provider) => ({
      success: false as const,
      provider,
      error: `${provider} playback capture is unavailable in browser preview mode.`,
      code: "BROWSER_PREVIEW_UNSUPPORTED",
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    startPlaybackCapture: async (provider) => ({
      success: false as const,
      provider,
      error: `${provider} playback capture is unavailable in browser preview mode.`,
      code: "BROWSER_PREVIEW_UNSUPPORTED",
      hint: "Use the Electron desktop app for native library browsing and capture.",
    }),
    cancelPlaybackCapture: async () => ({ success: true }),
    getPlaybackCaptureStatus: async () => ({ success: true, sessions: [], backend: [] }),
    onPlaybackCaptureProgress: (callback) => youTubeProgressEmitter.subscribe(callback as any),
    resolveYouTubeUrl: async () => {
      youTubeProgressEmitter.emit({
        provider: "youtube",
        status: "error",
        error: "YouTube import is unavailable in browser preview mode.",
      })
      return {
        success: false as const,
        code: "BROWSER_PREVIEW_UNSUPPORTED",
        error: "YouTube import is unavailable in browser preview mode.",
        hint: "Use the Electron desktop app for native download support.",
      }
    },
    onYouTubeProgress: (callback) => youTubeProgressEmitter.subscribe(callback),
    getGpuDevices: async () => ({
      has_cuda: false,
      gpus: [],
      recommended_profile: { device: "cpu", vram_gb: 0 },
    }),
    getSystemRuntimeInfo: async () => ({
      fetchedAt: new Date().toISOString(),
      cache: { ttlMs: 30_000, gpuSource: "fresh", runtimeFingerprintSource: "fresh" },
      gpu: {
        has_cuda: false,
        gpus: [],
        recommended_profile: { device: "cpu", vram_gb: 0 },
      },
      runtimeFingerprint: {
        version: "browser-preview",
        platform: navigator.platform,
        torch: { version: "preview", cuda_available: false, cuda_device_count: 0 },
        neuralop: { version: "preview", fno1d_import_ok: false, fno_import_ok: false },
      },
      runtimeFingerprintError: null,
      previewCachePolicy: {
        baseDir: "browser://preview",
        keepLast: 8,
        maxAgeDays: 0,
        ephemeral: true,
      },
    }),
    getWorkflows: async () => ({ workflows: {} }),
    checkPresetModels: async (presetMappings) =>
      Object.fromEntries(
        Object.entries(presetMappings).map(([key, modelId]) => [
          key,
          !!hydrateModelList().find((model) => model.id === modelId)?.installed,
        ]),
      ),
    onDownloadProgress: (callback) => downloadProgressEmitter.subscribe(callback),
    onDownloadComplete: (callback) => downloadCompleteEmitter.subscribe(callback),
    onDownloadError: (callback) => downloadErrorEmitter.subscribe(callback),
    onDownloadPaused: (callback) => downloadPausedEmitter.subscribe(callback),
    onBackendError: (callback) => backendErrorEmitter.subscribe(callback),
    onBridgeReady: (callback) => {
      const unsubscribe = bridgeReadyEmitter.subscribe(callback)
      window.setTimeout(() => {
        callback({
          capabilities: ["browser-preview", "demo-separation", "virtual-export"],
          modelsCount: hydrateModelList().length,
          recipesCount: 0,
        })
      }, 0)
      return unsubscribe
    },
    onQualityProgress: (callback) => qualityProgressEmitter.subscribe(callback),
    onQualityComplete: (callback) => qualityCompleteEmitter.subscribe(callback),
    saveQueue: async (queue) => {
      writeJson(QUEUE_KEY, queue)
      return { success: true }
    },
    loadQueue: async () => readJson<any[] | null>(QUEUE_KEY, []),
    startWatchMode: async () => true,
    stopWatchMode: async () => true,
    onWatchFileDetected: (callback) => watchFileEmitter.subscribe(callback),
    setModelsDir: async (modelsDir) => {
      appConfig = { ...appConfig, modelsDir }
      persistConfig()
      return { success: true, modelsDir, models: hydrateModelList() }
    },
    saveAppConfig: async (config) => {
      appConfig = { ...appConfig, ...config }
      persistConfig()
      return true
    },
    getAppConfig: async () => appConfig,
    setHuggingFaceToken: async (token) => {
      appConfig = { ...appConfig, hfToken: token }
      persistConfig()
      return { success: true }
    },
    clearHuggingFaceToken: async () => {
      appConfig = { ...appConfig, hfToken: "" }
      persistConfig()
      return { success: true }
    },
    getHuggingFaceAuthStatus: async () => ({
      configured: typeof appConfig.hfToken === "string" && appConfig.hfToken.trim().length > 0,
    }),
    openExternalUrl: async (url) => {
      window.open(url, "_blank", "noopener,noreferrer")
      return true
    },
    resolveMediaUrl: (filePath: string) => {
      if (/^(blob:|data:|https?:)/i.test(filePath)) return filePath
      return getFileEntry(filePath)?.objectUrl || filePath
    },
  }

  ;(window as any).electronAPI = electronAPI
}
