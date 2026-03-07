export type SourceAudioProfile = {
  path: string
  container: string | null
  codec: string | null
  codecLongName: string | null
  sampleRate: number | null
  channels: number | null
  sampleFormat: string | null
  bitDepth: number | null
  durationSeconds: number | null
  isLossless: boolean
}

export type StagingDecision = {
  sourcePath: string
  workingPath: string
  sourceExt: string
  copiedDirectly: boolean
  workingCodec: 'original' | 'pcm_s16le' | 'pcm_s24le' | 'pcm_f32le'
  reason: string
}

export type PlaybackMetadata = {
  sourceKind: 'preview_cache' | 'saved_output' | 'missing_source'
  previewDir?: string
  savedDir?: string
}

export type SeparationProgressEvent = {
  jobId: string
  kind:
    | 'job_started'
    | 'progress'
    | 'step_started'
    | 'step_completed'
    | 'completed'
    | 'cancelled'
    | 'error'
  progress?: number
  message?: string
  phase?: string
  stepId?: string
  stepLabel?: string
  stepIndex?: number
  stepCount?: number
  modelId?: string
  chunksDone?: number
  chunksTotal?: number
  elapsedMs?: number
  ts?: number
  meta?: Record<string, any>
  error?: string
}

export type ExportProgressEvent = {
  requestId: string
  status: 'preflight' | 'copying' | 'transcoding' | 'completed' | 'failed'
  stem?: string
  fileIndex?: number
  fileCount?: number
  fileProgress?: number
  totalProgress?: number
  detail?: string
  format?: string
  outputPath?: string
  error?: string
}
