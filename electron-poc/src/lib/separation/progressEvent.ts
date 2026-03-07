import type { SeparationProgressEvent } from "@/types/media"
import type { QueueItem } from "@/types/store"

const clampProgress = (value: number | undefined, fallback = 0) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback
  return Math.max(0, Math.min(100, value))
}

export function queueUpdatesFromProgressEvent(
  current: QueueItem,
  event: SeparationProgressEvent,
  now = Date.now(),
): Partial<QueueItem> {
  const nextProgress = clampProgress(event.progress, current.progress ?? 0)
  const next: Partial<QueueItem> = {
    progress: nextProgress,
    message: event.message ?? current.message,
    lastProgressTime: now,
  }

  if (event.phase) next.activePhase = event.phase
  if (event.stepId) next.activeStepId = event.stepId
  if (event.stepLabel) next.activeStepLabel = event.stepLabel
  if (typeof event.stepIndex === "number") next.activeStepIndex = event.stepIndex
  if (typeof event.stepCount === "number") next.activeStepCount = event.stepCount
  if (event.modelId) next.activeModelId = event.modelId
  if (typeof event.chunksDone === "number") next.chunksDone = event.chunksDone
  if (typeof event.chunksTotal === "number") next.chunksTotal = event.chunksTotal
  if (typeof event.elapsedMs === "number") next.lastStepDurationMs = event.elapsedMs

  if (event.kind === "job_started") {
    next.status = "processing"
    next.startTime = current.startTime ?? now
  } else if (event.kind === "completed") {
    next.status = "completed"
    next.progress = 100
    next.message = event.message ?? "Complete"
  } else if (event.kind === "error" || event.kind === "cancelled") {
    next.status = event.kind === "cancelled" ? "cancelled" : "failed"
    next.error = event.message ?? current.error
  } else {
    next.status = current.status === "pending" ? "processing" : current.status
  }

  return next
}

export function getQueueStepLabel(item: QueueItem): string | null {
  if (item.activeStepLabel) {
    if (
      typeof item.activeStepIndex === "number" &&
      typeof item.activeStepCount === "number"
    ) {
      return `${item.activeStepLabel} (${item.activeStepIndex + 1}/${item.activeStepCount})`
    }
    return item.activeStepLabel
  }

  if (item.activePhase) {
    return item.activePhase
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  }

  return null
}

export function calculateQueueEta(
  item: QueueItem,
  now = Date.now(),
): string | null {
  if (typeof item.startTime !== "number") return null
  const progress = clampProgress(item.progress)
  if (progress < 5) return null

  const elapsedMs = now - item.startTime
  const stepProgressHint =
    typeof item.activeStepCount === "number" &&
    item.activeStepCount > 0 &&
    typeof item.activeStepIndex === "number"
      ? Math.max(progress / 100, (item.activeStepIndex + 0.35) / item.activeStepCount)
      : progress / 100

  if (!Number.isFinite(stepProgressHint) || stepProgressHint <= 0) return null
  const totalEstimateMs = elapsedMs / stepProgressHint
  const remainingMs = totalEstimateMs - elapsedMs
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) return null

  const seconds = Math.round(remainingMs / 1000)
  if (seconds < 60) return `~${seconds}s`
  const minutes = Math.round(seconds / 60)
  return `~${minutes}m`
}
