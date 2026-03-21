import { useEffect, useSyncExternalStore } from 'react'
import { TtlCache } from '../lib/systemRuntime/ttlCache'

type RuntimeSnapshot = {
  info: SystemRuntimeInfo | null
  loading: boolean
  error: string | null
  lastUpdated: number
}

const POLL_INTERVAL_MS = 30_000
const runtimeInfoCache = new TtlCache<SystemRuntimeInfo>(POLL_INTERVAL_MS)

let snapshot: RuntimeSnapshot = {
  info: null,
  loading: false,
  error: null,
  lastUpdated: 0,
}

let consumers = 0
let pollTimer: ReturnType<typeof setInterval> | null = null
let inflight: Promise<void> | null = null
let playbackCaptureBusy = false
let playbackCaptureListenerCleanup: (() => void) | null = null
const listeners = new Set<() => void>()

const notify = () => {
  for (const listener of listeners) {
    listener()
  }
}

const setSnapshot = (next: Partial<RuntimeSnapshot>) => {
  snapshot = { ...snapshot, ...next }
  notify()
}

const fetchRuntimeInfo = async (force = false): Promise<void> => {
  if (playbackCaptureBusy) {
    if (!snapshot.info) {
      setSnapshot({
        loading: false,
        error: null,
      })
    }
    return
  }

  if (!force) {
    const cached = runtimeInfoCache.get()
    if (cached) {
      setSnapshot({
        info: cached,
        loading: false,
        error: null,
      })
      return
    }
  }

  if (inflight && !force) return inflight

  inflight = (async () => {
    if (!snapshot.info) {
      setSnapshot({ loading: true })
    }

    try {
      if (window.electronAPI?.getSystemRuntimeInfo) {
        const info = await window.electronAPI.getSystemRuntimeInfo()
        runtimeInfoCache.set(info)
        setSnapshot({
          info,
          loading: false,
          error: null,
          lastUpdated: Date.now(),
        })
        return
      }

      if (window.electronAPI?.getGpuDevices) {
        const gpu = await window.electronAPI.getGpuDevices()
        const fallbackInfo: SystemRuntimeInfo = {
          fetchedAt: new Date().toISOString(),
          cache: { ttlMs: POLL_INTERVAL_MS, gpuSource: 'fresh', runtimeFingerprintSource: 'error' },
          gpu,
          runtimeFingerprint: null,
          runtimeFingerprintError: 'get_system_runtime_info is not available in this build',
          previewCachePolicy: { ephemeral: true },
        }
        runtimeInfoCache.set(fallbackInfo)
        setSnapshot({
          info: fallbackInfo,
          loading: false,
          error: null,
          lastUpdated: Date.now(),
        })
        return
      }

      setSnapshot({
        loading: false,
        error: 'Electron API is unavailable',
      })
    } catch (error) {
      setSnapshot({
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  })().finally(() => {
    inflight = null
  })

  return inflight
}

const subscribe = (listener: () => void) => {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

const getSnapshot = () => snapshot

const startPolling = () => {
  if (pollTimer) return
  if (!playbackCaptureListenerCleanup && window.electronAPI?.onPlaybackCaptureProgress) {
    playbackCaptureListenerCleanup = window.electronAPI.onPlaybackCaptureProgress((data) => {
      const status = String(data?.status || "").toLowerCase()
      playbackCaptureBusy =
        status === "launching" ||
        status === "awaiting_audio" ||
        status === "capturing" ||
        status === "saving" ||
        status === "cancelling"
      if (!playbackCaptureBusy) {
        void fetchRuntimeInfo(true)
      }
    })
  }
  runtimeInfoCache.clear()
  void fetchRuntimeInfo()
  pollTimer = setInterval(() => {
    void fetchRuntimeInfo()
  }, POLL_INTERVAL_MS)
}

const stopPolling = () => {
  if (!pollTimer) return
  clearInterval(pollTimer)
  pollTimer = null
  if (playbackCaptureListenerCleanup) {
    playbackCaptureListenerCleanup()
    playbackCaptureListenerCleanup = null
  }
  playbackCaptureBusy = false
}

export function useSystemRuntimeInfo() {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)

  useEffect(() => {
    consumers += 1
    startPolling()
    return () => {
      consumers -= 1
      if (consumers <= 0) {
        consumers = 0
        stopPolling()
      }
    }
  }, [])

  return {
    ...state,
    refresh: (force = true) => fetchRuntimeInfo(force),
  }
}
