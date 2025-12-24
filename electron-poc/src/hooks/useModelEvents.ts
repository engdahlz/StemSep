import { useEffect } from 'react'
import { useStore } from '../stores/useStore'
import { logger } from '../utils/logger'

export function useModelEvents() {
    const {
        setDownloadProgress,
        completeDownload,
        setDownloadError,
        pauseDownload
    } = useStore()

    useEffect(() => {
        if (!window.electronAPI) return

        // Progress listener
        const removeProgressListener = window.electronAPI.onDownloadProgress((data: { modelId: string, progress: number, speed?: number, eta?: number }) => {
            // logger.debug(`Download progress: ${data.modelId} - ${data.progress}%`, undefined, 'useModelEvents')
            setDownloadProgress(data)
        })

        // Complete listener
        const removeCompleteListener = window.electronAPI.onDownloadComplete((data: { modelId: string }) => {
            logger.info(`Download complete: ${data.modelId}`, undefined, 'useModelEvents')
            completeDownload(data.modelId)
        })

        // Error listener
        const removeErrorListener = window.electronAPI.onDownloadError((data: { modelId: string, error: string }) => {
            logger.error(`Download error: ${data.modelId}`, data.error, 'useModelEvents')
            setDownloadError(data.modelId, data.error)
        })

        // Paused listener
        const removePausedListener = window.electronAPI.onDownloadPaused((data: { modelId: string }) => {
            logger.info(`Download paused: ${data.modelId}`, undefined, 'useModelEvents')
            pauseDownload(data.modelId)
        })

        return () => {
            removeProgressListener()
            removeCompleteListener()
            removeErrorListener()
            removePausedListener()
        }
    }, [setDownloadProgress, completeDownload, setDownloadError, pauseDownload])
}
