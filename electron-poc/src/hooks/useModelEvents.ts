import { useEffect } from 'react'
import { useStore } from '../stores/useStore'
import { logger } from '../utils/logger'
import { normalizeModel } from '../lib/models/normalizeModel'

export function useModelEvents() {
    const {
        setDownloadProgress,
        completeDownload,
        setDownloadError,
        pauseDownload,
        mergeModel,
    } = useStore()

    useEffect(() => {
        if (!window.electronAPI) return

        // Progress listener
        const removeProgressListener = window.electronAPI.onDownloadProgress((data: {
            modelId: string,
            progress: number,
            stage?: string,
            artifactIndex?: number,
            artifactCount?: number,
            currentFile?: string,
            currentRelativePath?: string,
            currentSource?: string,
            verified?: boolean,
            message?: string,
            speed?: number,
            eta?: number,
        }) => {
            // logger.debug(`Download progress: ${data.modelId} - ${data.progress}%`, undefined, 'useModelEvents')
            setDownloadProgress(data)
        })

        // Complete listener
        const removeCompleteListener = window.electronAPI.onDownloadComplete(async (data: { modelId: string }) => {
            logger.info(`Download complete: ${data.modelId}`, undefined, 'useModelEvents')
            completeDownload(data.modelId)
            try {
                const catalog = await window.electronAPI?.getCatalog?.()
                const refreshed = Array.isArray(catalog?.models)
                    ? catalog.models.find((entry: { id?: string }) => entry?.id === data.modelId)
                    : null

                if (refreshed) {
                    mergeModel(data.modelId, normalizeModel(refreshed))
                    return
                }

                const installation = await window.electronAPI?.getSelectionInstallation?.("model", data.modelId)
                mergeModel(data.modelId, normalizeModel({
                    id: data.modelId,
                    installation,
                    installed: Boolean(installation?.installed),
                    downloading: false,
                    downloadPaused: false,
                    downloadProgress: 100,
                    downloadState: installation?.installed ? "installed" : "failed",
                    downloadVerified: Boolean(installation?.installed),
                }))
            } catch (error) {
                logger.error(`Failed to refresh installation after download: ${data.modelId}`, error, 'useModelEvents')
            }
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
    }, [setDownloadProgress, completeDownload, setDownloadError, pauseDownload, mergeModel])
}
