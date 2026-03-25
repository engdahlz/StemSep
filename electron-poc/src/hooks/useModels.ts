import { useEffect, useState } from 'react'
import { useStore } from '../stores/useStore'
import { logger } from '../utils/logger'
import { normalizeModel } from '../lib/models/normalizeModel'

export function useModels() {
    const { setModels } = useStore()
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const loadModels = async () => {
            if (!window.electronAPI?.getCatalog && !window.electronAPI?.getModels) {
                console.log('Electron API not available')
                setLoading(false)
                return
            }

            try {
                const catalog = window.electronAPI?.getCatalog
                    ? await window.electronAPI.getCatalog()
                    : null
                const backendModels = Array.isArray(catalog?.models)
                    ? catalog.models
                    : window.electronAPI?.getModels
                        ? await window.electronAPI.getModels()
                        : []

                const modelsArray = Array.isArray(backendModels) ? backendModels : Object.values(backendModels)

                const convertedModels = modelsArray.map(normalizeModel)

                setModels(convertedModels)
                logger.info(`Loaded ${convertedModels.length} models`, { count: convertedModels.length }, 'useModels')
            } catch (err) {
                const errorMessage = err instanceof Error ? err.message : String(err)
                logger.error('Failed to load models', errorMessage, 'useModels')
                setError(errorMessage)
            } finally {
                setLoading(false)
            }
        }

        loadModels()
    }, [setModels])

    return { loading, error }
}
