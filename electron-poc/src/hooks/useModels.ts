import { useEffect, useState } from 'react'
import { useStore } from '../stores/useStore'
import { logger } from '../utils/logger'

export function useModels() {
    const { setModels } = useStore()
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const loadModels = async () => {
            if (!window.electronAPI?.getModels) {
                console.log('Electron API not available')
                setLoading(false)
                return
            }

            try {
                const backendModels = await window.electronAPI.getModels()

                // Convert backend models to store format
                // We assume backendModels is an array based on usage in other files
                const modelsArray = Array.isArray(backendModels) ? backendModels : Object.values(backendModels)

                const convertedModels = modelsArray.map((m: any) => ({
                    ...m,
                    category: m.category || 'primary',
                    installed: m.installed || false,
                    downloading: false,
                    downloadPaused: false,
                    downloadProgress: 0,
                    recommended: m.recommended || false,
                }))

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
