import { useCallback, useMemo } from 'react'
import { useStore } from '../stores/useStore'
import { SeparationConfig } from '../components/ConfigurePage'
import { toast } from 'sonner'
import { ALL_PRESETS, Preset } from '../presets'

export const useSeparation = () => {
    const queue = useStore(state => state.separation.queue)
    const isPaused = useStore(state => state.separation.isPaused)
    const outputDirectory = useStore(state => state.settings.defaultOutputDir)
    const setOutputDirectory = useStore(state => state.setDefaultOutputDir)

    const setSeparationStatus = useStore(state => state.startSeparation)
    const setSeparationProgress = useStore(state => state.setSeparationProgress)
    const updateQueueItem = useStore(state => state.updateQueueItem)
    const completeSeparation = useStore(state => state.completeSeparation)
    const addLog = useStore(state => state.addLog)
    const addToHistory = useStore(state => state.addToHistory)
    const resumeQueue = useStore(state => state.resumeQueue)
    const phaseParams = useStore(state => state.settings.phaseParams)

    const recipes = useStore(state => state.recipes)

    const combinedPresets: Preset[] = useMemo(() => {
        // Safety check for recipes
        if (!Array.isArray(recipes)) {
            return ALL_PRESETS
        }

        // Convert recipes to Preset format
        const recipePresets = recipes.map(r => ({
            id: r.id,
            name: r.name,
            description: r.description,
            stems: r.target ? [r.target] : [],
            recommended: false,
            category: 'smart',
            modelId: undefined,
            ensembleConfig: {
                models: r.steps.map((s: any) => ({
                    model_id: s.model_id,
                    weight: s.weight || 1.0
                })),
                algorithm: r.algorithm || 'average'
            },
            isRecipe: true
        })) as unknown as Preset[]

        return [...ALL_PRESETS, ...recipePresets]
    }, [recipes])

    const handleSelectOutputDirectory = useCallback(async () => {
        if (!window.electronAPI) return

        const selectedPath = await window.electronAPI.selectOutputDirectory()
        if (selectedPath) {
            setOutputDirectory(selectedPath)
        }
    }, [setOutputDirectory])

    const startSeparation = useCallback(async (config: SeparationConfig) => {
        if (queue.length === 0) return
        if (!outputDirectory) {
            toast.error("Please select an output folder first")
            handleSelectOutputDirectory()
            return
        }

        // Ensure queue is running
        if (isPaused) {
            resumeQueue()
        }

        setSeparationStatus()
        addLog(`Starting batch separation of ${queue.length} files...`)

        let processedCount = 0
        const totalFiles = queue.length
        const allOutputs: Record<string, string[]> = {}

        // Process queue sequentially
        // Note: We access current queue from store inside loop? No, queue is from closure.
        // If queue changes during processing, this might be stale.
        // However, standard practice here usually involves blocking UI or locking queue.
        // Ideally we should ref the latest queue or use an index.
        // But for now we stick to the existing logic which iterates the queue captured at start.

        for (const item of queue) {
            // Check if item is already completed (e.g. from previous run)
            if (item.status === 'completed') {
                processedCount++
                continue
            }

            updateQueueItem(item.id, {
                status: 'processing',
                progress: 0,
                startTime: Date.now(),
                message: 'Pre-flight...'
            })
            const fileName = item.file.split(/[\\/]/).pop() || item.file
            setSeparationProgress(Math.round((processedCount / totalFiles) * 100), `Processing ${fileName}...`)

            try {
                // DEBUG: Log full config to trace ensemble issue
                console.log('[useSeparation] FULL CONFIG RECEIVED:', JSON.stringify(config, null, 2))

                // Determine model and stems
                let modelId = config.modelId || 'htdemucs'
                let stems = config.stems

                // If custom ensemble is configured, it takes priority over everything
                if (config.ensembleConfig && config.ensembleConfig.models && config.ensembleConfig.models.length > 0) {
                    // Ensemble mode - use ensemble marker as modelId
                    modelId = 'ensemble'
                    // Stems default to vocals/instrumental for ensembles
                    stems = stems || ['vocals', 'instrumental']
                    console.log('[useSeparation] Using ensemble config:', config.ensembleConfig)
                } else if (config.mode === 'simple' && config.presetId) {
                    // Simple mode with preset - use preset config
                    const preset = combinedPresets.find(p => p.id === config.presetId)
                    if (preset) {
                        modelId = preset.modelId || modelId
                        stems = preset.stems
                    }
                }

                if (!window.electronAPI) throw new Error('Electron API not available')

                const preflight = window.electronAPI.separationPreflight
                    ? await window.electronAPI.separationPreflight(
                        item.file,
                        modelId,
                        outputDirectory,
                        stems,
                        undefined,
                        config.advancedParams?.overlap,
                        config.advancedParams?.segmentSize,
                        config.advancedParams?.shifts,
                        config.outputFormat,
                        config.advancedParams?.bitrate,
                        config.advancedParams?.tta,
                        config.ensembleConfig,
                        config.ensembleConfig?.algorithm,
                        config.invert,
                        phaseParams?.enabled ? phaseParams : undefined
                    )
                    : null

                const warnings = (preflight as any)?.warnings as string[] | undefined
                if (warnings && warnings.length > 0 && processedCount === 0) {
                    toast.warning(warnings[0])
                    addLog(`Pre-flight warning: ${warnings[0]}`)
                }

                const canProceed = (preflight as any)?.can_proceed
                if (canProceed === false) {
                    const errors = (preflight as any)?.errors as string[] | undefined
                    throw new Error(errors?.[0] || 'Pre-flight failed')
                }

                // Call backend
                const result = await window.electronAPI.separateAudio(
                    item.file,
                    modelId,
                    outputDirectory,
                    stems,
                    undefined, // device (optional)
                    config.advancedParams?.overlap,
                    config.advancedParams?.segmentSize,
                    config.advancedParams?.shifts,
                    config.outputFormat,
                    config.advancedParams?.bitrate,
                    config.advancedParams?.tta,
                    config.ensembleConfig,
                    config.ensembleConfig?.algorithm,
                    config.invert,
                    phaseParams?.enabled ? phaseParams : undefined
                )

                console.log('Separation Result (Frontend):', result)

                if (!result.success) {
                    throw new Error(result.error || 'Separation failed')
                }

                const outputs = (result.outputFiles as unknown as Record<string, string>) || {}
                allOutputs[item.id] = Object.values(outputs)

                // Update item status
                updateQueueItem(item.id, {
                    status: 'completed',
                    progress: 100,
                    outputFiles: outputs,
                    backendJobId: result.jobId
                })

                // Add to history
                // Determine display name - ensemble takes priority
                const isCustomEnsemble = config.ensembleConfig?.models && config.ensembleConfig.models.length > 0
                const presetName = isCustomEnsemble
                    ? `Custom Ensemble (${config.ensembleConfig!.models.length} models)`
                    : config.presetId
                        ? combinedPresets.find(p => p.id === config.presetId)?.name
                        : config.modelId

                addToHistory({
                    inputFile: item.file,
                    outputDir: outputDirectory,
                    modelId: isCustomEnsemble ? 'custom-ensemble' : (config.presetId || 'custom'),
                    modelName: presetName || config.modelId || 'Unknown Model',
                    status: 'completed',
                    outputFiles: outputs,
                    backendJobId: result.jobId,
                    settings: {
                        stems: stems || [],
                        overlap: config.advancedParams?.overlap,
                        segmentSize: config.advancedParams?.segmentSize
                    }
                })

                toast.success(`Separated: ${fileName}`)

            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error'

                // Update item status to failed
                updateQueueItem(item.id, { status: 'failed', error: errorMessage })

                toast.error(`Failed to separate ${fileName}: ${errorMessage}`)
            }
            processedCount++
            // Note: progressMessage here is from outer scope (store value at render time).
            // Better to pass a fresh message or undefined if we want to keep current.
            // But setSeparationProgress updates the store message.
            setSeparationProgress(Math.round((processedCount / totalFiles) * 100), `Processing next file...`)
        }

        // Finalize batch
        completeSeparation({})

        addLog('Batch processing complete!')
        toast.success('Batch processing complete!')
    }, [
        queue,
        outputDirectory,
        isPaused,
        resumeQueue,
        setSeparationStatus,
        addLog,
        updateQueueItem,
        setSeparationProgress,
        combinedPresets,
        phaseParams,
        completeSeparation,
        addToHistory,
        handleSelectOutputDirectory
    ])

    return { startSeparation }
}
