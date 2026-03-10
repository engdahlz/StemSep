import { useState, useEffect, useMemo } from 'react'
import { X, Info, Settings2, Layers, Zap } from 'lucide-react'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { useStore } from '../stores/useStore'
import { getRequiredModels, Preset } from '../presets'
import type { SeparationConfig } from '../types/separation'
import { EnsembleBuilder } from './EnsembleBuilder'
import { PresetSelector } from './PresetSelector'
import { VRAMUsageMeter, estimateVRAMUsage } from './ui/vram-meter'
import { TTAWarning, CPUOnlyWarning, LowVRAMWarning, EnsembleTip } from './ui/warning-tip'
import { bestVolumeCompensation } from '../utils/volumeCompensation'
import { useSystemRuntimeInfo } from '../hooks/useSystemRuntimeInfo'

function buildPresetWorkflow(preset: Preset, outputFormat: 'wav' | 'mp3' | 'flac') {
    if (preset.isRecipe && preset.recipe) {
        return {
            version: 1 as const,
            id: preset.id,
            name: preset.name,
            kind: preset.recipe.type === 'pipeline' || preset.recipe.type === 'chained'
                ? 'pipeline' as const
                : preset.recipe.type === 'ensemble'
                    ? 'ensemble' as const
                    : 'single' as const,
            surface: preset.recipe.surface || (
                preset.recipe.target === 'restoration'
                ? 'restoration' as const
                : preset.recipe.target === 'drums' || preset.recipe.target === 'bass'
                    ? 'special_stem' as const
                    : 'workflow' as const
            ),
            family: preset.recipe.family,
            description: preset.workflowSummary || preset.description,
            stems: preset.stems,
            steps: preset.recipe.steps as any[],
            intermediateOutputs: preset.recipe.intermediate_outputs,
            operatingProfile: preset.recipe.operating_profile,
            fallbackPolicy: preset.recipe.fallback_policy ? {
                mode: preset.recipe.fallback_policy.mode,
                reason: preset.recipe.fallback_policy.reason,
                runtimeOrder: preset.recipe.fallback_policy.runtime_order,
                fallbackWorkflowId: preset.recipe.fallback_policy.fallback_workflow_id,
                fallbackOperatingProfile: preset.recipe.fallback_policy.fallback_operating_profile,
            } : undefined,
            runtimePolicy: preset.recipe.runtime_policy ? {
                required: preset.recipe.runtime_policy.required,
                fallbacks: preset.recipe.runtime_policy.fallbacks,
                allowManualModels: preset.recipe.runtime_policy.allow_manual_models,
                preferredRuntime: preset.recipe.runtime_policy.preferred_runtime,
            } : undefined,
            exportPolicy: {
                stems: preset.stems,
                outputFormat,
                intermediateOutputs: preset.recipe.export_policy?.intermediate_outputs || preset.recipe.intermediate_outputs,
            },
        }
    }

    if (preset.ensembleConfig) {
        return {
            version: 1 as const,
            id: preset.id,
            name: preset.name,
            kind: 'ensemble' as const,
            surface: 'ensemble' as const,
            description: preset.description,
            stems: preset.stems,
            models: preset.ensembleConfig.models.map(model => ({
                model_id: model.model_id,
                weight: model.weight,
                role: 'ensemble_partner' as const,
            })),
            blend: {
                algorithm: preset.ensembleConfig.algorithm,
                stemAlgorithms: preset.ensembleConfig.stemAlgorithms,
                phaseFixEnabled: preset.ensembleConfig.phaseFixEnabled,
                phaseFixParams: preset.ensembleConfig.phaseFixParams,
            },
            exportPolicy: {
                stems: preset.stems,
                outputFormat,
            },
        }
    }

    if (preset.modelId) {
        return {
            version: 1 as const,
            id: preset.id,
            name: preset.name,
            kind: 'single' as const,
            surface: 'single' as const,
            description: preset.description,
            stems: preset.stems,
            models: [{ model_id: preset.modelId, role: 'primary' as const }],
            exportPolicy: {
                stems: preset.stems,
                outputFormat,
            },
        }
    }

    return undefined
}

export type { SeparationConfig } from '../types/separation'

interface SeparationConfigDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    onConfirm: (config: SeparationConfig) => void
    initialPresetId?: string
    presets: Preset[]
    availability?: Record<string, any>
    modelMap?: Record<string, string>
    models?: any[]
    onNavigateToModels?: (modelId?: string) => void
    onShowModelDetails?: (modelId: string) => void
}

export default function SeparationConfigDialog({
    open,
    onOpenChange,
    onConfirm,
    initialPresetId,
    presets = [],
    availability,
    modelMap = {},
    models = [],
    onNavigateToModels,
    onShowModelDetails
}: SeparationConfigDialogProps) {
    type SimpleGoalFilter = 'all' | 'instrumental' | 'vocals' | 'karaoke' | 'cleanup' | 'instruments'
    type QualityIntentFilter = 'all' | 'fullness' | 'bleedless' | 'lead_back' | 'restoration' | 'special'

    const globalAdvancedSettings = useStore(state => state.settings.advancedSettings)
    const normalizeOverlap = (value: any) => {
        const n = typeof value === 'number' ? value : Number(value)
        if (!Number.isFinite(n)) return 4
        if (n < 1) {
            const denom = Math.max(1e-6, 1 - n)
            return Math.max(2, Math.min(50, Math.round(1 / denom)))
        }
        return Math.max(2, Math.min(50, Math.round(n)))
    }
    const [mode, setMode] = useState<'simple' | 'advanced'>('simple')
    const [selectedPresetId, setSelectedPresetId] = useState<string>(initialPresetId || (presets.length > 0 ? presets[0].id : ''))
    const [simpleGoalFilter, setSimpleGoalFilter] = useState<SimpleGoalFilter>('all')
    const [qualityIntentFilter, setQualityIntentFilter] = useState<QualityIntentFilter>('all')
    const [selectedModelId, setSelectedModelId] = useState<string>('')
    const [device, setDevice] = useState<string>((globalAdvancedSettings?.device === 'cuda' ? 'cuda:0' : globalAdvancedSettings?.device) || 'auto')
    // Preview/playback is always WAV. Export format is chosen later in Results.
    const [outputFormat] = useState<'wav' | 'mp3' | 'flac'>('wav')
    const [invert, setInvert] = useState(false)
    const [normalize, setNormalize] = useState(false)
    const [volumeCompEnabled, setVolumeCompEnabled] = useState(false)
    const [bitDepth, setBitDepth] = useState('16')
    const [splitFreq, setSplitFreq] = useState(750)
    const [advancedParams, setAdvancedParams] = useState({
        overlap: normalizeOverlap(globalAdvancedSettings?.overlap ?? 4),
        segmentSize: globalAdvancedSettings?.segmentSize ?? 0,
        shifts: globalAdvancedSettings?.shifts || 1,
        tta: false,
        bitrate: globalAdvancedSettings?.bitrate || '320k'
    })

    // Simple mode is locked to presets; phase-correction is only available in Advanced mode.

    // Ensemble State
    const [isEnsembleMode, setIsEnsembleMode] = useState(false)
    const [ensembleConfig, setEnsembleConfig] = useState<{ model_id: string; weight: number }[]>([])
    const [ensembleAlgorithm, setEnsembleAlgorithm] = useState<'average' | 'max_spec' | 'min_spec' | 'phase_fix' | 'frequency_split'>('average')
    const [ensembleStemAlgorithms, setEnsembleStemAlgorithms] = useState<{ vocals?: 'average' | 'max_spec' | 'min_spec'; instrumental?: 'average' | 'max_spec' | 'min_spec' }>()
    const [phaseFixEnabled, setPhaseFixEnabled] = useState(false)
    const [phaseFixParams, setPhaseFixParams] = useState<{ enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }>()
    const { info: runtimeInfo } = useSystemRuntimeInfo()
    const gpuInfo = runtimeInfo?.gpu ?? null

    // Update selected preset when initialPresetId changes
    useEffect(() => {
        if (initialPresetId) {
            setSelectedPresetId(initialPresetId)
        }
    }, [initialPresetId])

    const inferSimpleGoal = (preset: Preset): Exclude<SimpleGoalFilter, 'all'> | undefined => {
        if (preset.simpleGoal) return preset.simpleGoal
        if (preset.category === 'vocals') return 'vocals'
        if (preset.category === 'instrumental') return 'instrumental'
        if (preset.category === 'instruments') return 'instruments'
        if (preset.category === 'utility' && preset.tags?.includes('karaoke')) return 'karaoke'
        if (preset.category === 'utility') return 'cleanup'
        return undefined
    }

    const matchesQualityIntent = (preset: Preset, filter: QualityIntentFilter) => {
        if (filter === 'all') return true
        const haystack = [
            preset.name,
            preset.description,
            preset.workflowSummary,
            ...(preset.tags || []),
            ...(preset.recommendedFor || []),
        ]
            .filter(Boolean)
            .join(' ')
            .toLowerCase()

        if (filter === 'fullness') return /fullness|body|warm|detail/.test(haystack)
        if (filter === 'bleedless') return /bleed|clean|cleanup|debleed/.test(haystack)
        if (filter === 'lead_back') return /karaoke|lead\/back|lead|backing|harmony|duet/.test(haystack)
        if (filter === 'restoration') return /restoration|cleanup|dereverb|denoise|live/.test(haystack)
        if (filter === 'special') return /drum|bass|guitar|special/.test(haystack)
        return true
    }

    const filteredPresets = useMemo(() => {
        return presets.filter((preset) => {
            const goal = inferSimpleGoal(preset)
            const goalMatch = simpleGoalFilter === 'all' || goal === simpleGoalFilter
            const qualityMatch = matchesQualityIntent(preset, qualityIntentFilter)
            return goalMatch && qualityMatch
        })
    }, [presets, simpleGoalFilter, qualityIntentFilter])

    useEffect(() => {
        if (!filteredPresets.some((preset) => preset.id === selectedPresetId)) {
            setSelectedPresetId(filteredPresets[0]?.id || '')
        }
    }, [filteredPresets, selectedPresetId])

    // Update selected model when preset changes
    useEffect(() => {
        if (mode === 'simple' && selectedPresetId) {
            const preset = presets.find(p => p.id === selectedPresetId)
            if (preset) {
                // If it's an ensemble preset, we don't set a single model ID
                if (preset.ensembleConfig) {
                    // Logic for ensemble preset display handled in render
                } else {
                    const modelId = preset.modelId || modelMap[preset.id]
                    if (modelId) setSelectedModelId(modelId)
                }
            }
        }
    }, [selectedPresetId, mode, presets, modelMap])

    const handleConfirm = () => {
        const config: SeparationConfig = {
            mode,
            // Presets are only meaningful in Simple mode. In Advanced mode, avoid carrying a stale
            // presetId (it can cause Results to show the preset label even when a custom modelId ran).
            presetId: mode === 'simple' ? selectedPresetId : undefined,
            workflowId: mode === 'simple' ? selectedPresetId : undefined,
            modelId: isEnsembleMode ? undefined : selectedModelId,
            device,
            outputFormat,
            invert,
            normalize,
            volumeCompensation: volumeCompEnabled
                ? bestVolumeCompensation()
                : undefined,
            bitDepth,
            splitFreq: (isEnsembleMode && ensembleAlgorithm === 'frequency_split') ? splitFreq : undefined,
            advancedParams: mode === 'advanced' ? advancedParams : undefined,
            ensembleConfig: (mode === 'advanced' && isEnsembleMode) ? {
                models: ensembleConfig,
                algorithm: ensembleAlgorithm as any,
                stemAlgorithms: ensembleStemAlgorithms,
                phaseFixEnabled,
                phaseFixParams: phaseFixParams ? {
                    lowHz: phaseFixParams.lowHz,
                    highHz: phaseFixParams.highHz,
                    highFreqWeight: phaseFixParams.highFreqWeight,
                } : undefined,
            } : undefined,
            workflow: undefined,
        }

        // Simple mode is locked to preset definitions.
        // - If the preset defines an ensembleConfig, use it.
        // - Otherwise use the preset's modelId mapping (already reflected in selectedModelId).
        if (mode === 'simple') {
            const preset = presets.find(p => p.id === selectedPresetId)
            config.workflow = preset ? buildPresetWorkflow(preset, outputFormat) : undefined
            config.runtimePolicy = (preset?.isRecipe && preset.recipe?.runtime_policy) ? {
                required: preset.recipe.runtime_policy.required,
                fallbacks: preset.recipe.runtime_policy.fallbacks,
                allowManualModels: preset.recipe.runtime_policy.allow_manual_models,
                preferredRuntime: preset.recipe.runtime_policy.preferred_runtime,
            } : undefined
            config.exportPolicy = {
                stems: preset?.stems,
                outputFormat,
            }
            if (preset?.ensembleConfig) {
                config.ensembleConfig = preset.ensembleConfig
                config.modelId = undefined
            }
        } else if (mode === 'advanced' && isEnsembleMode) {
            config.workflow = {
                version: 1,
                name: 'Custom Ensemble',
                kind: 'ensemble',
                surface: 'ensemble',
                stems: ['vocals', 'instrumental'],
                models: ensembleConfig.map((model, index) => ({
                    model_id: model.model_id,
                    weight: model.weight,
                    role: phaseFixEnabled && index === 1 ? 'phase_reference' : 'ensemble_partner',
                })),
                blend: {
                    algorithm: ensembleAlgorithm as any,
                    stemAlgorithms: ensembleStemAlgorithms,
                    phaseFixEnabled,
                    phaseFixParams: phaseFixParams ? {
                        lowHz: phaseFixParams.lowHz,
                        highHz: phaseFixParams.highHz,
                        highFreqWeight: phaseFixParams.highFreqWeight,
                    } : undefined,
                    splitFreq: ensembleAlgorithm === 'frequency_split' ? splitFreq : undefined,
                },
                exportPolicy: {
                    stems: ['vocals', 'instrumental'],
                    outputFormat,
                },
            }
        } else if (mode === 'advanced' && selectedModelId) {
            config.workflow = {
                version: 1,
                kind: 'single',
                surface: 'single',
                models: [{ model_id: selectedModelId, role: 'primary' }],
                stems: ['vocals', 'instrumental'],
                exportPolicy: {
                    stems: ['vocals', 'instrumental'],
                    outputFormat,
                },
            }
        }

        onConfirm(config)
        onOpenChange(false)
    }

    const selectedPreset = filteredPresets.find(p => p.id === selectedPresetId) || presets.find(p => p.id === selectedPresetId)

    const cudaGpus = Array.isArray(gpuInfo?.gpus)
        ? gpuInfo.gpus.filter((g: any) => g?.type === 'cuda')
        : []
    const hasCuda = !!gpuInfo?.has_cuda || cudaGpus.length > 0
    const primaryCudaDevice = cudaGpus[0]
        ? `cuda:${Number.isFinite(cudaGpus[0].index) ? cudaGpus[0].index : 0}`
        : 'cuda:0'

    useEffect(() => {
        if (!hasCuda && device.startsWith('cuda')) {
            setDevice('cpu')
        }
        if (hasCuda && device === 'cuda') {
            setDevice(primaryCudaDevice)
        }
    }, [hasCuda, device, primaryCudaDevice])

    if (!open) return null


    // Check availability
    let isAvailable = true
    let missingModels: string[] = []

    if (mode === 'simple' && selectedPreset) {
        const requiredModels = getRequiredModels(selectedPreset)
        if (requiredModels.length > 0) {
            requiredModels.forEach(modelId => {
                if (availability && availability[modelId]?.available === false) {
                    isAvailable = false
                    missingModels.push(modelId)
                }
            })
        } else {
            const fallbackModelId = selectedPreset.modelId || modelMap[selectedPreset.id]
            if (fallbackModelId && availability && availability[fallbackModelId]?.available === false) {
                isAvailable = false
                missingModels.push(fallbackModelId)
            }
        }
    } else if (mode === 'advanced' && !isEnsembleMode && selectedModelId) {
        if (availability && availability[selectedModelId]?.available === false) {
            isAvailable = false
            missingModels.push(selectedModelId)
        }
    } else if (mode === 'advanced' && isEnsembleMode) {
        ensembleConfig.forEach(m => {
            if (availability && availability[m.model_id]?.available === false) {
                isAvailable = false
                missingModels.push(m.model_id)
            }
        })
    }


    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <Card className="w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl bg-background text-foreground border-border">
                <div className="flex items-center justify-between p-6 border-b">
                    <div>
                        <h2 className="text-2xl font-bold">Separation Configuration</h2>
                        <p className="text-muted-foreground">Configure how your audio will be processed</p>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)}>
                        <X className="h-5 w-5" />
                    </Button>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Mode Tabs */}
                    <div className="grid w-full grid-cols-2 mb-6 bg-muted p-1 rounded-lg">
                        <button
                            onClick={() => setMode('simple')}
                            className={`py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === 'simple' ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                        >
                            Simple Mode
                        </button>
                        <button
                            onClick={() => setMode('advanced')}
                            className={`py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === 'advanced' ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                        >
                            Advanced Mode
                        </button>
                    </div>

                    {mode === 'simple' ? (
                        <div className="space-y-6">
                            {/* Preset Selection - main choice */}
                            <div className="space-y-3">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Quality Mode</label>
                                    <div className="flex flex-wrap gap-2">
                                        {([
                                            ['all', 'All'],
                                            ['instrumental', 'Instrumental'],
                                            ['vocals', 'Vocals'],
                                            ['karaoke', 'Karaoke'],
                                            ['cleanup', 'Restoration'],
                                            ['instruments', 'Special Stems'],
                                        ] as Array<[SimpleGoalFilter, string]>).map(([value, label]) => (
                                            <button
                                                key={value}
                                                type="button"
                                                onClick={() => setSimpleGoalFilter(value)}
                                                className={`rounded-full border px-3 py-1 text-xs transition-colors ${simpleGoalFilter === value ? 'border-primary bg-primary/10 text-foreground' : 'border-border bg-background/70 text-muted-foreground hover:text-foreground'}`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Quality Goal</label>
                                    <div className="flex flex-wrap gap-2">
                                        {([
                                            ['all', 'All'],
                                            ['fullness', 'Max Fullness'],
                                            ['bleedless', 'Min Bleed'],
                                            ['lead_back', 'Lead/Back'],
                                            ['restoration', 'Restoration'],
                                            ['special', 'Special Stems'],
                                        ] as Array<[QualityIntentFilter, string]>).map(([value, label]) => (
                                            <button
                                                key={value}
                                                type="button"
                                                onClick={() => setQualityIntentFilter(value)}
                                                className={`rounded-full border px-3 py-1 text-xs transition-colors ${qualityIntentFilter === value ? 'border-primary bg-primary/10 text-foreground' : 'border-border bg-background/70 text-muted-foreground hover:text-foreground'}`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <label className="text-sm font-medium block">Select Preset</label>
                                <PresetSelector
                                    selectedPresetId={selectedPresetId}
                                    onSelectPreset={setSelectedPresetId}
                                    presets={filteredPresets}
                                    availability={availability}
                                />
                                <p className="text-xs text-muted-foreground">
                                    Showing {filteredPresets.length} preset{filteredPresets.length === 1 ? '' : 's'} for the current quality filters.
                                </p>
                            </div>

                            {/* Selected preset info */}
                            {selectedPreset && (
                                <div className="bg-secondary/30 p-4 rounded-lg space-y-2">
                                    <p className="text-sm text-muted-foreground">{selectedPreset.description}</p>
                                    {selectedPreset.workflowSummary && (
                                        <p className="text-sm text-foreground/80">
                                            {selectedPreset.workflowSummary}
                                        </p>
                                    )}
                                    <div className="flex gap-2 flex-wrap">
                                        {selectedPreset.difficulty && (
                                            <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-background/80">
                                                {selectedPreset.difficulty}
                                            </span>
                                        )}
                                        {selectedPreset.expectedRuntimeTier && (
                                            <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-background/80">
                                                Runtime: {selectedPreset.expectedRuntimeTier}
                                            </span>
                                        )}
                                        {selectedPreset.expectedVramTier && (
                                            <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-background/80">
                                                VRAM: {selectedPreset.expectedVramTier}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium">Target Stems:</span>
                                        <div className="flex gap-1 flex-wrap">
                                            {selectedPreset.stems.map(stem => (
                                                <span key={stem} className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-transparent bg-secondary text-secondary-foreground">
                                                    {stem}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {selectedPreset.ensembleConfig && (
                                        <div className="mt-2 pt-2 border-t border-border/50">
                                            <div className="flex items-center gap-2 mb-1">
                                                <Layers className="w-4 h-4 text-primary" />
                                                <span className="text-sm font-medium">Ensemble: {selectedPreset.ensembleConfig.models.length} models</span>
                                            </div>
                                        </div>
                                    )}

                                    {selectedPreset.recommendedFor?.length ? (
                                        <div className="mt-2 pt-2 border-t border-border/50">
                                            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground mb-1">Best For</div>
                                            <p className="text-sm text-foreground/80">{selectedPreset.recommendedFor.join(' • ')}</p>
                                        </div>
                                    ) : null}

                                    {selectedPreset.contraindications?.length ? (
                                        <div className="space-y-1">
                                            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Avoid When</div>
                                            <p className="text-sm text-muted-foreground">{selectedPreset.contraindications.join(' • ')}</p>
                                        </div>
                                    ) : null}

                                    {!isAvailable && (
                                        <div className="flex flex-col gap-2 mt-2 bg-destructive/10 p-3 rounded border border-destructive/20">
                                            <div className="flex items-center text-destructive text-sm">
                                                <Info className="w-4 h-4 mr-2 shrink-0" />
                                                <span>{missingModels.length} model(s) missing</span>
                                            </div>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="w-full border-destructive/30 hover:bg-destructive/10"
                                                onClick={() => {
                                                    if (onShowModelDetails) {
                                                        onShowModelDetails(missingModels[0])
                                                    } else {
                                                        onOpenChange(false)
                                                        onNavigateToModels?.(missingModels[0])
                                                    }
                                                }}
                                            >
                                                Download Missing Models
                                            </Button>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-6">
                            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 text-sm text-yellow-600 dark:text-yellow-400">
                                <div className="flex items-center gap-2 font-semibold mb-1">
                                    <Settings2 className="w-4 h-4" />
                                    Advanced Configuration
                                </div>
                                Manual control over model parameters. Use with caution.
                            </div>

                            {/* Single vs Ensemble Toggle */}
                            <div className="flex items-center space-x-4 mb-4">
                                <button
                                    onClick={() => setIsEnsembleMode(false)}
                                    className={`text-sm font-medium pb-1 border-b-2 transition-colors ${!isEnsembleMode ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'}`}
                                >
                                    Single Model
                                </button>
                                <button
                                    onClick={() => setIsEnsembleMode(true)}
                                    className={`text-sm font-medium pb-1 border-b-2 transition-colors ${isEnsembleMode ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'}`}
                                >
                                    Custom Ensemble
                                </button>
                            </div>

                            {isEnsembleMode ? (
                                <div className="space-y-4">
                                    <EnsembleBuilder
                                        models={models}
                                        config={ensembleConfig}
                                        algorithm={ensembleAlgorithm as any}
                                        phaseFixEnabled={phaseFixEnabled}
                                        stemAlgorithms={ensembleStemAlgorithms}
                                        phaseFixParams={phaseFixParams}
                                        onChange={(cfg, alg, stemAlgorithms, nextPhaseParams, nextPhaseFixEnabled) => {
                                            setEnsembleConfig(cfg)
                                            setEnsembleAlgorithm(alg as any)
                                            setEnsembleStemAlgorithms(stemAlgorithms)
                                            setPhaseFixParams(nextPhaseParams)
                                            setPhaseFixEnabled(nextPhaseFixEnabled ?? false)
                                        }}
                                    />

                                    {ensembleAlgorithm === 'frequency_split' && (
                                        <div className="p-4 rounded-lg border bg-secondary/10 space-y-3 animate-in fade-in slide-in-from-top-2">
                                            <div className="flex justify-between">
                                                <label className="text-sm font-medium">Crossover Frequency</label>
                                                <span className="text-sm font-mono bg-background px-2 rounded border">{splitFreq} Hz</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="200"
                                                max="5000"
                                                step="50"
                                                value={splitFreq}
                                                onChange={(e) => setSplitFreq(parseInt(e.target.value))}
                                                className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                                            />
                                            <p className="text-xs text-muted-foreground">
                                                Frequencies below {splitFreq}Hz use the first model (Low). Above use the second (High).
                                            </p>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <label className="text-sm font-medium block">Select Model</label>
                                    <select
                                        value={selectedModelId}
                                        onChange={(e) => setSelectedModelId(e.target.value)}
                                        className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                        <option value="">Select a model...</option>
                                        {models.map((model: any) => (
                                            <option key={model.id} value={model.id}>
                                                {model.name || model.id}
                                            </option>
                                        ))}
                                    </select>
                                    {selectedModelId && (
                                        <p className="text-xs text-muted-foreground">
                                            Model ID: {selectedModelId}
                                        </p>
                                    )}
                                </div>
                            )}

                            <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Overlap</label>
                                    <input
                                        type="number"
                                        value={advancedParams.overlap}
                                        onChange={(e) => {
                                            const next = parseFloat(e.target.value)
                                            if (!Number.isFinite(next)) return
                                            setAdvancedParams(prev => ({ ...prev, overlap: next }))
                                        }}
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        step={0.05}
                                        min={0}
                                        max={50}
                                    />
                                    <p className="text-xs text-muted-foreground">Overlap ratio (0-0.95) or divisor (2-50)</p>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Segment Size</label>
                                    <input
                                        type="number"
                                        value={advancedParams.segmentSize}
                                        onChange={(e) => setAdvancedParams(prev => ({ ...prev, segmentSize: parseInt(e.target.value) || 256 }))}
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        step={64}
                                    />
                                    <p className="text-xs text-muted-foreground">Segment size for processing</p>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Shifts</label>
                                    <input
                                        type="number"
                                        value={advancedParams.shifts}
                                        onChange={(e) => setAdvancedParams(prev => ({ ...prev, shifts: parseInt(e.target.value) || 1 }))}
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        min={1}
                                        max={10}
                                    />
                                    <p className="text-xs text-muted-foreground">Number of random shifts (1-10)</p>
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                <input
                                    type="checkbox"
                                    id="tta"
                                    checked={advancedParams.tta}
                                    onChange={(e) => setAdvancedParams(prev => ({ ...prev, tta: e.target.checked }))}
                                    className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                                />
                                <label htmlFor="tta" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                                    Test Time Augmentation (TTA)
                                </label>
                            </div>

                        </div>
                    )}

                        <div className="space-y-4 pt-4 border-t">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Preview Output</label>
                                <div className="text-xs text-muted-foreground">
                                    Preview stems are always written as WAV for lossless playback. Use Export in Results to create MP3/FLAC.
                                </div>
                                <div className="mt-2">
                                    <label className="text-xs font-medium block mb-1">Bit Depth (WAV)</label>
                                    <select
                                        value={bitDepth}
                                        onChange={(e) => setBitDepth(e.target.value)}
                                        className="flex h-8 w-full items-center justify-between rounded-md border border-input bg-background px-2 py-1 text-xs"
                                    >
                                        <option value="16">16-bit (Standard)</option>
                                        <option value="24">24-bit (High Res)</option>
                                        <option value="32">32-bit Float (Pro)</option>
                                    </select>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Processing Device</label>
                                <select
                                    value={device}
                                    onChange={(e) => setDevice(e.target.value)}
                                    className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    <option value="auto">Auto (Recommended)</option>
                                    <option value="cpu">CPU</option>
                                    {cudaGpus.length > 0 ? (
                                        cudaGpus.map((g: any, idx: number) => {
                                            const index = Number.isFinite(g?.index) ? g.index : idx
                                            const label = g?.name ? `GPU (CUDA): ${g.name}` : `GPU (CUDA) #${index}`
                                            return (
                                                <option key={`cuda:${index}`} value={`cuda:${index}`}>
                                                    {label}
                                                </option>
                                            )
                                        })
                                    ) : (
                                        <option value={primaryCudaDevice} disabled={!hasCuda}>
                                            GPU (CUDA)
                                        </option>
                                    )}
                                </select>
                            </div>
                        </div>

                        {/* Normalize Checkbox */}
                        <div className="mt-4 p-3 bg-secondary/30 rounded-lg border border-border/50">
                            <div className="flex items-start space-x-3">
                                <div className="flex items-center h-5">
                                    <input
                                        type="checkbox"
                                        id="normalize"
                                        checked={normalize}
                                        onChange={(e) => setNormalize(e.target.checked)}
                                        className="h-4 w-4 rounded border-primary text-primary focus:ring-primary"
                                    />
                                </div>
                                <div className="flex-1">
                                    <label htmlFor="normalize" className="text-sm font-medium text-foreground cursor-pointer">
                                        Normalize Output (-0.1 dB)
                                    </label>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Prevents clipping by scaling the peak volume. Recommended for ensembles.
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Volume Compensation */}
                        <div className="mt-2 p-3 bg-secondary/30 rounded-lg border border-border/50">
                            <div className="flex items-start space-x-3">
                                <div className="flex items-center h-5">
                                    <input
                                        type="checkbox"
                                        id="volumeCompensation"
                                        checked={volumeCompEnabled}
                                        onChange={(e) => setVolumeCompEnabled(e.target.checked)}
                                        className="h-4 w-4 rounded border-primary text-primary focus:ring-primary"
                                    />
                                </div>
                                <div className="flex-1">
                                    <label htmlFor="volumeCompensation" className="text-sm font-medium text-foreground cursor-pointer">
                                        Volume Compensation (VC)
                                    </label>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Adds headroom when combining multiple models (reduces clipping risk). When enabled, uses best defaults.
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Inversion Checkbox */}
                        <div className="mt-2 p-3 bg-secondary/30 rounded-lg border border-border/50">
                            <div className="flex items-start space-x-3">
                                <div className="flex items-center h-5">
                                    <input
                                        type="checkbox"
                                        id="invert"
                                        checked={invert}
                                        onChange={(e) => setInvert(e.target.checked)}
                                        className="h-4 w-4 rounded border-primary text-primary focus:ring-primary"
                                    />
                                </div>
                                <div className="flex-1">
                                    <label htmlFor="invert" className="text-sm font-medium text-foreground cursor-pointer">
                                        Spectral Inversion (Residual Mode)
                                    </label>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Generates complementary stems by subtracting output from original.
                                        Use for "True Instrumental" from Vocal models.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* VRAM Usage Meter */}
                <div className="px-6 py-3 border-t bg-secondary/5 space-y-3">
                    <VRAMUsageMeter
                        estimatedVRAM={estimateVRAMUsage(
                            selectedModelId || selectedPresetId || 'mdx',
                            advancedParams.segmentSize || 352800,
                            advancedParams.overlap,
                            advancedParams.tta
                        )}
                        availableVRAM={gpuInfo?.recommended_profile?.vram_gb || gpuInfo?.gpus?.[0]?.memory_gb || 0}
                    />

                    {/* Contextual Warnings */}
                    {advancedParams.tta && <TTAWarning />}

                    {(gpuInfo?.gpus?.length === 0 || !gpuInfo?.gpus?.[0]?.memory_gb) && (
                        <CPUOnlyWarning />
                    )}

                    {(() => {
                        const estimated = estimateVRAMUsage(
                            selectedModelId || selectedPresetId || 'mdx',
                            advancedParams.segmentSize || 352800,
                            advancedParams.overlap,
                            advancedParams.tta
                        )
                        const available = gpuInfo?.recommended_profile?.vram_gb || gpuInfo?.gpus?.[0]?.memory_gb || 0
                        return estimated > available * 0.9 && available > 0 ? (
                            <LowVRAMWarning available={available} required={estimated} />
                        ) : null
                    })()}

                    {selectedPreset?.ensembleConfig && <EnsembleTip />}
                </div>

                <div className="p-6 border-t bg-secondary/10 flex justify-end gap-3">
                    <Button variant="outline" onClick={() => onOpenChange(false)}>
                        Cancel
                    </Button>
                    <Button onClick={handleConfirm} disabled={(!selectedPresetId && mode === 'simple') || (mode === 'advanced' && !isEnsembleMode && !selectedModelId) || (mode === 'advanced' && isEnsembleMode && ensembleConfig.length === 0)}>
                        <Zap className="w-4 h-4 mr-2" />
                        Start Separation
                    </Button>
                </div>
            </Card >
        </div >
    )
}
