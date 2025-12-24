import { useState, useEffect, useMemo } from 'react'
import { ArrowLeft, Info, Settings2, Layers, Zap, Music, AudioWaveform, Mic2, Guitar, AlertTriangle, Download } from 'lucide-react'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { useStore } from '../stores/useStore'
import { Preset, PostProcessingStep, getRequiredModels } from '../presets'
import { EnsembleBuilder } from './EnsembleBuilder'
import { PresetSelector } from './PresetSelector'
import { ModelSelector } from './ModelSelector'
import { VRAMUsageMeter, estimateVRAMUsage } from './ui/vram-meter'
import { CPUOnlyWarning, LowVRAMWarning, EnsembleTip } from './ui/warning-tip'

export interface ConfigurePageProps {
    fileName: string
    filePath: string
    onBack: () => void
    onConfirm: (config: SeparationConfig) => void
    initialPresetId?: string
    presets: Preset[]
    availability?: Record<string, any>
    modelMap?: Record<string, string>
    models?: any[]
    onNavigateToModels?: (modelId?: string) => void
    onShowModelDetails?: (modelId: string) => void
}

export interface SeparationConfig {
    mode: 'simple' | 'advanced'
    presetId?: string
    modelId?: string
    device: string
    outputFormat: 'wav' | 'mp3' | 'flac'
    exportMixes?: string[]
    stems?: string[]
    invert?: boolean
    normalize?: boolean
    bitDepth?: string
    advancedParams?: {
        overlap?: number
        segmentSize?: number
        shifts?: number
        tta?: boolean
        bitrate?: string
    }
    ensembleConfig?: {
        models: { model_id: string; weight?: number }[]
        algorithm: 'average' | 'avg_wave' | 'max_spec' | 'min_spec'
        stemAlgorithms?: {
            vocals?: 'average' | 'max_spec' | 'min_spec'
            instrumental?: 'average' | 'max_spec' | 'min_spec'
        }
        phaseFixEnabled?: boolean
        phaseFixParams?: {
            lowHz: number
            highHz: number
            highFreqWeight: number
        }
    }
    splitFreq?: number
    postProcessingSteps?: PostProcessingStep[]
}

export function ConfigurePage({
    fileName,
    filePath: _filePath,
    onBack,
    onConfirm,
    initialPresetId,
    presets = [],
    availability,
    modelMap = {},
    models = [],
    onNavigateToModels,
    onShowModelDetails
}: ConfigurePageProps) {
    const globalAdvancedSettings = useStore(state => state.settings.advancedSettings)
    const [mode, setMode] = useState<'simple' | 'advanced'>('simple')
    const [selectedPresetId, setSelectedPresetId] = useState<string>(initialPresetId || (presets.length > 0 ? presets[0].id : ''))
    const [selectedModelId, setSelectedModelId] = useState<string>('')
    const [device, setDevice] = useState<string>('auto')
    const [outputFormat, setOutputFormat] = useState<'wav' | 'mp3' | 'flac'>(globalAdvancedSettings?.outputFormat || 'wav')
    const [invert, _setInvert] = useState(false)
    const [normalize, _setNormalize] = useState(false)
    const [bitDepth, _setBitDepth] = useState('16')
    const [_splitFreq, _setSplitFreq] = useState(750)
    const [advancedParams, setAdvancedParams] = useState({
        overlap: globalAdvancedSettings?.overlap || 0.25,
        segmentSize: globalAdvancedSettings?.segmentSize || 256,
        shifts: globalAdvancedSettings?.shifts || 1,
        tta: false,
        bitrate: globalAdvancedSettings?.bitrate || '320k'
    })

    // Phase Correction State
    const [usePhaseCorrection, setUsePhaseCorrection] = useState(false)

    // Auto Post-Processing Pipeline State
    const [enableAutoPipeline, setEnableAutoPipeline] = useState(true)

    // Ensemble State
    const [isEnsembleMode, setIsEnsembleMode] = useState(false)
    const [ensembleConfig, setEnsembleConfig] = useState<{ model_id: string; weight: number }[]>([])
    const [ensembleAlgorithm, setEnsembleAlgorithm] = useState<'average' | 'max_spec' | 'min_spec'>('average')
    const [phaseFixEnabled, setPhaseFixEnabled] = useState(false)
    const [stemAlgorithms, setStemAlgorithms] = useState<{ vocals?: 'average' | 'max_spec' | 'min_spec', instrumental?: 'average' | 'max_spec' | 'min_spec' } | undefined>(undefined)
    const [phaseFixParams, setPhaseFixParams] = useState<{ lowHz: number, highHz: number, highFreqWeight: number }>({ lowHz: 500, highHz: 5000, highFreqWeight: 2.0 })

    // GPU Info for VRAM estimation
    const [gpuInfo, setGpuInfo] = useState<any>(null)
    useEffect(() => {
        const loadGpuInfo = async () => {
            if (window.electronAPI?.getGpuDevices) {
                try {
                    const info = await window.electronAPI.getGpuDevices()
                    setGpuInfo(info)
                } catch (e) {
                    console.error('Failed to load GPU info:', e)
                }
            }
        }
        loadGpuInfo()
    }, [])

    // Update selected preset when initialPresetId changes
    useEffect(() => {
        if (initialPresetId) {
            setSelectedPresetId(initialPresetId)
        }
    }, [initialPresetId])

    // Update selected model when preset changes
    useEffect(() => {
        if (mode === 'simple' && selectedPresetId) {
            const preset = presets.find(p => p.id === selectedPresetId)
            if (preset) {
                if (preset.ensembleConfig) {
                    // Ensemble preset - handled in render
                } else {
                    const modelId = preset.modelId || modelMap[preset.id]
                    if (modelId) setSelectedModelId(modelId)
                }
            }
        }
    }, [selectedPresetId, mode, presets, modelMap])

    // Update processing parameters when model changes in Advanced Mode
    useEffect(() => {
        if (mode === 'advanced' && selectedModelId) {
            const model = models.find(m => m.id === selectedModelId)
            if (model) {
                const defaultOverlap = 0.25
                const defaultSegmentSize = 352800  // Safe default for most models

                // Use model's recommended settings if available
                const recommendedOverlap = model.recommended_settings?.overlap ?? defaultOverlap
                const recommendedSegmentSize = model.recommended_settings?.segment_size
                    ?? model.recommended_settings?.chunk_size
                    ?? model.chunk_size
                    ?? defaultSegmentSize

                setAdvancedParams(prev => ({
                    ...prev,
                    overlap: recommendedOverlap,
                    segmentSize: recommendedSegmentSize
                }))
            }
        }
    }, [selectedModelId, mode, models])

    // Compute which models are required but not installed for the selected preset
    const missingModels = useMemo(() => {
        if (mode !== 'simple' || !selectedPresetId) return []

        const preset = presets.find(p => p.id === selectedPresetId)
        if (!preset) return []

        const requiredModels = getRequiredModels(preset)
        const installedModelIds = new Set(models.filter(m => m.installed).map(m => m.id))

        return requiredModels.filter(modelId => !installedModelIds.has(modelId))
    }, [mode, selectedPresetId, presets, models])

    // Check if separation can proceed (no missing models)
    const canStartSeparation = missingModels.length === 0

    const handleConfirm = () => {
        // In Advanced + Ensemble mode, don't pass presetId - let ensembleConfig take priority
        const usePresetId = (mode === 'advanced' && isEnsembleMode) ? undefined : selectedPresetId

        const config: SeparationConfig = {
            mode,
            presetId: usePresetId,
            modelId: isEnsembleMode ? undefined : selectedModelId,
            device,
            outputFormat,
            // For ensemble mode, default to vocals+instrumental (2 stems)
            stems: isEnsembleMode ? ['vocals', 'instrumental'] : undefined,
            invert,
            normalize,
            bitDepth,
            splitFreq: undefined,
            advancedParams: mode === 'advanced' ? advancedParams : undefined,
            ensembleConfig: (mode === 'advanced' && isEnsembleMode) ? {
                models: ensembleConfig,
                algorithm: ensembleAlgorithm as any,
                stemAlgorithms: stemAlgorithms,
                phaseFixEnabled: phaseFixEnabled,
                phaseFixParams: phaseFixEnabled ? phaseFixParams : undefined
            } : undefined
        }

        // Handle simple mode ensemble preset or phase correction
        if (mode === 'simple') {
            const preset = presets.find(p => p.id === selectedPresetId)

            if (usePhaseCorrection && !preset?.ensembleConfig && selectedModelId) {
                // Determine reference model for phase correction
                const refModel = selectedPresetId?.includes('instrumental') || selectedPresetId?.includes('inst')
                    ? 'mel-band-roformer-kim'
                    : 'bs-roformer-viperx-1297';

                config.ensembleConfig = {
                    models: [
                        { model_id: selectedModelId, weight: 1.0 },
                        { model_id: refModel, weight: 1.0 }
                    ],
                    algorithm: 'max_spec',
                    phaseFixEnabled: true
                }
                config.modelId = undefined

            } else if (preset?.ensembleConfig) {
                config.ensembleConfig = preset.ensembleConfig as any
            }

            // Add post-processing steps if enabled and preset has them
            if (enableAutoPipeline && preset?.postProcessingSteps && preset.postProcessingSteps.length > 0) {
                config.postProcessingSteps = preset.postProcessingSteps
            }
        }

        onConfirm(config)
    }

    const selectedPreset = presets.find(p => p.id === selectedPresetId)

    // isAvailable is now derived from missingModels (computed via useMemo above)
    const isAvailable = missingModels.length === 0

    // Get VRAM info
    const availableVRAM = gpuInfo?.gpus?.[0]?.memory_gb || 0
    const modelType = selectedPreset?.name || selectedModelId || 'unknown'
    const estimatedVRAM = estimateVRAMUsage(modelType, advancedParams.segmentSize, advancedParams.overlap, advancedParams.tta)
    const isCPUOnly = !gpuInfo?.has_cuda

    return (
        <div className="h-full flex flex-col bg-background">
            {/* Header */}
            <div className="border-b bg-card/50 px-6 py-4">
                <div className="flex items-center gap-4">
                    <Button variant="ghost" size="icon" onClick={onBack}>
                        <ArrowLeft className="h-5 w-5" />
                    </Button>
                    <div className="flex-1">
                        <h1 className="text-2xl font-bold">Configure Separation</h1>
                        <div className="flex items-center gap-2 text-muted-foreground">
                            <Music className="h-4 w-4" />
                            <span className="text-sm truncate max-w-md">{fileName}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-6xl mx-auto space-y-6">
                    {/* Mode Tabs */}
                    <div className="grid w-full grid-cols-2 bg-muted p-1 rounded-lg">
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
                            {/* Categorized Preset Selection - 4 Column Layout */}
                            <div className="grid grid-cols-4 gap-4">
                                {/* Instrumental Presets */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium flex items-center gap-2">
                                        <AudioWaveform className="w-4 h-4 text-muted-foreground" />
                                        Instrumental
                                    </label>
                                    <PresetSelector
                                        selectedPresetId={('category' in (presets.find(p => p.id === selectedPresetId) || {})) && (presets.find(p => p.id === selectedPresetId) as any)?.category === 'instrumental' ? selectedPresetId : ''}
                                        onSelectPreset={setSelectedPresetId}
                                        presets={presets.filter(p => ('category' in p) && (p as any).category === 'instrumental')}
                                        availability={availability}
                                    />
                                </div>

                                {/* Vocals Presets */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium flex items-center gap-2">
                                        <Mic2 className="w-4 h-4 text-muted-foreground" />
                                        Vocals
                                    </label>
                                    <PresetSelector
                                        selectedPresetId={('category' in (presets.find(p => p.id === selectedPresetId) || {})) && (presets.find(p => p.id === selectedPresetId) as any)?.category === 'vocals' ? selectedPresetId : ''}
                                        onSelectPreset={setSelectedPresetId}
                                        presets={presets.filter(p => ('category' in p) && (p as any).category === 'vocals')}
                                        availability={availability}
                                    />
                                </div>

                                {/* Instruments Presets (drums, guitar, etc) */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium flex items-center gap-2">
                                        <Guitar className="w-4 h-4 text-muted-foreground" />
                                        Instruments
                                    </label>
                                    <PresetSelector
                                        selectedPresetId={('category' in (presets.find(p => p.id === selectedPresetId) || {})) && (presets.find(p => p.id === selectedPresetId) as any)?.category === 'instruments' ? selectedPresetId : ''}
                                        onSelectPreset={setSelectedPresetId}
                                        presets={presets.filter(p => ('category' in p) && (p as any).category === 'instruments')}
                                        availability={availability}
                                    />
                                </div>

                                {/* Utility Presets (karaoke, de-reverb, de-noise) */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium flex items-center gap-2">
                                        <Settings2 className="w-4 h-4 text-muted-foreground" />
                                        Utility
                                    </label>
                                    <PresetSelector
                                        selectedPresetId={('category' in (presets.find(p => p.id === selectedPresetId) || {})) && (presets.find(p => p.id === selectedPresetId) as any)?.category === 'utility' ? selectedPresetId : ''}
                                        onSelectPreset={setSelectedPresetId}
                                        presets={presets.filter(p => ('category' in p) && (p as any).category === 'utility')}
                                        availability={availability}
                                    />
                                </div>
                            </div>

                            {/* Selected preset info */}
                            {selectedPreset && (
                                <Card className="p-4 space-y-3">
                                    {/* Header with quality badge */}
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <p className="text-sm text-muted-foreground">{selectedPreset.description}</p>
                                        </div>
                                        {'qualityLevel' in selectedPreset && (
                                            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ml-3 shrink-0 ${selectedPreset.qualityLevel === 'ultra' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                                                selectedPreset.qualityLevel === 'quality' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                                                    selectedPreset.qualityLevel === 'balanced' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                                                        'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                                                }`}>
                                                {selectedPreset.qualityLevel === 'ultra' ? '⭐ Ultra' :
                                                    selectedPreset.qualityLevel === 'quality' ? '✓ Quality' :
                                                        selectedPreset.qualityLevel === 'balanced' ? '◎ Balanced' :
                                                            '⚡ Fast'}
                                            </span>
                                        )}
                                    </div>

                                    {/* VRAM and Stems Row */}
                                    <div className="flex items-center justify-between gap-4">
                                        <div className="flex items-center gap-4">
                                            {'estimatedVram' in selectedPreset && (
                                                <div className="flex items-center gap-1.5 text-sm">
                                                    <span className="text-muted-foreground">VRAM:</span>
                                                    <span className={`font-medium ${selectedPreset.estimatedVram >= 12 ? 'text-orange-400' :
                                                        selectedPreset.estimatedVram >= 8 ? 'text-yellow-400' :
                                                            'text-emerald-400'
                                                        }`}>
                                                        {selectedPreset.estimatedVram}GB
                                                    </span>
                                                </div>
                                            )}
                                            <span className="text-sm font-medium text-muted-foreground">Stems:</span>
                                        </div>
                                        <div className="flex gap-1 flex-wrap">
                                            {selectedPreset.stems.map(stem => (
                                                <span key={stem} className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-transparent bg-secondary text-secondary-foreground">
                                                    {stem}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Tags */}
                                    {'tags' in selectedPreset && selectedPreset.tags && selectedPreset.tags.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5">
                                            {selectedPreset.tags.map((tag: string) => (
                                                <span key={tag} className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                                                    #{tag}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {/* Ensemble Info */}
                                    {selectedPreset.ensembleConfig && (
                                        <div className="pt-2 border-t">
                                            <div className="flex items-center gap-2">
                                                <Layers className="w-4 h-4 text-primary" />
                                                <span className="text-sm font-medium">Ensemble: {selectedPreset.ensembleConfig.models.length} models</span>
                                                <span className="text-xs text-muted-foreground">
                                                    ({selectedPreset.ensembleConfig.algorithm === 'max_spec' ? 'Max Spec' :
                                                        selectedPreset.ensembleConfig.algorithm === 'min_spec' ? 'Min Spec' : 'Average'})
                                                </span>
                                            </div>
                                        </div>
                                    )}

                                    {/* Post-Processing Pipeline Toggle */}
                                    {'postProcessingSteps' in selectedPreset && selectedPreset.postProcessingSteps && selectedPreset.postProcessingSteps.length > 0 ? (
                                        <div className="pt-2 border-t">
                                            <button
                                                onClick={() => setEnableAutoPipeline(!enableAutoPipeline)}
                                                className={`w-full flex items-start gap-3 text-left p-3 rounded-lg transition-all ${enableAutoPipeline
                                                    ? 'bg-emerald-500/10 border border-emerald-500/30'
                                                    : 'bg-muted/50 border border-transparent hover:border-muted-foreground/20'
                                                    }`}
                                            >
                                                <div className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${enableAutoPipeline
                                                    ? 'border-emerald-500 bg-emerald-500'
                                                    : 'border-muted-foreground/40'
                                                    }`}>
                                                    {enableAutoPipeline && (
                                                        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                                        </svg>
                                                    )}
                                                </div>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-sm font-medium ${enableAutoPipeline ? 'text-emerald-400' : 'text-muted-foreground'}`}>
                                                            Auto Post-Processing
                                                        </span>
                                                        {enableAutoPipeline && (
                                                            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">ON</span>
                                                        )}
                                                    </div>
                                                    <div className="text-xs text-muted-foreground mt-1">
                                                        {selectedPreset.postProcessingSteps.map((step: any, i: number) => (
                                                            <span key={i}>
                                                                {i > 0 && ' → '}
                                                                {step.description}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            </button>
                                        </div>
                                    ) : (
                                        // Show regular pipeline note if no postProcessingSteps
                                        'pipelineNote' in selectedPreset && selectedPreset.pipelineNote && (
                                            <div className="pt-2 border-t">
                                                <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 rounded p-2">
                                                    <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                                                    <span>{selectedPreset.pipelineNote}</span>
                                                </div>
                                            </div>
                                        )
                                    )}

                                    {!isAvailable && (
                                        <div className="flex flex-col gap-2 bg-destructive/10 p-3 rounded border border-destructive/20">
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
                                                        onBack()
                                                        onNavigateToModels?.(missingModels[0])
                                                    }
                                                }}
                                            >
                                                Download Missing Models
                                            </Button>
                                        </div>
                                    )}
                                </Card>
                            )}

                            {/* Device Selection (GPU/CPU) */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Processing Device</label>
                                <div className="flex gap-2">
                                    <Button
                                        variant={device === 'auto' ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => setDevice('auto')}
                                    >
                                        Auto
                                    </Button>
                                    {gpuInfo?.has_cuda && (
                                        <Button
                                            variant={device === 'cuda' ? 'default' : 'outline'}
                                            size="sm"
                                            onClick={() => setDevice('cuda')}
                                        >
                                            GPU (CUDA)
                                        </Button>
                                    )}
                                    <Button
                                        variant={device === 'cpu' ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => setDevice('cpu')}
                                    >
                                        CPU {!gpuInfo?.has_cuda && '(No GPU detected)'}
                                    </Button>
                                </div>
                                {device === 'cpu' && gpuInfo?.has_cuda && (
                                    <p className="text-xs text-muted-foreground">
                                        CPU processing is slower but uses less memory
                                    </p>
                                )}
                            </div>

                            {/* Output Format */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Output Format</label>
                                <div className="flex gap-2">
                                    {(['wav', 'mp3', 'flac'] as const).map(fmt => (
                                        <Button
                                            key={fmt}
                                            variant={outputFormat === fmt ? 'default' : 'outline'}
                                            size="sm"
                                            onClick={() => setOutputFormat(fmt)}
                                        >
                                            {fmt.toUpperCase()}
                                        </Button>
                                    ))}
                                </div>
                            </div>

                            {/* Phase Correction Toggle */}
                            {mode === 'simple' && selectedPreset && !selectedPreset.ensembleConfig && (
                                <div className="space-y-2">
                                    <label className="text-sm font-medium block">Phase Correction</label>
                                    <button
                                        onClick={() => setUsePhaseCorrection(!usePhaseCorrection)}
                                        className={`w-full flex items-start gap-3 text-left p-3 rounded-lg transition-all ${usePhaseCorrection
                                            ? 'bg-blue-500/10 border border-blue-500/30'
                                            : 'bg-muted/50 border border-transparent hover:border-muted-foreground/20'
                                            }`}
                                    >
                                        <div className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${usePhaseCorrection
                                            ? 'border-blue-500 bg-blue-500'
                                            : 'border-muted-foreground/40'
                                            }`}>
                                            {usePhaseCorrection && (
                                                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                </svg>
                                            )}
                                        </div>
                                        <div className="flex-1">
                                            <div className={`font-medium ${usePhaseCorrection ? 'text-blue-500' : 'text-foreground'}`}>
                                                Phase Swap Enhancement
                                            </div>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Uses a reference model to improve phase accuracy. Reduces artifacts in high frequencies.
                                            </p>
                                        </div>
                                    </button>
                                </div>
                            )}

                            {/* Warnings */}
                            <div className="space-y-2">
                                {isCPUOnly && <CPUOnlyWarning />}
                                {availableVRAM > 0 && estimatedVRAM > availableVRAM * 0.8 && (
                                    <LowVRAMWarning available={availableVRAM} required={estimatedVRAM} />
                                )}
                                {selectedPreset?.ensembleConfig && <EnsembleTip />}
                            </div>
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
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${!isEnsembleMode ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground'}`}
                                >
                                    Single Model
                                </button>
                                <button
                                    onClick={() => setIsEnsembleMode(true)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${isEnsembleMode ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground'}`}
                                >
                                    <Layers className="w-4 h-4 inline mr-1" />
                                    Ensemble
                                </button>
                            </div>

                            {isEnsembleMode ? (
                                <EnsembleBuilder
                                    models={models}
                                    config={ensembleConfig}
                                    algorithm={ensembleAlgorithm}
                                    phaseFixEnabled={phaseFixEnabled}
                                    stemAlgorithms={stemAlgorithms}
                                    phaseFixParams={{ ...phaseFixParams, enabled: phaseFixEnabled }}
                                    onChange={(config, alg, newStemAlgos, newPhaseFixParams, newPhaseFixEnabled) => {
                                        setEnsembleConfig(config)
                                        setEnsembleAlgorithm(alg)
                                        if (newStemAlgos) setStemAlgorithms(newStemAlgos)
                                        if (newPhaseFixParams) setPhaseFixParams({
                                            lowHz: newPhaseFixParams.lowHz,
                                            highHz: newPhaseFixParams.highHz,
                                            highFreqWeight: newPhaseFixParams.highFreqWeight
                                        })
                                        if (newPhaseFixEnabled !== undefined) setPhaseFixEnabled(newPhaseFixEnabled)
                                    }}
                                />
                            ) : (
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Select Model</label>
                                        <ModelSelector
                                            selectedModelId={selectedModelId}
                                            onSelectModel={setSelectedModelId}
                                            models={models}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Advanced Parameters */}
                            <Card className="p-4 space-y-4">
                                <h4 className="font-medium">Processing Parameters</h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-xs text-muted-foreground">Overlap</label>
                                        <input
                                            type="number"
                                            step="0.05"
                                            min="0"
                                            max="50"
                                            className="w-full p-2 rounded border bg-background"
                                            value={advancedParams.overlap}
                                            onChange={(e) => {
                                                const next = parseFloat(e.target.value)
                                                if (!Number.isFinite(next)) return
                                                setAdvancedParams(p => ({ ...p, overlap: next }))
                                            }}
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs text-muted-foreground">Segment Size</label>
                                        <input
                                            type="number"
                                            className="w-full p-2 rounded border bg-background"
                                            value={advancedParams.segmentSize}
                                            onChange={(e) => setAdvancedParams(p => ({ ...p, segmentSize: parseInt(e.target.value) }))}
                                        />
                                    </div>
                                </div>
                            </Card>

                            {/* VRAM Meter */}
                            {availableVRAM > 0 && (
                                <VRAMUsageMeter
                                    availableVRAM={availableVRAM}
                                    estimatedVRAM={estimatedVRAM}
                                />
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Missing Models Warning */}
            {missingModels.length > 0 && (
                <div className="border-t bg-destructive/5 px-6 py-4">
                    <div className="max-w-2xl mx-auto">
                        <Card className="border-destructive/50 bg-destructive/10 p-4">
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                                <div className="flex-1">
                                    <h4 className="font-medium text-destructive">Missing Models</h4>
                                    <p className="text-sm text-muted-foreground mt-1">
                                        This preset requires models that are not installed:
                                    </p>
                                    <ul className="text-sm mt-2 space-y-1">
                                        {missingModels.map(modelId => {
                                            const model = models.find(m => m.id === modelId)
                                            return (
                                                <li key={modelId} className="flex items-center gap-2">
                                                    <span className="text-destructive">•</span>
                                                    <span>{model?.name || modelId}</span>
                                                    {model?.file_size && (
                                                        <span className="text-muted-foreground text-xs">
                                                            ({(model.file_size / (1024 * 1024)).toFixed(0)} MB)
                                                        </span>
                                                    )}
                                                </li>
                                            )
                                        })}
                                    </ul>
                                    <div className="flex gap-2 mt-3">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => onNavigateToModels?.(missingModels[0])}
                                        >
                                            <Download className="w-4 h-4 mr-2" />
                                            Go to Model Browser
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    </div>
                </div>
            )}

            {/* Footer */}
            <div className="border-t bg-card/50 px-6 py-4">
                <div className="max-w-2xl mx-auto flex justify-between">
                    <Button variant="outline" onClick={onBack}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleConfirm}
                        disabled={!canStartSeparation || (!selectedPresetId && mode === 'simple') || (mode === 'advanced' && !isEnsembleMode && !selectedModelId) || (mode === 'advanced' && isEnsembleMode && ensembleConfig.length === 0)}
                    >
                        <Zap className="w-4 h-4 mr-2" />
                        Start Separation
                    </Button>
                </div>
            </div>
        </div>
    )
}
