import { useState, useEffect } from 'react'
import { X, Info, Settings2, Layers, Zap } from 'lucide-react'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { useStore } from '../stores/useStore'
import { Preset } from '../presets'
import { EnsembleBuilder } from './EnsembleBuilder'
import { PresetSelector } from './PresetSelector'
import { PhaseSwapControls } from './PhaseSwapControls'
import { VRAMUsageMeter, estimateVRAMUsage } from './ui/vram-meter'
import { TTAWarning, CPUOnlyWarning, LowVRAMWarning, EnsembleTip } from './ui/warning-tip'

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
        algorithm: 'average' | 'avg_wave' | 'max_spec' | 'min_spec' | 'phase_fix'
    }
    splitFreq?: number
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
    const globalAdvancedSettings = useStore(state => state.settings.advancedSettings)
    const [mode, setMode] = useState<'simple' | 'advanced'>('simple')
    const [selectedPresetId, setSelectedPresetId] = useState<string>(initialPresetId || (presets.length > 0 ? presets[0].id : ''))
    const [selectedModelId, setSelectedModelId] = useState<string>('')
    const [device, setDevice] = useState<string>('auto')
    const [outputFormat, setOutputFormat] = useState<'wav' | 'mp3' | 'flac'>(globalAdvancedSettings?.outputFormat || 'wav')
    const [invert, setInvert] = useState(false)
    const [normalize, setNormalize] = useState(false)
    const [bitDepth, setBitDepth] = useState('16')
    const [splitFreq, setSplitFreq] = useState(750)
    const [advancedParams, setAdvancedParams] = useState({
        overlap: globalAdvancedSettings?.overlap || 0.25,
        segmentSize: globalAdvancedSettings?.segmentSize || 256,
        shifts: globalAdvancedSettings?.shifts || 1,
        tta: false,
        bitrate: globalAdvancedSettings?.bitrate || '320k'
    })

    // Phase Correction State
    const [usePhaseCorrection, _setUsePhaseCorrection] = useState(false)

    // Ensemble State
    const [isEnsembleMode, setIsEnsembleMode] = useState(false)
    const [ensembleConfig, setEnsembleConfig] = useState<{ model_id: string; weight: number }[]>([])
    const [ensembleAlgorithm, setEnsembleAlgorithm] = useState<'average' | 'max_spec' | 'min_spec' | 'phase_fix' | 'frequency_split'>('average')

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
            presetId: selectedPresetId,
            modelId: isEnsembleMode ? undefined : selectedModelId,
            device,
            outputFormat,
            invert,
            normalize,
            bitDepth,
            splitFreq: (isEnsembleMode && ensembleAlgorithm === 'frequency_split') ? splitFreq : undefined,
            advancedParams: mode === 'advanced' ? advancedParams : undefined,
            ensembleConfig: (mode === 'advanced' && isEnsembleMode) ? {
                models: ensembleConfig,
                algorithm: ensembleAlgorithm as any
            } : undefined
        }

        // Handle simple mode ensemble preset or phase correction
        if (mode === 'simple') {
            const preset = presets.find(p => p.id === selectedPresetId)

            if (usePhaseCorrection && !preset?.ensembleConfig && selectedModelId) {
                const refModel = getPhaseRefModelId() || 'bs-roformer-viperx-1297';

                config.ensembleConfig = {
                    models: [
                        { model_id: selectedModelId, weight: 1.0 }, // Target (Magnitude)
                        { model_id: refModel, weight: 1.0 } // Reference (Phase)
                    ],
                    algorithm: 'phase_fix' as any
                }
                // Clear modelId since we are sending an ensemble
                config.modelId = undefined

            } else if (preset?.ensembleConfig) {
                config.ensembleConfig = preset.ensembleConfig
            }
        }

        onConfirm(config)
        onOpenChange(false)
    }

    if (!open) return null

    const selectedPreset = presets.find(p => p.id === selectedPresetId)

    // Determine reference model for phase correction
    const getPhaseRefModelId = () => {
        if (!selectedPresetId) return null;
        if (selectedPresetId.includes('instrumental') || selectedPresetId.includes('inst')) {
            return 'mel-band-roformer-kim';
        }
        return 'bs-roformer-viperx-1297'; // Default for vocals
    }

    const phaseRefModelId = getPhaseRefModelId();
    const isPhaseRefAvailable = phaseRefModelId ? availability?.[phaseRefModelId]?.available !== false : true;

    // Check availability
    let isAvailable = true
    let missingModels: string[] = []

    if (mode === 'simple' && selectedPreset) {
        if (selectedPreset.ensembleConfig) {
            // Check all models in ensemble
            selectedPreset.ensembleConfig.models.forEach(m => {
                if (availability && availability[m.model_id]?.available === false) {
                    isAvailable = false
                    missingModels.push(m.model_id)
                }
            })
        } else {
            const modelId = selectedPreset.modelId || modelMap[selectedPreset.id]
            if (modelId && availability && availability[modelId]?.available === false) {
                isAvailable = false
                missingModels.push(modelId)
            }

            // Check Phase Ref model if enabled
            if (usePhaseCorrection && phaseRefModelId && !isPhaseRefAvailable) {
                isAvailable = false;
                if (!missingModels.includes(phaseRefModelId)) {
                    missingModels.push(phaseRefModelId);
                }
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
                                <label className="text-sm font-medium block">Select Preset</label>
                                <PresetSelector
                                    selectedPresetId={selectedPresetId}
                                    onSelectPreset={setSelectedPresetId}
                                    presets={presets}
                                    availability={availability}
                                />
                            </div>

                            {/* Selected preset info */}
                            {selectedPreset && (
                                <div className="bg-secondary/30 p-4 rounded-lg space-y-2">
                                    <p className="text-sm text-muted-foreground">{selectedPreset.description}</p>
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
                                        onChange={(cfg, alg) => {
                                            setEnsembleConfig(cfg)
                                            setEnsembleAlgorithm(alg as any)
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

                            {/* Phase Swap Controls */}
                            <div className="pt-4 border-t border-border/50">
                                <PhaseSwapControls />
                            </div>
                        </div>
                    )}

                    <div className="space-y-4 pt-4 border-t">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Output Format</label>
                                <select
                                    value={outputFormat}
                                    onChange={(e) => setOutputFormat(e.target.value as any)}
                                    className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    <option value="wav">WAV (Lossless)</option>
                                    <option value="flac">FLAC (Compressed Lossless)</option>
                                    <option value="mp3">MP3 (Compressed)</option>
                                </select>
                                {outputFormat === 'mp3' ? (
                                    <div className="mt-2">
                                        <label className="text-xs font-medium block mb-1">Bitrate</label>
                                        <select
                                            value={advancedParams.bitrate}
                                            onChange={(e) => setAdvancedParams(prev => ({ ...prev, bitrate: e.target.value }))}
                                            className="flex h-8 w-full items-center justify-between rounded-md border border-input bg-background px-2 py-1 text-xs"
                                        >
                                            <option value="128k">128k (Good)</option>
                                            <option value="192k">192k (Better)</option>
                                            <option value="256k">256k (High)</option>
                                            <option value="320k">320k (Best)</option>
                                        </select>
                                    </div>
                                ) : (
                                    <div className="mt-2">
                                        <label className="text-xs font-medium block mb-1">Bit Depth</label>
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
                                )}
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium block">Processing Device</label>
                                <select
                                    value={device}
                                    onChange={(e) => setDevice(e.target.value)}
                                    className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    <option value="auto">Auto (Recommended)</option>
                                    <option value="cuda">GPU (CUDA)</option>
                                    <option value="cpu">CPU</option>
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
