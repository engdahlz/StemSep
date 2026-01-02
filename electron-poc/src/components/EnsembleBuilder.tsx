import { useState } from 'react'
import { Plus, Trash2, Info, Settings2, Sliders } from 'lucide-react'

import { Button } from './ui/button'
import { Card } from './ui/card'

interface Model {
    id: string
    name: string
    description?: string
    installed?: boolean
}

interface EnsembleModelConfig {
    model_id: string
    weight: number
}

interface StemAlgorithms {
    vocals?: 'average' | 'max_spec' | 'min_spec'
    instrumental?: 'average' | 'max_spec' | 'min_spec'
}

interface PhaseFixParams {
    enabled: boolean
    lowHz: number
    highHz: number
    highFreqWeight: number
}

// Pre-defined stem algorithm presets based on user guide
const STEM_ALGORITHM_PRESETS = {
    balanced: { vocals: 'average', instrumental: 'average', description: 'Balanced SDR (safest)' },
    best_instrumental: { vocals: 'average', instrumental: 'max_spec', description: 'Max detail on instrumental' },
    best_vocals: { vocals: 'max_spec', instrumental: 'average', description: 'Fuller vocals with max detail' },
    bleedless: { vocals: 'max_spec', instrumental: 'min_spec', description: 'Cleanest instrumental' },
} as const

// Phase Fixer presets
const PHASE_FIX_PRESETS = {
    standard: { lowHz: 500, highHz: 5000, highFreqWeight: 2.0, description: 'Standard (Roformer buzzing)' },
    hyperace: { lowHz: 3000, highHz: 5000, highFreqWeight: 2.0, description: 'HyperACE optimized' },
    gentle: { lowHz: 500, highHz: 5000, highFreqWeight: 0.8, description: 'Gentle (less aggressive)' },
    wide: { lowHz: 250, highHz: 8000, highFreqWeight: 1.5, description: 'Wide range (extreme)' },
} as const

interface EnsembleBuilderProps {
    models: Model[]
    config: EnsembleModelConfig[]
    algorithm: 'average' | 'max_spec' | 'min_spec' | 'frequency_split'
    phaseFixEnabled?: boolean
    volumeCompEnabled?: boolean
    stemAlgorithms?: StemAlgorithms
    phaseFixParams?: PhaseFixParams
    onVolumeCompEnabledChange?: (enabled: boolean) => void
    onChange: (
        config: EnsembleModelConfig[],
        algorithm: 'average' | 'max_spec' | 'min_spec' | 'frequency_split',
        stemAlgorithms?: StemAlgorithms,
        phaseFixParams?: PhaseFixParams,
        phaseFixEnabled?: boolean
    ) => void
}

export function EnsembleBuilder({
    models,
    config,
    algorithm,
    phaseFixEnabled = false,
    volumeCompEnabled = false,
    stemAlgorithms,
    phaseFixParams,
    onVolumeCompEnabledChange,
    onChange
}: EnsembleBuilderProps) {
    const availableModels = models.filter(m => m.installed !== false)
    const [showAdvanced, setShowAdvanced] = useState(false)
    const [stemAlgoPreset, setStemAlgoPreset] = useState<string>('balanced')
    const [phasePreset, setPhasePreset] = useState<string>('standard')

    const handleAddModel = () => {
        const defaultModel = availableModels[0]?.id || ''
        onChange([...config, { model_id: defaultModel, weight: 1.0 }], algorithm, stemAlgorithms, phaseFixParams)
    }

    const handleRemoveModel = (index: number) => {
        const newConfig = [...config]
        newConfig.splice(index, 1)
        onChange(newConfig, algorithm, stemAlgorithms, phaseFixParams)
    }

    const handleUpdateModel = (index: number, field: keyof EnsembleModelConfig, value: any) => {
        const newConfig = [...config]
        newConfig[index] = { ...newConfig[index], [field]: value }
        onChange(newConfig, algorithm, stemAlgorithms, phaseFixParams)
    }

    const handleAlgorithmChange = (newAlg: string) => {
        onChange(config, newAlg as any, stemAlgorithms, phaseFixParams)
    }

    const handleStemAlgoPresetChange = (preset: string) => {
        setStemAlgoPreset(preset)
        const presetData = STEM_ALGORITHM_PRESETS[preset as keyof typeof STEM_ALGORITHM_PRESETS]
        if (presetData) {
            onChange(config, algorithm, {
                vocals: presetData.vocals as any,
                instrumental: presetData.instrumental as any
            }, phaseFixParams)
        }
    }

    const handlePhasePresetChange = (preset: string) => {
        setPhasePreset(preset)
        const presetData = PHASE_FIX_PRESETS[preset as keyof typeof PHASE_FIX_PRESETS]
        if (presetData) {
            onChange(config, algorithm, stemAlgorithms, {
                enabled: true,
                lowHz: presetData.lowHz,
                highHz: presetData.highHz,
                highFreqWeight: presetData.highFreqWeight
            })
        }
    }

    const handlePhaseParamChange = (key: keyof PhaseFixParams, value: number | boolean) => {
        onChange(config, algorithm, stemAlgorithms, {
            enabled: phaseFixParams?.enabled ?? false,
            lowHz: phaseFixParams?.lowHz ?? 500,
            highHz: phaseFixParams?.highHz ?? 5000,
            highFreqWeight: phaseFixParams?.highFreqWeight ?? 2.0,
            [key]: value
        })
    }

    return (
        <div className="space-y-4">
            {/* Phase Fix Checkbox */}
            <div className="flex items-center gap-3 p-3 rounded-lg border bg-card/50">
                <input
                    type="checkbox"
                    id="phaseFixEnabled"
                    checked={phaseFixEnabled}
                    onChange={(e) => onChange(config, algorithm, stemAlgorithms, phaseFixParams, e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
                <div>
                    <label htmlFor="phaseFixEnabled" className="text-sm font-medium cursor-pointer">
                        Enable Phase Fix
                    </label>
                    <p className="text-xs text-muted-foreground">
                        Reduces metallic Roformer buzzing. Applied before ensemble.
                    </p>
                </div>
            </div>

            {/* Volume Compensation Checkbox */}
            <div className="flex items-center gap-3 p-3 rounded-lg border bg-card/50">
                <input
                    type="checkbox"
                    id="volumeCompEnabled"
                    checked={volumeCompEnabled}
                    onChange={(e) => onVolumeCompEnabledChange?.(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
                <div>
                    <label htmlFor="volumeCompEnabled" className="text-sm font-medium cursor-pointer">
                        Enable VC
                    </label>
                    <p className="text-xs text-muted-foreground">
                        Adds headroom when combining multiple models (reduces clipping risk). When enabled, uses best defaults.
                    </p>
                </div>
            </div>

            {/* Algorithm Selection */}
            <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Ensemble Algorithm</label>
                <select
                    value={algorithm}
                    onChange={(e) => handleAlgorithmChange(e.target.value)}
                    className="h-9 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                    <option value="average">Average (Balanced SDR)</option>
                    <option value="max_spec">Max Spec (Fullest)</option>
                    <option value="min_spec">Min Spec (Bleedless)</option>
                    <option value="frequency_split">Frequency Split (Low/High)</option>
                </select>
            </div>

            {/* Models List */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Models ({config.length})</label>
                    <Button variant="outline" size="sm" onClick={handleAddModel} className="h-8">
                        <Plus className="w-3 h-3 mr-1" />
                        Add Model
                    </Button>
                </div>

                {config.length === 0 && (
                    <div className="text-center p-8 border-2 border-dashed rounded-lg text-muted-foreground text-sm">
                        No models selected. Add models to create an ensemble.
                    </div>
                )}

                {config.length > 4 && (
                    <div className="bg-yellow-500/10 border border-yellow-500/30 p-2 rounded-md text-xs text-yellow-600 dark:text-yellow-400">
                        ⚠️ Using {config.length} models. Quality may degrade after 4-5 models.
                    </div>
                )}

                {config.map((item, index) => (
                    <Card key={index} className="p-3 flex items-center gap-3">
                        <div className="flex-1">
                            <select
                                value={item.model_id}
                                onChange={(e) => handleUpdateModel(index, 'model_id', e.target.value)}
                                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                            >
                                <option value="" disabled>Select Model</option>
                                {availableModels.map(m => (
                                    <option key={m.id} value={m.id}>
                                        {m.name || m.id}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {algorithm === 'average' && (
                            <div className="w-24">
                                <input
                                    type="number"
                                    value={item.weight}
                                    onChange={(e) => handleUpdateModel(index, 'weight', parseFloat(e.target.value) || 1)}
                                    className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                                    placeholder="Weight"
                                    step="0.1"
                                    min="0.1"
                                />
                            </div>
                        )}

                        {phaseFixEnabled && (
                            <div className="w-32 text-xs text-muted-foreground">
                                {index === 0 ? '🎵 Magnitude' : index === 1 ? '📊 Phase Ref' : '(ignored)'}
                            </div>
                        )}

                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-9 w-9 text-muted-foreground hover:text-destructive"
                            onClick={() => handleRemoveModel(index)}
                        >
                            <Trash2 className="w-4 h-4" />
                        </Button>
                    </Card>
                ))}
            </div>

            {/* Advanced Settings Toggle */}
            <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-muted-foreground"
                onClick={() => setShowAdvanced(!showAdvanced)}
            >
                <Settings2 className="w-4 h-4 mr-2" />
                {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </Button>

            {/* Advanced Settings Panel */}
            {showAdvanced && (
                <Card className="p-4 space-y-4 bg-secondary/20">
                    {/* Per-Stem Algorithm Selection */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium flex items-center gap-2">
                            <Sliders className="w-4 h-4" />
                            Per-Stem Algorithm
                        </label>
                        <select
                            value={stemAlgoPreset}
                            onChange={(e) => handleStemAlgoPresetChange(e.target.value)}
                            className="h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                        >
                            {Object.entries(STEM_ALGORITHM_PRESETS).map(([key, preset]) => (
                                <option key={key} value={key}>{preset.description}</option>
                            ))}
                        </select>
                        <div className="text-xs text-muted-foreground">
                            Vocals: {stemAlgorithms?.vocals || 'average'} | Instrumental: {stemAlgorithms?.instrumental || 'average'}
                        </div>
                    </div>

                    {/* Phase Fixer Controls */}
                    {phaseFixEnabled && (
                        <div className="space-y-3 pt-2 border-t">
                            <label className="text-sm font-medium">Phase Fixer Settings</label>

                            <select
                                value={phasePreset}
                                onChange={(e) => handlePhasePresetChange(e.target.value)}
                                className="h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm"
                            >
                                {Object.entries(PHASE_FIX_PRESETS).map(([key, preset]) => (
                                    <option key={key} value={key}>{preset.description}</option>
                                ))}
                            </select>

                            <div className="grid grid-cols-3 gap-2">
                                <div>
                                    <label className="text-xs text-muted-foreground">Low Hz</label>
                                    <input
                                        type="number"
                                        value={phaseFixParams?.lowHz ?? 500}
                                        onChange={(e) => handlePhaseParamChange('lowHz', parseInt(e.target.value))}
                                        className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
                                        min={100}
                                        max={5000}
                                        step={100}
                                    />
                                </div>
                                <div>
                                    <label className="text-xs text-muted-foreground">High Hz</label>
                                    <input
                                        type="number"
                                        value={phaseFixParams?.highHz ?? 5000}
                                        onChange={(e) => handlePhaseParamChange('highHz', parseInt(e.target.value))}
                                        className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
                                        min={1000}
                                        max={20000}
                                        step={500}
                                    />
                                </div>
                                <div>
                                    <label className="text-xs text-muted-foreground">Weight</label>
                                    <input
                                        type="number"
                                        value={phaseFixParams?.highFreqWeight ?? 2.0}
                                        onChange={(e) => handlePhaseParamChange('highFreqWeight', parseFloat(e.target.value))}
                                        className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
                                        min={0.1}
                                        max={3.0}
                                        step={0.1}
                                    />
                                </div>
                            </div>
                        </div>
                    )}
                </Card>
            )}

            {/* Algorithm Info */}
            <div className="bg-secondary/20 p-3 rounded-md text-xs text-muted-foreground flex gap-2">
                <Info className="w-4 h-4 shrink-0 mt-0.5" />
                <div>
                    {algorithm === 'max_spec' && "Max Spec takes the highest magnitude for each frequency bin. Great for fullness but may increase bleed."}
                    {algorithm === 'average' && "Average blends waveforms evenly. Highest SDR, safest default. Use weights to favor certain models."}
                    {algorithm === 'min_spec' && "Min Spec takes the lowest magnitude. Best for removing bleed, but may sound 'thin'."}
                    {phaseFixEnabled && "Phase Fix reduces metallic buzzing from Roformer models. Applied before ensemble algorithm."}
                </div>
            </div>
        </div>
    )
}
