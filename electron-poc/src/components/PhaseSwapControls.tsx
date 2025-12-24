import { Slider } from './ui/slider'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Settings2, RotateCcw } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { useState } from 'react'
import { cn } from '@/lib/utils'

const DEFAULT_PHASE_PARAMS = {
    enabled: false,
    lowHz: 500,
    highHz: 5000,
    highFreqWeight: 0.8
}

interface SliderRowProps {
    label: string
    value: number
    min: number
    max: number
    step: number
    unit: string
    onChange: (value: number) => void
    disabled?: boolean
}

function SliderRow({ label, value, min, max, step, unit, onChange, disabled }: SliderRowProps) {
    return (
        <div className="space-y-2">
            <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-mono text-foreground">{value}{unit}</span>
            </div>
            <Slider
                value={[value]}
                min={min}
                max={max}
                step={step}
                onValueChange={(v) => onChange(v[0])}
                disabled={disabled}
            />
        </div>
    )
}

export function PhaseSwapControls() {
    const [isExpanded, setIsExpanded] = useState(false)
    const phaseParams = useStore(state => state.settings.phaseParams)
    const setPhaseParams = useStore(state => state.setPhaseParams)

    const handleToggle = () => {
        setPhaseParams({ ...phaseParams, enabled: !phaseParams.enabled })
    }

    const handleReset = () => {
        setPhaseParams(DEFAULT_PHASE_PARAMS)
    }

    const isEnabled = phaseParams?.enabled ?? false
    const lowHz = phaseParams?.lowHz ?? 500
    const highHz = phaseParams?.highHz ?? 5000
    const highFreqWeight = phaseParams?.highFreqWeight ?? 0.8

    return (
        <Card className="border-border/50 bg-card/50">
            <div
                className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/30 transition-colors"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium">Phase Swap (Advanced)</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className={cn(
                        "text-xs px-2 py-0.5 rounded-full",
                        isEnabled
                            ? "bg-primary/20 text-primary"
                            : "bg-muted text-muted-foreground"
                    )}>
                        {isEnabled ? 'Manual' : 'Auto'}
                    </span>
                </div>
            </div>

            {isExpanded && (
                <CardContent className="pt-0 pb-4 space-y-4">
                    <div className="flex items-center justify-between">
                        <p className="text-xs text-muted-foreground">
                            Override preset phase swap settings with custom frequency cutoffs.
                        </p>
                        <div className="flex gap-2">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleReset}
                                className="h-7 text-xs"
                            >
                                <RotateCcw className="h-3 w-3 mr-1" />
                                Reset
                            </Button>
                            <Button
                                variant={isEnabled ? "default" : "outline"}
                                size="sm"
                                onClick={handleToggle}
                                className="h-7 text-xs"
                            >
                                {isEnabled ? 'Enabled' : 'Enable Manual'}
                            </Button>
                        </div>
                    </div>

                    <div className={cn(
                        "space-y-4 transition-opacity",
                        !isEnabled && "opacity-50 pointer-events-none"
                    )}>
                        <SliderRow
                            label="Low Cutoff"
                            value={lowHz}
                            min={0}
                            max={1000}
                            step={50}
                            unit=" Hz"
                            onChange={(v) => setPhaseParams({ ...phaseParams, lowHz: v })}
                            disabled={!isEnabled}
                        />
                        <SliderRow
                            label="High Cutoff"
                            value={highHz}
                            min={1000}
                            max={20000}
                            step={500}
                            unit=" Hz"
                            onChange={(v) => setPhaseParams({ ...phaseParams, highHz: v })}
                            disabled={!isEnabled}
                        />
                        <SliderRow
                            label="High Freq Weight"
                            value={highFreqWeight}
                            min={0}
                            max={5}
                            step={0.1}
                            unit=""
                            onChange={(v) => setPhaseParams({ ...phaseParams, highFreqWeight: v })}
                            disabled={!isEnabled}
                        />
                    </div>

                    <p className="text-xs text-muted-foreground/80 border-t border-border/50 pt-3">
                        💡 <strong>Tip:</strong> For Fullness models, try Low=200, High=500, Weight=2.0
                    </p>
                </CardContent>
            )}
        </Card>
    )
}
