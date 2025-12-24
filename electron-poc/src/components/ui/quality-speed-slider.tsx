import { Zap, Sparkles, Crown } from 'lucide-react'
import { cn } from '../../lib/utils'

interface QualitySpeedSliderProps {
    value: number
    onChange: (value: number) => void
    disabled?: boolean
    className?: string
}

export interface QualityLevel {
    level: number
    name: string
    description: string
    icon: React.ReactNode
    modelCount: number
    useTTA: boolean
    estimatedTime: string
}

export const qualityLevels: QualityLevel[] = [
    {
        level: 0,
        name: 'Fast',
        description: 'Single model, fastest processing',
        icon: <Zap className="h-4 w-4" />,
        modelCount: 1,
        useTTA: false,
        estimatedTime: '~1x'
    },
    {
        level: 1,
        name: 'Balanced',
        description: 'Ensemble of 2 models',
        icon: <Sparkles className="h-4 w-4" />,
        modelCount: 2,
        useTTA: false,
        estimatedTime: '~2x'
    },
    {
        level: 2,
        name: 'Quality',
        description: 'Ensemble of 3-4 models',
        icon: <Crown className="h-4 w-4" />,
        modelCount: 4,
        useTTA: false,
        estimatedTime: '~4x'
    },
    {
        level: 3,
        name: 'Maximum',
        description: 'Full ensemble + TTA',
        icon: <Crown className="h-4 w-4 text-amber-500" />,
        modelCount: 4,
        useTTA: true,
        estimatedTime: '~8x'
    }
]

export function QualitySpeedSlider({
    value,
    onChange,
    disabled = false,
    className
}: QualitySpeedSliderProps) {
    const currentLevel = qualityLevels[value] || qualityLevels[0]

    const getSliderColor = (level: number) => {
        switch (level) {
            case 0: return 'bg-emerald-500'
            case 1: return 'bg-blue-500'
            case 2: return 'bg-purple-500'
            case 3: return 'bg-amber-500'
            default: return 'bg-blue-500'
        }
    }

    return (
        <div className={cn("space-y-3", className)}>
            <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Quality / Speed</label>
                <div className={cn(
                    "flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium",
                    value === 0 && "bg-emerald-500/20 text-emerald-400",
                    value === 1 && "bg-blue-500/20 text-blue-400",
                    value === 2 && "bg-purple-500/20 text-purple-400",
                    value === 3 && "bg-amber-500/20 text-amber-400"
                )}>
                    {currentLevel.icon}
                    {currentLevel.name}
                </div>
            </div>

            {/* Slider Track */}
            <div className="relative">
                <input
                    type="range"
                    min={0}
                    max={3}
                    step={1}
                    value={value}
                    onChange={(e) => onChange(parseInt(e.target.value))}
                    disabled={disabled}
                    className={cn(
                        "w-full h-2 rounded-full appearance-none cursor-pointer",
                        "bg-secondary",
                        "[&::-webkit-slider-thumb]:appearance-none",
                        "[&::-webkit-slider-thumb]:w-5",
                        "[&::-webkit-slider-thumb]:h-5",
                        "[&::-webkit-slider-thumb]:rounded-full",
                        "[&::-webkit-slider-thumb]:border-2",
                        "[&::-webkit-slider-thumb]:border-white",
                        "[&::-webkit-slider-thumb]:shadow-lg",
                        "[&::-webkit-slider-thumb]:cursor-pointer",
                        "[&::-webkit-slider-thumb]:transition-transform",
                        "[&::-webkit-slider-thumb]:hover:scale-110",
                        value === 0 && "[&::-webkit-slider-thumb]:bg-emerald-500",
                        value === 1 && "[&::-webkit-slider-thumb]:bg-blue-500",
                        value === 2 && "[&::-webkit-slider-thumb]:bg-purple-500",
                        value === 3 && "[&::-webkit-slider-thumb]:bg-amber-500",
                        disabled && "opacity-50 cursor-not-allowed"
                    )}
                />

                {/* Step Indicators */}
                <div className="absolute top-1/2 -translate-y-1/2 w-full flex justify-between px-0.5 pointer-events-none">
                    {qualityLevels.map((_, i) => (
                        <div
                            key={i}
                            className={cn(
                                "w-1.5 h-1.5 rounded-full transition-colors",
                                i <= value ? getSliderColor(value) : "bg-secondary-foreground/30"
                            )}
                        />
                    ))}
                </div>
            </div>

            {/* Description */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{currentLevel.description}</span>
                <span className="tabular-nums">{currentLevel.estimatedTime}</span>
            </div>

            {/* Labels */}
            <div className="flex justify-between text-[10px] text-muted-foreground/60 px-0.5">
                <span>Faster</span>
                <span>Better Quality</span>
            </div>
        </div>
    )
}
