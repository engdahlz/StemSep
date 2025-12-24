import { Mic, Drum, Guitar, Piano, Music, Check } from 'lucide-react'
import { cn } from '../../lib/utils'

export type StemType = 'vocals' | 'drums' | 'bass' | 'other' | 'instrumental'

interface StemOption {
    id: StemType
    label: string
    icon: React.ReactNode
    description: string
}

const stemOptions: StemOption[] = [
    {
        id: 'vocals',
        label: 'Vocals',
        icon: <Mic className="h-6 w-6" />,
        description: 'Singing & speech'
    },
    {
        id: 'drums',
        label: 'Drums',
        icon: <Drum className="h-6 w-6" />,
        description: 'Percussion'
    },
    {
        id: 'bass',
        label: 'Bass',
        icon: <Guitar className="h-6 w-6" />,
        description: 'Bass guitar & low end'
    },
    {
        id: 'other',
        label: 'Other',
        icon: <Piano className="h-6 w-6" />,
        description: 'Keys, synths, etc.'
    },
    {
        id: 'instrumental',
        label: 'Full Band',
        icon: <Music className="h-6 w-6" />,
        description: 'All instruments'
    }
]

interface StemSelectorProps {
    selectedStems: StemType[]
    onChange: (stems: StemType[]) => void
    disabled?: boolean
    className?: string
}

export function StemSelector({
    selectedStems,
    onChange,
    disabled = false,
    className
}: StemSelectorProps) {

    const toggleStem = (stemId: StemType) => {
        if (disabled) return

        // Special handling for "instrumental" - it's exclusive
        if (stemId === 'instrumental') {
            onChange(['instrumental'])
            return
        }

        // If instrumental was selected, clear it when selecting individual stems
        const filtered = selectedStems.filter(s => s !== 'instrumental')

        if (filtered.includes(stemId)) {
            // Remove stem
            const newStems = filtered.filter(s => s !== stemId)
            onChange(newStems.length > 0 ? newStems : ['vocals']) // Default to vocals if empty
        } else {
            // Add stem
            onChange([...filtered, stemId])
        }
    }

    return (
        <div className={cn("space-y-3", className)}>
            <label className="text-sm font-medium">What do you want to extract?</label>

            <div className="grid grid-cols-5 gap-2">
                {stemOptions.map((stem) => {
                    const isSelected = selectedStems.includes(stem.id)

                    return (
                        <button
                            key={stem.id}
                            onClick={() => toggleStem(stem.id)}
                            disabled={disabled}
                            className={cn(
                                "relative flex flex-col items-center gap-1.5 p-3 rounded-xl border-2 transition-all",
                                "hover:scale-105 active:scale-95",
                                isSelected
                                    ? "border-primary bg-primary/10 text-primary shadow-lg shadow-primary/20"
                                    : "border-border bg-secondary/30 text-muted-foreground hover:border-primary/50 hover:bg-secondary/50",
                                disabled && "opacity-50 cursor-not-allowed hover:scale-100"
                            )}
                        >
                            {/* Selection indicator */}
                            {isSelected && (
                                <div className="absolute -top-1 -right-1 bg-primary rounded-full p-0.5">
                                    <Check className="h-3 w-3 text-primary-foreground" />
                                </div>
                            )}

                            {/* Icon */}
                            <div className={cn(
                                "p-2 rounded-lg transition-colors",
                                isSelected ? "bg-primary/20" : "bg-secondary"
                            )}>
                                {stem.icon}
                            </div>

                            {/* Label */}
                            <span className="text-xs font-medium">{stem.label}</span>
                        </button>
                    )
                })}
            </div>

            {/* Description */}
            <p className="text-xs text-muted-foreground text-center">
                {selectedStems.includes('instrumental')
                    ? 'Extract full instrumental (everything except vocals)'
                    : selectedStems.length === 1
                        ? `Extract ${stemOptions.find(s => s.id === selectedStems[0])?.description || selectedStems[0]}`
                        : `Extract ${selectedStems.length} stems: ${selectedStems.join(', ')}`
                }
            </p>
        </div>
    )
}

// Helper function to auto-select preset based on stems and quality
export function getPresetForStems(stems: StemType[], qualityLevel: number): string {
    // Quality: 0=fast (single model), 1=balanced, 2=quality (ensemble), 3=max (ensemble+TTA)
    const useEnsemble = qualityLevel >= 2

    // Vocal extraction
    if (stems.length === 1 && stems[0] === 'vocals') {
        return useEnsemble ? 'best_vocals' : 'clarity_vocals'
    }

    // Instrumental extraction  
    if (stems.includes('instrumental') ||
        (stems.length === 1 && stems[0] === 'other')) {
        return useEnsemble ? 'best_instrumental' : 'balanced_instrumental'
    }

    // Karaoke (no vocals = backing track)
    if (stems.includes('drums') && stems.includes('bass') && !stems.includes('vocals')) {
        return 'best_karaoke'
    }

    // Multi-stem or vocals + something else
    if (stems.includes('vocals')) {
        return useEnsemble ? 'best_vocals' : 'clarity_vocals'
    }

    // Default to instrumental
    return useEnsemble ? 'best_instrumental' : 'balanced_instrumental'
}
