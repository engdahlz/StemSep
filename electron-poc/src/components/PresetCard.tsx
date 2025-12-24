import { Star, Zap, Layers, Music, Mic, Guitar, Activity, Clock, Cpu } from 'lucide-react'
import { Badge } from './ui/badge'
import { cn } from '../lib/utils'

export interface Preset {
    id: string
    name: string
    description: string
    stems: string[]
    recommended: boolean
    category?: string
    workflow?: string
    vram_required?: number
    speed?: string
}

interface PresetCardProps {
    preset: Preset
    isFavorite: boolean
    onToggleFavorite: (id: string) => void
    onClick?: (preset: Preset) => void
}

export function PresetCard({ preset, isFavorite, onToggleFavorite, onClick }: PresetCardProps) {

    const getWorkflowIcon = () => {
        switch (preset.workflow) {
            case 'ensemble': return <Layers className="w-3 h-3" />
            case 'sequential': return <Activity className="w-3 h-3" />
            case 'compare': return <Activity className="w-3 h-3" /> // Or another icon
            default: return <Zap className="w-3 h-3" />
        }
    }

    const getCategoryIcon = () => {
        switch (preset.category) {
            case 'vocals': return <Mic className="w-4 h-4" />
            case 'instrumental': return <Guitar className="w-4 h-4" />
            case 'karaoke': return <Music className="w-4 h-4" />
            default: return <Layers className="w-4 h-4" />
        }
    }

    return (
        <div
            className="group relative flex flex-col rounded-xl border border-white/10 bg-white/5 backdrop-blur-md transition-all duration-300 hover:bg-white/10 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 overflow-hidden cursor-pointer"
            onClick={() => onClick?.(preset)}
        >
            {/* Selection/Hover Overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

            {/* Favorite Button */}
            <button
                onClick={(e) => {
                    e.stopPropagation()
                    onToggleFavorite(preset.id)
                }}
                className="absolute top-3 right-3 z-20 p-1.5 rounded-full hover:bg-white/10 transition-colors"
            >
                <Star
                    className={cn(
                        "w-4 h-4 transition-all duration-300",
                        isFavorite ? "fill-yellow-400 text-yellow-400 scale-110" : "text-muted-foreground group-hover:text-yellow-400/70"
                    )}
                />
            </button>

            {/* Recommended Badge */}
            {preset.recommended && (
                <div className="absolute top-3 left-3 z-20">
                    <Badge variant="default" className="bg-primary/90 hover:bg-primary text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md px-2 py-0.5 h-5">
                        Recommended
                    </Badge>
                </div>
            )}

            <div className="p-5 flex-1 flex flex-col gap-4 relative z-10">
                {/* Header */}
                <div className="space-y-1.5 pt-4">
                    <div className="flex items-center gap-2 text-xs text-primary/80 font-medium uppercase tracking-wider mb-1">
                        {getCategoryIcon()}
                        <span>{preset.category}</span>
                    </div>
                    <h3 className="font-bold text-lg leading-tight tracking-tight text-foreground/90 group-hover:text-primary transition-colors">
                        {preset.name}
                    </h3>

                    {/* Metadata Chips */}
                    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground mt-2">
                        <span className="flex items-center gap-1 bg-white/5 border border-white/5 px-2 py-0.5 rounded-md">
                            {getWorkflowIcon()}
                            <span className="capitalize">{preset.workflow}</span>
                        </span>
                        {preset.vram_required && (
                            <span className="flex items-center gap-1 bg-white/5 border border-white/5 px-2 py-0.5 rounded-md">
                                <Cpu className="w-3 h-3" /> {preset.vram_required}GB
                            </span>
                        )}
                        {preset.speed && (
                            <span className="flex items-center gap-1 bg-white/5 border border-white/5 px-2 py-0.5 rounded-md">
                                <Clock className="w-3 h-3" /> <span className="capitalize">{preset.speed.replace('_', ' ')}</span>
                            </span>
                        )}
                    </div>
                </div>

                {/* Description */}
                <p className="text-sm text-muted-foreground line-clamp-2 min-h-[2.5rem]">
                    {preset.description}
                </p>

                {/* Stems */}
                <div className="mt-auto pt-3 border-t border-white/5">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 font-medium">Output Stems</p>
                    <div className="flex flex-wrap gap-1.5">
                        {preset.stems.slice(0, 4).map((stem) => (
                            <Badge key={stem} variant="outline" className="text-[10px] capitalize bg-white/5 border-white/10 hover:bg-white/10 transition-colors">
                                {stem}
                            </Badge>
                        ))}
                        {preset.stems.length > 4 && (
                            <Badge variant="outline" className="text-[10px] bg-white/5 border-white/10">
                                +{preset.stems.length - 4}
                            </Badge>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
