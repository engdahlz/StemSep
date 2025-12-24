import { cn } from '@/lib/utils'
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

interface VRAMUsageMeterProps {
    estimatedVRAM: number // In GB
    availableVRAM: number // In GB
    className?: string
}

// VRAM estimation based on NotebookLM recommendations
export function estimateVRAMUsage(
    modelType: string,
    chunkSize: number,
    overlap: number = 2,
    tta: boolean = false
): number {
    let baseVRAM = 2.0 // Baseline for most models

    // Model-specific base requirements
    if (modelType.toLowerCase().includes('roformer')) {
        // BS-Roformer VRAM scales with chunk_size
        if (chunkSize >= 485100) {
            baseVRAM = 10.0
        } else if (chunkSize >= 352800) {
            baseVRAM = 6.0
        } else if (chunkSize >= 112455) {
            baseVRAM = 3.5
        } else {
            baseVRAM = 3.0
        }
    } else if (modelType.toLowerCase().includes('mdx')) {
        baseVRAM = 2.5
    } else if (modelType.toLowerCase().includes('demucs')) {
        baseVRAM = 4.0
    } else if (modelType.toLowerCase().includes('scnet')) {
        baseVRAM = 3.0
    }

    // Overlap multiplier (higher overlap = more parallel processing)
    const overlapMultiplier = overlap > 4 ? 1.3 : overlap > 2 ? 1.15 : 1.0

    // TTA triples memory peaks momentarily
    const ttaMultiplier = tta ? 1.5 : 1.0

    return baseVRAM * overlapMultiplier * ttaMultiplier
}

type VRAMStatus = 'safe' | 'warning' | 'critical'

function getVRAMStatus(estimated: number, available: number): VRAMStatus {
    if (available <= 0) return 'warning' // Unknown VRAM
    const usage = estimated / available
    if (usage > 0.9) return 'critical'
    if (usage > 0.7) return 'warning'
    return 'safe'
}

export function VRAMUsageMeter({ estimatedVRAM, availableVRAM, className }: VRAMUsageMeterProps) {
    const status = getVRAMStatus(estimatedVRAM, availableVRAM)
    const percentage = availableVRAM > 0
        ? Math.min((estimatedVRAM / availableVRAM) * 100, 100)
        : 50

    const statusConfig = {
        safe: {
            color: 'bg-green-500',
            bgColor: 'bg-green-500/10',
            textColor: 'text-green-500',
            icon: CheckCircle,
            label: 'Safe',
            message: 'Plenty of VRAM available'
        },
        warning: {
            color: 'bg-yellow-500',
            bgColor: 'bg-yellow-500/10',
            textColor: 'text-yellow-500',
            icon: AlertTriangle,
            label: 'Warning',
            message: 'Close other apps for best performance'
        },
        critical: {
            color: 'bg-red-500',
            bgColor: 'bg-red-500/10',
            textColor: 'text-red-500',
            icon: XCircle,
            label: 'Critical',
            message: 'May cause out-of-memory errors'
        }
    }

    const config = statusConfig[status]
    const Icon = config.icon

    return (
        <div className={cn("space-y-2", className)}>
            <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground flex items-center gap-1.5">
                    <Icon className={cn("w-3.5 h-3.5", config.textColor)} />
                    VRAM Usage
                </span>
                <span className={cn("font-medium", config.textColor)}>
                    ~{estimatedVRAM.toFixed(1)}GB / {availableVRAM > 0 ? `${availableVRAM}GB` : '?'}
                </span>
            </div>

            <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                    className={cn(
                        "h-full rounded-full transition-all duration-500",
                        config.color
                    )}
                    style={{ width: `${percentage}%` }}
                />
            </div>

            {status !== 'safe' && (
                <p className={cn("text-[10px]", config.textColor)}>
                    ⚠️ {config.message}
                </p>
            )}
        </div>
    )
}
