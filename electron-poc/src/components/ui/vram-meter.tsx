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
            color: 'from-emerald-400 to-emerald-500',
            bgColor: 'bg-emerald-50/82 border-emerald-300/55',
            textColor: 'text-emerald-700',
            icon: CheckCircle,
            label: 'Safe',
            message: 'Plenty of VRAM available'
        },
        warning: {
            color: 'from-amber-400 to-amber-500',
            bgColor: 'bg-amber-50/82 border-amber-300/55',
            textColor: 'text-amber-700',
            icon: AlertTriangle,
            label: 'Warning',
            message: 'Close other apps for best performance'
        },
        critical: {
            color: 'from-rose-400 to-rose-500',
            bgColor: 'bg-rose-50/82 border-rose-300/55',
            textColor: 'text-rose-700',
            icon: XCircle,
            label: 'Critical',
            message: 'May cause out-of-memory errors'
        }
    }

    const config = statusConfig[status]
    const Icon = config.icon

    return (
        <div className={cn("rounded-[1.3rem] border p-4 shadow-[0_16px_36px_rgba(141,150,179,0.1)] backdrop-blur-md", config.bgColor, className)}>
            <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 text-slate-600">
                    <Icon className={cn("w-3.5 h-3.5", config.textColor)} />
                    VRAM Usage
                </span>
                <span className={cn("font-medium", config.textColor)}>
                    ~{estimatedVRAM.toFixed(1)}GB / {availableVRAM > 0 ? `${availableVRAM}GB` : '?'}
                </span>
            </div>

            <div className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-white/60 shadow-[inset_0_1px_0_rgba(255,255,255,0.52)]">
                <div
                    className={cn(
                        "h-full rounded-full bg-gradient-to-r transition-all duration-500",
                        config.color
                    )}
                    style={{ width: `${percentage}%` }}
                />
            </div>

            <div className="mt-2 flex items-center justify-between text-[10px]">
                <span className={cn("font-medium", config.textColor)}>{config.label}</span>
                {status !== 'safe' ? (
                    <p className={cn(config.textColor)}>
                        {config.message}
                    </p>
                ) : (
                    <p className="text-slate-500">{config.message}</p>
                )}
            </div>
        </div>
    )
}
