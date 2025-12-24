import { memo } from 'react'
import { X, Check, AlertCircle, Clock, Cpu, Zap, PlayCircle, Ban, Pause, Loader2, Save } from 'lucide-react'
import { Button } from './ui/button'
import { cn } from '../lib/utils'
import { QueueItem } from '../types/store'
import { CircularProgress } from './ui/circular-progress'

// Phase detection from progress message
type ProcessingPhase = 'loading' | 'processing' | 'finalizing' | 'unknown'

const getPhase = (message: string | undefined): ProcessingPhase => {
    if (!message) return 'unknown'
    const lower = message.toLowerCase()
    if (lower.includes('loading') || lower.includes('model')) return 'loading'
    if (lower.includes('finaliz') || lower.includes('saving') || lower.includes('stitch')) return 'finalizing'
    if (lower.includes('process') || lower.includes('chunk') || lower.includes('segment')) return 'processing'
    return 'processing' // Default to processing if message exists
}

const getPhaseLabel = (phase: ProcessingPhase): string => {
    switch (phase) {
        case 'loading': return 'Loading Model...'
        case 'processing': return 'Processing Audio'
        case 'finalizing': return 'Saving Stems...'
        default: return 'Processing...'
    }
}

// Calculate ETR (Estimated Time Remaining)
const calculateETR = (progress: number, startTime: number | undefined): string | null => {
    if (!startTime || progress < 10) return null // Don't show ETR until 10%
    const elapsed = Date.now() - startTime
    const totalEstimate = (elapsed / progress) * 100
    const remainingMs = totalEstimate - elapsed

    if (remainingMs < 0 || !isFinite(remainingMs)) return null

    const seconds = Math.round(remainingMs / 1000)
    if (seconds < 60) return `~${seconds}s`
    const minutes = Math.round(seconds / 60)
    return `~${minutes}m`
}

// Helper to safely get filename
const getFileName = (path: string) => {
    return path.replace(/^.*[\/]/, '')
}

interface BatchQueueItemProps {
    item: QueueItem
    onRemove: (id: string) => void
    onPreview?: (item: QueueItem) => void
}

export const BatchQueueItem = memo(({ item, onRemove, onPreview }: BatchQueueItemProps) => {
    const isProcessingItem = item.status === 'processing';
    const isCompleted = item.status === 'completed';
    const isFailed = item.status === 'failed';
    const progress = item.progress || 0;
    const phase = getPhase(item.message);
    const etr = calculateETR(progress, item.startTime);

    return (
        <div
            className={cn(
                "group relative flex items-center gap-4 rounded-xl border bg-card p-3 shadow-sm transition-all hover:shadow-md",
                isProcessingItem && "border-primary/50 ring-1 ring-primary/20",
                isCompleted && "border-green-500/30 bg-green-500/5",
                isFailed && "border-destructive/30 bg-destructive/5"
            )}
        >
            {/* Status Icon / Circular Progress */}
            {isProcessingItem ? (
                phase === 'loading' ? (
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                        <Loader2 className="w-5 h-5 animate-spin" />
                    </div>
                ) : phase === 'finalizing' ? (
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-green-500/10 text-green-500">
                        <Save className="w-5 h-5 animate-pulse" />
                    </div>
                ) : (
                    <CircularProgress value={progress} size={40} strokeWidth={3}>
                        <span className="text-[9px] font-bold text-primary">
                            {Math.round(progress)}%
                        </span>
                    </CircularProgress>
                )
            ) : (
                <div className={cn(
                    "flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition-colors",
                    isCompleted && "bg-green-500/10 text-green-500",
                    isFailed && "bg-destructive/10 text-destructive",
                    item.status === 'pending' && "bg-secondary text-muted-foreground",
                    item.status === 'queued' && "bg-yellow-500/10 text-yellow-500",
                    item.status === 'cancelled' && "bg-muted text-muted-foreground"
                )}>
                    {isCompleted && <Check className="w-5 h-5" />}
                    {isFailed && <AlertCircle className="w-5 h-5" />}
                    {item.status === 'pending' && <Clock className="w-5 h-5" />}
                    {item.status === 'queued' && <Pause className="w-5 h-5" />}
                    {item.status === 'cancelled' && <Ban className="w-5 h-5" />}
                </div>
            )}

            {/* Main Info */}
            <div className="flex-1 min-w-0 grid gap-1">
                <div className="flex items-center justify-between">
                    <span className="text-sm font-medium truncate text-foreground" title={item.file}>
                        {getFileName(item.file)}
                    </span>

                    {/* Progress Badge with ETR */}
                    {isProcessingItem && (
                        <span className="text-xs font-bold text-primary bg-primary/10 px-2 py-0.5 rounded-full flex items-center gap-1">
                            {phase !== 'loading' && `${Math.round(progress)}%`}
                            {etr && <span className="text-muted-foreground font-normal">{etr}</span>}
                        </span>
                    )}
                </div>

                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    {/* Phase-based status message */}
                    {isProcessingItem ? (
                        <span className="truncate max-w-[200px] font-medium text-primary/80">
                            {getPhaseLabel(phase)}
                        </span>
                    ) : (
                        <span className="truncate max-w-[200px] font-medium">{item.modelId || 'Unknown Model'}</span>
                    )}
                    {item.device && (
                        <span className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-secondary">
                            {item.device === 'GPU' ? <Zap className="w-3 h-3" /> : <Cpu className="w-3 h-3" />}
                            {item.device}
                        </span>
                    )}
                    {isFailed && <span className="text-destructive truncate">{item.error}</span>}
                </div>

                {/* Progress Bar (Inline) */}
                {isProcessingItem && (
                    <div className="h-1 w-full bg-secondary rounded-full overflow-hidden mt-1">
                        <div
                            className="h-full bg-primary rounded-full transition-all duration-300"
                            style={{ width: `${item.progress || 0}%` }}
                        />
                    </div>
                )}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                {isCompleted && onPreview && item.outputFiles && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-muted-foreground hover:text-foreground"
                        onClick={() => onPreview(item)}
                    >
                        <PlayCircle className="w-4 h-4" />
                    </Button>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => onRemove(item.id)}
                    disabled={isProcessingItem}
                >
                    <X className="w-4 h-4" />
                </Button>
            </div>
        </div>
    )
}, (prevProps, nextProps) => {
    // Custom comparison function for React.memo
    // Returns true if props are equal (no re-render needed)
    return (
        prevProps.item === nextProps.item && // Referentially equal item (Zustand updates should handle immutability)
        prevProps.onRemove === nextProps.onRemove &&
        prevProps.onPreview === nextProps.onPreview
    )
})

BatchQueueItem.displayName = 'BatchQueueItem'
