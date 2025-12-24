import { Play, Pause } from 'lucide-react'
import { Button } from './ui/button'
import { QueueItem } from '../types/store'
import { BatchQueueItem } from './BatchQueueItem'

interface BatchQueueListProps {
    queue: QueueItem[]
    onRemoveItem: (id: string) => void
    onClearQueue: () => void
    onPreviewItem?: (item: QueueItem) => void
    isProcessing: boolean
    isPaused: boolean
    onPause: () => void
    onResume: () => void
}

export function BatchQueueList({ queue, onRemoveItem, onClearQueue, onPreviewItem, isProcessing, isPaused, onPause, onResume }: BatchQueueListProps) {
    if (!queue || queue.length === 0) return null

    return (
        <div className="mt-8 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between px-1">
                <div className="flex items-center gap-2">
                    <h3 className="text-lg font-semibold text-foreground">Queue</h3>
                    <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-muted px-1.5 text-[10px] font-medium text-muted-foreground">
                        {queue.length}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    {(isProcessing || isPaused) && (
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={isPaused ? onResume : onPause}
                            className="h-8 text-xs"
                        >
                            {isPaused ? <Play className="w-3.5 h-3.5 mr-1.5" /> : <Pause className="w-3.5 h-3.5 mr-1.5" />}
                            {isPaused ? "Resume" : "Pause"}
                        </Button>
                    )}
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={onClearQueue}
                        disabled={isProcessing && !isPaused}
                        className="h-8 text-xs text-muted-foreground hover:text-destructive"
                    >
                        Clear All
                    </Button>
                </div>
            </div>

            {/* List */}
            <div className="space-y-2">
                {queue.map((item) => (
                    <BatchQueueItem
                        key={item.id}
                        item={item}
                        onRemove={onRemoveItem}
                        onPreview={onPreviewItem}
                    />
                ))}
            </div>
        </div>
    )
}
