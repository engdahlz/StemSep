import { useState, useMemo } from 'react'
import { X, Clock, FolderOpen, Trash2, Star, Zap } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { cn } from '@/lib/utils'

interface HistoryDialogProps {
  isOpen: boolean
  onClose: () => void
}

export default function HistoryDialog({ isOpen, onClose }: HistoryDialogProps) {
  const history = useStore(state => state.history)
  const removeFromHistory = useStore(state => state.removeFromHistory)
  const clearHistory = useStore(state => state.clearHistory)
  const toggleHistoryFavorite = useStore(state => state.toggleHistoryFavorite)

  const [filter, setFilter] = useState<'all' | 'favorites'>('all')

  const filteredHistory = useMemo(() => {
    if (filter === 'favorites') {
      return history.filter(item => item.isFavorite)
    }
    return history
  }, [history, filter])

  const stats = useMemo(() => {
    const successful = history.filter(h => h.status === 'completed')

    // Calculate most used model (using modelId)
    const modelCounts = history.reduce((acc, entry) => {
      const id = entry.modelId || 'unknown'
      acc[id] = (acc[id] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const mostUsedModelId = Object.entries(modelCounts).sort((a, b) => b[1] - a[1])[0]?.[0]
    const mostUsedModelName = history.find(h => h.modelId === mostUsedModelId)?.modelName || mostUsedModelId || 'N/A'

    return {
      totalSeparations: history.length,
      successfulSeparations: successful.length,
      failedSeparations: history.length - successful.length,
      averageDuration: successful.length > 0
        ? successful.reduce((sum, h) => sum + (h.duration || 0), 0) / successful.length
        : 0,
      mostUsedModel: mostUsedModelName
    }
  }, [history])

  const handleDelete = (id: string) => {
    removeFromHistory(id)
  }

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all history? This cannot be undone.')) {
      clearHistory()
    }
  }

  const handleOpenFolder = async (folderPath: string) => {
    if (!(window as any).electronAPI) {
      alert('Opening folders only works in the Electron app')
      return
    }

    try {
      await (window as any).electronAPI.openFolder(folderPath)
    } catch (error) {
      console.error('Failed to open folder:', error)
    }
  }

  const formatTimestamp = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col m-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-card">
          <div>
            <h2 className="text-2xl font-bold">Separation History</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {history.length} total separations
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex bg-secondary rounded-lg p-1 mr-2">
              <button
                onClick={() => setFilter('all')}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-all",
                  filter === 'all' ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                All
              </button>
              <button
                onClick={() => setFilter('favorites')}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-all flex items-center gap-1",
                  filter === 'favorites' ? "bg-background shadow-sm text-yellow-500" : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Star className={cn("w-3 h-3", filter === 'favorites' && "fill-current")} />
                Favorites
              </button>
            </div>

            {history.length > 0 && (
              <button
                onClick={handleClearAll}
                className="px-3 py-2 rounded-lg text-sm text-destructive hover:bg-destructive/10 transition-colors flex items-center gap-1"
              >
                <Trash2 className="w-4 h-4" />
                Clear All
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-secondary transition-colors"
              aria-label="Close history"
            >
              <X className="w-5 h-5 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* Stats */}
        {stats && stats.totalSeparations > 0 && (
          <div className="p-4 border-b border-border bg-secondary/30">
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-500">
                  {stats.successfulSeparations}
                </div>
                <div className="text-xs text-muted-foreground">Successful</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-destructive">
                  {stats.failedSeparations}
                </div>
                <div className="text-xs text-muted-foreground">Failed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-primary">
                  {stats.averageDuration > 0 ? formatDuration(stats.averageDuration) : 'N/A'}
                </div>
                <div className="text-xs text-muted-foreground">Avg Duration</div>
              </div>
              <div>
                <div className="text-lg font-bold text-blue-500 truncate px-2" title={stats.mostUsedModel}>
                  {stats.mostUsedModel}
                </div>
                <div className="text-xs text-muted-foreground">Most Used Model</div>
              </div>
            </div>
          </div>
        )}

        {/* History List */}
        <div className="flex-1 overflow-y-auto p-6 bg-background/50">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-12">
              <Clock className="w-16 h-16 mx-auto text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">
                {filter === 'favorites' ? 'No favorite items found' : 'No separation history yet'}
              </p>
              {filter === 'all' && (
                <p className="text-sm text-muted-foreground/70 mt-2">
                  Your completed separations will appear here
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {filteredHistory.map((entry) => {
                const fileName = entry.inputFile.split(/[/\\]/).pop() || entry.inputFile
                return (
                  <div
                    key={entry.id}
                    className={cn(
                      "group p-4 rounded-xl border transition-all duration-200 hover:shadow-md",
                      entry.status === 'completed'
                        ? "border-border bg-card hover:border-primary/30"
                        : "border-destructive/30 bg-destructive/5"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <span className={cn(
                            "w-2.5 h-2.5 rounded-full shadow-sm",
                            entry.status === 'completed' ? "bg-green-500 shadow-green-500/50" : "bg-destructive shadow-destructive/50"
                          )} />
                          <h3 className="font-semibold text-foreground truncate text-lg">
                            {fileName}
                          </h3>
                          <span className="text-xs text-muted-foreground bg-secondary px-2 py-0.5 rounded-full">
                            {formatTimestamp(entry.date)}
                          </span>
                        </div>

                        <div className="text-sm text-muted-foreground space-y-1 ml-5">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground/80">Model:</span>
                            <span>{entry.modelName}</span>
                          </div>
                          {entry.settings.stems && (
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-foreground/80">Stems:</span>
                              <span>{entry.settings.stems.length} ({entry.settings.stems.join(', ')})</span>
                            </div>
                          )}
                          {entry.duration && (
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-foreground/80">Duration:</span>
                              <span>{formatDuration(entry.duration)}</span>
                            </div>
                          )}
                          {entry.status === 'failed' && (
                            <div className="flex items-center gap-2 text-destructive">
                              <span className="font-medium">Error:</span>
                              <span className="text-xs">Failed to process</span>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 ml-4 self-start">
                        <button
                          onClick={() => toggleHistoryFavorite(entry.id)}
                          className={cn(
                            "p-2 rounded-lg transition-all duration-200",
                            entry.isFavorite
                              ? "text-yellow-500 bg-yellow-500/10 hover:bg-yellow-500/20"
                              : "text-muted-foreground hover:text-yellow-500 hover:bg-secondary opacity-0 group-hover:opacity-100"
                          )}
                          title={entry.isFavorite ? "Remove from favorites" : "Add to favorites"}
                        >
                          <Star className={cn("w-5 h-5", entry.isFavorite && "fill-current")} />
                        </button>

                        <button
                          onClick={() => {
                            // Re-queue logic
                            const queueItem = {
                              id: crypto.randomUUID(),
                              file: entry.inputFile,
                              status: 'pending' as const,
                              device: 'cpu' // Default, will be updated by separate page logic
                            }
                            useStore.getState().addToQueue([queueItem])
                            // Notify user (optional toast)
                            alert('Added to queue!')
                          }}
                          className="p-2 rounded-lg text-muted-foreground hover:bg-primary/10 hover:text-primary transition-colors opacity-0 group-hover:opacity-100"
                          title="Run again (Re-queue)"
                        >
                          <Zap className="w-5 h-5" />
                        </button>

                        {entry.status === 'completed' && (
                          <>
                            <button
                              onClick={async () => {
                                // Open in Player logic
                                // Check if output directory exists
                                try {
                                  // We need a way to check if files exist. For now, we'll just try to load it.
                                  // Ideally, we should check existence first.
                                  // Assuming we can just load it into the session.
                                  useStore.getState().loadSession(entry)
                                  onClose()
                                } catch (e) {
                                  console.error(e)
                                  alert('Could not load session. Files might be missing.')
                                }
                              }}
                              className="p-2 rounded-lg text-muted-foreground hover:bg-primary/10 hover:text-primary transition-colors opacity-0 group-hover:opacity-100"
                              title="Open in Player"
                            >
                              <FolderOpen className="w-5 h-5" />
                              {/* Using FolderOpen icon for now, maybe Play icon is better? */}
                            </button>

                            <button
                              onClick={() => handleOpenFolder(entry.outputDir)}
                              className="p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors btn-hover"
                              title="Open output folder"
                            >
                              <FolderOpen className="w-5 h-5" />
                            </button>
                          </>
                        )}
                        <button
                          onClick={() => handleDelete(entry.id)}
                          className="p-2 rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete entry"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

