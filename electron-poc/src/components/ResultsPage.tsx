import { useState, useMemo } from 'react'
import { useStore } from '../stores/useStore'
import { MultiTrackPlayer } from './MultiTrackPlayer'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { ScrollArea } from './ui/scroll-area'
import { Clock, ArrowLeft, Music2, Download, Trash2 } from 'lucide-react'
import { Badge } from './ui/badge'
import { formatDistanceToNow } from 'date-fns'
import ExportDialog from './ExportDialog'
import { SeparationHistory } from '../utils/separationHistory'
import { ALL_PRESETS } from '../presets'

interface ResultsPageProps {
    onBack?: () => void
}

export function ResultsPage({ onBack }: ResultsPageProps) {
    const history = useStore(state => state.history)
    const settings = useStore(state => state.settings)
    const sessionToLoad = useStore(state => state.sessionToLoad)
    const loadSession = useStore(state => state.loadSession)
    const removeFromHistory = useStore(state => state.removeFromHistory)
    const clearHistory = useStore(state => state.clearHistory)
    const defaultExportDir = useStore(state => state.settings.defaultExportDir)
    const setDefaultExportDir = useStore(state => state.setDefaultExportDir)
    const models = useStore(state => state.models)

    const [showExportDialog, setShowExportDialog] = useState(false)

    // Helper to get display name for a history item
    // Handles old entries that have preset ID in modelName instead of preset object
    const getDisplayName = (item: typeof history[0]) => {
        // If we have the preset object (new format), use it
        if (item.preset?.name) return item.preset.name

        // Try to look up by modelName (which might be a preset ID for old entries)
        const preset = ALL_PRESETS.find(p => p.id === item.modelName || p.id === item.modelId)
        if (preset) return preset.name

        // Try to look up as a model
        const model = models.find(m => m.id === item.modelId || m.id === item.modelName)
        if (model) return model.name

        // Fallback: format the ID nicely (replace underscores, capitalize)
        return item.modelName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    }

    // Combine completed queue items and history
    const completedItems = useMemo(() => {
        // Start with history (most recent first usually)
        const items = [...history].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
        return items
    }, [history])

    const activeSession = sessionToLoad || (completedItems.length > 0 ? completedItems[0] : null)

    return (
        <div className="h-full flex flex-col bg-background text-foreground">
            {/* Header */}
            <header className="border-b border-border/40 bg-background/95 backdrop-blur p-4 flex items-center gap-4 sticky top-0 z-10">
                <Button variant="ghost" size="icon" onClick={onBack}>
                    <ArrowLeft className="w-5 h-5" />
                </Button>
                <h1 className="text-2xl font-bold">Results Studio</h1>
            </header>

            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar List */}
                <div className="w-80 border-r border-border/40 flex flex-col bg-card/30">
                    <div className="p-4 border-b border-border/20 flex items-center justify-between">
                        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Recent Separations</h2>
                        {completedItems.length > 0 && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
                                onClick={() => {
                                    if (confirm('Clear all recent separations? This cannot be undone.')) {
                                        SeparationHistory.clearHistory()
                                        clearHistory()
                                    }
                                }}
                            >
                                <Trash2 className="h-3 w-3 mr-1" />
                                Clear All
                            </Button>
                        )}
                    </div>
                    <ScrollArea className="flex-1">
                        <div className="flex flex-col p-2 gap-2">
                            {completedItems.map(item => (
                                <div
                                    key={item.id}
                                    onClick={() => loadSession(item)}
                                    role="button"
                                    tabIndex={0}
                                    className={`group flex flex-col gap-1 p-3 rounded-lg text-left transition-all hover:bg-accent/50 border overflow-hidden min-w-0 cursor-pointer ${activeSession?.id === item.id
                                        ? 'bg-accent border-primary/50 shadow-sm'
                                        : 'bg-card/50 border-transparent'
                                        }`}
                                >
                                    <div
                                        className="font-medium truncate"
                                        style={{
                                            maxWidth: '260px'
                                        }}
                                        title={item.inputFile.split(/[\\/]/).pop()}
                                    >
                                        {item.inputFile.split(/[\\/]/).pop()}
                                    </div>
                                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                                        <span
                                            style={{
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                                maxWidth: '140px'
                                            }}
                                            title={getDisplayName(item)}
                                        >
                                            {getDisplayName(item)}
                                        </span>
                                        <span>{formatDistanceToNow(new Date(item.date), { addSuffix: true })}
                                        </span>
                                    </div>
                                    <div className="flex gap-1 mt-1 justify-between">
                                        <div className="flex gap-1 flex-wrap">
                                            {item.outputFiles && Object.keys(item.outputFiles).map(stem => (
                                                <Badge key={stem} variant="secondary" className="text-[10px] px-1 py-0 h-4">
                                                    {stem}
                                                </Badge>
                                            ))}
                                        </div>
                                        <span
                                            className="h-5 w-5 p-0 flex items-center justify-center text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer rounded hover:bg-destructive/10"
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                if (confirm('Delete this separation from history?')) {
                                                    removeFromHistory(item.id)
                                                }
                                            }}
                                            title="Delete from history"
                                        >
                                            <Trash2 className="h-3 w-3" />
                                        </span>
                                    </div>
                                </div>
                            ))}
                            {completedItems.length === 0 && (
                                <div className="p-8 text-center text-muted-foreground text-sm">
                                    No completed separations yet.
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                </div>

                {/* Main Player Area */}
                <div className="flex-1 p-6 overflow-y-auto bg-background/50 relative">
                    {/* Ambient Glow */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 blur-[100px] rounded-full pointer-events-none" />

                    {activeSession && activeSession.outputFiles ? (
                        <div className="max-w-5xl mx-auto space-y-6 relative z-10">
                            <div className="flex items-end justify-between mb-2">
                                <div>
                                    <h2 className="text-3xl font-bold tracking-tight">
                                        {activeSession.inputFile.split(/[\\/]/).pop()}
                                    </h2>
                                    <div className="flex items-center gap-4 mt-2 text-muted-foreground">
                                        <span className="flex items-center gap-1 text-sm">
                                            <Clock className="w-4 h-4" />
                                            {new Date(activeSession.date).toLocaleString()}
                                        </span>
                                        <span className="flex items-center gap-1 text-sm">
                                            <Badge variant="outline">{getDisplayName(activeSession)}</Badge>
                                        </span>
                                    </div>
                                </div>
                                <Button onClick={() => setShowExportDialog(true)}>
                                    <Download className="w-4 h-4 mr-2" />
                                    Export
                                </Button>
                            </div>

                            <MultiTrackPlayer
                                stems={activeSession.outputFiles}
                                jobId={activeSession.backendJobId || activeSession.id}
                                onDiscard={() => {
                                    removeFromHistory(activeSession.id)
                                }}
                            />

                            {/* Metadata / Details Card */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                                <Card className="bg-card/30 backdrop-blur-sm">
                                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Settings</CardTitle></CardHeader>
                                    <CardContent className="text-sm space-y-1 text-muted-foreground">
                                        <div>Overlap: {activeSession.settings.overlap || 'Default'}</div>
                                        <div>Segment: {activeSession.settings.segmentSize || 'Default'}</div>
                                    </CardContent>
                                </Card>
                                <Card className="bg-card/30 backdrop-blur-sm">
                                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Paths</CardTitle></CardHeader>
                                    <CardContent className="text-sm space-y-1 text-muted-foreground">
                                        <div className="truncate" title={activeSession.inputFile}>Input: {activeSession.inputFile}</div>
                                        <div className="truncate" title={activeSession.outputDir}>Output: {activeSession.outputDir}</div>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>
                    ) : (
                        <div className="h-full flex items-center justify-center text-muted-foreground">
                            <div className="text-center space-y-4">
                                <Music2 className="w-16 h-16 mx-auto opacity-20" />
                                <p>Select a session from the list to preview stems.</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Export Dialog */}
            {activeSession?.outputFiles && (
                <ExportDialog
                    isOpen={showExportDialog}
                    onClose={() => setShowExportDialog(false)}
                    outputFiles={activeSession.outputFiles}
                    defaultExportDir={defaultExportDir}
                    onDefaultDirChange={setDefaultExportDir}
                    defaultExportFormat={settings?.advancedSettings?.outputFormat}
                    defaultExportBitrate={settings?.advancedSettings?.bitrate}
                />
            )}
        </div>
    )
}

