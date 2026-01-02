import { format } from 'date-fns'
import { FolderOpen, Play, Trash2, Clock, AlertCircle, CheckCircle2, FileAudio, Music } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from './ui/table'
import { Badge } from './ui/badge'
import { useStore, HistoryItem } from '../stores/useStore'
import { cn } from '../lib/utils'
import type { Page } from '../types/navigation'

interface HistoryPageProps {
    onNavigate: (page: Page) => void
}

export function HistoryPage({ onNavigate }: HistoryPageProps) {
    const history = useStore(state => state.history)
    const removeFromHistory = useStore(state => state.removeFromHistory)
    const clearHistory = useStore(state => state.clearHistory)
    const loadSession = useStore(state => state.loadSession)

    const handleOpenFolder = async (path: string) => {
        if (window.electronAPI) {
            await window.electronAPI.openFolder(path)
        }
    }

    const handleRerun = (item: HistoryItem) => {
        loadSession(item)
        onNavigate('home')
    }

    return (
        <div className="h-full overflow-auto p-8">
            <div className="max-w-6xl mx-auto space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold mb-2">History</h1>
                        <p className="text-muted-foreground">
                            View and manage your past separation jobs
                        </p>
                    </div>
                    {history.length > 0 && (
                        <Button
                            variant="destructive"
                            size="sm"
                            onClick={clearHistory}
                        >
                            <Trash2 className="mr-2 h-4 w-4" />
                            Clear History
                        </Button>
                    )}
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Clock className="h-5 w-5" />
                            Recent Jobs
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {history.length === 0 ? (
                            <div className="text-center py-12 text-muted-foreground">
                                <Clock className="h-12 w-12 mx-auto mb-4 opacity-20" />
                                <p className="text-lg font-medium">No history yet</p>
                                <p className="text-sm">Your completed separation jobs will appear here</p>
                            </div>
                        ) : (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Date</TableHead>
                                        <TableHead>File</TableHead>
                                        <TableHead>Model / Preset</TableHead>
                                        <TableHead>Status</TableHead>
                                        <TableHead className="text-right">Actions</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {history.map((item) => (
                                        <TableRow key={item.id}>
                                            <TableCell className="font-medium">
                                                <div className="flex flex-col">
                                                    <span>{format(new Date(item.date), 'MMM d, yyyy')}</span>
                                                    <span className="text-xs text-muted-foreground">
                                                        {format(new Date(item.date), 'h:mm a')}
                                                    </span>
                                                </div>
                                            </TableCell>
                                            <TableCell>
                                                <div className="flex items-center gap-2" title={item.inputFile}>
                                                    <FileAudio className="h-4 w-4 text-muted-foreground" />
                                                    <span className="truncate max-w-[200px]">
                                                        {item.inputFile.split(/[/\\]/).pop()}
                                                    </span>
                                                </div>
                                            </TableCell>
                                            <TableCell>
                                                <div className="flex items-center gap-2">
                                                    <Music className="h-4 w-4 text-muted-foreground" />
                                                    <span>{item.modelName}</span>
                                                </div>
                                            </TableCell>
                                            <TableCell>
                                                <Badge
                                                    variant={item.status === 'completed' ? 'default' : 'destructive'}
                                                    className={cn(
                                                        "capitalize",
                                                        item.status === 'completed' && "bg-green-500/15 text-green-600 hover:bg-green-500/25 border-green-500/20",
                                                        item.status === 'failed' && "bg-red-500/15 text-red-600 hover:bg-red-500/25 border-red-500/20"
                                                    )}
                                                >
                                                    {item.status === 'completed' ? (
                                                        <CheckCircle2 className="mr-1 h-3 w-3" />
                                                    ) : (
                                                        <AlertCircle className="mr-1 h-3 w-3" />
                                                    )}
                                                    {item.status}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <div className="flex justify-end gap-2">
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => handleOpenFolder(item.outputDir)}
                                                        title="Open Output Folder"
                                                    >
                                                        <FolderOpen className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => handleRerun(item)}
                                                        title="Re-run with same settings"
                                                    >
                                                        <Play className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-destructive hover:text-destructive"
                                                        onClick={() => removeFromHistory(item.id)}
                                                        title="Remove from history"
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
