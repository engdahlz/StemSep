import { useState, useEffect } from 'react'
import { X, Download, FolderOpen, Check } from 'lucide-react'
import { Button } from './ui/button'
import { toast } from 'sonner'

interface ExportDialogProps {
    isOpen: boolean
    onClose: () => void
    outputFiles: Record<string, string>
    defaultExportDir?: string
    onDefaultDirChange?: (dir: string) => void
}

const FORMATS = [
    { value: 'wav', label: 'WAV (Lossless)' },
    { value: 'flac', label: 'FLAC (Lossless Compressed)' },
    { value: 'mp3', label: 'MP3' },
]

const BITRATES = [
    { value: '128k', label: '128 kbps' },
    { value: '192k', label: '192 kbps' },
    { value: '256k', label: '256 kbps' },
    { value: '320k', label: '320 kbps (Best)' },
]

// Stem display names and colors
const STEM_COLORS: Record<string, string> = {
    vocals: 'bg-blue-500',
    instrumental: 'bg-red-500',
    drums: 'bg-orange-500',
    bass: 'bg-purple-500',
    other: 'bg-green-500',
    guitar: 'bg-yellow-500',
    piano: 'bg-pink-500',
}

export default function ExportDialog({
    isOpen,
    onClose,
    outputFiles,
    defaultExportDir,
    onDefaultDirChange
}: ExportDialogProps) {
    const [format, setFormat] = useState('mp3')
    const [bitrate, setBitrate] = useState('320k')
    const [exportDir, setExportDir] = useState(defaultExportDir || '')
    const [rememberPath, setRememberPath] = useState(!!defaultExportDir)
    const [isExporting, setIsExporting] = useState(false)
    // Track which stems are selected for export (all selected by default)
    const [selectedStems, setSelectedStems] = useState<Set<string>>(new Set(Object.keys(outputFiles)))

    // Update local state when defaultExportDir changes
    useEffect(() => {
        if (defaultExportDir && !exportDir) {
            setExportDir(defaultExportDir)
            setRememberPath(true)
        }
    }, [defaultExportDir])

    // Update selected stems when outputFiles changes
    useEffect(() => {
        setSelectedStems(new Set(Object.keys(outputFiles)))
    }, [outputFiles])

    const handleBrowse = async () => {
        if (!window.electronAPI?.selectOutputDirectory) return
        const dir = await window.electronAPI.selectOutputDirectory()
        if (dir) {
            setExportDir(dir)
        }
    }

    const toggleStem = (stem: string) => {
        const newSelected = new Set(selectedStems)
        if (newSelected.has(stem)) {
            newSelected.delete(stem)
        } else {
            newSelected.add(stem)
        }
        setSelectedStems(newSelected)
    }

    const selectAll = () => {
        setSelectedStems(new Set(Object.keys(outputFiles)))
    }

    const selectNone = () => {
        setSelectedStems(new Set())
    }

    const handleExport = async () => {
        if (!exportDir) {
            toast.error('Please select an export directory')
            return
        }

        if (selectedStems.size === 0) {
            toast.error('Please select at least one stem to export')
            return
        }

        if (!window.electronAPI?.exportFiles) {
            toast.error('Export not available')
            return
        }

        setIsExporting(true)

        try {
            // Filter outputFiles to only include selected stems
            const filesToExport: Record<string, string> = {}
            for (const stem of selectedStems) {
                if (outputFiles[stem]) {
                    filesToExport[stem] = outputFiles[stem]
                }
            }

            await window.electronAPI.exportFiles(filesToExport, exportDir, format, bitrate)

            // Save path if requested
            if (rememberPath && onDefaultDirChange) {
                onDefaultDirChange(exportDir)
            }

            toast.success(`Exported ${selectedStems.size} file${selectedStems.size > 1 ? 's' : ''} as ${format.toUpperCase()}`)
            onClose()
        } catch (error) {
            toast.error('Export failed')
        } finally {
            setIsExporting(false)
        }
    }

    if (!isOpen) return null

    const stemCount = Object.keys(outputFiles).length
    const allStems = Object.keys(outputFiles)

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-5 border-b border-neutral-200 dark:border-neutral-700">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-primary/10">
                            <Download className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-neutral-900 dark:text-white">Export Stems</h2>
                            <p className="text-xs text-muted-foreground">{selectedStems.size} of {stemCount} selected</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                        aria-label="Close"
                    >
                        <X className="w-5 h-5 text-neutral-400" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-5 space-y-5">
                    {/* Stem Selection */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-neutral-900 dark:text-white">Select Stems</label>
                            <div className="flex gap-2 text-xs">
                                <button
                                    onClick={selectAll}
                                    className="text-primary hover:underline"
                                >
                                    All
                                </button>
                                <span className="text-neutral-400 dark:text-neutral-600">|</span>
                                <button
                                    onClick={selectNone}
                                    className="text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300"
                                >
                                    None
                                </button>
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            {allStems.map(stem => (
                                <label
                                    key={stem}
                                    className={`flex items-center gap-2 p-2.5 rounded-lg border cursor-pointer transition-all ${selectedStems.has(stem)
                                        ? 'bg-neutral-100 dark:bg-neutral-800 border-primary/50'
                                        : 'bg-neutral-50 dark:bg-neutral-850 border-neutral-200 dark:border-neutral-700 opacity-60 hover:opacity-80'
                                        }`}
                                >
                                    <div
                                        onClick={() => toggleStem(stem)}
                                        className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-all ${selectedStems.has(stem)
                                            ? 'bg-primary border-primary'
                                            : 'border-neutral-400 dark:border-neutral-600'
                                            }`}
                                    >
                                        {selectedStems.has(stem) && <Check className="w-2.5 h-2.5 text-white" />}
                                    </div>
                                    <div className={`w-2 h-2 rounded-full ${STEM_COLORS[stem] || 'bg-gray-500'}`} />
                                    <span className="text-sm text-neutral-900 dark:text-white capitalize">{stem}</span>
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Format Selection */}
                    <div>
                        <label className="block text-sm font-medium text-neutral-900 dark:text-white mb-2">Format</label>
                        <select
                            value={format}
                            onChange={(e) => setFormat(e.target.value)}
                            className="w-full bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg p-3 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
                        >
                            {FORMATS.map(f => (
                                <option key={f.value} value={f.value}>{f.label}</option>
                            ))}
                        </select>
                    </div>

                    {/* Bitrate (only for MP3) */}
                    <div className={format !== 'mp3' ? 'opacity-50 pointer-events-none' : ''}>
                        <label className="block text-sm font-medium text-neutral-900 dark:text-white mb-2">
                            Bitrate {format !== 'mp3' && <span className="text-muted-foreground">(MP3 only)</span>}
                        </label>
                        <select
                            value={bitrate}
                            onChange={(e) => setBitrate(e.target.value)}
                            disabled={format !== 'mp3'}
                            className="w-full bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg p-3 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent transition-all disabled:cursor-not-allowed"
                        >
                            {BITRATES.map(b => (
                                <option key={b.value} value={b.value}>{b.label}</option>
                            ))}
                        </select>
                    </div>

                    <div className="border-t border-neutral-200 dark:border-neutral-700/50 pt-5">
                        {/* Export Directory */}
                        <label className="block text-sm font-medium text-neutral-900 dark:text-white mb-2">Export to</label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={exportDir}
                                readOnly
                                placeholder="Select a folder..."
                                className="flex-1 bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg px-3 py-2.5 text-sm text-neutral-700 dark:text-neutral-300 placeholder-neutral-400 dark:placeholder-neutral-500 truncate"
                            />
                            <Button variant="outline" size="sm" onClick={handleBrowse} className="shrink-0">
                                <FolderOpen className="w-4 h-4 mr-2" />
                                Browse
                            </Button>
                        </div>

                        {/* Remember Path Checkbox */}
                        <label className="flex items-center gap-2 mt-3 cursor-pointer group">
                            <div
                                onClick={() => setRememberPath(!rememberPath)}
                                className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${rememberPath
                                    ? 'bg-primary border-primary'
                                    : 'border-neutral-400 dark:border-neutral-600 group-hover:border-neutral-500'
                                    }`}
                            >
                                {rememberPath && <Check className="w-3 h-3 text-white" />}
                            </div>
                            <span className="text-sm text-neutral-700 dark:text-neutral-300 select-none">Remember this path as default</span>
                        </label>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 p-5 border-t border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-900/50">
                    <Button variant="outline" onClick={onClose} disabled={isExporting}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleExport}
                        disabled={!exportDir || isExporting || selectedStems.size === 0}
                        className="min-w-[100px]"
                    >
                        {isExporting ? (
                            <span className="flex items-center gap-2">
                                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Exporting...
                            </span>
                        ) : (
                            <>
                                <Download className="w-4 h-4 mr-2" />
                                Export {selectedStems.size > 0 && `(${selectedStems.size})`}
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    )
}
