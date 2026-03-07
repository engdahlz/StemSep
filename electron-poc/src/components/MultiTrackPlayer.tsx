import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, Loader2, Repeat, FastForward, Save, Trash2 } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { WaveformTrack, type WaveformTrackHandle } from './WaveformTrack'
import { toast } from 'sonner'

interface MultiTrackPlayerProps {
    stems: Record<string, string>
    jobId?: string
    onClose?: () => void
    onDiscard?: () => void
}

const STEM_COLORS: Record<string, string> = {
    vocals: '#3b82f6',
    instrumental: '#ef4444',
    drums: '#eab308',
    bass: '#22c55e',
    other: '#a855f7',
    piano: '#ec4899',
    guitar: '#f97316',
}

const DEFAULT_COLOR = '#64748b'

export function MultiTrackPlayer({ stems, jobId, onClose, onDiscard }: MultiTrackPlayerProps) {
    const stemEntries = useMemo(() => Object.entries(stems), [stems])
    const stemNames = useMemo(() => stemEntries.map(([stem]) => stem), [stemEntries])
    const preferredMasterStem = useMemo(() => {
        if (stems.instrumental) return 'instrumental'
        if (stems.vocals) return 'vocals'
        return stemNames[0] || null
    }, [stemNames, stems.instrumental, stems.vocals])

    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [readyStems, setReadyStems] = useState<string[]>([])
    const [trackStates, setTrackStates] = useState<Record<string, {
        volume: number
        muted: boolean
        soloed: boolean
    }>>({})
    const [playbackRate, setPlaybackRate] = useState(1.0)
    const [isLooping, setIsLooping] = useState(false)
    const [loopStart, setLoopStart] = useState<number | null>(null)
    const [loopEnd, setLoopEnd] = useState<number | null>(null)
    const [isSaving, setIsSaving] = useState(false)
    const [syncVersion, setSyncVersion] = useState(0)

    const trackRefs = useRef<Record<string, WaveformTrackHandle | null>>({})

    useEffect(() => {
        const initialStates: Record<string, { volume: number; muted: boolean; soloed: boolean }> = {}
        stemNames.forEach((stem) => {
            initialStates[stem] = {
                volume: 1,
                muted: false,
                soloed: false,
            }
        })
        trackRefs.current = {}
        setTrackStates(initialStates)
        setReadyStems([])
        setIsPlaying(false)
        setCurrentTime(0)
        setDuration(0)
        setSyncVersion(0)
        setLoopStart(null)
        setLoopEnd(null)
    }, [stemNames])

    const isAllReady = readyStems.length >= stemEntries.length

    const requestSync = useCallback(() => {
        setSyncVersion((prev) => prev + 1)
    }, [])

    const handleTrackReady = useCallback((stem: string, trackDuration: number) => {
        setReadyStems((prev) => (prev.includes(stem) ? prev : [...prev, stem]))
        setDuration((prev) => Math.max(prev, trackDuration))
    }, [])

    const handlePlayPause = () => {
        if (!isAllReady) return
        setIsPlaying((prev) => !prev)
        requestSync()
    }

    const handleSeek = useCallback((time: number) => {
        const boundedTime = Math.max(0, Math.min(duration || time, time))
        setCurrentTime(boundedTime)
        requestSync()
    }, [duration, requestSync])

    const handleMasterTimeUpdate = useCallback((time: number) => {
        if (isLooping && loopStart !== null && loopEnd !== null && time >= loopEnd) {
            setCurrentTime(loopStart)
            requestSync()
            return
        }

        if (!isLooping && duration > 0 && time >= duration) {
            setIsPlaying(false)
            setCurrentTime(0)
            requestSync()
            return
        }

        setCurrentTime(time)
    }, [duration, isLooping, loopEnd, loopStart, requestSync])

    const handleMasterEnded = useCallback(() => {
        if (isLooping) {
            setCurrentTime(loopStart ?? 0)
            requestSync()
            return
        }
        setIsPlaying(false)
        setCurrentTime(0)
        requestSync()
    }, [isLooping, loopStart, requestSync])

    const toggleMute = (stem: string) => {
        setTrackStates(prev => ({
            ...prev,
            [stem]: { ...prev[stem], muted: !prev[stem].muted }
        }))
    }

    const toggleSolo = (stem: string) => {
        setTrackStates(prev => {
            const newSoloed = !prev[stem].soloed
            return {
                ...prev,
                [stem]: { ...prev[stem], soloed: newSoloed }
            }
        })
    }

    const getEffectiveMute = (stem: string) => {
        const state = trackStates[stem]
        if (!state) return false
        const anySoloed = Object.values(trackStates).some(s => s.soloed)
        if (anySoloed) {
            return !state.soloed
        }
        return state.muted
    }

    const toggleLoop = () => {
        setIsLooping(!isLooping)
    }

    const setLoopPoint = (point: 'start' | 'end') => {
        if (point === 'start') {
            setLoopStart(currentTime)
            if (loopEnd !== null && loopEnd < currentTime) {
                setLoopEnd(null)
            }
        } else {
            setLoopEnd(currentTime)
            if (loopStart !== null && loopStart > currentTime) {
                setLoopStart(null)
            }
            setIsLooping(true)
        }
    }

    const clearLoop = () => {
        setLoopStart(null)
        setLoopEnd(null)
        setIsLooping(false)
    }

    const handleSave = async () => {
        if (!jobId || !window.electronAPI?.saveJobOutput) return
        setIsSaving(true)
        try {
            const result = await window.electronAPI.saveJobOutput(jobId)
            if (result.success) {
                toast.success("Files saved to output directory")
                onClose?.()
            } else {
                toast.error("Failed to save: " + result.error)
            }
        } catch (e) {
            toast.error("Error saving files")
            console.error(e)
        } finally {
            setIsSaving(false)
        }
    }

    const handleDiscard = () => {
        if (!confirm("Discard this separation? This will remove it from history and delete the cached stems.")) return

        if (jobId && window.electronAPI?.discardJobOutput) {
            window.electronAPI.discardJobOutput(jobId)
                .then((result) => {
                    if (!result?.success) {
                        toast.error("Failed to discard output: " + (result?.error || "Unknown error"))
                        return
                    }
                    onDiscard?.()
                    toast.success("Separation removed from history")
                })
                .catch((e) => {
                    toast.error("Error discarding output")
                    console.error(e)
                })
            return
        }

        onDiscard?.()
        toast.success("Separation removed from history")
    }

    return (
        <Card className="w-full border-primary/20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <CardHeader className="pb-4">
                <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <span>Multi-Track Preview</span>
                        {!isAllReady && (
                            <span className="text-sm font-normal text-muted-foreground flex items-center gap-2">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                Loading stems ({readyStems.length}/{stemEntries.length})...
                            </span>
                        )}
                    </div>
                    {onClose && (
                        <Button variant="ghost" size="sm" onClick={onClose}>✕</Button>
                    )}
                </CardTitle>
            </CardHeader>

            <CardContent className="space-y-6">
                <div className="flex items-center gap-4 p-4 bg-secondary/50 rounded-xl">
                    <Button
                        size="icon"
                        className="h-12 w-12 rounded-full shadow-lg"
                        onClick={handlePlayPause}
                        disabled={!isAllReady}
                    >
                        {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6 ml-1" />}
                    </Button>

                    <div className="flex items-center gap-2">
                        <FastForward className="h-4 w-4 text-muted-foreground" />
                        <select
                            className="bg-background border rounded px-1 text-xs h-8 w-16"
                            value={playbackRate}
                            onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
                        >
                            <option value="0.5">0.5x</option>
                            <option value="0.75">0.75x</option>
                            <option value="1.0">1.0x</option>
                            <option value="1.25">1.25x</option>
                            <option value="1.5">1.5x</option>
                        </select>
                    </div>

                    <div className="h-8 w-px bg-border mx-2" />

                    <div className="flex items-center gap-2">
                        <Button
                            variant={isLooping ? "default" : "ghost"}
                            size="icon"
                            className="h-8 w-8"
                            onClick={toggleLoop}
                            title="Toggle Loop"
                        >
                            <Repeat className="h-4 w-4" />
                        </Button>
                        <div className="flex flex-col gap-1">
                            <div className="flex gap-1">
                                <Button
                                    variant={loopStart !== null ? "secondary" : "outline"}
                                    size="sm"
                                    className="h-5 text-[10px] px-2"
                                    onClick={() => setLoopPoint('start')}
                                >
                                    A {loopStart !== null ? formatTime(loopStart) : ''}
                                </Button>
                                <Button
                                    variant={loopEnd !== null ? "secondary" : "outline"}
                                    size="sm"
                                    className="h-5 text-[10px] px-2"
                                    onClick={() => setLoopPoint('end')}
                                >
                                    B {loopEnd !== null ? formatTime(loopEnd) : ''}
                                </Button>
                            </div>
                        </div>
                        {(loopStart !== null || loopEnd !== null) && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                                onClick={clearLoop}
                            >
                                ✕
                            </Button>
                        )}
                    </div>

                    <div className="flex-1 flex flex-col justify-center gap-1">
                        <div className="text-xs text-muted-foreground font-mono text-right">
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </div>
                    </div>
                </div>

                <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                    {stemEntries.map(([stem, url]) => (
                        <WaveformTrack
                            key={stem}
                            ref={(handle) => {
                                trackRefs.current[stem] = handle
                            }}
                            name={stem}
                            url={url}
                            color={STEM_COLORS[stem.toLowerCase()] || DEFAULT_COLOR}
                            isPlaying={isPlaying}
                            currentTime={currentTime}
                            syncVersion={syncVersion}
                            isMaster={stem === preferredMasterStem}
                            volume={trackStates[stem]?.volume ?? 1}
                            muted={getEffectiveMute(stem)}
                            soloed={trackStates[stem]?.soloed ?? false}
                            playbackRate={playbackRate}
                            onReady={(trackDuration) => handleTrackReady(stem, trackDuration)}
                            onSeek={handleSeek}
                            onTimeUpdate={handleMasterTimeUpdate}
                            onEnded={handleMasterEnded}
                            onToggleMute={() => toggleMute(stem)}
                            onToggleSolo={() => toggleSolo(stem)}
                        />
                    ))}
                </div>

                {jobId && (
                    <div className="flex justify-end gap-3 pt-4 border-t">
                        <Button variant="outline" onClick={handleDiscard} className="text-destructive hover:text-destructive">
                            <Trash2 className="w-4 h-4 mr-2" />
                            Discard
                        </Button>
                        <Button onClick={handleSave} disabled={isSaving}>
                            {isSaving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                            Save Outputs
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

function formatTime(seconds: number) {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
}
