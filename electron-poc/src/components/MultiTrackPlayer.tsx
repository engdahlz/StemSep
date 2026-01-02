import { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, Loader2, Repeat, FastForward, Save, Trash2 } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { WaveformTrack } from './WaveformTrack'
import { toast } from 'sonner'

interface MultiTrackPlayerProps {
    stems: Record<string, string>
    jobId?: string
    onClose?: () => void
    onDiscard?: () => void
}

// ... (colors remain same)

const STEM_COLORS: Record<string, string> = {
    vocals: '#3b82f6', // Blue
    instrumental: '#ef4444', // Red
    drums: '#eab308', // Yellow
    bass: '#22c55e', // Green
    other: '#a855f7', // Purple
    piano: '#ec4899', // Pink
    guitar: '#f97316', // Orange
}

const DEFAULT_COLOR = '#64748b' // Slate

export function MultiTrackPlayer({ stems, jobId, onClose, onDiscard }: MultiTrackPlayerProps) {
    // ... (state remains same)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [readyCount, setReadyCount] = useState(0)

    // Track states
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

    const animationFrameRef = useRef<number | null>(null)
    const startTimeRef = useRef<number>(0)
    const lastPlayTimeRef = useRef<number>(0)

    // ... (effects and logic remain same until handlers) ...
    // Initialize track states
    useEffect(() => {
        const initialStates: Record<string, any> = {}
        Object.keys(stems).forEach(key => {
            initialStates[key] = {
                volume: 1,
                muted: false,
                soloed: false
            }
        })
        setTrackStates(initialStates)
        setReadyCount(0)
        setIsPlaying(false)
        setCurrentTime(0)
    }, [stems])

    const totalTracks = Object.keys(stems).length
    const isAllReady = readyCount >= totalTracks

    // Playback Timer Logic
    const startTimer = useCallback(() => {
        startTimeRef.current = performance.now() - (lastPlayTimeRef.current * 1000 / playbackRate)

        const loop = () => {
            const now = performance.now()
            const newTime = (now - startTimeRef.current) * playbackRate / 1000

            if (isLooping && loopStart !== null && loopEnd !== null) {
                if (newTime >= loopEnd) {
                    setCurrentTime(loopStart)
                    lastPlayTimeRef.current = loopStart
                    startTimeRef.current = performance.now() - (loopStart * 1000 / playbackRate)
                    animationFrameRef.current = requestAnimationFrame(loop)
                    return
                }
            }

            if (newTime >= duration && duration > 0) {
                if (isLooping) {
                    setCurrentTime(0)
                    lastPlayTimeRef.current = 0
                    startTimeRef.current = performance.now()
                    animationFrameRef.current = requestAnimationFrame(loop)
                    return
                }
                setIsPlaying(false)
                setCurrentTime(0)
                lastPlayTimeRef.current = 0
                return
            }

            setCurrentTime(newTime)
            animationFrameRef.current = requestAnimationFrame(loop)
        }

        animationFrameRef.current = requestAnimationFrame(loop)
    }, [duration, isLooping, loopStart, loopEnd, playbackRate])

    const stopTimer = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current)
        }
        lastPlayTimeRef.current = currentTime
    }, [currentTime])

    useEffect(() => {
        if (isPlaying) {
            startTimer()
        } else {
            stopTimer()
        }
        return () => stopTimer()
    }, [isPlaying, startTimer, stopTimer])

    // Handlers
    const handlePlayPause = () => {
        setIsPlaying(!isPlaying)
    }

    const handleSeek = useCallback((time: number) => {
        lastPlayTimeRef.current = time
        setCurrentTime(time)
        // Note: startTimeRef update happens in startTimer based on isPlaying state
    }, [])

    const handleTrackReady = useCallback((trackDuration: number) => {
        setReadyCount(prev => prev + 1)
        setDuration(prev => Math.max(prev, trackDuration))
    }, [])

    const toggleMute = (stem: string) => {
        setTrackStates(prev => ({
            ...prev,
            [stem]: { ...prev[stem], muted: !prev[stem].muted }
        }))
    }

    const toggleSolo = (stem: string) => {
        setTrackStates(prev => {
            const newSoloed = !prev[stem].soloed
            const newState = { ...prev }
            newState[stem] = { ...newState[stem], soloed: newSoloed }
            return newState
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

        // Best-effort cleanup of staged preview outputs.
        // Uses backend job id (passed via jobId) when available.
        if (jobId && window.electronAPI?.discardJobOutput) {
            window.electronAPI.discardJobOutput(jobId)
                .then((result) => {
                    if (!result?.success) {
                        toast.error("Failed to discard output: " + (result?.error || "Unknown error"))
                        return
                    }
                    if (onDiscard) {
                        onDiscard()
                        toast.success("Separation removed from history")
                    }
                })
                .catch((e) => {
                    toast.error("Error discarding output")
                    console.error(e)
                })
            return
        }

        if (onDiscard) {
            onDiscard()
            toast.success("Separation removed from history")
        }
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
                                Loading stems ({readyCount}/{totalTracks})...
                            </span>
                        )}
                    </div>
                    {onClose && (
                        <Button variant="ghost" size="sm" onClick={onClose}>✕</Button>
                    )}
                </CardTitle>
            </CardHeader>

            <CardContent className="space-y-6">
                {/* Global Controls */}
                <div className="flex items-center gap-4 p-4 bg-secondary/50 rounded-xl">
                    <Button
                        size="icon"
                        className="h-12 w-12 rounded-full shadow-lg"
                        onClick={handlePlayPause}
                        disabled={!isAllReady}
                    >
                        {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6 ml-1" />}
                    </Button>

                    {/* ... Speed and Loop controls remain similar ... */}
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

                    {/* Loop Controls */}
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
                                    A {loopStart ? formatTime(loopStart) : ''}
                                </Button>
                                <Button
                                    variant={loopEnd !== null ? "secondary" : "outline"}
                                    size="sm"
                                    className="h-5 text-[10px] px-2"
                                    onClick={() => setLoopPoint('end')}
                                >
                                    B {loopEnd ? formatTime(loopEnd) : ''}
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
                            {formatTime(currentTime)}
                        </div>
                    </div>
                </div>

                {/* Tracks */}
                <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                    {Object.entries(stems).map(([stem, url]) => (
                        <WaveformTrack
                            key={stem}
                            name={stem}
                            url={url}
                            color={STEM_COLORS[stem.toLowerCase()] || DEFAULT_COLOR}
                            isPlaying={isPlaying}
                            volume={trackStates[stem]?.volume ?? 1}
                            muted={getEffectiveMute(stem)}
                            soloed={trackStates[stem]?.soloed ?? false}
                            playbackRate={playbackRate}
                            onReady={handleTrackReady}
                            onSeek={handleSeek}
                            onToggleMute={() => toggleMute(stem)}
                            onToggleSolo={() => toggleSolo(stem)}
                        />
                    ))}
                </div>

                {/* Save / Discard Footer */}
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
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
}
