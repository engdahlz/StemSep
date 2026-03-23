import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, Loader2, Repeat, FastForward, Save, SkipBack, SkipForward, Trash2 } from 'lucide-react'
import { WaveformTrack, type WaveformTrackHandle } from './WaveformTrack'
import { Slider } from './ui/slider'
import { toast } from 'sonner'

interface MultiTrackPlayerProps {
    stems: Record<string, string>
    jobId?: string
    outputDir?: string
    onClose?: () => void
    onDiscard?: () => void
    isResolvingPlayback?: boolean
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

export function MultiTrackPlayer({
    stems,
    jobId,
    outputDir,
    onClose,
    onDiscard,
    isResolvingPlayback = false,
}: MultiTrackPlayerProps) {
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

    const activeStemLabel = useMemo(() => {
        const soloedStem = stemNames.find((stem) => trackStates[stem]?.soloed)
        return soloedStem || preferredMasterStem || stemNames[0] || null
    }, [preferredMasterStem, stemNames, trackStates])
    const timelineTicks = useMemo(() => {
        if (!Number.isFinite(duration) || duration <= 0) return []
        const steps = duration >= 180 ? 6 : duration >= 60 ? 5 : 4
        return Array.from({ length: steps + 1 }, (_, index) => {
            const value = (duration / steps) * index
            return {
                value,
                left: `${(index / steps) * 100}%`,
            }
        })
    }, [duration])

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

    const isAllReady = stemEntries.length > 0 && readyStems.length >= stemEntries.length

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

    const handleTimelineChange = useCallback((values: number[]) => {
        const nextTime = values[0]
        if (typeof nextTime !== 'number') return
        handleSeek(nextTime)
    }, [handleSeek])

    const handleSkipBy = useCallback((delta: number) => {
        handleSeek(currentTime + delta)
    }, [currentTime, handleSeek])

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

    const setVolume = (stem: string, volume: number) => {
        setTrackStates(prev => ({
            ...prev,
            [stem]: { ...prev[stem], volume }
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
        if (!jobId || !window.electronAPI?.exportSelectionJob) return
        if (!outputDir) {
            toast.error("No output directory is configured for this separation")
            return
        }
        setIsSaving(true)
        try {
            const result = await window.electronAPI.exportSelectionJob(jobId, outputDir)
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

        if (jobId && window.electronAPI?.discardSelectionJob) {
            window.electronAPI.discardSelectionJob(jobId)
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
        <div className="w-full rounded-[2rem] border border-white/70 bg-[rgba(250,248,252,0.8)] p-6 text-slate-800 shadow-[0_40px_120px_rgba(141,150,179,0.22)] backdrop-blur-2xl">
            <div className="mb-5 flex items-start justify-between gap-4">
                <div>
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span className="stemsep-config-chip">
                            Results
                        </span>
                        <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
                            {stemEntries.length} tracks
                        </span>
                    </div>
                    <h2 className="text-[20px] tracking-[-0.6px] text-slate-800">
                        Multi-Track Preview
                    </h2>
                    <p className="mt-1 text-[13px] tracking-[-0.2px] text-slate-500">
                        Mix, inspect and export your separated stems
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {(isResolvingPlayback || (stemEntries.length > 0 && !isAllReady)) && (
                        <span className="inline-flex items-center gap-2 rounded-full border border-white/70 bg-white/76 px-3 py-1.5 text-[12px] tracking-[-0.2px] text-slate-600">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            {isResolvingPlayback
                                ? "Resolving playback..."
                                : `Loading ${readyStems.length}/${stemEntries.length}`}
                        </span>
                    )}
                    {onClose && (
                        <button
                            type="button"
                            onClick={onClose}
                            className="stemsep-config-secondary rounded-[999px] border border-white/70 bg-white/72 px-4 py-2.5 text-[13px] text-slate-700 transition-all hover:bg-white/88 hover:text-slate-900"
                        >
                            Close
                        </button>
                    )}
                </div>
            </div>

            <div className="stemsep-player-scroll max-h-[470px] space-y-4 overflow-y-auto pr-2">
                {stemEntries.length === 0 && (
                    <div className="flex min-h-[160px] items-center justify-center rounded-[1.35rem] border border-dashed border-white/65 bg-white/36 px-6 text-center text-[13px] tracking-[-0.2px] text-slate-500">
                        {isResolvingPlayback
                            ? "Preparing playable preview files..."
                            : "No playable stems are available for this session yet."}
                    </div>
                )}
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
                            onVolumeChange={(value) => setVolume(stem, value)}
                            onToggleMute={() => toggleMute(stem)}
                            onToggleSolo={() => toggleSolo(stem)}
                        />
                    ))}
            </div>

            <div className="mt-6 rounded-[1.5rem] border border-white/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.92),rgba(247,244,250,0.88))] p-5 shadow-[0_24px_48px_rgba(141,150,179,0.14)] backdrop-blur-xl">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <div className="text-[11px] font-medium uppercase tracking-[0.2em] text-slate-500">
                            Playback Dock
                        </div>
                        <div className="mt-1 text-[16px] tracking-[-0.35px] text-slate-800">
                            Listen before export
                        </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {stemNames.slice(0, 4).map((stem) => (
                            <span
                                key={stem}
                                className="rounded-full border border-white/75 bg-white/82 px-2.5 py-1 text-[11px] tracking-[-0.15px] text-slate-600 shadow-[0_8px_18px_rgba(141,150,179,0.08)]"
                            >
                                {stem}
                            </span>
                        ))}
                    </div>
                </div>

                <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-center gap-3">
                        <button
                            type="button"
                            onClick={() => handleSkipBy(-5)}
                            disabled={!isAllReady}
                            className="flex h-10 w-10 items-center justify-center rounded-full border border-white/70 bg-white/78 text-slate-600 shadow-[0_10px_22px_rgba(141,150,179,0.1)] transition-all hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                            title="Back 5 seconds"
                        >
                            <SkipBack className="h-4 w-4" />
                        </button>
                        <button
                            type="button"
                            className="stemsep-config-action relative flex h-14 w-14 items-center justify-center overflow-hidden rounded-full border border-white/75 bg-white text-[#23324c] shadow-[0_16px_28px_rgba(141,150,179,0.16)] transition-all hover:scale-[1.02] hover:bg-white active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
                            onClick={handlePlayPause}
                            disabled={!isAllReady}
                        >
                            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
                        </button>
                        <button
                            type="button"
                            onClick={() => handleSkipBy(5)}
                            disabled={!isAllReady}
                            className="flex h-10 w-10 items-center justify-center rounded-full border border-white/70 bg-white/78 text-slate-600 shadow-[0_10px_22px_rgba(141,150,179,0.1)] transition-all hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                            title="Forward 5 seconds"
                        >
                            <SkipForward className="h-4 w-4" />
                        </button>
                    </div>

                    <div className="rounded-[1.15rem] border border-white/70 bg-white/76 px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                        <div className="mb-3 flex items-center justify-between gap-3">
                            <div>
                                <div className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
                                    Now Playing
                                </div>
                                <div className="mt-1 text-[13px] tracking-[-0.2px] text-slate-700">
                                    {activeStemLabel ? `${activeStemLabel} stem` : 'Playback preview'}
                                </div>
                            </div>
                            {isAllReady && duration > 0 && (
                                <div className="text-[11px] text-slate-500">
                                    {stemEntries.length} synced tracks
                                </div>
                            )}
                        </div>
                        <div className="mb-2 flex items-center justify-between gap-3">
                            <span className="text-[12px] font-medium tabular-nums tracking-[-0.2px] text-slate-600">
                                {formatTime(currentTime)}
                            </span>
                            <span className="text-[12px] font-medium tabular-nums tracking-[-0.2px] text-slate-500">
                                {formatTime(duration)}
                            </span>
                        </div>
                        <Slider
                            value={[Math.min(currentTime, duration || currentTime || 0)]}
                            max={Math.max(duration, 0.1)}
                            step={0.01}
                            disabled={!isAllReady || duration <= 0}
                            onValueChange={handleTimelineChange}
                            className="w-full"
                            trackClassName="h-3 rounded-full bg-[linear-gradient(90deg,rgba(203,213,225,0.45),rgba(226,232,240,0.9))] shadow-[inset_0_1px_3px_rgba(15,23,42,0.08)]"
                            rangeClassName="bg-[linear-gradient(90deg,#64748b,#334155)] shadow-[0_0_22px_rgba(51,65,85,0.24)]"
                            thumbClassName="h-5 w-5 border-[3px] border-white bg-[#23324c] shadow-[0_8px_20px_rgba(35,50,76,0.35)]"
                        />
                        {timelineTicks.length > 0 && (
                            <div className="relative mt-3 h-4">
                                {timelineTicks.map((tick) => (
                                    <div
                                        key={tick.left}
                                        className="absolute top-0 -translate-x-1/2 text-[10px] text-slate-400"
                                        style={{ left: tick.left }}
                                    >
                                        <div className="mx-auto mb-1 h-1.5 w-px bg-slate-300/80" />
                                        {formatTime(tick.value)}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div className="flex items-center gap-2">
                            <FastForward className="h-4 w-4 text-slate-400" />
                            <select
                                className="h-9 rounded-full border border-white/70 bg-white/78 px-3 text-[12px] text-slate-700 outline-none shadow-[0_10px_22px_rgba(141,150,179,0.08)]"
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

                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                onClick={toggleLoop}
                                className={`flex h-9 w-9 items-center justify-center rounded-full transition-all ${
                                    isLooping
                                        ? 'bg-white text-slate-900 shadow-[0_10px_22px_rgba(141,150,179,0.16)]'
                                        : 'bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800'
                                }`}
                                title="Toggle Loop"
                            >
                                <Repeat className="h-3.5 w-3.5" />
                            </button>
                            <button
                                type="button"
                                onClick={() => setLoopPoint('start')}
                                className={`rounded-full px-3 py-1.5 text-[11px] tracking-[-0.15px] transition-all ${
                                    loopStart !== null
                                        ? 'bg-white text-slate-900 shadow-[0_10px_22px_rgba(141,150,179,0.16)]'
                                        : 'bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800'
                                }`}
                            >
                                A {loopStart !== null ? formatTime(loopStart) : ''}
                            </button>
                            <button
                                type="button"
                                onClick={() => setLoopPoint('end')}
                                className={`rounded-full px-3 py-1.5 text-[11px] tracking-[-0.15px] transition-all ${
                                    loopEnd !== null
                                        ? 'bg-white text-slate-900 shadow-[0_10px_22px_rgba(141,150,179,0.16)]'
                                        : 'bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800'
                                }`}
                            >
                                B {loopEnd !== null ? formatTime(loopEnd) : ''}
                            </button>
                            {(loopStart !== null || loopEnd !== null) && (
                                <button
                                    type="button"
                                    onClick={clearLoop}
                                    className="rounded-full px-3 py-1.5 text-[11px] tracking-[-0.15px] text-slate-500 transition-all hover:bg-white hover:text-slate-800"
                                >
                                    Reset
                                </button>
                            )}
                        </div>

                        {isLooping && loopStart !== null && loopEnd !== null && (
                            <div className="text-[12px] tracking-[-0.15px] text-slate-500">
                                {`Looping ${formatTime(loopStart)} to ${formatTime(loopEnd)}`}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {jobId && (
                <div className="mt-5 flex justify-end gap-3 border-t border-white/60 pt-5">
                    <button
                        type="button"
                        onClick={handleDiscard}
                        className="inline-flex items-center gap-2 rounded-[999px] border border-rose-300/55 bg-rose-50/84 px-4 py-2.5 text-[13px] tracking-[-0.2px] text-rose-700 transition-all hover:bg-rose-100"
                    >
                        <Trash2 className="h-4 w-4" />
                        Discard
                    </button>
                    <button
                        type="button"
                        onClick={handleSave}
                        disabled={isSaving}
                        className="stemsep-config-action relative inline-flex items-center gap-2 overflow-hidden rounded-[999px] border border-white/75 bg-white/86 px-4 py-2.5 text-[13px] tracking-[-0.2px] text-[#23324c] transition-all hover:bg-white disabled:opacity-60"
                    >
                        {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                        Save Outputs
                    </button>
                </div>
            )}
        </div>
    )
}

function formatTime(seconds: number) {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
}
