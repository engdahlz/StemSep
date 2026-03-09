import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, Loader2, Repeat, FastForward, Save, Trash2 } from 'lucide-react'
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
        <div className="w-full rounded-[2rem] border border-white/70 bg-[rgba(250,248,252,0.8)] p-6 text-slate-800 shadow-[0_40px_120px_rgba(141,150,179,0.22)] backdrop-blur-2xl">
            <div className="mb-5 flex items-start justify-between gap-4">
                <div>
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span className="stemsep-config-chip">
                            Result Studio
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
                    {!isAllReady && (
                        <span className="inline-flex items-center gap-2 rounded-full border border-white/70 bg-white/76 px-3 py-1.5 text-[12px] tracking-[-0.2px] text-slate-600">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            Loading {readyStems.length}/{stemEntries.length}
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

            <div className="mb-6 flex items-center gap-4 rounded-[1.35rem] border border-white/70 bg-white/54 p-4 shadow-[0_18px_38px_rgba(141,150,179,0.1)] backdrop-blur-xl">
                <button
                    type="button"
                    className="stemsep-config-action relative flex h-12 w-12 items-center justify-center overflow-hidden rounded-[1.1rem] border border-white/75 bg-white/86 text-[#23324c] transition-all hover:scale-[1.02] hover:bg-white active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={handlePlayPause}
                    disabled={!isAllReady}
                >
                    {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
                </button>

                <div className="flex items-center gap-2">
                    <FastForward className="h-4 w-4 text-slate-400" />
                    <select
                        className="h-9 rounded-xl border border-white/70 bg-white/76 px-3 text-[12px] text-slate-700 outline-none"
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

                <div className="h-8 w-px bg-white/60" />

                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        onClick={toggleLoop}
                        className={`flex h-8 w-8 items-center justify-center rounded-lg transition-all ${
                            isLooping
                                ? 'bg-white text-slate-900 shadow-[0_8px_18px_rgba(141,150,179,0.14)]'
                                : 'bg-white/65 text-slate-500 hover:bg-white/86 hover:text-slate-800'
                        }`}
                        title="Toggle Loop"
                    >
                        <Repeat className="h-3.5 w-3.5" />
                    </button>
                    <button
                        type="button"
                        onClick={() => setLoopPoint('start')}
                        className={`rounded-lg px-2 py-1 text-[11px] tracking-[-0.2px] transition-all ${
                            loopStart !== null
                                ? 'bg-white text-slate-900 shadow-[0_8px_18px_rgba(141,150,179,0.14)]'
                                : 'bg-white/65 text-slate-500 hover:bg-white/86 hover:text-slate-800'
                        }`}
                    >
                        A {loopStart !== null ? formatTime(loopStart) : ''}
                    </button>
                    <button
                        type="button"
                        onClick={() => setLoopPoint('end')}
                        className={`rounded-lg px-2 py-1 text-[11px] tracking-[-0.2px] transition-all ${
                            loopEnd !== null
                                ? 'bg-white text-slate-900 shadow-[0_8px_18px_rgba(141,150,179,0.14)]'
                                : 'bg-white/65 text-slate-500 hover:bg-white/86 hover:text-slate-800'
                        }`}
                    >
                        B {loopEnd !== null ? formatTime(loopEnd) : ''}
                    </button>
                    {(loopStart !== null || loopEnd !== null) && (
                        <button
                            type="button"
                            onClick={clearLoop}
                            className="rounded-lg px-2 py-1 text-[11px] tracking-[-0.2px] text-slate-500 transition-all hover:bg-white/66 hover:text-slate-800"
                        >
                            Reset
                        </button>
                    )}
                </div>

                <div className="ml-auto text-right">
                    <div className="text-[12px] font-medium tracking-[-0.2px] text-slate-700">
                        {formatTime(currentTime)} / {formatTime(duration)}
                    </div>
                    <div className="text-[11px] tracking-[-0.15px] text-slate-500">
                        {stemEntries.length} active tracks
                    </div>
                </div>
            </div>

            <div className="stemsep-player-scroll max-h-[470px] space-y-3 overflow-y-auto pr-2">
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
