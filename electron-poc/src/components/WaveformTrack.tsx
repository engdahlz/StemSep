import {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useRef,
    useState
} from 'react'
import WaveSurfer from 'wavesurfer.js'
import Spectrogram from 'wavesurfer.js/dist/plugins/spectrogram.esm.js'
import { Volume2, VolumeX, Activity, AlertTriangle, AudioLines, Drum, Guitar, Mic2, Piano } from 'lucide-react'
import { cn } from '../lib/utils'
import { previewLoadMessage } from '../lib/previewErrors'
import { toMediaUrl } from '../lib/media/toMediaUrl'

export interface WaveformTrackHandle {
    getCurrentTime: () => number
    seekTo: (time: number) => void
}

interface WaveformTrackProps {
    url: string
    name: string
    color: string
    height?: number
    isPlaying: boolean
    currentTime: number
    syncVersion: number
    isMaster?: boolean
    volume: number
    muted: boolean
    soloed: boolean
    playbackRate: number
    onReady: (duration: number) => void
    onSeek: (time: number) => void
    onTimeUpdate?: (time: number) => void
    onEnded?: () => void
    onVolumeChange: (value: number) => void
    onToggleMute: () => void
    onToggleSolo: () => void
}

const getTrackIcon = (name: string) => {
    const value = name.toLowerCase()
    if (value.includes('vocal')) return Mic2
    if (value.includes('drum') || value.includes('kick') || value.includes('snare')) return Drum
    if (value.includes('guitar') || value.includes('bass')) return Guitar
    if (value.includes('piano') || value.includes('key') || value.includes('synth')) return Piano
    return AudioLines
}

const REMOTE_MEDIA_PATTERN = /^(blob:|data:|https?:)/i
const localAudioBlobCache = new Map<string, string>()

function base64ToBlobUrl(base64: string, mimeType: string): string {
    const binary = atob(base64)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index)
    }
    return URL.createObjectURL(new Blob([bytes], { type: mimeType }))
}

function getCachedLocalAudioUrl(cacheKey: string) {
    return localAudioBlobCache.get(cacheKey) || null
}

function cacheLocalAudioUrl(cacheKey: string, url: string) {
    localAudioBlobCache.set(cacheKey, url)
}

function withAlpha(hex: string, alphaHex: string) {
    return `${hex}${alphaHex}`
}

function createStemWaveformRenderer(color: string) {
    const fillColor = adjustColorBrightness(color, 8)
    const glowColor = withAlpha(color, '24')

    return (channelData: unknown, ctx: CanvasRenderingContext2D) => {
        const canvasContext = ctx as CanvasRenderingContext2D & {
            roundRect?: (x: number, y: number, width: number, height: number, radii?: number | DOMPointInit | Iterable<number | DOMPointInit>) => void
        }
        const channels = Array.isArray(channelData) ? channelData as Float32Array[] : []
        if (channels.length === 0) return

        const width = canvasContext.canvas.width
        const height = canvasContext.canvas.height
        const centerY = height / 2
        const bars = Math.max(48, Math.floor(width / 6))
        const step = Math.max(1, Math.floor((channels[0]?.length || 0) / bars))
        const amplitudes: number[] = []

        for (let barIndex = 0; barIndex < bars; barIndex += 1) {
            const start = barIndex * step
            const end = Math.min(start + step, channels[0]?.length || 0)
            let peak = 0

            for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
                let mixed = 0
                for (const channel of channels) {
                    mixed += Math.abs(channel[sampleIndex] || 0)
                }
                mixed /= channels.length || 1
                if (mixed > peak) peak = mixed
            }

            amplitudes.push(peak)
        }

        const maxAmplitude = Math.max(...amplitudes, 0.001)
        const drawableWidth = width / bars
        const barWidth = Math.max(2, drawableWidth * 0.72)

        canvasContext.clearRect(0, 0, width, height)
        canvasContext.fillStyle = fillColor
        canvasContext.shadowColor = glowColor
        canvasContext.shadowBlur = 12

        for (let barIndex = 0; barIndex < amplitudes.length; barIndex += 1) {
            const normalized = Math.min(1, amplitudes[barIndex] / maxAmplitude)
            const eased = Math.pow(normalized, 0.65)
            const barHeight = Math.max(6, eased * (height * 0.42))
            const x = barIndex * drawableWidth + (drawableWidth - barWidth) / 2
            const y = centerY - barHeight

            if (typeof canvasContext.roundRect === 'function') {
                canvasContext.beginPath()
                canvasContext.roundRect(x, y, barWidth, barHeight * 2, barWidth / 2)
                canvasContext.fill()
            } else {
                canvasContext.fillRect(x, y, barWidth, barHeight * 2)
            }
        }

        canvasContext.shadowBlur = 0
    }
}

export const WaveformTrack = forwardRef<WaveformTrackHandle, WaveformTrackProps>(function WaveformTrack({
    url,
    name,
    color,
    height = 92,
    isPlaying,
    currentTime,
    syncVersion,
    isMaster = false,
    volume,
    muted,
    soloed,
    playbackRate,
    onReady,
    onSeek,
    onTimeUpdate,
    onEnded,
    onVolumeChange,
    onToggleMute,
    onToggleSolo
}, ref) {
    const containerRef = useRef<HTMLDivElement>(null)
    const wavesurferRef = useRef<WaveSurfer | null>(null)
    const [isReady, setIsReady] = useState(false)
    const [showSpectrogram, setShowSpectrogram] = useState(false)
    const [loadError, setLoadError] = useState<string | null>(null)
    const isMasterRef = useRef(isMaster)
    const currentTimeRef = useRef(currentTime)
    const onReadyRef = useRef(onReady)
    const onSeekRef = useRef(onSeek)
    const onTimeUpdateRef = useRef(onTimeUpdate)
    const onEndedRef = useRef(onEnded)
    const TrackIcon = getTrackIcon(name)

    useEffect(() => {
        isMasterRef.current = isMaster
        currentTimeRef.current = currentTime
        onReadyRef.current = onReady
        onSeekRef.current = onSeek
        onTimeUpdateRef.current = onTimeUpdate
        onEndedRef.current = onEnded
    }, [currentTime, isMaster, onEnded, onReady, onSeek, onTimeUpdate])

    useImperativeHandle(ref, () => ({
        getCurrentTime: () => wavesurferRef.current?.getCurrentTime() ?? 0,
        seekTo: (time: number) => {
            const ws = wavesurferRef.current
            if (!ws || !isReady) return
            ws.setTime(Math.max(0, time))
        }
    }), [isReady])

    useEffect(() => {
        if (!containerRef.current) return

        let mounted = true
        let objectUrl: string | null = null
        let shouldRevokeObjectUrl = false
        setLoadError(null)
        setIsReady(false)

        const plugins: any[] = []
        if (showSpectrogram) {
            plugins.push(Spectrogram.create({
                labels: true,
                height: 128,
                splitChannels: false,
                frequencyMin: 0,
                frequencyMax: 22000,
                fftSamples: 1024,
            }))
        }

        const getBlobFallbackUrl = async () => {
            if (
                REMOTE_MEDIA_PATTERN.test(url) ||
                !window.electronAPI?.readAudioFile
            ) {
                return null
            }

            const audioFile = await window.electronAPI.readAudioFile(url)
            if (!mounted) return null

            if (!audioFile.success) {
                throw new Error(
                    previewLoadMessage(
                        audioFile.code,
                        audioFile.hint || audioFile.error,
                    ),
                )
            }

            const cacheKey = audioFile.resolvedPath || url
            const cachedObjectUrl = getCachedLocalAudioUrl(cacheKey)
            if (cachedObjectUrl) {
                return cachedObjectUrl
            }

            objectUrl = base64ToBlobUrl(audioFile.data, audioFile.mimeType)
            cacheLocalAudioUrl(cacheKey, objectUrl)
            shouldRevokeObjectUrl = false
            return objectUrl
        }

        const createWaveform = async (
            sourceUrl: string,
            allowBlobFallback: boolean,
        ) => {
            try {
                const ws = WaveSurfer.create({
                    container: containerRef.current!,
                    waveColor: withAlpha(color, '55'),
                    progressColor: adjustColorBrightness(color, -12),
                    cursorColor: 'transparent',
                    cursorWidth: 0,
                    barWidth: showSpectrogram ? undefined : 0,
                    barGap: showSpectrogram ? undefined : 0,
                    barRadius: showSpectrogram ? undefined : 999,
                    height: showSpectrogram ? 72 : height,
                    normalize: !showSpectrogram,
                    minPxPerSec: showSpectrogram ? 85 : 140,
                    fillParent: false,
                    interact: true,
                    hideScrollbar: true,
                    autoScroll: true,
                    autoCenter: true,
                    renderFunction: showSpectrogram
                        ? undefined
                        : createStemWaveformRenderer(color),
                    url: sourceUrl,
                    plugins,
                })

                ws.on('ready', () => {
                    if (!mounted) return
                    setIsReady(true)
                    onReadyRef.current(ws.getDuration())
                })

                ws.on('error', async (error) => {
                    if (error instanceof Error && error.name === 'AbortError') return

                    if (allowBlobFallback) {
                        try {
                            const fallbackUrl = await getBlobFallbackUrl()
                            if (fallbackUrl && mounted) {
                                ws.destroy()
                                if (wavesurferRef.current === ws) {
                                    wavesurferRef.current = null
                                }
                                void createWaveform(fallbackUrl, false)
                                return
                            }
                        } catch (fallbackError) {
                            console.error('[WaveformTrack] Blob fallback failed:', name, fallbackError)
                            setLoadError(
                                previewLoadMessage(
                                    undefined,
                                    fallbackError instanceof Error
                                        ? fallbackError.message
                                        : String(fallbackError),
                                ),
                            )
                            return
                        }
                    }

                    console.error('[WaveformTrack] Error loading audio:', name, error)
                    setLoadError(previewLoadMessage(undefined, error instanceof Error ? error.message : String(error)))
                })

                ws.on('interaction', (newTime) => {
                    onSeekRef.current(newTime)
                })

                ws.on('timeupdate', (time) => {
                    if (!isMasterRef.current) return
                    onTimeUpdateRef.current?.(time)
                })

                ws.on('finish', () => {
                    if (!isMasterRef.current) return
                    onEndedRef.current?.()
                })

                wavesurferRef.current = ws
            } catch (error) {
                console.error('[WaveformTrack] Failed to create WaveSurfer:', error)
                setLoadError(previewLoadMessage(undefined, error instanceof Error ? error.message : String(error)))
            }
        }

        const initializeWaveform = async () => {
            const sourceUrl = toMediaUrl(url)
            await createWaveform(sourceUrl, true)
        }

        void initializeWaveform()

        return () => {
            mounted = false
            wavesurferRef.current?.destroy()
            wavesurferRef.current = null
            if (objectUrl && shouldRevokeObjectUrl) {
                URL.revokeObjectURL(objectUrl)
            }
            setIsReady(false)
            setLoadError(null)
        }
    }, [color, height, name, showSpectrogram, url])

    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        ws.setTime(Math.max(0, currentTimeRef.current))
    }, [isReady, syncVersion])

    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        if (isPlaying) {
            void ws.play().catch((error) => {
                if (error instanceof Error && error.name === 'AbortError') return
                console.error('[WaveformTrack] Play failed:', name, error)
            })
        } else {
            ws.pause()
        }
    }, [isPlaying, isReady, name])

    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return
        ws.setPlaybackRate(playbackRate)
    }, [isReady, playbackRate])

    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return
        ws.setVolume(muted ? 0 : volume)
    }, [isReady, muted, volume])

    return (
        <div className={cn(
            "flex items-center gap-4 rounded-[1.45rem] border p-4 backdrop-blur-md transition-all duration-300 shadow-[0_18px_34px_rgba(141,150,179,0.1)]",
            muted ? "border-white/45 bg-white/34" : "border-white/70 bg-white/64"
        )}>
            <div className="flex w-32 shrink-0 items-center gap-3">
                <div
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[1rem] border border-white/60 bg-white/65 shadow-[0_10px_24px_rgba(141,150,179,0.08)]"
                    style={{ backgroundColor: `${color}33` }}
                >
                    <TrackIcon className="h-4 w-4" style={{ color }} />
                </div>
                <div className="min-w-0">
                    <p className={cn(
                        "truncate text-[13px] tracking-[-0.2px]",
                        muted ? "text-slate-500" : "text-slate-800"
                    )}>
                        {name}
                    </p>
                    <p className="text-[11px] tracking-[-0.15px] text-slate-500">
                        {showSpectrogram ? 'Spectrogram' : 'Waveform'}
                    </p>
                </div>
            </div>

            <div
                className={cn(
                    "relative min-h-[96px] flex-1 overflow-hidden rounded-[1.15rem] border border-white/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.95),rgba(241,245,249,0.92))] shadow-[inset_0_1px_0_rgba(255,255,255,0.7)] transition-all ease-in-out",
                    showSpectrogram && "min-h-[196px]"
                )}
                style={{ height: showSpectrogram ? 196 : Math.max(height, 72) }}
                ref={containerRef}
            >
                <div className="pointer-events-none absolute inset-y-2 left-1/2 z-10 w-px -translate-x-1/2 bg-[linear-gradient(180deg,rgba(255,255,255,0),rgba(15,23,42,0.55),rgba(255,255,255,0))]" />
                <div className="pointer-events-none absolute inset-y-5 left-1/2 z-10 w-3 -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(255,255,255,0.95)_0%,rgba(255,255,255,0)_72%)] opacity-90" />
                {!isReady && (
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
                        {loadError ? (
                            <div className="flex items-center gap-2 px-3 text-center text-[11px] text-rose-700">
                                <AlertTriangle className="h-4 w-4 shrink-0" />
                                <span>{loadError}</span>
                            </div>
                        ) : (
                            'Loading...'
                        )}
                    </div>
                )}
            </div>

            <div className="flex shrink-0 items-center gap-2 rounded-[1rem] border border-white/65 bg-white/72 px-2 py-2 shadow-[0_10px_24px_rgba(141,150,179,0.08)]">
                <button
                    type="button"
                    onClick={onToggleSolo}
                    className={cn(
                        "flex h-8 w-8 items-center justify-center rounded-lg text-[11px] tracking-[-0.2px] transition-all",
                        soloed
                            ? "bg-[#f2d675] text-[#1d1d1d]"
                            : "bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800"
                    )}
                    title="Toggle Solo"
                >
                    S
                </button>
                <button
                    type="button"
                    onClick={onToggleMute}
                    className={cn(
                        "flex h-8 w-8 items-center justify-center rounded-lg transition-all",
                        muted
                            ? "bg-[#f47174] text-white"
                            : "bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800"
                    )}
                    title="Toggle Mute"
                >
                    {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                </button>
                <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={volume}
                    onChange={(event) => onVolumeChange(Number(event.target.value))}
                    className="h-1.5 w-20 cursor-pointer accent-slate-600"
                    title="Track volume"
                />
                <button
                    type="button"
                    onClick={() => setShowSpectrogram(!showSpectrogram)}
                    className={cn(
                        "flex h-8 w-8 items-center justify-center rounded-lg transition-all",
                        showSpectrogram
                            ? "bg-white text-slate-900 shadow-[0_8px_18px_rgba(141,150,179,0.14)]"
                            : "bg-white/70 text-slate-500 hover:bg-white hover:text-slate-800"
                    )}
                    title="Toggle Spectrogram"
                >
                    <Activity className="h-3.5 w-3.5" />
                </button>
            </div>
        </div>
    )
})

function adjustColorBrightness(hex: string, percent: number) {
    const num = parseInt(hex.replace('#', ''), 16)
    const amt = Math.round(2.55 * percent)
    const R = (num >> 16) + amt
    const B = ((num >> 8) & 0x00ff) + amt
    const G = (num & 0x0000ff) + amt
    return '#' + (0x1000000 + (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 + (B < 255 ? (B < 1 ? 0 : B) : 255) * 0x100 + (G < 255 ? (G < 1 ? 0 : G) : 255)).toString(16).slice(1)
}
