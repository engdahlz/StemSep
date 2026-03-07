import {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useRef,
    useState
} from 'react'
import WaveSurfer from 'wavesurfer.js'
import Spectrogram from 'wavesurfer.js/dist/plugins/spectrogram.esm.js'
import { Volume2, VolumeX, Headphones, Activity, AlertTriangle } from 'lucide-react'
import { Button } from './ui/button'
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
    onToggleMute: () => void
    onToggleSolo: () => void
}

export const WaveformTrack = forwardRef<WaveformTrackHandle, WaveformTrackProps>(function WaveformTrack({
    url,
    name,
    color,
    height = 64,
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
    onToggleMute,
    onToggleSolo
}, ref) {
    const containerRef = useRef<HTMLDivElement>(null)
    const wavesurferRef = useRef<WaveSurfer | null>(null)
    const [isReady, setIsReady] = useState(false)
    const [showSpectrogram, setShowSpectrogram] = useState(false)
    const [loadError, setLoadError] = useState<string | null>(null)
    const isMasterRef = useRef(isMaster)
    const onTimeUpdateRef = useRef(onTimeUpdate)
    const onEndedRef = useRef(onEnded)

    useEffect(() => {
        isMasterRef.current = isMaster
        onTimeUpdateRef.current = onTimeUpdate
        onEndedRef.current = onEnded
    }, [isMaster, onEnded, onTimeUpdate])

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

        try {
            const ws = WaveSurfer.create({
                container: containerRef.current,
                waveColor: color,
                progressColor: adjustColorBrightness(color, -20),
                cursorColor: '#ef4444',
                height: showSpectrogram ? 64 : height,
                normalize: false,
                minPxPerSec: 50,
                interact: true,
                hideScrollbar: true,
                url: toMediaUrl(url),
                plugins,
            })

            ws.on('ready', () => {
                if (!mounted) return
                setIsReady(true)
                onReady(ws.getDuration())
            })

            ws.on('error', (error) => {
                if (error instanceof Error && error.name === 'AbortError') return
                console.error('[WaveformTrack] Error loading audio:', name, error)
                setLoadError(previewLoadMessage(undefined, error instanceof Error ? error.message : String(error)))
            })

            ws.on('interaction', (newTime) => {
                onSeek(newTime)
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

        return () => {
            mounted = false
            wavesurferRef.current?.destroy()
            wavesurferRef.current = null
            setIsReady(false)
            setLoadError(null)
        }
    }, [color, height, name, onReady, onSeek, showSpectrogram, url])

    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        ws.setTime(Math.max(0, currentTime))
        if (isPlaying) {
            void ws.play().catch((error) => {
                if (error instanceof Error && error.name === 'AbortError') return
                console.error('[WaveformTrack] Play failed:', name, error)
            })
        } else {
            ws.pause()
        }
    }, [currentTime, isPlaying, isReady, name, syncVersion])

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
        <div className="flex items-center gap-4 p-2 bg-secondary/30 rounded-lg border border-border/50 transition-all duration-300">
            <div className="w-32 flex flex-col gap-2 shrink-0">
                <div className="flex items-center gap-2">
                    <span
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: color }}
                    />
                    <span className="text-sm font-medium truncate capitalize flex-1">{name}</span>
                </div>

                <div className="flex gap-1">
                    <Button
                        variant={muted ? "destructive" : "ghost"}
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={onToggleMute}
                        title="Toggle Mute"
                    >
                        {muted ? <VolumeX className="h-3 w-3" /> : <Volume2 className="h-3 w-3" />}
                    </Button>
                    <Button
                        variant={soloed ? "default" : "ghost"}
                        size="sm"
                        className={cn(
                            "h-6 w-6 p-0",
                            soloed && "bg-yellow-500 hover:bg-yellow-600"
                        )}
                        onClick={onToggleSolo}
                        title="Toggle Solo"
                    >
                        <Headphones className="h-3 w-3" />
                    </Button>
                    <Button
                        variant={showSpectrogram ? "default" : "ghost"}
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => setShowSpectrogram(!showSpectrogram)}
                        title="Toggle Spectrogram"
                    >
                        <Activity className="h-3 w-3" />
                    </Button>
                </div>
            </div>

            <div
                className={cn(
                    "flex-1 rounded overflow-hidden relative min-h-[64px] transition-all ease-in-out",
                    showSpectrogram && "min-h-[192px]"
                )}
                style={{ height: showSpectrogram ? 192 : height }}
                ref={containerRef}
            >
                {!isReady && (
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                        {loadError ? (
                            <div className="flex items-center gap-2 px-3 text-center text-[11px] text-destructive">
                                <AlertTriangle className="h-4 w-4 shrink-0" />
                                <span>{loadError}</span>
                            </div>
                        ) : (
                            'Loading...'
                        )}
                    </div>
                )}
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
