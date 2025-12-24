import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import Spectrogram from 'wavesurfer.js/dist/plugins/spectrogram.esm.js'
import { Volume2, VolumeX, Headphones, Activity } from 'lucide-react'
import { Button } from './ui/button'
import { cn } from '../lib/utils'

interface WaveformTrackProps {
    url: string
    name: string
    color: string
    height?: number
    isPlaying: boolean
    volume: number
    muted: boolean
    soloed: boolean
    playbackRate: number
    onReady: (duration: number) => void
    onSeek: (time: number) => void
    onToggleMute: () => void
    onToggleSolo: () => void
}

export function WaveformTrack({
    url,
    name,
    color,
    height = 64,
    isPlaying,
    volume,
    muted,
    soloed,
    playbackRate,
    onReady,
    onSeek,
    onToggleMute,
    onToggleSolo
}: WaveformTrackProps) {
    const containerRef = useRef<HTMLDivElement>(null)
    const wavesurferRef = useRef<WaveSurfer | null>(null)
    const [isReady, setIsReady] = useState(false)
    const [showSpectrogram, setShowSpectrogram] = useState(false)

    // Initialize WaveSurfer - only runs once per URL
    useEffect(() => {
        if (!containerRef.current) return

        let mounted = true
        let blobUrl: string | null = null

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

        const loadAudio = async () => {
            console.log('[WaveformTrack] Loading audio via IPC:', url)

            if (!window.electronAPI?.readAudioFile) {
                console.error('[WaveformTrack] electronAPI.readAudioFile not available')
                return
            }

            try {
                const result = await window.electronAPI.readAudioFile(url)

                if (!mounted) {
                    console.log('[WaveformTrack] Component unmounted, aborting load')
                    return
                }

                if (!result.success || !result.data) {
                    console.error('[WaveformTrack] Failed to read audio file:', result.error)
                    return
                }

                const binaryData = atob(result.data)
                const bytes = new Uint8Array(binaryData.length)
                for (let i = 0; i < binaryData.length; i++) {
                    bytes[i] = binaryData.charCodeAt(i)
                }
                const blob = new Blob([bytes], { type: result.mimeType })
                blobUrl = URL.createObjectURL(blob)

                if (!mounted) {
                    URL.revokeObjectURL(blobUrl)
                    return
                }

                console.log('[WaveformTrack] Created blob URL for:', name)

                const ws = WaveSurfer.create({
                    container: containerRef.current!,
                    waveColor: color,
                    progressColor: adjustColorBrightness(color, -20),
                    cursorColor: '#ef4444',
                    height: showSpectrogram ? 64 : height,
                    normalize: false,
                    minPxPerSec: 50,
                    interact: true,
                    hideScrollbar: true,
                    url: blobUrl,
                    plugins: plugins,
                })

                ws.on('ready', () => {
                    if (!mounted) return
                    console.log('[WaveformTrack] Audio ready:', name)
                    setIsReady(true)
                    onReady(ws.getDuration())
                })

                ws.on('error', (error) => {
                    if (error instanceof Error && error.name === 'AbortError') return
                    console.error('[WaveformTrack] Error loading audio:', name, error)
                })

                ws.on('interaction', (newTime) => {
                    onSeek(newTime)
                })

                wavesurferRef.current = ws
                    ; (ws as any)._blobUrl = blobUrl
            } catch (error) {
                if (error instanceof Error && error.name === 'AbortError') return
                console.error('[WaveformTrack] Error loading audio via IPC:', error)
            }
        }

        loadAudio()

        return () => {
            mounted = false
            if (wavesurferRef.current) {
                const storedBlobUrl = (wavesurferRef.current as any)._blobUrl
                if (storedBlobUrl) {
                    URL.revokeObjectURL(storedBlobUrl)
                }
                wavesurferRef.current.destroy()
                wavesurferRef.current = null
            } else if (blobUrl) {
                URL.revokeObjectURL(blobUrl)
            }
            setIsReady(false)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [url, showSpectrogram])

    // Sync Playback State - simply play/pause
    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        if (isPlaying) {
            ws.play()
        } else {
            ws.pause()
        }
    }, [isPlaying, isReady])

    // Sync Playback Rate
    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        ws.setPlaybackRate(playbackRate)
    }, [playbackRate, isReady])

    // Sync Volume/Mute
    useEffect(() => {
        const ws = wavesurferRef.current
        if (!ws || !isReady) return

        ws.setVolume(muted ? 0 : volume)
    }, [volume, muted, isReady])

    return (
        <div className="flex items-center gap-4 p-2 bg-secondary/30 rounded-lg border border-border/50 transition-all duration-300">
            {/* Track Controls */}
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
                        className={cn("h-6 w-6 p-0")}
                        onClick={() => setShowSpectrogram(!showSpectrogram)}
                        title="Toggle Spectrogram"
                    >
                        <Activity className="h-3 w-3" />
                    </Button>
                </div>
            </div>

            {/* Waveform */}
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
                        Loading...
                    </div>
                )}
            </div>
        </div>
    )
}

// Helper to darken/lighten hex color
function adjustColorBrightness(hex: string, percent: number) {
    const num = parseInt(hex.replace('#', ''), 16)
    const amt = Math.round(2.55 * percent)
    const R = (num >> 16) + amt
    const B = ((num >> 8) & 0x00ff) + amt
    const G = (num & 0x0000ff) + amt
    return '#' + (0x1000000 + (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 + (B < 255 ? (B < 1 ? 0 : B) : 255) * 0x100 + (G < 255 ? (G < 1 ? 0 : G) : 255)).toString(16).slice(1)
}
