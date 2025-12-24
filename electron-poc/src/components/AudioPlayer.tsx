import { useState, useRef, useEffect } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Play, Pause, Volume2, VolumeX } from 'lucide-react'

interface AudioPlayerProps {
  stems: Record<string, string>  // { stemName: filePath }
  onClose?: () => void
}

export function AudioPlayer({ stems, onClose }: AudioPlayerProps) {
  const [selectedStem, setSelectedStem] = useState<string>(Object.keys(stems)[0])
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Update audio source when stem changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.load()
      setIsPlaying(false)
      setCurrentTime(0)
    }
  }, [selectedStem])

  // Update playback time
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const updateTime = () => setCurrentTime(audio.currentTime)
    const updateDuration = () => setDuration(audio.duration)
    const handleEnded = () => setIsPlaying(false)

    audio.addEventListener('timeupdate', updateTime)
    audio.addEventListener('loadedmetadata', updateDuration)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('timeupdate', updateTime)
      audio.removeEventListener('loadedmetadata', updateDuration)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [])

  // Draw waveform visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      const width = canvas.width
      const height = canvas.height

      // Clear canvas
      ctx.clearRect(0, 0, width, height)

      // Draw background
      ctx.fillStyle = 'rgba(100, 100, 100, 0.1)'
      ctx.fillRect(0, 0, width, height)

      // Draw progress bar
      const progress = duration > 0 ? currentTime / duration : 0
      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'
      ctx.fillRect(0, 0, width * progress, height)

      // Draw animated waveform (simplified - real implementation would use Web Audio API)
      if (isPlaying) {
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)'
        ctx.lineWidth = 2
        ctx.beginPath()

        const segments = 50
        const segmentWidth = width / segments
        for (let i = 0; i < segments; i++) {
          const x = i * segmentWidth
          const amplitude = Math.sin((currentTime * 10 + i) * 0.5) * (height * 0.3)
          const y = height / 2 + amplitude
          
          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        }
        ctx.stroke()
      } else {
        // Static waveform when paused
        ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()
      }

      // Draw current time indicator
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(width * progress, 0)
      ctx.lineTo(width * progress, height)
      ctx.stroke()
    }

    draw()
    const animationId = isPlaying ? setInterval(draw, 50) : null

    return () => {
      if (animationId) clearInterval(animationId)
    }
  }, [isPlaying, currentTime, duration])

  const togglePlayPause = () => {
    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return
    const value = parseFloat(e.target.value)
    const newTime = (value / 100) * duration
    audioRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return
    const newVolume = parseFloat(e.target.value) / 100
    audioRef.current.volume = newVolume
    setVolume(newVolume)
    setIsMuted(newVolume === 0)
  }

  const toggleMute = () => {
    if (!audioRef.current) return
    if (isMuted) {
      audioRef.current.volume = volume
      setIsMuted(false)
    } else {
      audioRef.current.volume = 0
      setIsMuted(true)
    }
  }

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Audio Preview</span>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Stem Selector */}
        <div className="flex gap-2 flex-wrap">
          {Object.keys(stems).map((stem) => (
            <Button
              key={stem}
              variant={selectedStem === stem ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedStem(stem)}
              className="capitalize"
            >
              {stem}
            </Button>
          ))}
        </div>

        {/* Waveform Visualization */}
        <canvas
          ref={canvasRef}
          width={600}
          height={100}
          className="w-full rounded border border-border bg-background cursor-pointer"
          onClick={(e) => {
            if (!audioRef.current) return
            const rect = e.currentTarget.getBoundingClientRect()
            const x = e.clientX - rect.left
            const progress = x / rect.width
            const newTime = progress * duration
            audioRef.current.currentTime = newTime
            setCurrentTime(newTime)
          }}
        />

        {/* Playback Controls */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={togglePlayPause}
              className="w-12 h-12"
            >
              {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
            </Button>

            <div className="flex-1">
              <input
                type="range"
                value={duration > 0 ? (currentTime / duration) * 100 : 0}
                onChange={handleSeek}
                min={0}
                max={100}
                step={0.1}
                className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-accent"
                style={{
                  background: `linear-gradient(to right, rgb(59, 130, 246) 0%, rgb(59, 130, 246) ${duration > 0 ? (currentTime / duration) * 100 : 0}%, rgba(100, 100, 100, 0.3) ${duration > 0 ? (currentTime / duration) * 100 : 0}%, rgba(100, 100, 100, 0.3) 100%)`
                }}
              />
            </div>

            <span className="text-sm text-muted-foreground w-24 text-right">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          {/* Volume Control */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleMute}
              className="w-10"
            >
              {isMuted || volume === 0 ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>

            <input
              type="range"
              value={isMuted ? 0 : volume * 100}
              onChange={handleVolumeChange}
              min={0}
              max={100}
              step={1}
              className="w-32 h-2 rounded-lg appearance-none cursor-pointer bg-accent"
              style={{
                background: `linear-gradient(to right, rgb(59, 130, 246) 0%, rgb(59, 130, 246) ${isMuted ? 0 : volume * 100}%, rgba(100, 100, 100, 0.3) ${isMuted ? 0 : volume * 100}%, rgba(100, 100, 100, 0.3) 100%)`
              }}
            />
          </div>
        </div>

        {/* Hidden Audio Element */}
        <audio
          ref={audioRef}
          src={`file://${stems[selectedStem]}`}
          preload="metadata"
        />
      </CardContent>
    </Card>
  )
}
