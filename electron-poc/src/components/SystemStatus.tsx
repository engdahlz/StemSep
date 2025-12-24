import { useEffect, useState } from 'react'
import { Cpu, Zap, Activity, FolderInput, HardDrive } from 'lucide-react'
import { useStore } from '../stores/useStore'

export function SystemStatus() {
    const [gpuInfo, setGpuInfo] = useState<any>(null)
    const isProcessing = useStore(state => state.separation.isProcessing)
    const isWatchMode = useStore(state => state.watchModeEnabled)

    useEffect(() => {
        const loadInfo = async () => {
            if (window.electronAPI?.getGpuDevices) {
                try {
                    const info = await window.electronAPI.getGpuDevices()
                    setGpuInfo(info)
                } catch (e) {
                    console.error("Failed to load GPU info:", e)
                    setGpuInfo({ gpus: [{ name: "Standard Driver (CPU)", type: "cpu", memory_gb: null }] })
                }
            }
        }
        loadInfo()
        const interval = setInterval(loadInfo, 30000)
        return () => clearInterval(interval)
    }, [])

    const activeGpu = gpuInfo?.gpus?.find((g: any) => g.recommended) || gpuInfo?.gpus?.[0]
    const system = gpuInfo?.system_info
    const isCuda = activeGpu?.type === 'cuda'

    // Efficiency Rating
    const getRating = () => {
        if (!gpuInfo) return { label: 'Analyzing...', color: 'text-muted-foreground', bg: 'bg-muted' }

        const vram = activeGpu?.memory_gb || 0
        const ram = system?.memory_total_gb || 0

        if (vram >= 8) return { label: 'Ultra', color: 'text-green-500', bg: 'bg-green-500/10' }
        if (vram >= 6) return { label: 'High', color: 'text-cyan-500', bg: 'bg-cyan-500/10' }
        if (vram >= 4) return { label: 'Medium', color: 'text-yellow-500', bg: 'bg-yellow-500/10' }
        if (ram >= 16 && !activeGpu) return { label: 'Decent (CPU)', color: 'text-blue-500', bg: 'bg-blue-500/10' }
        return { label: 'Low (CPU)', color: 'text-orange-500', bg: 'bg-orange-500/10' }
    }

    const rating = getRating()

    return (
        <div className="p-4 mt-auto border-t border-border/40 bg-transparent space-y-4">
            <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                    <Activity className="w-3 h-3" />
                    System Health
                </span>
                {isProcessing && (
                    <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                    </span>
                )}
            </div>

            <div className="space-y-3 text-xs">
                {/* CPU Info */}
                <div className="flex items-start gap-2.5">
                    <div className="p-1.5 rounded-md bg-secondary/50 text-muted-foreground shrink-0">
                        <Cpu className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0">
                        <p className="font-medium text-foreground truncate" title={system?.processor || 'Unknown CPU'}>
                            {system?.processor || 'CPU'}
                        </p>
                        <p className="text-[10px] text-muted-foreground">
                            {system?.cpu_count ? `${system.cpu_count} Cores` : '...'}
                        </p>
                    </div>
                </div>

                {/* RAM Info */}
                <div className="flex items-start gap-2.5">
                    <div className="p-1.5 rounded-md bg-secondary/50 text-muted-foreground shrink-0">
                        <HardDrive className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0">
                        <p className="font-medium text-foreground">System RAM</p>
                        <p className="text-[10px] text-muted-foreground">
                            {system?.memory_total_gb ? `${system.memory_total_gb} GB Total` : '...'}
                        </p>
                    </div>
                </div>

                {/* GPU Info */}
                <div className="flex items-start gap-2.5">
                    <div className={`p-1.5 rounded-md shrink-0 ${isCuda ? 'bg-yellow-500/10 text-yellow-500' : 'bg-secondary/50 text-muted-foreground'}`}>
                        {isCuda ? <Zap className="w-3.5 h-3.5" /> : <Zap className="w-3.5 h-3.5 opacity-50" />}
                    </div>
                    <div className="min-w-0">
                        <p className="font-medium text-foreground truncate" title={activeGpu?.name || 'No GPU Detected'}>
                            {activeGpu?.name || 'No GPU Detected'}
                        </p>
                        <p className="text-[10px] text-muted-foreground">
                            {activeGpu?.memory_gb ? `${activeGpu.memory_gb} GB VRAM` : 'Using CPU'}
                        </p>
                    </div>
                </div>
            </div>

            {/* Efficiency Rating */}
            <div className="pt-3 border-t border-border/30">
                <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-medium text-muted-foreground">Performance Rating</span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${rating.bg} ${rating.color}`}>
                        {rating.label}
                    </span>
                </div>
                <div className="h-1 w-full bg-secondary rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full ${rating.bg.replace('/10', '')}`}
                        style={{ width: rating.label.includes('Ultra') ? '100%' : rating.label.includes('High') ? '80%' : rating.label.includes('Medium') ? '60%' : '30%' }}
                    />
                </div>
            </div>



            {isWatchMode && (
                <div className="pt-2">
                    <div className="flex items-center gap-2 p-2 bg-primary/5 rounded-md border border-primary/10">
                        <FolderInput className="w-3.5 h-3.5 text-primary animate-pulse" />
                        <div className="flex-1 min-w-0">
                            <p className="text-[10px] font-medium text-primary truncate">Watch Mode Active</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}