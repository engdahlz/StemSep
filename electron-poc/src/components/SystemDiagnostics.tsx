import { Cpu, HardDrive, Zap } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

interface SystemInfo {
  gpu: string
  vram: string
  cpu: string
  ram: string
  cudaAvailable: boolean
}

interface SystemDiagnosticsProps {
  systemInfo?: SystemInfo
}

export function SystemDiagnostics({ systemInfo }: SystemDiagnosticsProps) {
  // Mock data for now - will connect to Python backend
  const info: SystemInfo = systemInfo || {
    gpu: 'NVIDIA RTX 2070',
    vram: '7.6 GB',
    cpu: 'AMD Ryzen 7 5800X',
    ram: '21.2 GB',
    cudaAvailable: true,
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5" />
          System Information
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* GPU */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <Zap className="w-4 h-4" />
              <span>GPU</span>
            </div>
            <p className="font-semibold">{info.gpu}</p>
            <p className="text-sm text-muted-foreground">
              {info.vram} VRAM
              {info.cudaAvailable && (
                <span className="ml-2 text-xs bg-green-500/20 text-green-500 px-2 py-0.5 rounded">
                  CUDA
                </span>
              )}
            </p>
          </div>

          {/* CPU */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <Cpu className="w-4 h-4" />
              <span>CPU</span>
            </div>
            <p className="font-semibold">{info.cpu}</p>
          </div>

          {/* RAM */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <HardDrive className="w-4 h-4" />
              <span>RAM</span>
            </div>
            <p className="font-semibold">{info.ram} Available</p>
          </div>

          {/* Performance Indicator */}
          <div className="space-y-1">
            <div className="text-muted-foreground text-sm">
              Performance
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-green-500 w-[85%]" />
              </div>
              <span className="text-sm font-semibold">Good</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
