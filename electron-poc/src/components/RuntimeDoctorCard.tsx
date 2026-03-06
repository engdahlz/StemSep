import { Activity, CheckCircle2, AlertTriangle, Wrench } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { useSystemRuntimeInfo } from '../hooks/useSystemRuntimeInfo'

interface RuntimeDoctorCardProps {
  compact?: boolean
}

const Status = ({
  ok,
  label,
  detail,
}: {
  ok: boolean
  label: string
  detail: string
}) => (
  <div className="flex items-start justify-between gap-3 rounded-md border p-3">
    <div className="space-y-1">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-muted-foreground break-all">{detail}</div>
    </div>
    <Badge variant={ok ? 'secondary' : 'destructive'} className="shrink-0">
      {ok ? 'OK' : 'Action'}
    </Badge>
  </div>
)

export function RuntimeDoctorCard({ compact = false }: RuntimeDoctorCardProps) {
  const { info, loading, error } = useSystemRuntimeInfo()
  const runtime = info?.runtimeFingerprint
  const torch = runtime?.torch
  const neuralop = runtime?.neuralop

  const fnoSupported = neuralop?.fno1d_import_ok !== false
  const cudaAvailable = torch?.cuda_available !== false
  const pythonVersion = runtime?.version || 'Unknown'
  const torchVersion = torch?.version || 'Unknown'
  const neuralopVersion = neuralop?.version || 'Not detected'

  return (
    <Card>
      <CardHeader className={compact ? 'pb-3' : undefined}>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Model / Env Doctor
        </CardTitle>
        <CardDescription>
          Preflight checks for Python, Torch, CUDA and FNO compatibility.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && (
          <div className="text-sm text-muted-foreground">Collecting runtime diagnostics...</div>
        )}

        {!loading && (
          <>
            <Status ok={!!runtime?.version} label="Python" detail={pythonVersion} />
            <Status
              ok={torchVersion !== 'Unknown'}
              label="Torch / CUDA"
              detail={`${torchVersion} • CUDA ${cudaAvailable ? 'available' : 'not available'}`}
            />
            <Status
              ok={fnoSupported}
              label="FNO (neuralop.models.FNO1d)"
              detail={
                fnoSupported
                  ? `${neuralopVersion} • compatible`
                  : (neuralop?.fno1d_import_error || 'FNO1d import failed')
              }
            />

            {(info?.runtimeFingerprintError || error) && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive mt-0.5" />
                  <div className="text-muted-foreground">
                    Runtime probe issue: {info?.runtimeFingerprintError || error}
                  </div>
                </div>
              </div>
            )}

            {!fnoSupported && (
              <div className="rounded-md border border-yellow-500/40 bg-yellow-500/10 p-3 text-sm">
                <div className="flex items-start gap-2">
                  <Wrench className="h-4 w-4 text-yellow-500 mt-0.5" />
                  <div className="text-muted-foreground">
                    Install a neuraloperator/neuralop build that exposes{' '}
                    <code>neuralop.models.FNO1d</code>, then restart StemSep.
                  </div>
                </div>
              </div>
            )}

            {fnoSupported && cudaAvailable && (
              <div className="rounded-md border border-green-500/30 bg-green-500/10 p-3 text-sm text-muted-foreground">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5" />
                  <div>Environment looks healthy for standard GPU separation workloads.</div>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

