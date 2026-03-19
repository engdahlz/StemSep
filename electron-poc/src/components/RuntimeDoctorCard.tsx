import { Activity, CheckCircle2, AlertTriangle, Wrench } from 'lucide-react'
import { Badge } from './ui/badge'
import { useSystemRuntimeInfo } from '../hooks/useSystemRuntimeInfo'

interface RuntimeDoctorCardProps {
  compact?: boolean
  showHeader?: boolean
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
  <div className="flex items-start justify-between gap-3 rounded-[1rem] border border-white/55 bg-[rgba(255,255,255,0.56)] p-3">
    <div className="space-y-1">
      <div className="text-sm font-medium text-slate-800">{label}</div>
      <div className="break-all text-xs text-slate-500">{detail}</div>
    </div>
    <Badge
      variant={ok ? 'secondary' : 'destructive'}
      className={ok ? 'shrink-0 border-0 bg-emerald-500/14 text-emerald-700' : 'shrink-0 border-0 bg-rose-500/14 text-rose-700'}
    >
      {ok ? 'OK' : 'Action'}
    </Badge>
  </div>
)

export function RuntimeDoctorCard({
  compact = false,
  showHeader = true,
}: RuntimeDoctorCardProps) {
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
    <div className="rounded-[1.6rem] border border-white/55 bg-[rgba(255,255,255,0.5)] p-5 shadow-[0_24px_80px_rgba(141,150,179,0.14)] backdrop-blur-xl">
      {showHeader && (
        <div className={compact ? 'pb-3' : 'pb-4'}>
          <div className="flex items-center gap-2 text-[16px] font-medium text-slate-800">
            <Activity className="h-5 w-5" />
            Model / Env Doctor
          </div>
          <div className="mt-1 text-sm text-slate-500">
            Preflight checks for Python, Torch, CUDA and FNO compatibility.
          </div>
        </div>
      )}
      <div className="space-y-3">
        {loading && (
          <div className="text-sm text-slate-500">Collecting runtime diagnostics...</div>
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
              <div className="rounded-xl border border-rose-300/45 bg-rose-50/78 p-3 text-sm">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="mt-0.5 h-4 w-4 text-rose-500" />
                  <div className="text-rose-800/80">
                    Runtime probe issue: {info?.runtimeFingerprintError || error}
                  </div>
                </div>
              </div>
            )}

            {!fnoSupported && (
              <div className="rounded-xl border border-amber-300/55 bg-amber-50/82 p-3 text-sm">
                <div className="flex items-start gap-2">
                  <Wrench className="mt-0.5 h-4 w-4 text-amber-600" />
                  <div className="text-amber-900/78">
                    Install a neuraloperator/neuralop build that exposes{' '}
                    <code>neuralop.models.FNO1d</code>, then restart StemSep.
                  </div>
                </div>
              </div>
            )}

            {fnoSupported && cudaAvailable && (
              <div className="rounded-xl border border-emerald-300/45 bg-emerald-50/80 p-3 text-sm text-emerald-900/75">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-600" />
                  <div>Environment looks healthy for standard GPU separation workloads.</div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
