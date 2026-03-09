import { AlertTriangle, CheckCircle2, Cpu, HardDrive, Loader2, Wand2 } from "lucide-react"
import { Badge } from "./ui/badge"
import type { SeparationPreflightReport } from "@/types/preflight"

type SeparationPlanCardProps = {
  report?: SeparationPreflightReport | null
  loading?: boolean
  mode?: "simple" | "advanced"
}

const formatOverlap = (value?: number | null) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Auto"
  if (value >= 1) return `${Math.round(value)}`
  return `${Math.round(value * 100)}%`
}

const formatVram = (value?: number | null) => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return "Unknown"
  }
  return `${value.toFixed(1)} GB`
}

export function SeparationPlanCard({
  report,
  loading = false,
  mode = "simple",
}: SeparationPlanCardProps) {
  const plan = report?.plan
  const requiredModels = plan?.required_models || []
  const runtimeBlocks = plan?.runtime_blocks || []
  const recommendedAdjustments = plan?.recommended_adjustments || []
  const warnings = report?.warnings || []
  const errors = report?.errors || []

  if (!loading && !report) {
    return null
  }

  return (
    <div className="rounded-[1.6rem] border border-white/55 bg-[rgba(255,255,255,0.5)] p-5 shadow-[0_24px_80px_rgba(141,150,179,0.14)] backdrop-blur-xl">
      <div className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <h3 className="text-base font-medium text-slate-800">Execution Plan</h3>
            <div className="text-sm text-slate-500">
              {plan?.workflow_name || "Resolving workflow"}
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {loading && (
              <Badge variant="secondary" className="gap-1.5 border-0 bg-slate-900/6 text-slate-600">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Refreshing
              </Badge>
            )}
            {plan?.difficulty && <Badge variant="outline" className="border-white/60 bg-white/60 text-slate-600">{plan.difficulty}</Badge>}
            {plan?.expected_runtime_tier && (
              <Badge variant="secondary" className="border-0 bg-slate-900/6 text-slate-600">{plan.expected_runtime_tier}</Badge>
            )}
            {plan?.should_use_simple && (
              <Badge className="border-0 bg-white text-[#111111]">{mode === "simple" ? "Simple fit" : "Prefers Simple"}</Badge>
            )}
            {plan?.should_use_advanced && (
              <Badge variant="secondary" className="border-0 bg-slate-900/6 text-slate-600">
                {mode === "advanced" ? "Advanced fit" : "Prefers Advanced"}
              </Badge>
            )}
          </div>
        </div>
      </div>
      <div className="space-y-4">
        {plan && (
          <div className="grid gap-3 md:grid-cols-4">
            <div className="rounded-xl border border-white/55 bg-[rgba(255,255,255,0.6)] p-3">
              <div className="text-xs text-slate-500">Workflow</div>
              <div className="mt-1 font-medium capitalize text-slate-800">
                {plan.workflow_type || "single"}
              </div>
            </div>
            <div className="rounded-xl border border-white/55 bg-[rgba(255,255,255,0.6)] p-3">
              <div className="text-xs text-slate-500">Resolved Device</div>
              <div className="mt-1 flex items-center gap-2 font-medium text-slate-800">
                <Cpu className="h-4 w-4 text-slate-400" />
                {plan.resolved_device || "auto"}
              </div>
            </div>
            <div className="rounded-xl border border-white/55 bg-[rgba(255,255,255,0.6)] p-3">
              <div className="text-xs text-slate-500">Estimated VRAM</div>
              <div className="mt-1 flex items-center gap-2 font-medium text-slate-800">
                <HardDrive className="h-4 w-4 text-slate-400" />
                {formatVram(plan.estimated_vram_gb)}
              </div>
            </div>
            <div className="rounded-xl border border-white/55 bg-[rgba(255,255,255,0.6)] p-3">
              <div className="text-xs text-slate-500">Overlap / Segment</div>
              <div className="mt-1 font-medium text-slate-800">
                {formatOverlap(plan.resolved_overlap)} /{" "}
                {plan.resolved_segment_size || "Auto"}
              </div>
            </div>
          </div>
        )}

        {!!plan?.fallback_reason && (
          <div className="flex items-start gap-2 rounded-xl border border-amber-300/60 bg-amber-50/82 p-3 text-sm text-amber-900/80">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{plan.fallback_reason}</span>
          </div>
        )}

        {errors.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-rose-700">
              Blocking issues
            </div>
            <div className="space-y-2">
              {errors.map((error, index) => (
                <div
                  key={`${error}-${index}`}
                  className="rounded-xl border border-rose-300/45 bg-rose-50/82 p-3 text-sm text-rose-800/80"
                >
                  {error}
                </div>
              ))}
            </div>
          </div>
        )}

        {runtimeBlocks.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-slate-800">Runtime blocks</div>
            <div className="flex flex-wrap gap-2">
              {runtimeBlocks.map((block, index) => (
                <Badge key={`${block}-${index}`} variant="destructive" className="border-0 bg-rose-500/14 text-rose-700">
                  {block}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {requiredModels.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-slate-800">Required models</div>
            <div className="flex flex-wrap gap-2">
              {requiredModels.map((model, index) => (
                <Badge
                  key={`${model.id || model.name || "required-model"}-${index}`}
                  variant={model.installed ? "secondary" : "outline"}
                  className={
                    model.installed
                      ? "border-0 bg-emerald-500/14 text-emerald-700"
                      : "border-amber-300/55 bg-amber-50/80 text-amber-800"
                  }
                >
                  {model.name || model.id}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {recommendedAdjustments.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-800">
              <Wand2 className="h-4 w-4 text-slate-500" />
              Recommended adjustments
            </div>
            <div className="space-y-2">
              {recommendedAdjustments.map((adjustment, index) => (
                <div
                  key={`${adjustment}-${index}`}
                  className="rounded-xl border border-white/55 bg-[rgba(255,255,255,0.6)] p-3 text-sm text-slate-600"
                >
                  {adjustment}
                </div>
              ))}
            </div>
          </div>
        )}

        {warnings.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-slate-800">Warnings</div>
            <div className="space-y-2">
              {warnings.map((warning, index) => (
                <div
                  key={`${warning}-${index}`}
                  className="rounded-xl border border-amber-300/55 bg-amber-50/80 p-3 text-sm text-amber-800"
                >
                  {warning}
                </div>
              ))}
            </div>
          </div>
        )}

        {report?.can_proceed === true && errors.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-emerald-700">
            <CheckCircle2 className="h-4 w-4" />
            Backend preflight is green for the current plan.
          </div>
        )}
      </div>
    </div>
  )
}
