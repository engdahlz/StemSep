import { AlertTriangle, CheckCircle2, Cpu, HardDrive, Loader2, Wand2 } from "lucide-react"
import { Badge } from "./ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
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
    <Card className="border-primary/20 bg-card/80">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base">Execution Plan</CardTitle>
            <div className="text-sm text-muted-foreground">
              {plan?.workflow_name || "Resolving workflow"}
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {loading && (
              <Badge variant="secondary" className="gap-1.5">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Refreshing
              </Badge>
            )}
            {plan?.difficulty && <Badge variant="outline">{plan.difficulty}</Badge>}
            {plan?.expected_runtime_tier && (
              <Badge variant="secondary">{plan.expected_runtime_tier}</Badge>
            )}
            {plan?.should_use_simple && (
              <Badge>{mode === "simple" ? "Simple fit" : "Prefers Simple"}</Badge>
            )}
            {plan?.should_use_advanced && (
              <Badge variant="secondary">
                {mode === "advanced" ? "Advanced fit" : "Prefers Advanced"}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {plan && (
          <div className="grid gap-3 md:grid-cols-4">
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Workflow</div>
              <div className="mt-1 font-medium capitalize">
                {plan.workflow_type || "single"}
              </div>
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Resolved Device</div>
              <div className="mt-1 flex items-center gap-2 font-medium">
                <Cpu className="h-4 w-4 text-muted-foreground" />
                {plan.resolved_device || "auto"}
              </div>
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Estimated VRAM</div>
              <div className="mt-1 flex items-center gap-2 font-medium">
                <HardDrive className="h-4 w-4 text-muted-foreground" />
                {formatVram(plan.estimated_vram_gb)}
              </div>
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Overlap / Segment</div>
              <div className="mt-1 font-medium">
                {formatOverlap(plan.resolved_overlap)} /{" "}
                {plan.resolved_segment_size || "Auto"}
              </div>
            </div>
          </div>
        )}

        {!!plan?.fallback_reason && (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{plan.fallback_reason}</span>
          </div>
        )}

        {errors.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-destructive">
              Blocking issues
            </div>
            <div className="space-y-2">
              {errors.map((error, index) => (
                <div
                  key={`${error}-${index}`}
                  className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm"
                >
                  {error}
                </div>
              ))}
            </div>
          </div>
        )}

        {runtimeBlocks.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Runtime blocks</div>
            <div className="flex flex-wrap gap-2">
              {runtimeBlocks.map((block, index) => (
                <Badge key={`${block}-${index}`} variant="destructive">
                  {block}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {requiredModels.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Required models</div>
            <div className="flex flex-wrap gap-2">
              {requiredModels.map((model) => (
                <Badge
                  key={model.id}
                  variant={model.installed ? "secondary" : "outline"}
                  className={
                    model.installed
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
                      : "border-amber-500/30 bg-amber-500/10 text-amber-200"
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
            <div className="flex items-center gap-2 text-sm font-medium">
              <Wand2 className="h-4 w-4 text-primary" />
              Recommended adjustments
            </div>
            <div className="space-y-2">
              {recommendedAdjustments.map((adjustment, index) => (
                <div
                  key={`${adjustment}-${index}`}
                  className="rounded-md border bg-muted/30 p-3 text-sm"
                >
                  {adjustment}
                </div>
              ))}
            </div>
          </div>
        )}

        {warnings.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Warnings</div>
            <div className="space-y-2">
              {warnings.map((warning, index) => (
                <div
                  key={`${warning}-${index}`}
                  className="rounded-md border border-amber-500/20 bg-amber-500/5 p-3 text-sm"
                >
                  {warning}
                </div>
              ))}
            </div>
          </div>
        )}

        {report?.can_proceed === true && errors.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-emerald-300">
            <CheckCircle2 className="h-4 w-4" />
            Backend preflight is green for the current plan.
          </div>
        )}
      </CardContent>
    </Card>
  )
}
