import {
  Download,
  Info,
  Trash2,
  AlertCircle,
  RotateCw,
  Pause,
  Play,
  Sparkles,
} from "lucide-react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";
import { Model } from "../stores/useStore";
import {
  formatCardMetricValue,
  formatCatalogStatus,
  formatMetricsSource,
  getCatalogSupportCopy,
  getCardMetricSlots,
} from "../lib/models/cardMetrics";

interface ModelCardProps {
  model: Model;
  onDownload: (id: string) => void;
  onRemove: (id: string) => void;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  onDetails: (model: Model) => void;
  isSelected: boolean;
  onToggleSelection: (id: string) => void;
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};

const normalizeLabel = (value: string | undefined | null): string => {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
};

const isNumericMetricValue = (value: number | string | null | undefined): boolean => {
  if (typeof value === "number" && Number.isFinite(value)) return true;
  if (typeof value !== "string") return false;
  return /^-?\d+(\.\d+)?$/.test(value.trim());
};

export function ModelCard({
  model,
  onDownload,
  onRemove,
  onPause,
  onResume,
  onDetails,
  isSelected,
  onToggleSelection,
}: ModelCardProps) {
  const stems = Array.isArray(model.stems) ? model.stems : [];

  const phaseFixValid = !!model.phase_fix?.is_valid_reference;
  const phaseFixTitle = (() => {
    if (!phaseFixValid) return undefined;
    const p = model.phase_fix?.recommended_params;
    const ref = model.phase_fix?.reference_model_id;
    const parts: string[] = [];
    if (ref) parts.push(`Reference: ${ref}`);
    if (p && (p.lowHz || p.highHz || p.highFreqWeight)) {
      parts.push(
        `Recommended: lowHz=${p.lowHz ?? "-"}, highHz=${p.highHz ?? "-"}, highFreqWeight=${p.highFreqWeight ?? "-"}`,
      );
    }
    return parts.join("\n");
  })();

  const tags = Array.isArray((model as any).tags)
    ? (((model as any).tags as any[]).filter(
        (t) => typeof t === "string",
      ) as string[])
    : [];
  const visibleTags = tags.slice(0, 4);
  const metricSlots = getCardMetricSlots(model);
  const metricsSource = formatMetricsSource(model);
  const catalogStatus = formatCatalogStatus(model.catalog_status);
  const supportCopy = getCatalogSupportCopy(model);

  const statusTone = (() => {
    if (model.downloadError) {
      return {
        shell: "border-destructive/40 bg-destructive/[0.03]",
        pill: "bg-destructive/12 text-destructive border-destructive/25",
        label: "Needs Attention",
      };
    }
    if (model.downloading || model.downloadPaused) {
      return {
        shell: "border-primary/35 bg-primary/[0.03]",
        pill: "bg-primary/12 text-primary border-primary/25",
        label: model.downloadPaused ? "Paused" : "Downloading",
      };
    }
    if (model.installed) {
      return {
        shell: "border-emerald-500/30 bg-emerald-500/[0.04]",
        pill: "bg-emerald-500/12 text-emerald-600 border-emerald-500/25",
        label: "Installed",
      };
    }
    return {
      shell: "border-border/60 bg-card",
      pill: "bg-muted text-muted-foreground border-border/70",
      label: "Available",
    };
  })();

  const readiness = model.status?.readiness
    ? normalizeLabel(model.status.readiness)
    : null;
  const categoryLabel = normalizeLabel(model.category || model.architecture);
  const speedLabel =
    typeof model.speed === "string" && model.speed.trim()
      ? `${normalizeLabel(model.speed)} Speed`
      : null;
  const sizeLabel =
    typeof model.file_size === "number" && model.file_size > 0
      ? formatBytes(model.file_size)
      : null;
  const runtimeAllowedLabel = Array.isArray(model.runtime?.allowed)
    ? model.runtime.allowed[0]
    : null;
  const focusItems =
    stems.length > 0
      ? stems.slice(0, 4).map((stem) => normalizeLabel(stem))
      : visibleTags.length > 0
        ? visibleTags.slice(0, 4).map((tag) => normalizeLabel(tag))
        : [];
  const profileItems = [
    { label: "Runtime", value: normalizeLabel(model.runtime?.preferred || runtimeAllowedLabel || "Unknown") },
    { label: "VRAM", value: `${model.vram_required || 0} GB` },
    { label: "Source", value: metricsSource || "Registry" },
  ];
  const metaChips = [catalogStatus, categoryLabel, speedLabel, readiness].filter(
    (value): value is string => Boolean(value && value.trim()),
  );

  return (
    <div
      className={cn(
        "group relative flex h-full flex-col overflow-hidden rounded-[2rem] border bg-card shadow-[0_20px_55px_-34px_rgba(15,23,42,0.24)] transition-all duration-300",
        statusTone.shell,
        isSelected
          ? "ring-2 ring-primary/60 ring-offset-2 ring-offset-background"
          : "hover:-translate-y-0.5 hover:border-primary/20 hover:shadow-[0_26px_80px_-42px_rgba(15,23,42,0.32)]",
      )}
    >
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/35 to-transparent" />

      <div className="flex items-start justify-between gap-3 border-b border-border/50 px-5 py-4">
        <label className="model-card-chip flex items-center gap-3 text-sm text-muted-foreground">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => onToggleSelection(model.id)}
            className="h-4 w-4 rounded border-primary/40 bg-background text-primary focus:ring-primary cursor-pointer"
          />
          <span>Select</span>
        </label>

        <div className="flex items-center gap-2">
          {phaseFixValid && (
            <Badge
              variant="outline"
              className="border-amber-500/25 bg-amber-500/10 text-[10px] font-semibold uppercase tracking-[0.18em] text-amber-600"
              title={phaseFixTitle}
            >
              Phase Fix
            </Badge>
          )}
          {model.recommended && (
            <Badge
              variant="outline"
              className="border-primary/25 bg-primary/10 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary"
            >
              Guide Pick
            </Badge>
          )}
          {model.is_custom && (
            <Badge
              variant="outline"
              className="border-sky-500/25 bg-sky-500/10 text-[10px] font-semibold uppercase tracking-[0.18em] text-sky-600"
            >
              Custom
            </Badge>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onDetails(model)}
            className="h-8 w-8 rounded-full text-muted-foreground hover:text-foreground"
            title="Details"
          >
            <Info className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-5 px-5 py-5">
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="model-card-chip text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              {normalizeLabel(model.architecture)}
            </div>
            <Badge
              variant="outline"
              className={cn(
                "model-card-chip shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em]",
                statusTone.pill,
              )}
            >
              {statusTone.label}
            </Badge>
          </div>

          <div className="space-y-2">
            <h3
              className="model-card-heading max-w-[14rem] text-[1.55rem] leading-[1.06] text-foreground transition-colors group-hover:text-primary"
              title={model.name}
            >
              {model.name}
            </h3>
            <p className="text-[0.95rem] leading-6 text-muted-foreground">
              {model.description || "No description available for this model yet."}
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {metaChips.slice(0, 4).map((chip) => (
              <Badge
                key={chip}
                variant={chip === categoryLabel ? "secondary" : "outline"}
                className="model-card-chip rounded-full px-2.5 py-1 text-[11px] font-medium"
              >
                {chip}
              </Badge>
            ))}
            {sizeLabel && (
              <Badge variant="outline" className="model-card-chip rounded-full px-2.5 py-1 text-[11px] font-medium">
                {sizeLabel}
              </Badge>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[1.4rem] border border-border/60 bg-muted/[0.22] p-4">
            <div className="model-card-section-label text-[0.68rem] uppercase text-muted-foreground">
              Output Focus
            </div>
            <div className="mt-3 flex min-h-[3.5rem] flex-wrap items-start gap-2">
              {focusItems.length > 0 ? (
                focusItems.map((item) => (
                  <span
                    key={item}
                    className="model-card-chip rounded-full border border-border/70 bg-background/80 px-2.5 py-1 text-[11px] text-foreground/80"
                  >
                    {item}
                  </span>
                ))
              ) : (
                <span className="text-sm leading-6 text-muted-foreground">Stem info unavailable</span>
              )}
            </div>
          </div>

          <div className="rounded-[1.4rem] border border-border/60 bg-background/70 p-4">
            <div className="model-card-section-label text-[0.68rem] uppercase text-muted-foreground">
              Model Profile
            </div>
            <div className="mt-3 space-y-2.5">
              {profileItems.map((item) => (
                <div key={item.label} className="flex items-center justify-between gap-3">
                  <span className="model-card-chip text-[11px] text-muted-foreground">
                    {item.label}
                  </span>
                  <span className="text-right text-[0.82rem] font-medium text-foreground">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {metricSlots.map((slot) => (
            <div
              key={slot.label}
              className="flex min-h-[6.25rem] flex-col justify-between rounded-[1.35rem] border border-border/60 bg-background/75 px-3 py-3.5"
              title={model.card_metrics?.evidence_note || undefined}
            >
              <div className="model-card-metric-label text-[0.62rem] uppercase text-muted-foreground">
                {slot.label}
              </div>
              {isNumericMetricValue(slot.value) ? (
                <div className="model-card-metric-value mt-3 text-[1.75rem] tabular-nums text-foreground">
                  {formatCardMetricValue(slot.value)}
                </div>
              ) : (
                <div className="mt-3 inline-flex max-w-full items-center rounded-full bg-secondary px-2.5 py-1.5">
                  <span className="model-card-metric-value-text truncate text-[0.84rem] text-foreground">
                    {formatCardMetricValue(slot.value)}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>

        {model.downloading && (
          <div className="space-y-2 rounded-2xl border border-primary/20 bg-primary/[0.04] p-4">
            <div className="flex items-center justify-between text-xs font-medium">
              <span className="text-primary">Downloading model assets</span>
              <span>{Math.round(model.downloadProgress)}%</span>
            </div>
            <Progress value={model.downloadProgress} className="h-1.5" />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>
                {model.downloadSpeed
                  ? formatBytes(model.downloadSpeed) + "/s"
                  : "--"}
              </span>
              <span>
                ETA:{" "}
                {model.downloadEta ? Math.round(model.downloadEta) + "s" : "--"}
              </span>
            </div>
          </div>
        )}

        {model.downloadError && (
          <div className="flex items-start gap-2 rounded-2xl border border-destructive/20 bg-destructive/10 p-4 text-xs">
            <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-bold text-destructive">Failed</p>
              <p
                className="text-muted-foreground mt-0.5 line-clamp-1"
                title={model.downloadError}
              >
                {model.downloadError}
              </p>
            </div>
          </div>
        )}

        <div className="mt-auto flex items-center justify-between gap-3 border-t border-border/50 pt-4">
          <div className="flex max-w-[9rem] items-center gap-2 text-xs leading-5 text-muted-foreground">
            {model.recommended && <Sparkles className="h-3.5 w-3.5 text-primary" />}
            <span>{supportCopy}</span>
          </div>

          {model.installed ? (
            <Button
              variant="outline"
              className="border-destructive/20 text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => onRemove(model.id)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Remove
            </Button>
          ) : model.downloadPaused ? (
            <Button
              onClick={() => onResume(model.id)}
              className="min-w-[8.5rem]"
              variant="outline"
            >
              <Play className="mr-2 h-4 w-4" />
              Resume
            </Button>
          ) : (
            <Button
              onClick={
                model.downloading ? () => onPause(model.id) : () => onDownload(model.id)
              }
              className={cn(
                "min-w-[8.5rem] transition-all duration-300",
                model.downloading
                  ? "bg-secondary text-secondary-foreground"
                  : "bg-primary text-primary-foreground shadow-[0_14px_34px_-18px_rgba(0,0,0,0.55)] hover:shadow-[0_20px_40px_-18px_rgba(0,0,0,0.65)]",
              )}
              variant={model.downloadError ? "destructive" : "default"}
            >
              {model.downloading ? (
                <>
                  <Pause className="mr-2 h-4 w-4" /> Pause
                </>
              ) : model.downloadError ? (
                <>
                  <RotateCw className="mr-2 h-4 w-4" /> Retry
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" /> Download
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
