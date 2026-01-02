import {
  Download,
  Check,
  Info,
  Trash2,
  AlertCircle,
  RotateCw,
  Pause,
  Play,
  Cpu,
  Zap,
} from "lucide-react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";
import { Model } from "../stores/useStore";

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

const formatMetric = (
  value: number | null | undefined,
  decimals = 1,
): string => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0)
    return "—";
  return value.toFixed(decimals);
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

  const runtimeAllowed = model.runtime?.allowed;
  const runtimeBlockedReason = model.runtime?.blocking_reason;
  const isBlocked = runtimeAllowed === false || !!runtimeBlockedReason;

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

  const getStatusColor = () => {
    if (model.downloadError) return "border-destructive/50 bg-destructive/5";
    if (model.installed) return "border-green-500/30 bg-green-500/5";
    if (model.downloading) return "border-primary/50 bg-primary/5";
    return "border-border/40 bg-card/40";
  };

  return (
    <div
      className={cn(
        "group relative flex flex-col rounded-xl border backdrop-blur-sm transition-all duration-300 overflow-hidden",
        getStatusColor(),
        isSelected
          ? "ring-2 ring-primary ring-offset-2 ring-offset-background"
          : "hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5",
        isBlocked ? "opacity-70" : "",
      )}
      title={
        isBlocked
          ? runtimeBlockedReason ||
            "This model is blocked by runtime constraints"
          : undefined
      }
    >
      {/* Selection Overlay (visible on hover or selected) */}
      <div
        className={cn(
          "absolute top-3 left-3 z-20 transition-opacity duration-200",
          isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100",
        )}
      >
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onToggleSelection(model.id)}
          className="h-5 w-5 rounded border-primary/50 bg-background/50 text-primary focus:ring-primary cursor-pointer"
        />
      </div>

      {(model.recommended ||
        model.is_custom ||
        isBlocked ||
        phaseFixValid ||
        tags.length > 0) && (
        <div className="absolute top-3 right-3 z-20 flex flex-col gap-1 items-end">
          {isBlocked && (
            <Badge
              variant="default"
              className="bg-destructive/90 hover:bg-destructive text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
              title={
                runtimeBlockedReason ||
                "This model is blocked by runtime constraints"
              }
            >
              Blocked
            </Badge>
          )}
          {phaseFixValid && (
            <Badge
              variant="default"
              className="bg-amber-500/90 hover:bg-amber-500 text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
              title={phaseFixTitle}
            >
              Phase Fix
            </Badge>
          )}
          {tags.slice(0, 2).map((t) => (
            <Badge
              key={t}
              variant="default"
              className="bg-secondary/80 hover:bg-secondary text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
              title={t}
            >
              {t}
            </Badge>
          ))}
          {tags.length > 2 && (
            <Badge
              variant="default"
              className="bg-secondary/60 hover:bg-secondary text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
              title={tags.join(", ")}
            >
              +{tags.length - 2}
            </Badge>
          )}
          {model.recommended && (
            <Badge
              variant="default"
              className="bg-primary/90 hover:bg-primary text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
            >
              Recommended
            </Badge>
          )}
          {model.is_custom && (
            <Badge
              variant="default"
              className="bg-blue-600/90 hover:bg-blue-600 text-[10px] font-bold uppercase tracking-wider shadow-sm backdrop-blur-md"
            >
              Custom
            </Badge>
          )}
        </div>
      )}

      <div className="p-5 flex-1 flex flex-col gap-4">
        {/* Header */}
        <div className="space-y-1.5 pt-2">
          <div className="flex items-start justify-between gap-4">
            <h3
              className="font-bold text-lg leading-tight tracking-tight text-foreground/90 group-hover:text-primary transition-colors line-clamp-2"
              title={model.name}
            >
              {model.name}
            </h3>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1 bg-secondary/50 px-2 py-0.5 rounded-md">
              <Zap className="w-3 h-3" /> {model.architecture}
            </span>

            <span className="flex items-center gap-1 bg-secondary/50 px-2 py-0.5 rounded-md">
              <Cpu className="w-3 h-3" /> {model.vram_required}GB
            </span>
            {model.recommended_settings && (
              <span
                className="flex items-center gap-1 bg-blue-500/10 text-blue-500 px-2 py-0.5 rounded-md border border-blue-500/20"
                title="Has optimized settings"
              >
                <Zap className="w-3 h-3" /> Tuned
              </span>
            )}
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-muted-foreground line-clamp-2 min-h-[2.5rem]">
          {model.description}
        </p>

        {/* Stems */}
        <div className="flex flex-wrap gap-1.5">
          {stems.slice(0, 3).map((stem) => (
            <Badge
              key={stem}
              variant="outline"
              className="text-[10px] capitalize bg-background/30 border-white/10"
            >
              {stem}
            </Badge>
          ))}
          {stems.length > 3 && (
            <Badge
              variant="outline"
              className="text-[10px] bg-background/30 border-white/10"
            >
              +{stems.length - 3}
            </Badge>
          )}
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-2 py-3 border-t border-border/30 mt-auto">
          <div className="text-center p-2 rounded-lg bg-background/30">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              SDR
            </div>
            <div className="text-sm font-semibold tabular-nums">
              {formatMetric(model.sdr)}
            </div>
          </div>
          <div className="text-center p-2 rounded-lg bg-background/30">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Full
            </div>
            <div className="text-sm font-semibold tabular-nums">
              {formatMetric(model.fullness)}
            </div>
          </div>
          <div className="text-center p-2 rounded-lg bg-background/30">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Bleed
            </div>
            <div className="text-sm font-semibold tabular-nums">
              {formatMetric(model.bleedless)}
            </div>
          </div>
        </div>

        {/* Download State */}
        {model.downloading && (
          <div className="space-y-2 bg-background/40 p-3 rounded-lg border border-border/50">
            <div className="flex justify-between text-xs font-medium">
              <span className="text-primary">Downloading...</span>
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

        {/* Error State */}
        {model.downloadError && (
          <div className="flex items-start gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-xs">
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
      </div>

      {/* Footer Actions */}
      <div className="p-4 pt-0 flex gap-2 mt-auto">
        {model.installed ? (
          <>
            <Button
              variant="secondary"
              className="flex-1 bg-green-500/10 text-green-500 hover:bg-green-500/20 border border-green-500/20"
              disabled
            >
              <Check className="mr-2 h-4 w-4" />
              Installed
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onRemove(model.id)}
              className="text-muted-foreground hover:text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </>
        ) : model.downloadPaused ? (
          <Button
            onClick={() => onResume(model.id)}
            className="flex-1"
            variant="outline"
            disabled={isBlocked}
            title={isBlocked ? runtimeBlockedReason || "Blocked" : undefined}
          >
            <Play className="mr-2 h-4 w-4" />
            Resume
          </Button>
        ) : (
          <Button
            onClick={
              isBlocked
                ? undefined
                : model.downloading
                  ? () => onPause(model.id)
                  : () => onDownload(model.id)
            }
            className={cn(
              "flex-1 transition-all duration-300",
              model.downloading
                ? "bg-secondary text-secondary-foreground"
                : "bg-primary text-primary-foreground shadow-lg shadow-primary/20 hover:shadow-primary/40",
              isBlocked ? "cursor-not-allowed opacity-60" : "",
            )}
            variant={
              isBlocked
                ? "secondary"
                : model.downloadError
                  ? "destructive"
                  : "default"
            }
            disabled={isBlocked}
            title={
              isBlocked
                ? runtimeBlockedReason ||
                  "This model is blocked (missing checkpoints / unverified URL / manual steps required)"
                : undefined
            }
          >
            {isBlocked ? (
              <>
                <AlertCircle className="mr-2 h-4 w-4" /> Blocked
              </>
            ) : model.downloading ? (
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

        <Button
          variant="ghost"
          size="icon"
          onClick={() => onDetails(model)}
          className="text-muted-foreground hover:text-foreground"
        >
          <Info className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
