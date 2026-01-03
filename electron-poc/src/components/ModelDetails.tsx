import { useEffect, useState } from "react";
import {
  X,
  Download,
  Trash2,
  ExternalLink,
  Zap,
  Cpu,
  Layers,
} from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Model } from "../stores/useStore";

interface ModelDetailsProps {
  model: Model;
  onClose: () => void;
  onDownload?: (id: string) => void;
  onRemove?: (id: string) => void;
}

export function ModelDetails({
  model,
  onClose,
  onDownload,
  onRemove,
}: ModelDetailsProps) {
  const [resolvedModel, setResolvedModel] = useState<Model>(model);

  useEffect(() => {
    setResolvedModel(model);
    const run = async () => {
      if (!window.electronAPI?.getModelTech) return;
      try {
        const res = await window.electronAPI.getModelTech(model.id);
        const tech = res?.data || res;
        if (tech && typeof tech === "object") {
          setResolvedModel((prev) => ({
            ...prev,
            chunk_size: tech.chunk_size ?? prev.chunk_size,
            dim_f: tech.dim_f ?? prev.dim_f,
            dim_t: tech.dim_t ?? prev.dim_t,
            n_fft: tech.n_fft ?? prev.n_fft,
            hop_length: tech.hop_length ?? prev.hop_length,
          }));
        }
      } catch {}
    };
    void run();
  }, [model]);

  const phaseFixValid = !!resolvedModel.phase_fix?.is_valid_reference;
  const phaseFixRef = resolvedModel.phase_fix?.reference_model_id;
  const phaseFixParams = resolvedModel.phase_fix?.recommended_params;

  const chunkSizeDisplay =
    resolvedModel.chunk_size ||
    resolvedModel.recommended_settings?.segment_size ||
    "Auto";
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/20 backdrop-blur-[1px] p-4 animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border bg-card shadow-2xl animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute right-4 top-4 z-10"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>

        {/* Header */}
        <div className="p-6 pb-4 border-b border-border/50 space-y-4">
          <div className="space-y-1">
            <h2 className="text-2xl font-bold tracking-tight">
              {resolvedModel.name}
            </h2>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Badge variant="secondary" className="font-normal">
                {resolvedModel.id}
              </Badge>
              {resolvedModel.repo_id && (
                <a
                  href={`https://huggingface.co/${resolvedModel.repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 hover:text-primary transition-colors"
                >
                  {resolvedModel.repo_id} <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge
              variant="outline"
              className="bg-primary/5 border-primary/20 text-primary"
            >
              <Zap className="mr-1 h-3 w-3" /> {resolvedModel.architecture}
            </Badge>
            <Badge variant="outline" className="bg-secondary/50">
              <Cpu className="mr-1 h-3 w-3" /> {resolvedModel.vram_required} GB
              VRAM
            </Badge>
            <Badge variant="outline" className="bg-secondary/50">
              <Layers className="mr-1 h-3 w-3" />{" "}
              {resolvedModel.stems.join(", ")}
            </Badge>
            {phaseFixValid && (
              <Badge
                variant="outline"
                className="bg-amber-500/10 border-amber-500/20 text-amber-600"
                title={
                  phaseFixRef
                    ? `Reference: ${phaseFixRef}`
                    : "Phase fix reference available"
                }
              >
                Phase Fix
              </Badge>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Runtime / Compatibility Notes */}
          {phaseFixValid && (
            <div className="space-y-3 rounded-lg border border-border/50 bg-secondary/20 p-4">
              {phaseFixValid && (
                <div className="space-y-1">
                  <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                    Phase Fix Reference
                  </h3>
                  <div className="text-sm text-foreground/90 space-y-1">
                    {phaseFixRef && (
                      <div className="flex justify-between py-1 border-b border-border/30">
                        <span className="text-muted-foreground">
                          Reference Model
                        </span>
                        <span className="tabular-nums font-medium text-foreground/90">
                          {phaseFixRef}
                        </span>
                      </div>
                    )}
                    {phaseFixParams && (
                      <>
                        <div className="flex justify-between py-1 border-b border-border/30">
                          <span className="text-muted-foreground">lowHz</span>
                          <span className="tabular-nums font-medium text-foreground/90">
                            {phaseFixParams.lowHz ?? "-"}
                          </span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-border/30">
                          <span className="text-muted-foreground">highHz</span>
                          <span className="tabular-nums font-medium text-foreground/90">
                            {phaseFixParams.highHz ?? "-"}
                          </span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-border/30">
                          <span className="text-muted-foreground">
                            highFreqWeight
                          </span>
                          <span className="tabular-nums font-medium text-foreground/90">
                            {phaseFixParams.highFreqWeight ?? "-"}
                          </span>
                        </div>
                      </>
                    )}
                    {!phaseFixRef && !phaseFixParams && (
                      <p className="text-sm text-muted-foreground">
                        This model has phase-fix metadata but no details were
                        provided.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          {/* Description */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Description
            </h3>
            <p className="text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
              {resolvedModel.description || "No description available."}
            </p>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {resolvedModel.sdr > 0 && (
              <div className="p-4 rounded-lg bg-secondary/30 border border-border/50 text-center">
                <div className="text-xs text-muted-foreground uppercase mb-1">
                  SDR
                </div>
                <div className="text-xl font-semibold tabular-nums">
                  {resolvedModel.sdr.toFixed(2)}
                </div>
              </div>
            )}
            {(resolvedModel.fullness || 0) > 0 && (
              <div className="p-4 rounded-lg bg-secondary/30 border border-border/50 text-center">
                <div className="text-xs text-muted-foreground uppercase mb-1">
                  Fullness
                </div>
                <div className="text-xl font-semibold tabular-nums">
                  {resolvedModel.fullness?.toFixed(2)}
                </div>
              </div>
            )}
            {(resolvedModel.bleedless || 0) > 0 && (
              <div className="p-4 rounded-lg bg-secondary/30 border border-border/50 text-center">
                <div className="text-xs text-muted-foreground uppercase mb-1">
                  Bleedless
                </div>
                <div className="text-xl font-semibold tabular-nums">
                  {resolvedModel.bleedless?.toFixed(2)}
                </div>
              </div>
            )}
            <div className="p-4 rounded-lg bg-secondary/30 border border-border/50 text-center">
              <div className="text-xs text-muted-foreground uppercase mb-1">
                Speed
              </div>
              <div className="text-xl font-bold capitalize">
                {resolvedModel.speed || "-"}
              </div>
            </div>
          </div>

          {/* Download Progress - Moved here for visibility */}
          {resolvedModel.downloading && (
            <div className="space-y-2 bg-muted/50 p-4 rounded-lg border border-border/50 animate-in fade-in slide-in-from-top-2">
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium">Downloading...</span>
                <span className="text-muted-foreground">
                  {Math.round(resolvedModel.downloadProgress || 0)}%
                </span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300 ease-out"
                  style={{ width: `${resolvedModel.downloadProgress || 0}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground mt-2">
                <span>
                  {resolvedModel.downloadSpeed
                    ? `${(resolvedModel.downloadSpeed / 1024 / 1024).toFixed(1)} MB/s`
                    : "Starting..."}
                </span>
                <span>
                  {resolvedModel.downloadEta
                    ? `~${Math.ceil(resolvedModel.downloadEta)}s remaining`
                    : "Calculating..."}
                </span>
              </div>
            </div>
          )}

          {/* Technical Details */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Technical Details
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 text-sm">
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-muted-foreground">Chunk Size</span>
                <span className="tabular-nums font-medium text-foreground/90">
                  {chunkSizeDisplay}
                </span>
              </div>
              {resolvedModel.hop_length && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Hop Length</span>
                  <span className="tabular-nums font-medium text-foreground/90">
                    {resolvedModel.hop_length}
                  </span>
                </div>
              )}
              {resolvedModel.dim_f && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Dim F</span>
                  <span className="tabular-nums font-medium text-foreground/90">
                    {resolvedModel.dim_f}
                  </span>
                </div>
              )}
              {resolvedModel.dim_t && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Dim T</span>
                  <span className="tabular-nums font-medium text-foreground/90">
                    {resolvedModel.dim_t}
                  </span>
                </div>
              )}
              {resolvedModel.n_fft && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">N FFT</span>
                  <span className="tabular-nums font-medium text-foreground/90">
                    {resolvedModel.n_fft}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 pt-0 flex justify-between gap-2">
          <div className="flex gap-2">
            {!resolvedModel.installed ? (
              <Button
                onClick={() => onDownload?.(resolvedModel.id)}
                className="gap-2"
                disabled={resolvedModel.downloading}
              >
                {resolvedModel.downloading ? (
                  <>
                    <Download className="h-4 w-4 animate-bounce" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Download Model
                  </>
                )}
              </Button>
            ) : (
              <Button
                variant="destructive"
                onClick={() => onRemove?.(resolvedModel.id)}
                className="gap-2"
              >
                <Trash2 className="h-4 w-4" />
                Remove Model
              </Button>
            )}
          </div>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}
