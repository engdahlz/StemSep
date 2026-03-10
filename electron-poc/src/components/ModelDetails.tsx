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
import {
  formatCardMetricValue,
  formatCatalogStatus,
  formatMetricsSource,
  getCardMetricSlots,
} from "../lib/models/cardMetrics";

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
            ...tech,
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
  const metricSlots = getCardMetricSlots(resolvedModel);
  const metricsSource = formatMetricsSource(resolvedModel);
  const catalogStatus = formatCatalogStatus(resolvedModel.catalog_status);
  const qualityRoles = Array.isArray(resolvedModel.quality_role)
    ? resolvedModel.quality_role
    : resolvedModel.quality_role
      ? [resolvedModel.quality_role]
      : [];
  const workflowRoles = resolvedModel.workflow_roles || [];
  const runtimeRequired = resolvedModel.runtime?.required || [];
  const runtimeFallbacks = resolvedModel.runtime?.fallbacks || [];
  const runtimeHosts = resolvedModel.runtime?.hosts || [];
  const runtimeBurden = resolvedModel.runtime?.install_burden;
  const runtimeCustomFiles = resolvedModel.runtime?.requires_custom_repo_file || [];
  const phaseFixReferences = resolvedModel.phase_fix?.references || {};
  const qualityAxes = resolvedModel.quality_axes || {};
  const operatingProfiles = resolvedModel.operating_profiles || {};
  const downloadInfo = resolvedModel.download;
  const installation = resolvedModel.installation;
  const downloadSources = downloadInfo?.sources || [];
  const downloadArtifacts = downloadInfo?.artifacts || [];
  const manualInstructions = downloadInfo?.manual_instructions || [];

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
            {resolvedModel.catalog_status && (
              <Badge variant="outline" className="bg-secondary/50">
                {catalogStatus}
              </Badge>
            )}
            {metricsSource && (
              <Badge variant="outline" className="bg-secondary/50">
                {metricsSource}
              </Badge>
            )}
            {qualityRoles.map((role) => (
              <Badge key={role} variant="outline" className="bg-secondary/50">
                {String(role).replace(/[_-]+/g, " ")}
              </Badge>
            ))}
            {resolvedModel.install?.mode && (
              <Badge variant="outline" className="bg-secondary/50">
                Install: {resolvedModel.install.mode}
              </Badge>
            )}
            {resolvedModel.status?.curated && (
              <Badge variant="outline" className="bg-emerald-500/10 border-emerald-500/20 text-emerald-700">
                Curated
              </Badge>
            )}
            {resolvedModel.status?.support_tier === "supported_advanced" && (
              <Badge variant="outline" className="bg-slate-900/6">
                Supported Advanced
              </Badge>
            )}
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
            {metricSlots.map((slot) => (
              <div
                key={slot.label}
                className="p-4 rounded-lg bg-secondary/30 border border-border/50 text-center"
              >
                <div className="text-xs text-muted-foreground uppercase mb-1">
                  {slot.label}
                </div>
                <div className="text-xl font-semibold tabular-nums">
                  {formatCardMetricValue(slot.value)}
                </div>
              </div>
            ))}
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
              {runtimeRequired.length > 0 && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Runtime Required</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {runtimeRequired.join(", ")}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.engine && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Runtime Engine</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {resolvedModel.runtime.engine}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.model_type && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Model Type</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {resolvedModel.runtime.model_type}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.patch_profile && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Patch Profile</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {resolvedModel.runtime.patch_profile}
                  </span>
                </div>
              )}
              {runtimeFallbacks.length > 0 && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Runtime Fallbacks</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {runtimeFallbacks.join(", ")}
                  </span>
                </div>
              )}
              {runtimeHosts.length > 0 && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Best Hosts</span>
                  <span className="font-medium text-foreground/90 text-right">
                    {runtimeHosts.join(", ")}
                  </span>
                </div>
              )}
              {runtimeBurden && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Install Burden</span>
                  <span className="font-medium text-foreground/90">
                    {runtimeBurden}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.requires_manual_assets && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Manual Assets</span>
                  <span className="font-medium text-foreground/90">
                    Required
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.requires_patch && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Repo Patch</span>
                  <span className="font-medium text-foreground/90">
                    Required
                  </span>
                </div>
              )}
              {resolvedModel.vram_profile && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">VRAM Profile</span>
                  <span className="font-medium text-foreground/90">
                    {resolvedModel.vram_profile}
                  </span>
                </div>
              )}
              <div className="flex justify-between py-1 border-b border-border/30">
                <span className="text-muted-foreground">Chunk Size</span>
                <span className="tabular-nums font-medium text-foreground/90">
                  {chunkSizeDisplay}
                </span>
              </div>
              {resolvedModel.chunk_overlap_policy?.default_overlap && (
                <div className="flex justify-between py-1 border-b border-border/30">
                  <span className="text-muted-foreground">Default Overlap</span>
                  <span className="tabular-nums font-medium text-foreground/90">
                    {resolvedModel.chunk_overlap_policy.default_overlap}
                  </span>
                </div>
              )}
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
          {(workflowRoles.length > 0 ||
            resolvedModel.content_fit?.length ||
            Object.keys(qualityAxes).length > 0 ||
            runtimeCustomFiles.length > 0) && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Guide Fit
              </h3>
              <div className="space-y-2 text-sm text-foreground/90">
                {workflowRoles.length > 0 && (
                  <p>Workflow roles: {workflowRoles.join(", ")}</p>
                )}
                {resolvedModel.content_fit?.length ? (
                  <p>Content fit: {resolvedModel.content_fit.join(", ")}</p>
                ) : null}
                {runtimeCustomFiles.length > 0 && (
                  <p>Custom files: {runtimeCustomFiles.join(", ")}</p>
                )}
                {Object.keys(qualityAxes).length > 0 && (
                  <p>
                    Quality axes: {Object.entries(qualityAxes).map(([key, value]) => `${key}=${value}`).join(", ")}
                  </p>
                )}
              </div>
            </div>
          )}

          {(downloadInfo || installation) && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Installation
              </h3>
              <div className="rounded-lg border border-border/50 bg-secondary/20 p-4 space-y-3">
                <div className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                  {downloadInfo?.mode && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Download Mode</span>
                      <span className="font-medium text-foreground/90 capitalize">
                        {String(downloadInfo.mode).replace(/_/g, " ")}
                      </span>
                    </div>
                  )}
                  {downloadInfo?.source_policy && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Source Policy</span>
                      <span className="font-medium text-foreground/90 capitalize">
                        {String(downloadInfo.source_policy).replace(/_/g, " ")}
                      </span>
                    </div>
                  )}
                  {downloadInfo?.family && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Storage Family</span>
                      <span className="font-medium text-foreground/90">
                        {downloadInfo.family}
                      </span>
                    </div>
                  )}
                  {typeof downloadInfo?.artifact_count === "number" && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Artifacts</span>
                      <span className="font-medium text-foreground/90">
                        {downloadInfo.artifact_count}
                      </span>
                    </div>
                  )}
                  {typeof downloadInfo?.downloadable_artifact_count === "number" && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Direct Files</span>
                      <span className="font-medium text-foreground/90">
                        {downloadInfo.downloadable_artifact_count}
                      </span>
                    </div>
                  )}
                  {installation && (
                    <div className="flex justify-between py-1 border-b border-border/30">
                      <span className="text-muted-foreground">Installed</span>
                      <span className="font-medium text-foreground/90">
                        {installation.installed ? "Yes" : "No"}
                      </span>
                    </div>
                  )}
                </div>

                {downloadArtifacts.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Artifact Plan
                    </h4>
                    <div className="space-y-2">
                      {downloadArtifacts.map((artifact) => (
                        <div
                          key={`${artifact.relative_path}:${artifact.kind}`}
                          className="rounded-md border border-border/40 bg-background/60 px-3 py-2 text-sm"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <span className="font-medium text-foreground/90">
                              {artifact.filename}
                            </span>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <span>{artifact.kind}</span>
                              <span>{artifact.manual ? "Manual" : "Direct"}</span>
                              <span>{artifact.exists ? "Present" : "Missing"}</span>
                            </div>
                          </div>
                          <div className="mt-1 text-xs text-muted-foreground break-all">
                            {artifact.relative_path}
                          </div>
                          {artifact.source_host && (
                            <div className="mt-1 text-xs text-muted-foreground">
                              Source: {artifact.source_host}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {downloadSources.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Sources
                    </h4>
                    <div className="space-y-2">
                      {downloadSources.map((source) => (
                        <div
                          key={`${source.role}:${source.url}`}
                          className="flex items-center justify-between gap-3 rounded-md border border-border/40 bg-background/60 px-3 py-2 text-sm"
                        >
                          <div className="min-w-0">
                            <div className="font-medium text-foreground/90">
                              {source.role}
                            </div>
                            <div className="text-xs text-muted-foreground break-all">
                              {source.host}
                            </div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => window.electronAPI?.openExternalUrl?.(source.url)}
                            className="shrink-0 gap-1"
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                            Open
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {manualInstructions.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Manual Setup
                    </h4>
                    <div className="space-y-1 text-sm text-foreground/90">
                      {manualInstructions.map((instruction) => (
                        <p key={instruction}>{instruction}</p>
                      ))}
                    </div>
                    {installation?.missing_artifacts?.length ? (
                      <p className="text-xs text-muted-foreground">
                        Missing: {installation.missing_artifacts.join(", ")}
                      </p>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          )}
          {Object.keys(phaseFixReferences).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Phase Fix Pairings
              </h3>
              <div className="space-y-1 text-sm text-foreground/90">
                {Object.entries(phaseFixReferences).map(([stem, models]) => (
                  <p key={stem}>
                    {stem}: {Array.isArray(models) ? models.join(", ") : String(models)}
                  </p>
                ))}
              </div>
            </div>
          )}
          {Object.keys(operatingProfiles).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Operating Profiles
              </h3>
              <div className="space-y-2 text-sm text-foreground/90">
                {Object.entries(operatingProfiles).map(([profile, config]) => (
                  <p key={profile}>
                    {profile}: seg {config?.segment_size ?? "-"}, overlap {config?.overlap ?? "-"}, shifts {config?.shifts ?? "-"}
                  </p>
                ))}
              </div>
            </div>
          )}
          {(resolvedModel.best_for?.length || resolvedModel.artifacts_risk?.length || resolvedModel.workflow_groups?.length) && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Workflow Fit
              </h3>
              <div className="space-y-2 text-sm text-foreground/90">
                {resolvedModel.best_for?.length ? (
                  <p>
                    Best for: {resolvedModel.best_for.join(", ")}
                  </p>
                ) : null}
                {resolvedModel.workflow_groups?.length ? (
                  <p>
                    Groups: {resolvedModel.workflow_groups.join(", ")}
                  </p>
                ) : null}
                {resolvedModel.artifacts_risk?.length ? (
                  <p>
                    Risk flags: {resolvedModel.artifacts_risk.join(", ")}
                  </p>
                ) : null}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 pt-0 flex justify-between gap-2">
          <div className="flex gap-2">
            {!resolvedModel.installed ? (
              <Button
                onClick={() => {
                  if (downloadInfo?.mode === "manual") {
                    const primarySource = downloadSources[0]?.url;
                    if (primarySource) {
                      void window.electronAPI?.openExternalUrl?.(primarySource);
                    }
                    return;
                  }
                  onDownload?.(resolvedModel.id);
                }}
                className="gap-2"
                disabled={resolvedModel.downloading || downloadInfo?.mode === "unavailable"}
                variant={downloadInfo?.mode === "manual" ? "outline" : "default"}
              >
                {downloadInfo?.mode === "manual" ? (
                  <>
                    <ExternalLink className="h-4 w-4" />
                    Open Source
                  </>
                ) : resolvedModel.downloading ? (
                  <>
                    <Download className="h-4 w-4 animate-bounce" />
                    Downloading...
                  </>
                ) : downloadInfo?.mode === "unavailable" ? (
                  <>
                    <Download className="h-4 w-4" />
                    Unavailable
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    {downloadInfo?.artifact_count && downloadInfo.artifact_count > 1
                      ? `Download ${downloadInfo.artifact_count} Files`
                      : "Download Model"}
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
