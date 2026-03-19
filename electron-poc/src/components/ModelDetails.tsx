import { useEffect, useState } from "react";
import {
  X,
  Download,
  Upload,
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
  const canManualImport =
    resolvedModel.availability?.class === "manual_import" ||
    downloadInfo?.mode === "manual";

  const handleImportFiles = async () => {
    try {
      const selected = await window.electronAPI?.openModelFileDialog?.();
      if (!Array.isArray(selected) || selected.length === 0) return;
      const files = selected.map((filePath: string) => ({ path: filePath }));
      await window.electronAPI?.importModelFiles?.(resolvedModel.id, files, true);
      const tech = await window.electronAPI?.getModelTech?.(resolvedModel.id);
      const next = tech?.data || tech;
      if (next && typeof next === "object") {
        setResolvedModel((prev) => ({ ...prev, ...next, installed: !!next.installed }));
      }
    } catch (error) {
      console.error("Failed to import model files", error);
    }
  };

  const chunkSizeDisplay =
    resolvedModel.chunk_size ||
    resolvedModel.recommended_settings?.segment_size ||
    "Auto";
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-[rgba(24,28,40,0.34)] backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div
        className="relative max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-[2rem] border border-white/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(248,249,252,0.94))] shadow-[0_34px_96px_rgba(0,0,0,0.22)] animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        <Button
          variant="outline"
          size="icon"
          className="absolute right-5 top-5 z-10 rounded-full border-white/70 bg-white/82 text-slate-600 shadow-[0_10px_24px_rgba(0,0,0,0.08)] backdrop-blur-sm hover:bg-white hover:text-slate-900"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>

        <div className="space-y-5 border-b border-slate-200/70 bg-[linear-gradient(135deg,rgba(255,255,255,0.82),rgba(241,244,249,0.88))] p-7 pb-5">
          <div className="space-y-1">
            <h2 className="pr-12 text-[30px] font-medium tracking-[-1px] text-slate-950">
              {resolvedModel.name}
            </h2>
            <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
              <Badge
                variant="secondary"
                className="rounded-full border border-white/80 bg-white/82 font-normal text-slate-600 shadow-[inset_0_1px_0_rgba(255,255,255,0.65)]"
              >
                {resolvedModel.id}
              </Badge>
              {resolvedModel.repo_id && (
                <a
                  href={`https://huggingface.co/${resolvedModel.repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 transition-colors hover:text-slate-900"
                >
                  {resolvedModel.repo_id} <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge
              variant="outline"
              className="rounded-full border-violet-200 bg-violet-50 text-violet-700"
            >
              <Zap className="mr-1 h-3 w-3" /> {resolvedModel.architecture}
            </Badge>
            <Badge variant="outline" className="rounded-full border-slate-200 bg-white/75 text-slate-600">
              <Cpu className="mr-1 h-3 w-3" /> {resolvedModel.vram_required} GB
              VRAM
            </Badge>
            <Badge variant="outline" className="rounded-full border-slate-200 bg-white/75 text-slate-600">
              <Layers className="mr-1 h-3 w-3" />{" "}
              {resolvedModel.stems.join(", ")}
            </Badge>
            {resolvedModel.catalog_status && (
              <Badge variant="outline" className="rounded-full border-slate-200 bg-white/75 text-slate-600">
                {catalogStatus}
              </Badge>
            )}
            {metricsSource && (
              <Badge variant="outline" className="rounded-full border-slate-200 bg-white/75 text-slate-600">
                {metricsSource}
              </Badge>
            )}
            {qualityRoles.map((role) => (
              <Badge
                key={role}
                variant="outline"
                className="rounded-full border-slate-200 bg-white/75 text-slate-600"
              >
                {String(role).replace(/[_-]+/g, " ")}
              </Badge>
            ))}
            {resolvedModel.install?.mode && (
              <Badge variant="outline" className="rounded-full border-slate-200 bg-white/75 text-slate-600">
                Install: {resolvedModel.install.mode}
              </Badge>
            )}
            {resolvedModel.status?.curated && (
              <Badge variant="outline" className="rounded-full border-emerald-200 bg-emerald-50 text-emerald-700">
                Curated
              </Badge>
            )}
            {resolvedModel.status?.support_tier === "supported_advanced" && (
              <Badge variant="outline" className="rounded-full border-amber-200 bg-amber-50 text-amber-700">
                Supported Advanced
              </Badge>
            )}
            {phaseFixValid && (
              <Badge
                variant="outline"
                className="rounded-full border-amber-200 bg-amber-50 text-amber-700"
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

        <div className="space-y-6 p-7">
          {phaseFixValid && (
            <div className="space-y-3 rounded-[1.4rem] border border-amber-100 bg-amber-50/60 p-5">
              {phaseFixValid && (
                <div className="space-y-1">
                  <h3 className="text-sm font-medium uppercase tracking-wider text-amber-700/85">
                    Phase Fix Reference
                  </h3>
                  <div className="space-y-1 text-sm text-slate-700">
                    {phaseFixRef && (
                      <div className="flex justify-between border-b border-amber-100/80 py-1">
                        <span className="text-slate-500">
                          Reference Model
                        </span>
                        <span className="tabular-nums font-medium text-slate-900">
                          {phaseFixRef}
                        </span>
                      </div>
                    )}
                    {phaseFixParams && (
                      <>
                        <div className="flex justify-between border-b border-amber-100/80 py-1">
                          <span className="text-slate-500">lowHz</span>
                          <span className="tabular-nums font-medium text-slate-900">
                            {phaseFixParams.lowHz ?? "-"}
                          </span>
                        </div>
                        <div className="flex justify-between border-b border-amber-100/80 py-1">
                          <span className="text-slate-500">highHz</span>
                          <span className="tabular-nums font-medium text-slate-900">
                            {phaseFixParams.highHz ?? "-"}
                          </span>
                        </div>
                        <div className="flex justify-between border-b border-amber-100/80 py-1">
                          <span className="text-slate-500">
                            highFreqWeight
                          </span>
                          <span className="tabular-nums font-medium text-slate-900">
                            {phaseFixParams.highFreqWeight ?? "-"}
                          </span>
                        </div>
                      </>
                    )}
                    {!phaseFixRef && !phaseFixParams && (
                      <p className="text-sm text-slate-500">
                        This model has phase-fix metadata but no details were
                        provided.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="space-y-2 rounded-[1.4rem] border border-slate-200/70 bg-white/68 p-5">
            <h3 className="text-sm font-medium uppercase tracking-wider text-slate-500">
              Description
            </h3>
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
              {resolvedModel.description || "No description available."}
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {metricSlots.map((slot) => (
              <div
                key={slot.label}
                className="rounded-[1.2rem] border border-slate-200/70 bg-white/72 p-4 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]"
              >
                <div className="mb-1 text-xs uppercase text-slate-400">
                  {slot.label}
                </div>
                <div className="text-xl font-semibold tabular-nums text-slate-900">
                  {formatCardMetricValue(slot.value)}
                </div>
              </div>
            ))}
            <div className="rounded-[1.2rem] border border-slate-200/70 bg-white/72 p-4 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]">
              <div className="mb-1 text-xs uppercase text-slate-400">
                Speed
              </div>
              <div className="text-xl font-bold capitalize text-slate-900">
                {resolvedModel.speed || "-"}
              </div>
            </div>
          </div>

          {resolvedModel.downloading && (
            <div className="space-y-2 rounded-[1.35rem] border border-sky-100 bg-sky-50/80 p-4 animate-in fade-in slide-in-from-top-2">
              <div className="mb-1 flex justify-between text-sm text-sky-800">
                <span className="font-medium">Downloading...</span>
                <span className="text-sky-700/80">
                  {Math.round(resolvedModel.downloadProgress || 0)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-sky-100">
                <div
                  className="h-full bg-sky-500 transition-all duration-300 ease-out"
                  style={{ width: `${resolvedModel.downloadProgress || 0}%` }}
                />
              </div>
              <div className="mt-2 flex justify-between text-xs text-sky-700/80">
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

          <div className="space-y-2 rounded-[1.4rem] border border-slate-200/70 bg-white/68 p-5">
            <h3 className="text-sm font-medium uppercase tracking-wider text-slate-500">
              Technical Details
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 text-sm">
              {runtimeRequired.length > 0 && (
                <div className="flex justify-between border-b border-slate-200/70 py-1">
                  <span className="text-slate-500">Runtime Required</span>
                  <span className="text-right font-medium text-slate-900">
                    {runtimeRequired.join(", ")}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.engine && (
                <div className="flex justify-between border-b border-slate-200/70 py-1">
                  <span className="text-slate-500">Runtime Engine</span>
                  <span className="text-right font-medium text-slate-900">
                    {resolvedModel.runtime.engine}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.model_type && (
                <div className="flex justify-between border-b border-slate-200/70 py-1">
                  <span className="text-slate-500">Model Type</span>
                  <span className="text-right font-medium text-slate-900">
                    {resolvedModel.runtime.model_type}
                  </span>
                </div>
              )}
              {resolvedModel.runtime?.patch_profile && (
                <div className="flex justify-between border-b border-slate-200/70 py-1">
                  <span className="text-slate-500">Patch Profile</span>
                  <span className="text-right font-medium text-slate-900">
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
                    {canManualImport && (
                      <Button variant="outline" size="sm" onClick={handleImportFiles} className="gap-2">
                        <Upload className="h-3.5 w-3.5" />
                        Import Files
                      </Button>
                    )}
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

        <div className="flex justify-between gap-2 border-t border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.55),rgba(244,246,250,0.72))] p-7 pt-5">
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
                className="gap-2 rounded-full"
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
                className="gap-2 rounded-full"
              >
                <Trash2 className="h-4 w-4" />
                Remove Model
              </Button>
            )}
          </div>
          <Button variant="outline" onClick={onClose} className="rounded-full border-slate-200 bg-white/82">
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}
