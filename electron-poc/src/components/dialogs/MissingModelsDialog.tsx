import { useMemo } from "react";
import { AlertTriangle, Download, X } from "lucide-react";

import { Progress } from "../ui/progress";

export type MissingModelReason = "not_installed";

export type MissingModel = {
  modelId: string;
  reason: MissingModelReason;
  details?: string;
};

export type MissingModelsDialogModel = {
  id: string;
  name?: string;
  installed?: boolean;

  // Download state (from store model shape)
  downloading?: boolean;
  downloadPaused?: boolean;
  downloadProgress?: number;
  downloadError?: string;

  file_size?: number;
};

export type MissingModelsDialogProps = {
  open: boolean;
  title?: string;
  description?: string;

  missing: MissingModel[];

  /**
   * Models available in the UI store. Used for display name, size + download state.
   */
  models: MissingModelsDialogModel[];

  /**
   * Called when the user dismisses the dialog (click backdrop or close button).
   */
  onClose: () => void;

  /**
   * Quick download action. Typically calls store + electronAPI download/resume.
   */
  onQuickDownload: (modelId: string) => void;

  /**
   * Navigate to model browser (optionally focusing a model).
   */
  onNavigateToModels?: (modelId?: string) => void;
};

function formatSizeMb(bytes: number | undefined): string | null {
  if (!bytes || !Number.isFinite(bytes) || bytes <= 0) return null;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

export function MissingModelsDialog(props: MissingModelsDialogProps) {
  const {
    open,
    title = "Missing Models",
    description = "This run requires models that are not installed:",
    missing,
    models,
    onClose,
    onQuickDownload,
    onNavigateToModels,
  } = props;

  const missingIds = useMemo(() => missing.map((m) => m.modelId), [missing]);

  const firstMissingId = missingIds[0];

  if (!open || missingIds.length === 0) return null;

  return (
    <div className="fixed inset-0 z-[80]">
      <button
        aria-label="Close"
        className="absolute inset-0 bg-[rgba(122,130,162,0.26)] backdrop-blur-md"
        onClick={onClose}
      />

      <div className="absolute left-1/2 top-1/2 w-[min(640px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2">
        <div className="overflow-hidden rounded-[1.75rem] border border-white/70 bg-[rgba(252,249,255,0.88)] p-5 text-slate-800 shadow-[0_40px_120px_rgba(141,150,179,0.24)] backdrop-blur-2xl">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[1rem] border border-amber-300/55 bg-amber-50/88">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <span className="rounded-full border border-white/60 bg-white/64 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                      Setup Required
                    </span>
                    <span className="rounded-full border border-amber-300/55 bg-amber-50/88 px-3 py-1 text-[11px] text-amber-700">
                      {missingIds.length} missing
                    </span>
                  </div>
                  <h4 className="text-[20px] font-normal tracking-[-0.5px] text-slate-800">
                    {title}
                  </h4>
                  <p className="mt-1 text-sm text-slate-500">
                    {description}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close missing models"
                  className="inline-flex h-10 w-10 items-center justify-center rounded-[14px] border border-white/60 bg-white/62 text-slate-500 transition-all hover:bg-white/82 hover:text-slate-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="mt-5 space-y-3">
                {missing.map((m) => {
                  const model = models.find((x) => x.id === m.modelId);

                  const isDownloading = !!model?.downloading;
                  const isPaused = !!model?.downloadPaused;
                  const progress =
                    typeof model?.downloadProgress === "number"
                      ? model.downloadProgress
                      : 0;

                  const sizeText = formatSizeMb(model?.file_size);

                  return (
                    <div
                      key={m.modelId}
                      className="rounded-[1.2rem] border border-white/60 bg-white/62 p-4"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex min-w-0 items-center gap-2">
                            <span className="truncate text-[14px] font-medium text-slate-800">
                              {model?.name || m.modelId}
                            </span>
                            {sizeText && (
                              <span className="shrink-0 rounded-full border border-white/60 bg-white/70 px-2 py-0.5 text-[11px] text-slate-500">
                                {sizeText}
                              </span>
                            )}
                          </div>
                          <div className="mt-1 text-[12px] text-slate-500">
                            {m.details || "Required by the selected separation setup."}
                          </div>
                        </div>

                        <button
                          type="button"
                          disabled={isDownloading || model?.installed}
                          onClick={() => onQuickDownload(m.modelId)}
                          className={`inline-flex shrink-0 items-center gap-2 rounded-[14px] border px-4 py-2.5 text-[13px] transition-all ${
                            model?.installed
                              ? "border-emerald-300/55 bg-emerald-50/88 text-emerald-700"
                              : model?.downloadError
                                ? "border-rose-300/55 bg-rose-50/88 text-rose-700 hover:bg-rose-100/80"
                                : "border-white/60 bg-white/68 text-slate-700 hover:bg-white/84 hover:text-slate-900"
                          } disabled:cursor-not-allowed disabled:opacity-65`}
                        >
                          <Download className="h-4 w-4" />
                          {model?.installed
                            ? "Installed"
                            : isPaused
                              ? "Resume"
                              : model?.downloadError
                                ? "Retry"
                                : "Quick download"}
                        </button>
                      </div>

                      {(isDownloading || isPaused || progress > 0 || model?.downloadError) && (
                        <div className="mt-4 space-y-2">
                          <div className="flex items-center justify-between text-[12px] text-slate-500">
                            <span>
                              {model?.downloadError
                                ? "Download failed"
                                : isPaused
                                  ? "Download paused"
                                  : "Downloading"}
                            </span>
                            <span className="tabular-nums">
                              {Math.round(progress)}%
                            </span>
                          </div>
                          <Progress
                            value={progress}
                            className="h-1.5 bg-slate-900/8"
                          />
                          {model?.downloadError && (
                            <div className="text-[12px] text-rose-700/88">
                              {model.downloadError}
                            </div>
                          )}
                        </div>
                      )}

                      {!isDownloading && !model?.downloadError && !model?.installed && (
                        <div className="mt-3 text-[12px] text-slate-500">
                          Install this model to unlock the selected workflow.
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {onNavigateToModels && (
                <div className="mt-5 flex gap-3 border-t border-white/60 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      onClose();
                      onNavigateToModels(firstMissingId);
                    }}
                    className="inline-flex items-center gap-2 rounded-[14px] border border-white/60 bg-white/68 px-4 py-2.5 text-[13px] text-slate-700 transition-all hover:bg-white/84 hover:text-slate-900"
                  >
                    <Download className="h-4 w-4" />
                    Go to Model Browser
                  </button>
                  <button
                    type="button"
                    onClick={onClose}
                    className="inline-flex items-center rounded-[14px] border border-white/60 bg-transparent px-4 py-2.5 text-[13px] text-slate-500 transition-all hover:bg-white/68 hover:text-slate-800"
                  >
                    Close
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MissingModelsDialog;
