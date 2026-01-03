import { useMemo } from "react";
import { AlertTriangle, Download, X } from "lucide-react";

import { Button } from "../ui/button";
import { Card } from "../ui/card";
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
        className="absolute inset-0 bg-background/60 backdrop-blur-[2px]"
        onClick={onClose}
      />

      <div className="absolute left-1/2 top-1/2 w-[min(640px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2">
        <Card className="border-destructive/50 bg-background p-4 shadow-xl">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="font-medium text-destructive">{title}</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {description}
                  </p>
                </div>

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  aria-label="Close missing models"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <ul className="text-sm mt-3 space-y-2">
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
                    <li key={m.modelId} className="space-y-1">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-destructive">•</span>
                          <span className="truncate">
                            {model?.name || m.modelId}
                          </span>

                          {sizeText && (
                            <span className="text-muted-foreground text-xs shrink-0">
                              ({sizeText})
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {(isDownloading || isPaused || progress > 0) && (
                            <div className="flex items-center gap-2 w-[180px]">
                              <Progress value={progress} className="h-1.5" />
                              <span className="text-xs tabular-nums text-muted-foreground w-10 text-right">
                                {Math.round(progress)}%
                              </span>
                            </div>
                          )}

                          <Button
                            size="sm"
                            variant={
                              model?.downloadError ? "destructive" : "outline"
                            }
                            disabled={
                              isDownloading || model?.installed
                            }
                            onClick={() => onQuickDownload(m.modelId)}
                          >
                            <Download className="w-4 h-4 mr-2" />
                            {model?.installed
                              ? "Installed"
                              : isPaused
                                ? "Resume"
                                : model?.downloadError
                                  ? "Retry"
                                  : "Quick download"}
                          </Button>
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>

              {onNavigateToModels && (
                <div className="flex gap-2 mt-4">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      onClose();
                      onNavigateToModels(firstMissingId);
                    }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Go to Model Browser
                  </Button>
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default MissingModelsDialog;
