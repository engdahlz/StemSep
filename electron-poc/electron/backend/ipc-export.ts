import { randomUUID } from "crypto";
import type { IpcMain } from "electron";

export function registerExportIpcHandlers({
  ipcMain,
  emitExportProgress,
  exportFilesLocal,
  log,
}: {
  ipcMain: IpcMain;
  emitExportProgress: (payload: {
    requestId?: string;
    status: string;
    detail?: string;
    progress?: number;
    percent?: string;
    error?: string;
  }) => void;
  exportFilesLocal: (args: {
    sourceFiles: Record<string, string>;
    exportPath: string;
    format: string;
    bitrate: string;
    requestId?: string;
    onProgress?: (payload: any) => void;
    log?: (message: string, ...args: any[]) => void;
  }) => Promise<{ exported: Record<string, string> }>;
  log: (message: string, ...args: any[]) => void;
}) {
  ipcMain.handle(
    "export-files",
    async (
      _event,
      {
        sourceFiles,
        exportPath,
        format,
        bitrate,
        requestId,
      }: {
        sourceFiles: Record<string, string>;
        exportPath: string;
        format: string;
        bitrate: string;
        requestId?: string;
      },
    ) => {
      const resolvedRequestId = requestId || randomUUID().slice(0, 8);
      log("[export-files] local export requested", {
        requestId: resolvedRequestId,
        stems: Object.keys(sourceFiles || {}),
        exportPath,
        format,
        bitrate,
      });
      try {
        const res = await exportFilesLocal({
          sourceFiles,
          exportPath,
          format,
          bitrate,
          requestId: resolvedRequestId,
          onProgress: emitExportProgress,
          log,
        });
        log("[export-files] local export complete", {
          requestId: resolvedRequestId,
          exported: Object.keys(res?.exported || {}),
        });
        return {
          success: true,
          exported: res.exported,
          requestId: resolvedRequestId,
        };
      } catch (e: any) {
        emitExportProgress({
          requestId: resolvedRequestId,
          status: "failed",
          error: e?.message || String(e),
        });
        log("[export-files] local export FAILED", {
          requestId: resolvedRequestId,
          error: e?.message || String(e),
        });
        return {
          success: false,
          error: e?.message || String(e),
          code: e?.code || "MISSING_SOURCE_FILE",
          hint:
            e?.hint ||
            "Export source is unavailable. Run a new separation to refresh cache files.",
          requestId: resolvedRequestId,
        };
      }
    },
  );
}
