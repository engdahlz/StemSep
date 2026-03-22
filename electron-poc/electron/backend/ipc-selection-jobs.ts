import { randomUUID } from "crypto";
import type { IpcMain } from "electron";

type SendBackendCommand = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
) => Promise<any>;

type ExportFilesLocal = (args: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
  requestId: string;
}) => Promise<{ exported: Record<string, string> }>;

type ModelJobState = {
  requestedModelId: string;
  effectiveModelId: string;
  inputFile: string;
  finalOutputDir: string;
  previewDir: string;
  sourceAudioProfile?: any;
  stagingDecision?: any;
  outputFiles?: Record<string, string>;
};

export function registerSelectionJobIpcHandlers({
  ipcMain,
  sendBackendCommand,
  modelInfoByJobId,
  lastProgressByJobId,
  emitExportProgress,
  exportFilesLocal,
}: {
  ipcMain: IpcMain;
  sendBackendCommand: SendBackendCommand;
  modelInfoByJobId: Map<string, ModelJobState>;
  lastProgressByJobId: Map<string, number>;
  emitExportProgress: (payload: any) => void;
  exportFilesLocal: ExportFilesLocal;
}) {
  ipcMain.handle("cancel-separation", async (_event, jobId: string) => {
    return sendBackendCommand("cancel_selection_job", { job_id: jobId }, 15000);
  });

  ipcMain.handle("save-job-output", async (_event, jobId: string) => {
    const jobState = modelInfoByJobId.get(jobId);
    if (!jobState?.finalOutputDir) {
      return {
        success: false,
        error: "No final output directory is configured for this separation.",
      };
    }

    const requestId = `save-${jobId}`;
    try {
      const res = await sendBackendCommand(
        "export_selection_job",
        { job_id: jobId, export_path: jobState.finalOutputDir },
        120000,
      );
      const exported = Object.fromEntries(
        Object.entries((res?.output_files as Record<string, string>) || {}).map(
          ([stem, outputPath]) => [stem, String(outputPath)],
        ),
      );
      if (!res?.success || Object.keys(exported).length === 0) {
        throw new Error(res?.error || "Failed to export separation outputs.");
      }
      jobState.outputFiles = exported;
      return {
        success: true,
        outputFiles: exported,
        sourceAudioProfile: jobState.sourceAudioProfile,
        stagingDecision: jobState.stagingDecision,
      };
    } catch (error: any) {
      emitExportProgress({
        requestId,
        status: "failed",
        error: error?.message || String(error),
        totalProgress: 0,
      });
      return {
        success: false,
        error: error?.message || String(error),
      };
    }
  });

  ipcMain.handle("export-output", async (_event, payload) => {
    const { jobId, exportPath, format, bitrate, requestId } = payload as {
      jobId: string;
      exportPath: string;
      format: string;
      bitrate: string;
      requestId?: string;
    };
    const jobState = modelInfoByJobId.get(jobId);
    const sourceFiles = jobState?.outputFiles;
    if (!sourceFiles || Object.keys(sourceFiles).length === 0) {
      return sendBackendCommand(
        "export_selection_job",
        {
          job_id: jobId,
          export_path: exportPath,
          format,
          bitrate,
        },
        120000,
      );
    }

    const resolvedRequestId = requestId || randomUUID().slice(0, 8);
    try {
      const res = await exportFilesLocal({
        sourceFiles,
        exportPath,
        format,
        bitrate,
        requestId: resolvedRequestId,
      });
      return {
        success: true,
        exported: res.exported,
        requestId: resolvedRequestId,
        sourceAudioProfile: jobState.sourceAudioProfile,
        stagingDecision: jobState.stagingDecision,
      };
    } catch (error: any) {
      emitExportProgress({
        requestId: resolvedRequestId,
        status: "failed",
        error: error?.message || String(error),
      });
      return {
        success: false,
        error: error?.message || String(error),
        requestId: resolvedRequestId,
      };
    }
  });

  ipcMain.handle("discard-job-output", async (_event, jobId: string) => {
    try {
      const result = await sendBackendCommand(
        "discard_selection_job",
        { job_id: jobId },
        30000,
      );
      if (result?.success !== false) {
        modelInfoByJobId.delete(jobId);
        lastProgressByJobId.delete(jobId);
      }
      return result;
    } catch (error) {
      modelInfoByJobId.delete(jobId);
      lastProgressByJobId.delete(jobId);
      throw error;
    }
  });
}
