import type { IpcMain } from "electron";

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

type RemoveModelLocal = (modelId: string) => {
  success: true;
  removedFiles: string[];
};

type RegisterDownloadIpcHandlersArgs = {
  ipcMain: IpcMain;
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
  removeModelLocal: RemoveModelLocal;
  log: (message: string, ...args: any[]) => void;
};

export function registerDownloadIpcHandlers({
  ipcMain,
  sendBackendCommandWithRetry,
  removeModelLocal,
  log,
}: RegisterDownloadIpcHandlersArgs) {
  ipcMain.handle(
    "import-model-files",
    async (
      _event,
      {
        modelId,
        files,
        allowCopy,
      }: { modelId: string; files: Array<{ kind?: string; path: string }>; allowCopy?: boolean },
    ) => {
      return sendBackendCommandWithRetry(
        "import_model_files",
        { model_id: modelId, files, allow_copy: allowCopy !== false },
        120_000,
      );
    },
  );

  ipcMain.handle(
    "import-custom-model",
    async (
      _event,
      {
        filePath,
        modelName,
        architecture,
      }: { filePath: string; modelName: string; architecture?: string },
    ) => {
      return sendBackendCommandWithRetry("import_custom_model", {
        file_path: filePath,
        model_name: modelName,
        architecture: architecture || "Custom",
      }, 60_000);
    },
  );

  ipcMain.handle("remove-model", async (_event, modelId: string) => {
    try {
      const res = await sendBackendCommandWithRetry("remove_model", { model_id: modelId }, 60_000);
      log("remove-model completed", { modelId, res });
      return res;
    } catch (error: any) {
      const msg = (error?.message || String(error) || "").toLowerCase();
      log("remove-model failed", { modelId, error: error?.message || String(error) });

      if (
        msg.includes("backend_unavailable") ||
        msg.includes("python proxy unavailable") ||
        msg.includes("python-bridge.py") ||
        msg.includes("backend not available")
      ) {
        try {
          const local = removeModelLocal(modelId);
          log("remove-model local fallback succeeded", {
            modelId,
            removed: local.removedFiles.length,
          });
          return local;
        } catch (localErr: any) {
          log("remove-model local fallback FAILED", {
            modelId,
            error: localErr?.message || String(localErr),
          });
          throw new Error(
            `BACKEND_UNAVAILABLE: ${error?.message || String(error)}\n\nLocal delete failed: ${localErr?.message || String(localErr)}`,
          );
        }
      }

      throw error;
    }
  });
}
