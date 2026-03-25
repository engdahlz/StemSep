import type { IpcMain } from "electron";

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

type RegisterModelIpcHandlersArgs = {
  ipcMain: IpcMain;
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
};

export function registerModelIpcHandlers({
  ipcMain,
  sendBackendCommandWithRetry,
}: RegisterModelIpcHandlersArgs) {
  ipcMain.handle("check-preset-models", async (_event, presetMappings: Record<string, string>) => {
    return sendBackendCommandWithRetry("check-preset-models", {
      preset_mappings: presetMappings,
    }, 10_000);
  });

  ipcMain.handle("get-workflows", async () => {
    return sendBackendCommandWithRetry("get_workflows", {}, 10_000);
  });

  ipcMain.handle("get-models", async () => {
    return sendBackendCommandWithRetry("get_models", {}, 120_000);
  });

  ipcMain.handle("get-model-tech", async (_event, modelId: string) => {
    return sendBackendCommandWithRetry("get_model_tech", { model_id: modelId }, 20_000);
  });
}
