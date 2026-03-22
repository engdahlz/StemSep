import type { IpcMain } from "electron";

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

type RegisterQualityIpcHandlersArgs = {
  ipcMain: IpcMain;
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
};

export function registerQualityIpcHandlers({
  ipcMain,
  sendBackendCommandWithRetry,
}: RegisterQualityIpcHandlersArgs) {
  ipcMain.handle("get-recipes", async () => {
    return sendBackendCommandWithRetry("get_recipes", {}, 10_000);
  });

  ipcMain.handle("quality-baseline-create", async (_event, payload: Record<string, any>) => {
    return sendBackendCommandWithRetry(
      "quality_baseline_create",
      payload || {},
      5 * 60 * 1000,
    );
  });

  ipcMain.handle("quality-compare", async (_event, payload: Record<string, any>) => {
    return sendBackendCommandWithRetry(
      "quality_compare",
      payload || {},
      5 * 60 * 1000,
    );
  });
}
