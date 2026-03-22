import { app, type IpcMain } from "electron";
import fs from "fs";
import path from "path";

type LogFn = (message: string, ...args: any[]) => void;
type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;
type WriteAppConfig = (partial: Record<string, any>) => boolean;
type GetStoredHuggingFaceToken = () => string | null;
type SetStoredHuggingFaceToken = (token: string | null) => { success: boolean; error?: string };
type RequestBridgeRestart = (reason: string) => void;
type RestartBackendAndWait = (reason: string, timeoutMs?: number) => Promise<void>;

type RegisterConfigIpcHandlersArgs = {
  ipcMain: IpcMain;
  log: LogFn;
  writeAppConfig: WriteAppConfig;
  getStoredHuggingFaceToken: GetStoredHuggingFaceToken;
  setStoredHuggingFaceToken: SetStoredHuggingFaceToken;
  requestBridgeRestart: RequestBridgeRestart;
  restartBackendAndWait: RestartBackendAndWait;
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
};

export function registerConfigIpcHandlers({
  ipcMain,
  log,
  writeAppConfig,
  getStoredHuggingFaceToken,
  setStoredHuggingFaceToken,
  requestBridgeRestart,
  restartBackendAndWait,
  sendBackendCommandWithRetry,
}: RegisterConfigIpcHandlersArgs) {
  const getQueuePath = () => path.join(app.getPath("userData"), "queue_state.json");

  ipcMain.handle("set-models-dir", async (_event, modelsDir: string) => {
    const normalized = typeof modelsDir === "string" ? modelsDir.trim() : "";
    if (!normalized) {
      throw new Error("Models directory is required.");
    }

    fs.mkdirSync(normalized, { recursive: true });
    const saved = writeAppConfig({ modelsDir: normalized });
    if (!saved) {
      throw new Error("Failed to persist models directory.");
    }

    await restartBackendAndWait("updated models directory");
    const models = await sendBackendCommandWithRetry("get_models", {}, 120000);

    return {
      success: true,
      modelsDir: normalized,
      models,
    };
  });

  ipcMain.handle("save-app-config", async (_event, config: Record<string, any>) => {
    try {
      const configPath = path.join(app.getPath("userData"), "app-config.json");
      let existingConfig: Record<string, any> = {};

      if (fs.existsSync(configPath)) {
        existingConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      }

      const { hfToken: _ignoredHfToken, ...safeConfig } = config || {};
      const newConfig = { ...existingConfig, ...safeConfig };
      fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
      log("Saved app config:", Object.keys(safeConfig).join(", "));
      return true;
    } catch (error) {
      log("Failed to save app config:", error);
      return false;
    }
  });

  ipcMain.handle("get-app-config", async () => {
    try {
      const configPath = path.join(app.getPath("userData"), "app-config.json");
      if (fs.existsSync(configPath)) {
        const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
        if (config && typeof config === "object") {
          delete config.hfToken;
        }
        return config;
      }
    } catch (error) {
      log("Failed to read app config:", error);
    }
    return {};
  });

  ipcMain.handle("get-huggingface-auth-status", async () => {
    return { configured: !!getStoredHuggingFaceToken() };
  });

  ipcMain.handle("set-huggingface-token", async (_event, token: string) => {
    const res = setStoredHuggingFaceToken(token);
    if (res.success) {
      requestBridgeRestart("updated huggingface token");
    }
    return res;
  });

  ipcMain.handle("clear-huggingface-token", async () => {
    const res = setStoredHuggingFaceToken(null);
    if (res.success) {
      requestBridgeRestart("cleared huggingface token");
    }
    return res;
  });

  ipcMain.handle("save-queue", async (_event, queueData: any) => {
    try {
      await fs.promises.writeFile(
        getQueuePath(),
        JSON.stringify(queueData),
        "utf-8",
      );
      return { success: true };
    } catch (error) {
      console.error("Failed to save queue:", error);
      return { success: false, error: String(error) };
    }
  });

  ipcMain.handle("load-queue", async () => {
    try {
      const queuePath = getQueuePath();
      if (!fs.existsSync(queuePath)) {
        return null;
      }
      const data = await fs.promises.readFile(queuePath, "utf-8");
      return JSON.parse(data);
    } catch (error) {
      console.error("Failed to load queue:", error);
      return null;
    }
  });
}
