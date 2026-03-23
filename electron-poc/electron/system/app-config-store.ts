import { app } from "electron";
import fs from "fs";
import path from "path";

type LogFn = (message: string, ...args: any[]) => void;

export function createAppConfigStore({ log }: { log: LogFn }) {
  const getConfigPath = () => path.join(app.getPath("userData"), "app-config.json");

  function readAppConfig(): Record<string, any> {
    try {
      const configPath = getConfigPath();
      if (fs.existsSync(configPath)) {
        return JSON.parse(fs.readFileSync(configPath, "utf-8"));
      }
    } catch (error) {
      log("Could not read app-config.json:", error);
    }
    return {};
  }

  function writeAppConfig(partial: Record<string, any>): boolean {
    try {
      const configPath = getConfigPath();
      const existingConfig = readAppConfig();
      const newConfig = { ...existingConfig, ...partial };
      fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
      return true;
    } catch (error) {
      log("Could not write app-config.json:", error);
      return false;
    }
  }

  function getStoredModelsDir(): string | null {
    try {
      const config = readAppConfig();
      return config.modelsDir || null;
    } catch (error) {
      log("Could not read modelsDir from config:", error);
      return null;
    }
  }

  function getStoredCaptureOutputDeviceId(): string | null {
    const value = readAppConfig().captureOutputDeviceId;
    return typeof value === "string" && value.trim() ? value.trim() : null;
  }

  function getStoredHuggingFaceToken(): string | null {
    try {
      const config = readAppConfig();
      const token = config.hfToken;
      if (typeof token === "string" && token.trim()) return token.trim();
    } catch (error) {
      log("Could not read hfToken from config:", error);
    }
    return null;
  }

  function setStoredHuggingFaceToken(
    token: string | null,
  ): { success: boolean; error?: string } {
    try {
      const trimmed = typeof token === "string" ? token.trim() : "";
      const configPath = getConfigPath();
      const existingConfig = readAppConfig();

      if (!trimmed) {
        if (Object.prototype.hasOwnProperty.call(existingConfig, "hfToken")) {
          delete existingConfig.hfToken;
          fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
        }
        return { success: true };
      }

      if (trimmed.length < 20) {
        return { success: false, error: "Token looks too short." };
      }

      existingConfig.hfToken = trimmed;
      fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
      return { success: true };
    } catch (error: any) {
      log("Failed to set hfToken:", error);
      return { success: false, error: error?.message || "Failed to save token." };
    }
  }

  return {
    readAppConfig,
    writeAppConfig,
    getStoredModelsDir,
    getStoredCaptureOutputDeviceId,
    getStoredHuggingFaceToken,
    setStoredHuggingFaceToken,
  };
}
