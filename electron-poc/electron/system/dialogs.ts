import { app, dialog, shell, type BrowserWindow, type IpcMain } from "electron";
import fs from "fs";
import path from "path";

type LogFn = (message: string, ...args: any[]) => void;
type GetMainWindow = () => BrowserWindow | null;
type ResolvePlaybackFilePath = (filePath: string) => string | null;
type ClassifyMissingAudioPath = (filePath: string) => { code: string; hint: string };
type ResolvePlaybackStems = (
  outputFiles?: Record<string, string>,
  playback?: any,
) => { stems: Record<string, string>; issues: Record<string, any> };

type RegisterDialogIpcHandlersArgs = {
  ipcMain: IpcMain;
  getMainWindow: GetMainWindow;
  log: LogFn;
  resolvePlaybackFilePath: ResolvePlaybackFilePath;
  classifyMissingAudioPath: ClassifyMissingAudioPath;
  resolvePlaybackStems: ResolvePlaybackStems;
};

export function registerDialogIpcHandlers({
  ipcMain,
  getMainWindow,
  log,
  resolvePlaybackFilePath,
  classifyMissingAudioPath,
  resolvePlaybackStems,
}: RegisterDialogIpcHandlersArgs) {
  ipcMain.handle("open-audio-file-dialog", async () => {
    const win = getMainWindow();
    if (!win) return null;
    try {
      const result = await dialog.showOpenDialog(win, {
        properties: ["openFile", "multiSelections"],
        filters: [
          {
            name: "Audio Files",
            extensions: ["mp3", "wav", "flac", "m4a", "ogg", "aac", "wma", "aiff"],
          },
        ],
      });
      return result.filePaths || [];
    } catch (error) {
      log("Error opening file dialog:", error);
      throw new Error("Failed to open file dialog");
    }
  });

  ipcMain.handle("open-model-file-dialog", async () => {
    const win = getMainWindow();
    if (!win) return null;
    try {
      const result = await dialog.showOpenDialog(win, {
        properties: ["openFile", "multiSelections"],
        filters: [
          {
            name: "Model Files",
            extensions: ["ckpt", "pth", "pt", "onnx", "safetensors"],
          },
        ],
      });
      return result.filePaths || [];
    } catch (error) {
      log("Error opening model file dialog:", error);
      throw new Error("Failed to open model file dialog");
    }
  });

  ipcMain.handle("select-output-directory", async () => {
    const win = getMainWindow();
    if (!win) return null;
    try {
      const result = await dialog.showOpenDialog(win, {
        properties: ["openDirectory"],
      });
      return result.filePaths[0] || null;
    } catch (error) {
      log("Error selecting output directory:", error);
      throw new Error("Failed to open directory dialog");
    }
  });

  ipcMain.handle("select-models-directory", async () => {
    const win = getMainWindow();
    if (!win) return null;
    try {
      const result = await dialog.showOpenDialog(win, {
        properties: ["openDirectory", "createDirectory"],
        title: "Choose Models Directory",
      });
      return result.filePaths[0] || null;
    } catch (error) {
      log("Error selecting models directory:", error);
      throw new Error("Failed to open models directory dialog");
    }
  });

  ipcMain.handle("scan-directory", async (_event, folderPath: string) => {
    const audioExtensions = new Set([".mp3", ".wav", ".flac", ".ogg", ".m4a"]);
    const audioFiles: string[] = [];

    async function scan(dir: string) {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          await scan(fullPath);
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          if (audioExtensions.has(ext)) {
            audioFiles.push(fullPath);
          }
        }
      }
    }

    try {
      await scan(folderPath);
      return audioFiles;
    } catch (error) {
      console.error("Error scanning directory:", error);
      log("Error scanning directory:", error);
      throw new Error(
        `Failed to scan directory: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  });

  ipcMain.handle("open-folder", async (_event, folderPath: string) => {
    await shell.openPath(folderPath);
  });

  ipcMain.handle("check-file-exists", async (_event, filePath: string) => {
    try {
      if (!filePath || typeof filePath !== "string") return false;
      const resolvedPath = resolvePlaybackFilePath(filePath) || filePath;
      return fs.existsSync(resolvedPath);
    } catch {
      return false;
    }
  });

  ipcMain.handle("read-audio-file", async (_event, filePath: string) => {
    try {
      const ext = filePath.split(".").pop()?.toLowerCase();
      const allowedExtensions = new Set([
        "mp3",
        "wav",
        "flac",
        "ogg",
        "m4a",
        "aac",
        "wma",
        "aiff",
      ]);
      if (!ext || !allowedExtensions.has(ext)) {
        throw new Error(
          `Security violation: Attempted to read non-audio file extension '.${ext || ""}'`,
        );
      }

      const resolvedPath = resolvePlaybackFilePath(filePath) || filePath;

      if (!fs.existsSync(resolvedPath)) {
        const missing = classifyMissingAudioPath(filePath);
        return {
          success: false,
          error: "Audio file not found",
          code: missing.code,
          hint: missing.hint,
        };
      }

      const data = fs.readFileSync(resolvedPath);
      const base64 = data.toString("base64");
      const mimeTypes: Record<string, string> = {
        mp3: "audio/mpeg",
        wav: "audio/wav",
        flac: "audio/flac",
        ogg: "audio/ogg",
        m4a: "audio/mp4",
        aac: "audio/aac",
        wma: "audio/x-ms-wma",
        aiff: "audio/aiff",
      };
      const mimeType = mimeTypes[ext] || "audio/mpeg";
      return { success: true, data: base64, mimeType, resolvedPath };
    } catch (error: any) {
      log(`Failed to read audio file: ${filePath}`, error);
      if (error?.code === "ENOENT") {
        const missing = classifyMissingAudioPath(filePath);
        return {
          success: false,
          error: "Audio file not found",
          code: missing.code,
          hint: missing.hint,
        };
      }
      return {
        success: false,
        error: error?.message || String(error),
        code: "MISSING_SOURCE_FILE",
        hint: "Unable to read the audio file. Verify the file exists and retry.",
      };
    }
  });

  ipcMain.handle("resolve-playback-stems", async (_event, { outputFiles, playback }) => {
    try {
      const resolved = resolvePlaybackStems(outputFiles, playback);
      return {
        success: true,
        stems: resolved.stems,
        issues: resolved.issues,
      };
    } catch (error: any) {
      return {
        success: false,
        stems: outputFiles || {},
        issues: {},
        error: error?.message || String(error),
      };
    }
  });

  ipcMain.handle("open-external-url", async (_event, url: string) => {
    try {
      if (typeof url !== "string" || !url.trim()) return false;
      const u = new URL(url);
      if (u.protocol !== "https:" && u.protocol !== "http:") return false;
      await shell.openExternal(url);
      return true;
    } catch (e) {
      log("Failed to open external url:", e);
      return false;
    }
  });
}
