import chokidar, { type FSWatcher } from "chokidar";
import path from "path";
import type { BrowserWindow, IpcMain } from "electron";

type LogFn = (message: string, ...args: any[]) => void;
type GetMainWindow = () => BrowserWindow | null;

type RegisterWatchIpcHandlersArgs = {
  ipcMain: IpcMain;
  getMainWindow: GetMainWindow;
  log: LogFn;
};

export function registerWatchIpcHandlers({
  ipcMain,
  getMainWindow,
  log,
}: RegisterWatchIpcHandlersArgs) {
  let watcher: FSWatcher | null = null;

  ipcMain.handle("start-watch-mode", async (_event, folderPath: string) => {
    if (watcher) {
      await watcher.close();
    }

    log("Starting watch mode on:", folderPath);

    watcher = chokidar.watch(folderPath, {
      ignored: /(^|[\/\\])\../,
      persistent: true,
      ignoreInitial: true,
      awaitWriteFinish: {
        stabilityThreshold: 2000,
        pollInterval: 100,
      },
    });

    watcher.on("add", (filePath) => {
      const ext = path.extname(filePath).toLowerCase();
      if ([".wav", ".mp3", ".flac", ".m4a", ".ogg"].includes(ext)) {
        log("New file detected:", filePath);
        getMainWindow()?.webContents.send("watch-file-detected", filePath);
      }
    });

    return true;
  });

  ipcMain.handle("stop-watch-mode", async () => {
    if (watcher) {
      await watcher.close();
      watcher = null;
      log("Watch mode stopped");
    }
    return true;
  });
}
