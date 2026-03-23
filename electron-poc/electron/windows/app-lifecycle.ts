import {
  app,
  desktopCapturer,
  dialog,
  protocol,
  session,
  type BrowserWindow,
} from "electron";
import type { LogFn } from "../system/logger";

export function registerGlobalErrorHandlers({ log }: { log: LogFn }) {
  process.on("uncaughtException", (error) => {
    log("CRITICAL: Uncaught Exception:", error);
    dialog.showErrorBox(
      "Application Error",
      `An unexpected error occurred: ${error.message}\n\nPlease check the logs for more details.`,
    );
  });

  process.on("unhandledRejection", (reason) => {
    log("CRITICAL: Unhandled Rejection:", reason);
  });
}

export function enforceSingleInstance({
  getMainWindow,
  createWindow,
}: {
  getMainWindow: () => BrowserWindow | null;
  createWindow: () => BrowserWindow | null | undefined;
}) {
  const gotSingleInstanceLock = app.requestSingleInstanceLock();
  if (!gotSingleInstanceLock) {
    app.quit();
    process.exit(0);
  }

  app.on("second-instance", () => {
    app
      .whenReady()
      .then(() => {
        try {
          const mainWindow = getMainWindow();
          if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.show();
            mainWindow.focus();
          } else {
            createWindow();
          }
        } catch {
          // ignore
        }
      })
      .catch(() => {
        // ignore
      });
  });
}

export function registerAppLifecycleHandlers({
  log,
  getMainWindow,
  createWindow,
  onBeforeQuit,
}: {
  log: LogFn;
  getMainWindow: () => BrowserWindow | null;
  createWindow: () => BrowserWindow | null | undefined;
  onBeforeQuit: () => void;
}) {
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
      log("All windows closed, quitting app.");
      app.quit();
    }
  });

  app.on("before-quit", () => {
    onBeforeQuit();
  });

  app.on("activate", () => {
    if (getMainWindow() === null) {
      createWindow();
    }
  });
}

export async function initializeAppWhenReady({
  cleanupPreviewCache,
  log,
  createWindow,
  maybeRunQobuzCaptureSmokeTest,
  installApplicationMenu,
  startHealthChecks,
  maybeRunSmokeSeparation,
}: {
  cleanupPreviewCache: () => void;
  log: LogFn;
  createWindow: () => BrowserWindow | null | undefined;
  maybeRunQobuzCaptureSmokeTest: () => Promise<void>;
  installApplicationMenu: () => void;
  startHealthChecks: () => void;
  maybeRunSmokeSeparation: () => Promise<void>;
}) {
  cleanupPreviewCache();

  session.defaultSession.setDisplayMediaRequestHandler(
    async (_request, callback) => {
      try {
        const sources = await desktopCapturer.getSources({
          types: ["screen"],
          thumbnailSize: { width: 1, height: 1 },
          fetchWindowIcons: false,
        });
        callback({
          video: sources[0],
          audio: "loopback",
        });
      } catch (error: any) {
        log("[capture] failed to set display media handler", {
          error: error?.message || String(error),
        });
        callback({ video: undefined, audio: undefined });
      }
    },
    { useSystemPicker: false },
  );

  protocol.registerFileProtocol("media", (request, callback) => {
    let filePath = request.url.replace("media://", "");
    if (
      filePath.startsWith("/") &&
      filePath.length > 2 &&
      filePath[2] === ":"
    ) {
      filePath = filePath.slice(1);
    }
    try {
      const decodedPath = decodeURIComponent(filePath);
      console.log("[media protocol] Serving file:", decodedPath);
      callback(decodedPath);
    } catch (error) {
      console.error("Failed to serve media file:", error);
    }
  });

  log("App is ready, creating window.");
  createWindow();

  if (process.env.STEMSEP_QOBUZ_CAPTURE_SMOKETEST_QUERY) {
    setTimeout(() => {
      void maybeRunQobuzCaptureSmokeTest();
    }, 2500);
  }

  installApplicationMenu();
  startHealthChecks();
  void maybeRunSmokeSeparation();
}
