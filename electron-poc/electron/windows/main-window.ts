import { app, BrowserWindow, dialog, screen } from "electron";
import path from "path";

type LogFn = (message: string, ...args: any[]) => void;

function ensureWindowOnScreen(win: BrowserWindow) {
  try {
    const bounds = win.getBounds();
    const displays = screen.getAllDisplays();

    const intersectsSomeDisplay = displays.some((d) => {
      const wa = d.workArea;
      const xOverlap =
        Math.min(bounds.x + bounds.width, wa.x + wa.width) -
        Math.max(bounds.x, wa.x);
      const yOverlap =
        Math.min(bounds.y + bounds.height, wa.y + wa.height) -
        Math.max(bounds.y, wa.y);
      return xOverlap > 20 && yOverlap > 20;
    });

    if (intersectsSomeDisplay) return;

    const wa = screen.getPrimaryDisplay().workArea;
    const width = Math.min(bounds.width, wa.width);
    const height = Math.min(bounds.height, wa.height);
    const x = wa.x + Math.round((wa.width - width) / 2);
    const y = wa.y + Math.round((wa.height - height) / 2);
    win.setBounds({ x, y, width, height });
  } catch {
    // ignore
  }
}

export function createMainWindow({
  log,
  getMainWindow,
  setMainWindow,
  getIsCreatingMainWindow,
  setIsCreatingMainWindow,
}: {
  log: LogFn;
  getMainWindow: () => BrowserWindow | null;
  setMainWindow: (win: BrowserWindow | null) => void;
  getIsCreatingMainWindow: () => boolean;
  setIsCreatingMainWindow: (value: boolean) => void;
}) {
  const existingWindow = getMainWindow();
  const mainWindow = existingWindow;
  function assignWindow(win: BrowserWindow | null) {
    setMainWindow(win);
  }

  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
      return mainWindow;
    }
  } catch {
    // ignore
  }

  if (getIsCreatingMainWindow()) return mainWindow;
  setIsCreatingMainWindow(true);

  const createdWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });
  assignWindow(createdWindow);

  try {
    ensureWindowOnScreen(createdWindow);
    createdWindow.center();
    if (createdWindow.isMinimized()) createdWindow.restore();
    createdWindow.show();
    createdWindow.focus();
  } catch {
    // ignore
  }

  createdWindow.once("ready-to-show", () => {
    try {
      ensureWindowOnScreen(createdWindow);
      if (createdWindow.isMinimized()) createdWindow.restore();
      createdWindow.show();
      createdWindow.focus();
    } catch {
      // ignore
    }
  });

  const isDev = !app.isPackaged;

  if (isDev) {
    try {
      createdWindow.setAlwaysOnTop(true);
      setTimeout(() => {
        try {
          createdWindow.setAlwaysOnTop(false);
        } catch {
          // ignore
        }
      }, 1500);
    } catch {
      // ignore
    }

    const loadURLWithRetry = (url: string, retries = 10) => {
      createdWindow
        .loadURL(url)
        .then(() => {
          try {
            ensureWindowOnScreen(createdWindow);
            if (createdWindow.isMinimized()) createdWindow.restore();
            createdWindow.show();
            createdWindow.focus();
          } catch {
            // ignore
          }
        })
        .catch((e) => {
          if (retries > 0) {
            log(`Failed to load URL, retrying... (${retries} attempts left)`);
            setTimeout(() => loadURLWithRetry(url, retries - 1), 1000);
          } else {
            log("Failed to load URL after multiple attempts:", e);
          }
        });
    };

    const envDevUrl = (process.env.VITE_DEV_SERVER_URL || "").trim();
    const rawDevUrl = envDevUrl || "http://127.0.0.1:5173/";
    const normalizedDevUrl = rawDevUrl
      .replace(/^http:\/\/localhost(?=[:/]|$)/i, "http://127.0.0.1")
      .replace(/^https:\/\/localhost(?=[:/]|$)/i, "https://127.0.0.1");

    log("Dev renderer URL", {
      envDevUrl: envDevUrl || null,
      normalizedDevUrl,
    });

    loadURLWithRetry(normalizedDevUrl);
    createdWindow.webContents.openDevTools();
  } else {
    createdWindow
      .loadFile(path.join(__dirname, "../dist-renderer/index.html"))
      .catch((e) => {
        log("CRITICAL: Failed to load index.html:", e);
        dialog.showErrorBox(
          "Startup Error",
          "Failed to load application resources. Please reinstall the application.",
        );
      });
  }

  createdWindow.on("unresponsive", () => {
    log("WARNING: Main window became unresponsive");
    dialog
      .showMessageBox(createdWindow, {
        type: "warning",
        title: "App Unresponsive",
        message:
          "The application is not responding. You can wait or restart it.",
        buttons: ["Wait", "Restart"],
        defaultId: 0,
        cancelId: 0,
      })
      .then(({ response }) => {
        if (response === 1) {
          createdWindow.reload();
        }
      });
  });

  createdWindow.on("closed", () => {
    assignWindow(null);
    setIsCreatingMainWindow(false);
  });

  createdWindow.webContents.once("did-fail-load", () => {
    setIsCreatingMainWindow(false);
  });

  createdWindow.once("ready-to-show", () => {
    setIsCreatingMainWindow(false);
  });

  return createdWindow;
}
