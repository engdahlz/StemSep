import { BrowserWindow, screen, type Session } from "electron";

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";

type RemoteProviderConfig = {
  name: string;
  partition: string;
  loginUrl: string;
  libraryUrl: string;
};

export function createRemoteWindowController({
  getRemoteProviderConfig,
  getRemoteSession,
  getMainWindow,
  log,
  onObservedQobuzUrl,
  onQobuzAutomationClosed,
}: {
  getRemoteProviderConfig: (provider: RemoteLibraryProvider) => RemoteProviderConfig;
  getRemoteSession: (provider: RemoteLibraryProvider) => Session;
  getMainWindow: () => BrowserWindow | null;
  log: (message: string, ...args: any[]) => void;
  onObservedQobuzUrl?: (url: string) => void;
  onQobuzAutomationClosed?: () => void;
}) {
  const remoteAuthWindows = new Map<RemoteLibraryProvider, BrowserWindow>();
  let qobuzAutomationWindow: BrowserWindow | null = null;

  function configureProviderSessionPermissions(provider: RemoteLibraryProvider) {
    const ses = getRemoteSession(provider);
    if ((ses as any).__stemsepPermissionsConfigured) return;

    ses.setPermissionCheckHandler((_webContents, permission) => {
      if (String(permission) === "speaker-selection") return true;
      return false;
    });

    ses.setPermissionRequestHandler((_webContents, permission, callback) => {
      if (String(permission) === "speaker-selection") {
        callback(true);
        return;
      }
      callback(false);
    });

    (ses as any).__stemsepPermissionsConfigured = true;
  }

  function getExistingAuthWindow(provider: RemoteLibraryProvider) {
    const win = remoteAuthWindows.get(provider) || null;
    return win && !win.isDestroyed() ? win : null;
  }

  function getExistingQobuzAutomationWindow() {
    return qobuzAutomationWindow && !qobuzAutomationWindow.isDestroyed()
      ? qobuzAutomationWindow
      : null;
  }

  function getQobuzAutomationWindow() {
    const existing = getExistingQobuzAutomationWindow();
    if (existing) return existing;

    configureProviderSessionPermissions("qobuz");
    const displays = screen.getAllDisplays();
    const virtualBounds = displays.reduce(
      (acc, display) => {
        acc.minX = Math.min(acc.minX, display.bounds.x);
        acc.minY = Math.min(acc.minY, display.bounds.y);
        acc.maxX = Math.max(acc.maxX, display.bounds.x + display.bounds.width);
        acc.maxY = Math.max(acc.maxY, display.bounds.y + display.bounds.height);
        return acc;
      },
      {
        minX: 0,
        minY: 0,
        maxX: 0,
        maxY: 0,
      },
    );

    qobuzAutomationWindow = new BrowserWindow({
      show: true,
      x: virtualBounds.maxX + 1200,
      y: virtualBounds.maxY + 1200,
      width: 360,
      height: 240,
      title: "Qobuz Automation",
      skipTaskbar: true,
      focusable: false,
      webPreferences: {
        partition: getRemoteProviderConfig("qobuz").partition,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        backgroundThrottling: false,
      },
    });
    qobuzAutomationWindow.setAlwaysOnTop(false);
    qobuzAutomationWindow.on("closed", () => {
      qobuzAutomationWindow = null;
      onQobuzAutomationClosed?.();
    });

    return qobuzAutomationWindow;
  }

  function openRemoteSourceAuthWindow(provider: RemoteLibraryProvider) {
    const existing = getExistingAuthWindow(provider);
    if (existing) {
      existing.focus();
      return existing;
    }

    const config = getRemoteProviderConfig(provider);
    configureProviderSessionPermissions(provider);
    const authWindow = new BrowserWindow({
      width: 1180,
      height: 860,
      parent: getMainWindow() || undefined,
      title: `${config.name} Sign In`,
      webPreferences: {
        partition: config.partition,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
      },
    });

    remoteAuthWindows.set(provider, authWindow);

    if (provider === "qobuz") {
      const observeQobuzUrl = (url: string) => {
        const normalized = String(url || "");
        if (!/play\.qobuz\.com/i.test(normalized)) return;
        if (/\/login(\/|$)|\/signin(\/|$)/i.test(normalized)) return;
        onObservedQobuzUrl?.(normalized);
        log("[library] qobuz observed authenticated url", { url: normalized });
      };

      authWindow.webContents.on("did-navigate", (_event, url) => {
        observeQobuzUrl(url);
      });
      authWindow.webContents.on("did-redirect-navigation", (_event, url) => {
        observeQobuzUrl(url);
      });
      authWindow.webContents.on("did-navigate-in-page", (_event, url) => {
        observeQobuzUrl(url);
      });
    }

    authWindow.on("closed", () => {
      remoteAuthWindows.delete(provider);
    });

    authWindow.loadURL(config.loginUrl).catch(async () => {
      try {
        await authWindow.loadURL(config.libraryUrl);
      } catch (error) {
        log("[remote] failed to open auth window", {
          provider,
          error: (error as any)?.message || String(error),
        });
      }
    });

    return authWindow;
  }

  function getOpenProviderProbeWindows(
    provider: Extract<RemoteLibraryProvider, "spotify" | "qobuz">,
  ) {
    const windows: BrowserWindow[] = [];
    const authWindow = getExistingAuthWindow(provider);
    if (authWindow) {
      windows.push(authWindow);
    }
    const automationWindow =
      provider === "qobuz" ? getExistingQobuzAutomationWindow() : null;
    if (automationWindow && automationWindow !== authWindow) {
      windows.push(automationWindow);
    }
    return windows;
  }

  function closeAllRemoteWindows() {
    for (const win of Array.from(remoteAuthWindows.values())) {
      try {
        win.close();
      } catch {
        // ignore
      }
    }
    const automationWindow = getExistingQobuzAutomationWindow();
    if (automationWindow) {
      try {
        automationWindow.close();
      } catch {
        // ignore
      }
      qobuzAutomationWindow = null;
      onQobuzAutomationClosed?.();
    }
  }

  return {
    configureProviderSessionPermissions,
    getExistingAuthWindow,
    getExistingQobuzAutomationWindow,
    getQobuzAutomationWindow,
    openRemoteSourceAuthWindow,
    getOpenProviderProbeWindows,
    closeAllRemoteWindows,
  };
}
