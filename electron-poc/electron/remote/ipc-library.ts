import type { IpcMain } from "electron";

type ActiveLibraryProvider = "spotify" | "qobuz";
type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";

export function registerLibraryCaptureIpcHandlers({
  ipcMain,
  log,
  refreshQobuzPlaybackDeviceMappings,
  getCaptureEnvironmentStatusForQobuz,
  writeAppConfig,
  qobuzSinkIdByPlaybackDeviceId,
  getRemoteProviderConfig,
  checkLibraryProviderAuthenticated,
  isQobuzPlaybackCaptureActive,
  openRemoteSourceAuthWindow,
  scrapeProviderItems,
  searchQobuzCatalogViaApi,
  remoteCatalogCache,
  isPlaybackCaptureActive,
  getCachedLibraryItem,
  getNativePlaybackDevices,
  resolveCaptureLaunchTarget,
  computeCaptureQualityMode,
  startQobuzPlaybackCaptureForItem,
  sendBackendCommandWithRetry,
  playbackCaptureSessions,
  lastPlaybackCaptureProgressById,
  abortPlaybackCaptureSession,
}: {
  ipcMain: IpcMain;
  log: (message: string, ...args: any[]) => void;
  refreshQobuzPlaybackDeviceMappings: () => Promise<any>;
  getCaptureEnvironmentStatusForQobuz: () => Promise<any>;
  writeAppConfig: (partial: Record<string, any>) => boolean;
  qobuzSinkIdByPlaybackDeviceId: Map<string, string>;
  getRemoteProviderConfig: (provider: ActiveLibraryProvider | RemoteLibraryProvider) => any;
  checkLibraryProviderAuthenticated: (provider: ActiveLibraryProvider) => Promise<boolean>;
  isQobuzPlaybackCaptureActive: () => boolean;
  openRemoteSourceAuthWindow: (provider: RemoteLibraryProvider) => unknown;
  scrapeProviderItems: (provider: ActiveLibraryProvider, url: string) => Promise<any[]>;
  searchQobuzCatalogViaApi: (query: string) => Promise<any[]>;
  remoteCatalogCache: Map<string, any[]>;
  isPlaybackCaptureActive: () => boolean;
  getCachedLibraryItem: (provider: ActiveLibraryProvider, trackId: string) => any;
  getNativePlaybackDevices: () => Promise<any[]>;
  resolveCaptureLaunchTarget: (provider: ActiveLibraryProvider, item: any) => any;
  computeCaptureQualityMode: (provider: ActiveLibraryProvider, item: any) => any;
  startQobuzPlaybackCaptureForItem: (item: any, deviceId: string) => Promise<any>;
  sendBackendCommandWithRetry: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
    maxRetries?: number,
  ) => Promise<any>;
  playbackCaptureSessions: Map<string, any>;
  lastPlaybackCaptureProgressById: Map<string, any>;
  abortPlaybackCaptureSession: (
    captureId?: string,
    reason?: string,
  ) => Promise<any>;
}) {
  ipcMain.handle("detect-playback-devices", async () => {
    const state = await refreshQobuzPlaybackDeviceMappings();
    return state.devices;
  });

  ipcMain.handle("get-capture-environment-status", async () => {
    return getCaptureEnvironmentStatusForQobuz();
  });

  ipcMain.handle(
    "set-capture-output-device",
    async (_event, { deviceId }: { deviceId: string }) => {
      const trimmed = typeof deviceId === "string" ? deviceId.trim() : "";
      if (!trimmed) {
        return {
          success: false,
          error: "Select a silent output device first.",
        };
      }

      const state = await refreshQobuzPlaybackDeviceMappings();
      const device = state.devices.find((entry: any) => entry.id === trimmed);
      if (!device) {
        return {
          success: false,
          error: "The selected playback device is no longer available.",
        };
      }
      if (!qobuzSinkIdByPlaybackDeviceId.has(trimmed)) {
        return {
          success: false,
          error:
            "Qobuz could not route playback to the selected silent output. Authenticate first, then try saving the device again.",
        };
      }

      if (!writeAppConfig({ captureOutputDeviceId: trimmed })) {
        return {
          success: false,
          error: "Failed to persist the selected silent output device.",
        };
      }

      return {
        success: true,
        deviceId: trimmed,
        label: device.label,
      };
    },
  );

  ipcMain.handle(
    "auth-library-provider",
    async (_event, { provider }: { provider: ActiveLibraryProvider }) => {
      try {
        const config = getRemoteProviderConfig(provider);
        const authenticated = await checkLibraryProviderAuthenticated(provider);
        if (authenticated) {
          return {
            success: true,
            provider,
            authenticated: true,
            message: `${config.name} session already active. Opening your library view.`,
          };
        }

        openRemoteSourceAuthWindow(provider);
        return {
          success: true,
          provider,
          authenticated: false,
          message:
            provider === "qobuz"
              ? "Sign in to the Qobuz web player in the opened window, then refresh your library."
              : `Sign in to ${config.name} in the opened window, then refresh your library.`,
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
        };
      }
    },
  );

  ipcMain.handle(
    "get-library-auth-status",
    async (_event, { provider }: { provider: ActiveLibraryProvider }) => {
      try {
        const config = getRemoteProviderConfig(provider);
        const authenticated =
          provider === "qobuz" && isQobuzPlaybackCaptureActive()
            ? true
            : await checkLibraryProviderAuthenticated(provider);
        return {
          success: true,
          provider,
          authenticated,
          message: authenticated
            ? `${config.name} session is active.`
            : `${config.name} is not signed in in StemSep yet.`,
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
        };
      }
    },
  );

  ipcMain.handle(
    "list-library-collection",
    async (
      _event,
      { provider, scope }: { provider: ActiveLibraryProvider; scope?: string },
    ) => {
      try {
        if (scope) {
          log("[library] scope requested", { provider, scope });
        }
        const authenticated = await checkLibraryProviderAuthenticated(provider);
        if (!authenticated) {
          const config = getRemoteProviderConfig(provider);
          return {
            success: false,
            provider,
            authenticated: false,
            error: `${config.name} is not signed in in StemSep yet.`,
            hint:
              provider === "qobuz"
                ? "Click Authenticate, finish the Qobuz web player login in the opened window, then refresh Qobuz."
                : `Click Connect, finish the login in the provider window, then refresh ${config.name}.`,
          };
        }
        const config = getRemoteProviderConfig(provider);
        const items = await scrapeProviderItems(provider, config.libraryUrl);
        return {
          success: true,
          provider,
          authenticated: true,
          items,
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
          hint:
            provider === "spotify"
              ? "Open Spotify sign-in, confirm your saved tracks are visible in the web player, then refresh."
              : "Open Qobuz sign-in, confirm your favorites/library page is visible, then refresh.",
        };
      }
    },
  );

  ipcMain.handle(
    "search-library",
    async (
      _event,
      { provider, query }: { provider: ActiveLibraryProvider; query: string },
    ) => {
      const trimmed = String(query || "").trim();
      if (!trimmed) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: "Enter a search query first.",
        };
      }

      try {
        const config = getRemoteProviderConfig(provider);
        const authenticated = await checkLibraryProviderAuthenticated(provider);
        if (!authenticated) {
          return {
            success: false,
            provider,
            authenticated: false,
            error: `${config.name} is not signed in in StemSep yet.`,
            hint: `Click Connect, finish the login in the provider window, then retry the search.`,
          };
        }
        const targetUrl = config.searchUrl?.(trimmed) || config.libraryUrl;
        log("[library] search requested", { provider, query: trimmed, targetUrl });
        const items =
          provider === "qobuz"
            ? await searchQobuzCatalogViaApi(trimmed)
            : await scrapeProviderItems(provider, targetUrl);
        const filtered = items.filter((item) =>
          [item.title, item.artist, item.album]
            .filter(Boolean)
            .some((value) =>
              String(value).toLowerCase().includes(trimmed.toLowerCase()),
            ),
        );
        log("[library] search results", {
          provider,
          query: trimmed,
          scraped: items.length,
          filtered: filtered.length,
        });
        remoteCatalogCache.set(provider, filtered);
        return {
          success: true,
          provider,
          authenticated: true,
          items: filtered,
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
          hint: "Sign in to the provider first, then retry the search.",
        };
      }
    },
  );

  ipcMain.handle(
    "prepare-playback-capture",
    async (
      _event,
      { provider, trackId }: { provider: ActiveLibraryProvider; trackId: string },
    ) => {
      try {
        if (isPlaybackCaptureActive()) {
          throw new Error(
            "Another playback capture is already running. Wait for it to finish or cancel it first.",
          );
        }

        const item = getCachedLibraryItem(provider, trackId);
        if (!item) {
          throw new Error("Track not found in the current provider list. Refresh and try again.");
        }
        const devices = await getNativePlaybackDevices();
        if (devices.length === 0) {
          throw new Error("No Windows playback device is available for hidden capture.");
        }
        const launch = resolveCaptureLaunchTarget(provider, item);
        const quality = computeCaptureQualityMode(provider, item);
        return {
          success: true,
          provider,
          trackId: item.trackId,
          displayName: [item.artist, item.title].filter(Boolean).join(" - ") || item.title,
          launchUrl: launch.launchUrl,
          launchUri:
            provider === "spotify" ? item.playbackUri || undefined : undefined,
          playbackSurface: launch.playbackSurface,
          deviceRequired: true,
          qualityLabel:
            item.qualityLabel ||
            (quality.verified
              ? "Verified Lossless"
              : provider === "qobuz"
                ? "Lossless Source Capture"
                : "Best Available Capture"),
          qualityMode: quality.qualityMode,
          verifiedLossless: quality.verified,
          estimatedDurationSec: item.durationSec,
          sourceMeta: {
            provider,
            providerTrackId: item.trackId,
            title: item.title,
            artist: item.artist,
            album: item.album,
            artworkUrl: item.artworkUrl,
            durationSec: item.durationSec,
            canonicalUrl: item.canonicalUrl || item.playbackUrl || item.pageUrl,
            qualityLabel:
              item.qualityLabel ||
              (quality.verified
                ? "Verified Lossless"
                : provider === "qobuz"
                  ? "Lossless Source Capture"
                  : "Best Available Capture"),
            isLossless: item.isLossless,
            ingestMode: "desktop_capture",
            playbackSurface: launch.playbackSurface,
            qualityMode: quality.qualityMode,
            verifiedLossless: quality.verified,
          },
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          error: error?.message || String(error),
          code: "CAPTURE_PREPARE_FAILED",
        };
      }
    },
  );

  ipcMain.handle(
    "start-playback-capture",
    async (
      _event,
      {
        provider,
        trackId,
        deviceId,
      }: {
        provider: ActiveLibraryProvider;
        trackId: string;
        deviceId: string;
      },
    ) => {
      try {
        const item = getCachedLibraryItem(provider, trackId);
        if (!item) {
          throw new Error("Track not found in the current provider list. Refresh and try again.");
        }
        if (provider !== "qobuz") {
          throw new Error(
            "Hidden library capture is only enabled for Qobuz in this build.",
          );
        }

        const authenticated = await checkLibraryProviderAuthenticated(provider);
        if (!authenticated) {
          throw new Error("Authenticate Qobuz in StemSep before starting a hidden capture.");
        }

        const deviceState = await refreshQobuzPlaybackDeviceMappings();
        const device = deviceState.devices.find((entry: any) => entry.id === deviceId);
        if (!device) {
          throw new Error("The selected playback device is no longer available.");
        }
        if (!qobuzSinkIdByPlaybackDeviceId.has(deviceId)) {
          throw new Error(
            "The selected silent output could not be routed from the hidden Qobuz player. Re-run Capture Setup and save the device again.",
          );
        }

        return await startQobuzPlaybackCaptureForItem(item, deviceId);
      } catch (error: any) {
        const message = error?.message || String(error);
        return {
          success: false,
          provider,
          error: /cancelled/i.test(message) ? "Capture cancelled" : message,
          code: /cancelled/i.test(message)
            ? "CAPTURE_CANCELLED"
            : "CAPTURE_START_FAILED",
        };
      }
    },
  );

  ipcMain.handle(
    "cancel-playback-capture",
    async (_event, { captureId }: { captureId?: string }) => {
      return abortPlaybackCaptureSession(captureId, "Capture cancelled.");
    },
  );

  ipcMain.handle(
    "get-playback-capture-status",
    async (_event, { captureId }: { captureId?: string } = {}) => {
      const localSessions = (captureId
        ? Array.from(playbackCaptureSessions.values()).filter(
            (session) => session.captureId === captureId,
          )
        : Array.from(playbackCaptureSessions.values())
      ).map((session) => ({
        captureId: session.captureId,
        provider: session.provider,
        trackId: session.trackId,
        item: session.item,
        startedAt: session.startedAt,
        backendCaptureStarted: session.backendCaptureStarted,
        progress: lastPlaybackCaptureProgressById.get(session.captureId) || null,
      }));

      try {
        const backendStatus = await sendBackendCommandWithRetry(
          "get_playback_capture_status",
          captureId ? { capture_id: captureId } : {},
          10_000,
          0,
        ).catch(() => null);
        return {
          success: true,
          sessions: localSessions,
          backend: backendStatus,
        };
      } catch (error: any) {
        return {
          success: false,
          error: error?.message || String(error),
          sessions: localSessions,
        };
      }
    },
  );
}
