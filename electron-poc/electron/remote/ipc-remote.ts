import type { IpcMain } from "electron";

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";

export function registerRemoteImportIpcHandlers({
  ipcMain,
  log,
  getRemoteProviderConfig,
  openRemoteSourceAuthWindow,
  scrapeRemoteCollection,
  remoteCatalogCache,
  emitRemoteResolveProgress,
  resolveRemoteDownloadInfo,
  downloadRemoteFile,
  extractArchiveForRemoteTrack,
  probeAudioFile,
  sendPythonCommand,
}: {
  ipcMain: IpcMain;
  log: (message: string, ...args: any[]) => void;
  getRemoteProviderConfig: (provider: RemoteLibraryProvider) => any;
  openRemoteSourceAuthWindow: (provider: RemoteLibraryProvider) => unknown;
  scrapeRemoteCollection: (provider: RemoteLibraryProvider) => Promise<any[]>;
  remoteCatalogCache: Map<string, any[]>;
  emitRemoteResolveProgress: (payload: any) => void;
  resolveRemoteDownloadInfo: (provider: RemoteLibraryProvider, pageUrl: string) => Promise<any>;
  downloadRemoteFile: (
    provider: RemoteLibraryProvider,
    url: string,
    title: string,
  ) => Promise<string>;
  extractArchiveForRemoteTrack: (
    provider: RemoteLibraryProvider,
    archivePath: string,
    title: string,
  ) => Promise<string | null>;
  probeAudioFile: (filePath: string) => Promise<any>;
  sendPythonCommand: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
  ) => Promise<any>;
}) {
  ipcMain.handle(
    "auth-remote-source",
    async (_event, { provider }: { provider: RemoteLibraryProvider }) => {
      try {
        openRemoteSourceAuthWindow(provider);
        const config = getRemoteProviderConfig(provider);
        return {
          success: true,
          provider,
          authenticated: false,
          message: `Sign in to ${config.name} in the opened window, then click Refresh Library.`,
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
    "list-remote-collection",
    async (
      _event,
      {
        provider,
        scope,
      }: { provider: RemoteLibraryProvider; scope?: string },
    ) => {
      try {
        const items = await scrapeRemoteCollection(provider);
        const authenticated = items.length > 0;
        if (scope && scope.trim()) {
          log("[remote] collection scope ignored", { provider, scope });
        }
        return {
          success: true,
          provider,
          authenticated,
          items,
        };
      } catch (error: any) {
        const config = getRemoteProviderConfig(provider);
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
          hint: `Open the ${config.name} sign-in window, log in, and make sure your purchases/downloads page is available.`,
        };
      }
    },
  );

  ipcMain.handle(
    "search-remote-catalog",
    async (
      _event,
      {
        provider,
        query,
      }: { provider: RemoteLibraryProvider; query: string },
    ) => {
      try {
        const items =
          remoteCatalogCache.get(provider) || (await scrapeRemoteCollection(provider));
        const trimmed = String(query || "").trim().toLowerCase();
        const filtered = !trimmed
          ? items
          : items.filter((item) =>
              [item.title, item.artist, item.album]
                .filter(Boolean)
                .some((value) => String(value).toLowerCase().includes(trimmed)),
            );
        return {
          success: true,
          provider,
          authenticated: items.length > 0,
          items: filtered,
        };
      } catch (error: any) {
        return {
          success: false,
          provider,
          authenticated: false,
          error: error?.message || String(error),
          hint: "Refresh the provider library first.",
        };
      }
    },
  );

  ipcMain.handle(
    "resolve-remote-track",
    async (
      _event,
      {
        provider,
        trackId,
        variantId,
      }: {
        provider: RemoteLibraryProvider;
        trackId: string;
        variantId?: string;
      },
    ) => {
      emitRemoteResolveProgress({
        provider,
        status: "starting",
        detail: "Preparing import...",
      });

      try {
        const items =
          remoteCatalogCache.get(provider) || (await scrapeRemoteCollection(provider));
        const item = items.find((entry) => entry.trackId === trackId);
        if (!item) {
          throw new Error("Remote item not found. Refresh the provider library and try again.");
        }
        if (variantId && item.variantId && variantId !== item.variantId) {
          throw new Error("Requested provider variant is no longer available.");
        }

        let downloadUrl = item.sourceUrl;
        let resolvedTitle = item.title;
        let resolvedArtist = item.artist;
        let resolvedAlbum = item.album;
        let resolvedArtwork = item.artworkUrl;
        let resolvedQuality = item.qualityLabel || "Lossless";
        let resolvedCanonicalUrl = item.canonicalUrl || item.pageUrl || item.sourceUrl;

        if (!downloadUrl || !/\.(flac|wav|aif|aiff|zip)(\?|$)/i.test(downloadUrl)) {
          if (!item.pageUrl) {
            throw new Error("No downloadable source was found for this item.");
          }
          emitRemoteResolveProgress({
            provider,
            status: "resolving",
            detail: "Resolving provider download link...",
          });
          const resolved = await resolveRemoteDownloadInfo(provider, item.pageUrl);
          downloadUrl = resolved.downloadUrl || downloadUrl;
          resolvedTitle = resolved.title || resolvedTitle;
          resolvedArtist = resolved.artist || resolvedArtist;
          resolvedAlbum = resolved.album || resolvedAlbum;
          resolvedArtwork = resolved.artworkUrl || resolvedArtwork;
          resolvedQuality = resolved.qualityLabel || resolvedQuality;
          resolvedCanonicalUrl = resolved.canonicalUrl || resolvedCanonicalUrl;
        }

        if (!downloadUrl) {
          throw new Error(
            "The provider page did not expose a downloadable lossless file for this item.",
          );
        }

        const downloadedPath = await downloadRemoteFile(
          provider,
          downloadUrl,
          resolvedTitle || item.title,
        );

        let effectivePath = downloadedPath;
        if (/\.(zip)(\?|$)/i.test(downloadedPath)) {
          emitRemoteResolveProgress({
            provider,
            status: "extracting",
            detail: "Extracting downloaded archive...",
          });
          const extractedPath = await extractArchiveForRemoteTrack(
            provider,
            downloadedPath,
            resolvedTitle || item.title,
          );
          if (!extractedPath) {
            throw new Error(
              "The provider returned an archive, but no matching audio track could be extracted.",
            );
          }
          effectivePath = extractedPath;
        }

        emitRemoteResolveProgress({
          provider,
          status: "probing",
          detail: "Verifying audio quality...",
        });
        const profile = await probeAudioFile(effectivePath);
        if (!profile.isLossless) {
          throw new Error(
            "The resolved provider file is not lossless and cannot be used for true lossless ingest.",
          );
        }

        emitRemoteResolveProgress({
          provider,
          status: "completed",
          detail: "Import ready.",
          percent: "100%",
          progress: 1,
        });

        return {
          success: true,
          provider,
          file_path: effectivePath,
          display_name:
            [resolvedArtist, resolvedTitle].filter(Boolean).join(" - ") ||
            resolvedTitle ||
            item.title,
          source_url: downloadUrl,
          canonical_url: resolvedCanonicalUrl,
          artist: resolvedArtist,
          album: resolvedAlbum,
          artwork_url: resolvedArtwork,
          duration_sec: item.durationSec,
          quality_label: resolvedQuality,
          is_lossless: true,
          provider_track_id: item.trackId,
          download_origin: item.downloadOrigin || "purchase",
        };
      } catch (error: any) {
        emitRemoteResolveProgress({
          provider,
          status: "failed",
          error: error?.message || String(error),
        });
        return {
          success: false,
          provider,
          error: error?.message || String(error),
          code: "REMOTE_RESOLVE_FAILED",
          hint: "Refresh the provider library, then try the import again from a purchased/downloadable lossless item.",
        };
      }
    },
  );

  ipcMain.handle("resolve-youtube-url", async (_event, { url }: { url: string }) => {
    try {
      const result = await sendPythonCommand("resolve_youtube", { url }, 10 * 60 * 1000);
      return {
        success: true,
        file_path: result?.file_path,
        title: result?.title,
        source_url: result?.source_url,
        channel: result?.channel,
        duration_sec: result?.duration_sec,
        thumbnail_url: result?.thumbnail_url,
        canonical_url: result?.canonical_url,
      };
    } catch (e: any) {
      const raw = e?.message || String(e);
      const lower = String(raw).toLowerCase();
      if (
        lower.includes("backend_unavailable") &&
        (lower.includes("python proxy unavailable") ||
          lower.includes("python-bridge.py"))
      ) {
        return {
          success: false,
          code: "YOUTUBE_UNAVAILABLE_IN_RUST_MODE",
          error:
            "YouTube import is unavailable in this build configuration (rust backend without python-bridge proxy).",
          hint:
            "Use local audio files for now, or enable the python bridge backend for YouTube resolution.",
        };
      }
      return {
        success: false,
        code: "YOUTUBE_RESOLVE_FAILED",
        error: raw,
      };
    }
  });
}
