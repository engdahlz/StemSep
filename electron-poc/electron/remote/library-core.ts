import type { BrowserWindow, Session } from "electron";

type ActiveLibraryProvider = "spotify" | "qobuz";
type RemoteLibraryProvider = ActiveLibraryProvider | "bandcamp";

type RemoteProviderConfig = {
  name: string;
  partition: string;
  loginUrl: string;
  libraryUrl: string;
};

type PlaybackSurface = "desktop_app" | "browser" | "none";
type CaptureQualityMode = "best_available" | "verified_lossless";

type RemoteCatalogItem = {
  provider: "qobuz" | "spotify" | "bandcamp";
  trackId: string;
  albumId?: string;
  title: string;
  artist?: string;
  album?: string;
  trackNumber?: number;
  discNumber?: number;
  artworkUrl?: string;
  durationSec?: number;
  canonicalUrl?: string;
  qualityLabel?: string;
  isLossless?: boolean;
  downloadOrigin?: string;
  pageUrl?: string;
  playbackUrl?: string;
  playbackUri?: string;
  playbackSurface?: PlaybackSurface;
  ingestMode?: "local_file" | "remote_download" | "desktop_capture";
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
};

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function getLibraryScrapeScript(provider: ActiveLibraryProvider) {
  const common = String.raw`
    const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim();
    const textOf = (el) => normalizeText(el && (el.innerText || el.textContent));
    const durationFromText = (value) => {
      const match = String(value || "").match(/(\d+):(\d{2})(?::(\d{2}))?/);
      if (!match) return undefined;
      if (match[3]) return (Number(match[1]) * 3600) + (Number(match[2]) * 60) + Number(match[3]);
      return (Number(match[1]) * 60) + Number(match[2]);
    };
    const withOrigin = (href) => {
      try { return new URL(href, window.location.origin).toString(); } catch { return String(href || ""); }
    };
  `;

  if (provider === "spotify") {
    return `(() => {
      ${common}
      const seen = new Set();
      const items = [];
      const rows = Array.from(document.querySelectorAll('[data-testid="tracklist-row"], div[role="row"]'));
      const pushItem = (node) => {
        const trackAnchor = Array.from(node.querySelectorAll('a[href]')).find((anchor) => /\\/track\\//i.test(anchor.getAttribute('href') || ''));
        if (!trackAnchor) return;
        const trackHref = withOrigin(trackAnchor.getAttribute('href') || trackAnchor.href || '');
        const trackIdMatch = trackHref.match(/\\/track\\/([A-Za-z0-9]+)/i);
        const trackId = trackIdMatch ? trackIdMatch[1] : '';
        if (!trackId || seen.has(trackId)) return;
        seen.add(trackId);
        const artistAnchors = Array.from(node.querySelectorAll('a[href]')).filter((anchor) => /\\/artist\\//i.test(anchor.getAttribute('href') || ''));
        const albumAnchor = Array.from(node.querySelectorAll('a[href]')).find((anchor) => /\\/album\\//i.test(anchor.getAttribute('href') || ''));
        const img = node.querySelector('img');
        const title = normalizeText(trackAnchor.textContent) || textOf(node.querySelector('[aria-colindex="2"], [data-encore-id="text"]'));
        const artist = artistAnchors.map((anchor) => normalizeText(anchor.textContent)).filter(Boolean).join(', ');
        const album = normalizeText(albumAnchor && albumAnchor.textContent);
        const qualityLabel = "Best Available Capture";
        const durationSec = durationFromText(textOf(node));
        if (!title) return;
        items.push({
          provider: "spotify",
          trackId,
          title,
          artist: artist || undefined,
          album: album || undefined,
          artworkUrl: img && img.src ? img.src : undefined,
          durationSec,
          canonicalUrl: trackHref,
          qualityLabel,
          isLossless: false,
          playbackUrl: trackHref,
          playbackUri: \`spotify:track:\${trackId}\`,
          playbackSurface: "desktop_app",
          ingestMode: "desktop_capture",
          qualityMode: "best_available",
          verifiedLossless: false,
        });
      };

      rows.forEach(pushItem);
      if (items.length === 0) {
        Array.from(document.querySelectorAll('a[href*="/track/"]')).forEach((anchor) => {
          const href = withOrigin(anchor.getAttribute('href') || anchor.href || '');
          const trackIdMatch = href.match(/\\/track\\/([A-Za-z0-9]+)/i);
          const trackId = trackIdMatch ? trackIdMatch[1] : '';
          const title = normalizeText(anchor.textContent);
          if (!trackId || !title || seen.has(trackId)) return;
          seen.add(trackId);
          const container = anchor.closest('section, article, div, li') || document;
          const img = container.querySelector('img');
          items.push({
            provider: "spotify",
            trackId,
            title,
            artworkUrl: img && img.src ? img.src : undefined,
            canonicalUrl: href,
            qualityLabel: "Best Available Capture",
            isLossless: false,
            playbackUrl: href,
            playbackUri: \`spotify:track:\${trackId}\`,
            playbackSurface: "desktop_app",
            ingestMode: "desktop_capture",
            qualityMode: "best_available",
            verifiedLossless: false,
          });
        });
      }
      return items.slice(0, 200);
    })()`;
  }

  return `(() => {
    ${common}
    const seen = new Set();
    const items = [];
    const nodes = Array.from(document.querySelectorAll('article, li, .track, .item, .album, [data-testid], [class*="track"], [class*="product"]'));
    for (const node of nodes) {
      const anchors = Array.from(node.querySelectorAll('a[href]'));
      const trackAnchor = anchors.find((anchor) => /\\/track\\//i.test(anchor.getAttribute('href') || anchor.href || ''));
      if (!trackAnchor) continue;
      const canonicalUrl = withOrigin(trackAnchor.getAttribute('href') || trackAnchor.href || '');
      const trackIdMatch = canonicalUrl.match(/\\/track\\/([^/?#]+)/i);
      const trackId = trackIdMatch ? trackIdMatch[1] : '';
      if (!trackId || seen.has(trackId)) continue;
      const title = normalizeText(trackAnchor.textContent) || textOf(node.querySelector('h1, h2, h3, h4, [class*="title"]'));
      if (!title) continue;
      seen.add(trackId);
      const text = textOf(node);
      const artist = textOf(node.querySelector('[class*="artist"], [data-testid*="artist"], [class*="subtitle"], [class*="meta"]'));
      const album = textOf(node.querySelector('[class*="album"], [data-testid*="album"]'));
      const img = node.querySelector('img');
      const hiResMatch = text.match(/(\\d+[- ]?bit[^\\d]{0,8}\\d+(?:[.,]\\d+)?\\s?k?hz)/i);
      const qualityLabel = hiResMatch ? hiResMatch[1].replace(/\\s+/g, ' ') : (/lossless/i.test(text) ? "Lossless Capture" : "Best Available Capture");
      items.push({
        provider: "qobuz",
        trackId,
        title,
        artist: artist || undefined,
        album: album || undefined,
        artworkUrl: img && img.src ? img.src : undefined,
        durationSec: durationFromText(text),
        canonicalUrl,
        qualityLabel,
        isLossless: /lossless|24[- ]?bit|flac|wav/i.test(text),
        playbackUrl: canonicalUrl,
        playbackSurface: "browser",
        ingestMode: "desktop_capture",
        qualityMode: "best_available",
        verifiedLossless: false,
      });
    }
    return items.slice(0, 200);
  })()`;
}

function isLikelyUnauthedProviderUrl(
  provider: ActiveLibraryProvider,
  url: string,
) {
  const normalized = String(url || "").toLowerCase();
  if (provider === "spotify") {
    return (
      normalized.includes("accounts.spotify.com") ||
      normalized.includes("/login") ||
      normalized.includes("/signup")
    );
  }
  return normalized.includes("/signin") || normalized.includes("/login");
}

export function createRemoteLibraryController({
  getRemoteProviderConfig,
  getRemoteSession,
  getQobuzAutomationWindow,
  openRemoteSourceAuthWindow,
  getOpenProviderProbeWindows,
  remoteCatalogCache,
  log,
  isQobuzPlaybackCaptureActive,
}: {
  getRemoteProviderConfig: (provider: RemoteLibraryProvider) => RemoteProviderConfig;
  getRemoteSession: (provider: RemoteLibraryProvider) => Session;
  getQobuzAutomationWindow: () => BrowserWindow;
  openRemoteSourceAuthWindow: (provider: RemoteLibraryProvider) => BrowserWindow;
  getOpenProviderProbeWindows: (
    provider: Extract<RemoteLibraryProvider, "spotify" | "qobuz">,
  ) => BrowserWindow[];
  remoteCatalogCache: Map<RemoteLibraryProvider, RemoteCatalogItem[]>;
  log: (message: string, ...args: any[]) => void;
  isQobuzPlaybackCaptureActive: () => boolean;
}) {
  let qobuzObservedPlaySession = false;

  function markQobuzPlaySessionObserved() {
    qobuzObservedPlaySession = true;
  }

  function resetQobuzPlaySessionObserved() {
    qobuzObservedPlaySession = false;
  }

  function waitForDidFinishLoad(
    win: BrowserWindow,
    timeoutMs = 15_000,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error("Timed out while loading provider page."));
      }, timeoutMs);

      const cleanup = () => {
        clearTimeout(timer);
        win.webContents.removeListener("did-finish-load", handleFinish);
        win.webContents.removeListener("did-fail-load", handleFail);
      };

      const handleFinish = () => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve();
      };

      const handleFail = (
        _event: any,
        code: number,
        description: string,
        _validatedURL?: string,
        isMainFrame?: boolean,
      ) => {
        if (isMainFrame === false) {
          return;
        }
        if (code === -3) {
          return;
        }
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error(`Provider page failed to load (${code}): ${description}`));
      };

      win.webContents.once("did-finish-load", handleFinish);
      win.webContents.once("did-fail-load", handleFail);
    });
  }

  async function loadURLAndWait(
    win: BrowserWindow,
    url: string,
    timeoutMs = 20_000,
  ) {
    const waiter = waitForDidFinishLoad(win, timeoutMs);
    try {
      await win.loadURL(url);
    } catch (error: any) {
      const message = String(error?.message || error || "");
      if (!message.includes("ERR_ABORTED") && !message.includes("(-3)")) {
        throw error;
      }
    }
    await waiter;
  }

  async function navigateRemoteWindow(
    provider: RemoteLibraryProvider,
    targetUrl: string,
  ) {
    const win =
      provider === "qobuz"
        ? getQobuzAutomationWindow()
        : openRemoteSourceAuthWindow(provider);
    const currentUrl = win.webContents.getURL();
    if (currentUrl !== targetUrl) {
      await loadURLAndWait(win, targetUrl);
    } else {
      await win.webContents.reloadIgnoringCache();
      await waitForDidFinishLoad(win);
    }
    await sleep(1200);
    return win;
  }

  async function probeQobuzAuthenticatedFromWindow(
    win: BrowserWindow,
  ): Promise<boolean> {
    try {
      const currentUrl = String(win.webContents.getURL() || "");
      if (!/play\.qobuz\.com/i.test(currentUrl)) return false;
      const currentPath = new URL(currentUrl).pathname || "";
      const onLoginPath = /^\/(?:login|signin)(?:\/|$)/i.test(currentPath);
      const authenticated = !onLoginPath;

      if (authenticated) {
        qobuzObservedPlaySession = true;
      }

      log("[library] qobuz auth probe", {
        currentUrl,
        currentPath,
        authenticated,
        observedSession: qobuzObservedPlaySession,
      });

      return authenticated;
    } catch (error) {
      log("[library] qobuz window auth probe failed", {
        error: (error as any)?.message || String(error),
      });
      return false;
    }
  }

  async function checkLibraryProviderAuthenticated(
    provider: ActiveLibraryProvider,
  ) {
    try {
      if (provider === "qobuz") {
        if (isQobuzPlaybackCaptureActive()) {
          return true;
        }

        const windows = getOpenProviderProbeWindows(provider);
        if (windows.length > 0) {
          for (const win of windows) {
            if (await probeQobuzAuthenticatedFromWindow(win)) {
              return true;
            }
          }
        }

        if (qobuzObservedPlaySession) {
          return true;
        }

        const automationWindow = getQobuzAutomationWindow();
        const currentUrl = String(automationWindow.webContents.getURL() || "");
        if (
          !/play\.qobuz\.com/i.test(currentUrl) ||
          /\/login(\/|$)|\/signin(\/|$)/i.test(currentUrl)
        ) {
          await loadURLAndWait(
            automationWindow,
            getRemoteProviderConfig("qobuz").libraryUrl,
            20_000,
          );
          await sleep(900);
        }

        return probeQobuzAuthenticatedFromWindow(automationWindow);
      }

      const sessionForProvider = getRemoteSession(provider);
      const config = getRemoteProviderConfig(provider);
      const providerUrl = new URL(config.libraryUrl);
      const rootHostname = providerUrl.hostname.replace(/^www\./i, "");
      const cookies = await sessionForProvider.cookies.get({});
      const providerCookies = cookies.filter((cookie) => {
        const domain = String(cookie.domain || "")
          .toLowerCase()
          .replace(/^\./, "");
        return (
          domain === providerUrl.hostname.toLowerCase() ||
          domain === rootHostname ||
          domain.endsWith(`.${rootHostname}`)
        );
      });

      if (provider === "spotify") {
        return providerCookies.some((cookie) =>
          ["sp_dc", "sp_key", "sp_t"].includes(String(cookie.name || "").toLowerCase()),
        );
      }

      return false;
    } catch (error) {
      log("[library] auth check dom probe failed", {
        provider,
        error: (error as any)?.message || String(error),
      });
      return false;
    }
  }

  async function scrapeProviderItems(
    provider: ActiveLibraryProvider,
    targetUrl: string,
  ): Promise<RemoteCatalogItem[]> {
    const win = await navigateRemoteWindow(provider, targetUrl);
    const rawItems = (await win.webContents.executeJavaScript(
      getLibraryScrapeScript(provider),
      true,
    )) as RemoteCatalogItem[];
    const items = Array.isArray(rawItems)
      ? rawItems.filter((item) => item?.title && item?.trackId)
      : [];
    remoteCatalogCache.set(provider, items);
    return items;
  }

  async function searchQobuzCatalogViaApi(
    query: string,
  ): Promise<RemoteCatalogItem[]> {
    const win = getQobuzAutomationWindow();
    const currentUrl = String(win.webContents.getURL() || "");
    if (!/qobuz\.com/i.test(currentUrl)) {
      await loadURLAndWait(win, getRemoteProviderConfig("qobuz").libraryUrl, 20_000);
      await sleep(1200);
    }

    const result = (await win.webContents.executeJavaScript(
      `(async () => {
        const query = ${JSON.stringify(query)};
        const env = String(window.__ENVIRONMENT__ || "production").toLowerCase();
        const appIds = {
          integration: "377257687",
          nightly: "377257687",
          recette: "724307056",
          production: "798273057",
        };
        const appId = appIds[env] || appIds.production;

        const findTokenInObject = (value, depth = 0, seen = new WeakSet()) => {
          if (!value || depth > 5) return "";
          if (typeof value === "string") {
            const direct = value.match(/"user_auth_token"\\s*:\\s*"([^"]+)"/i);
            if (direct) return direct[1];
            const jsonToken = value.match(/"token"\\s*:\\s*"([^"]+)"/i);
            if (jsonToken && /"infos"\\s*:/.test(value)) return jsonToken[1];
            return "";
          }
          if (typeof value !== "object") return "";
          if (seen.has(value)) return "";
          seen.add(value);

          if (typeof value.user_auth_token === "string" && value.user_auth_token) {
            return value.user_auth_token;
          }
          if (
            typeof value.token === "string" &&
            value.token &&
            (value.infos || value.user || value.session_id || value.expires_at)
          ) {
            return value.token;
          }

          for (const entry of Object.values(value)) {
            const found = findTokenInObject(entry, depth + 1, seen);
            if (found) return found;
          }
          return "";
        };

        const storagePayloads = [];
        for (const storage of [window.localStorage, window.sessionStorage]) {
          try {
            for (let index = 0; index < storage.length; index += 1) {
              const key = storage.key(index);
              if (!key) continue;
              const raw = storage.getItem(key);
              if (!raw) continue;
              storagePayloads.push(raw);
              try {
                storagePayloads.push(JSON.parse(raw));
              } catch {}
            }
          } catch {}
        }

        const globalCandidates = [
          window.__INITIAL_STATE__,
          window.__PRELOADED_STATE__,
          window.__STORE__?.getState?.(),
          window.store?.getState?.(),
          window.__NEXT_DATA__,
          window.__NUXT__,
        ];

        let token = "";
        for (const candidate of [...storagePayloads, ...globalCandidates]) {
          token = findTokenInObject(candidate);
          if (token) break;
        }

        const response = await fetch(
          "https://www.qobuz.com/api.json/0.2/track/search?" +
            new URLSearchParams({ query, offset: "0", limit: "25" }).toString(),
          {
            method: "GET",
            headers: token
              ? { "X-App-Id": appId, "X-User-Auth-Token": token }
              : { "X-App-Id": appId },
            credentials: "include",
          },
        );

        const payload = await response.json().catch(() => ({}));
        const rawTracks = Array.isArray(payload?.tracks?.items)
          ? payload.tracks.items
          : Array.isArray(payload?.tracks)
            ? payload.tracks
            : [];

        const normalize = (value) =>
          String(value || "").replace(/\\s+/g, " ").trim();

        const items = rawTracks
          .map((track) => {
            const id = track?.id;
            if (!id) return null;

            const performer =
              normalize(track?.performer?.name) ||
              normalize(track?.artist?.name) ||
              normalize(track?.album?.artist?.name);

            const albumTitle =
              normalize(track?.album?.title) ||
              normalize(track?.release?.title);
            const albumArtist =
              normalize(track?.album?.artist?.name) ||
              normalize(track?.release?.artist?.name);
            const albumId = track?.album?.id || track?.release?.id;
            const releaseYear = Number(
              track?.album?.release_date_original?.slice?.(0, 4) ||
                track?.album?.released_at?.slice?.(0, 4) ||
                track?.release?.released_at?.slice?.(0, 4) ||
                0,
            );
            const trackNumber = Number(
              track?.track_number ||
                track?.trackNumber ||
                track?.track_number_on_album ||
                0,
            );
            const discNumber = Number(
              track?.media_number ||
                track?.disc_number ||
                track?.discNumber ||
                0,
            );

            const maximumSamplingRate = Number(
              track?.maximum_sampling_rate ||
                track?.maximumSamplingRate ||
                track?.audio_info?.maximum_sampling_rate ||
                0,
            );
            const maximumBitDepth = Number(
              track?.maximum_bit_depth ||
                track?.maximumBitDepth ||
                track?.audio_info?.maximum_bit_depth ||
                0,
            );

            const qualityLabel =
              maximumSamplingRate && maximumBitDepth
                ? \`\${maximumBitDepth}-bit / \${maximumSamplingRate} kHz\`
                : track?.hires_streamable || track?.hires
                  ? "Hi-Res Capture"
                  : track?.lossless || track?.streamable
                    ? "Lossless Capture"
                    : "Best Available Capture";

            return {
              provider: "qobuz",
              trackId: String(id),
              albumId: albumId ? String(albumId) : undefined,
              title: normalize(track?.title),
              artist: performer || undefined,
              album: albumTitle || undefined,
              albumArtist: albumArtist || undefined,
              trackNumber:
                Number.isFinite(trackNumber) && trackNumber > 0
                  ? trackNumber
                  : undefined,
              discNumber:
                Number.isFinite(discNumber) && discNumber > 0
                  ? discNumber
                  : undefined,
              artworkUrl:
                track?.album?.image?.large ||
                track?.album?.image?.extralarge ||
                track?.album?.image?.small ||
                track?.image?.large ||
                track?.image?.small ||
                undefined,
              durationSec:
                typeof track?.duration === "number" ? track.duration : undefined,
              releaseYear:
                Number.isFinite(releaseYear) && releaseYear > 0
                  ? releaseYear
                  : undefined,
              canonicalUrl: \`https://play.qobuz.com/track/\${id}\`,
              pageUrl: albumId ? \`https://play.qobuz.com/album/\${albumId}\` : undefined,
              qualityLabel,
              isLossless: Boolean(
                track?.lossless ||
                  track?.hires ||
                  track?.hires_streamable ||
                  maximumSamplingRate >= 44.1,
              ),
              playbackUrl: \`https://play.qobuz.com/track/\${id}\`,
              playbackSurface: "browser",
              ingestMode: "desktop_capture",
              qualityMode: "best_available",
              verifiedLossless: false,
            };
          })
          .filter((item) => item && item.title);

        return {
          ok: response.ok,
          status: response.status,
          env,
          appId,
          hasToken: Boolean(token),
          count: items.length,
          items,
          firstTrackPreview: rawTracks[0]
            ? {
                id: rawTracks[0]?.id,
                url: rawTracks[0]?.url,
                relative_url: rawTracks[0]?.relative_url,
                permalink: rawTracks[0]?.permalink,
                slug: rawTracks[0]?.slug,
                albumId: rawTracks[0]?.album?.id,
                albumUrl: rawTracks[0]?.album?.url,
                albumRelativeUrl: rawTracks[0]?.album?.relative_url,
                albumSlug: rawTracks[0]?.album?.slug,
              }
            : null,
          payloadPreview: {
            hasTracksArray: Array.isArray(payload?.tracks?.items) || Array.isArray(payload?.tracks),
            keys: payload && typeof payload === "object" ? Object.keys(payload).slice(0, 12) : [],
          },
        };
      })()`,
      true,
    )) as {
      ok?: boolean;
      status?: number;
      env?: string;
      appId?: string;
      hasToken?: boolean;
      count?: number;
      items?: RemoteCatalogItem[];
      firstTrackPreview?: {
        id?: string | number;
        url?: string;
        relative_url?: string;
        permalink?: string;
        slug?: string;
        albumId?: string | number;
        albumUrl?: string;
        albumRelativeUrl?: string;
        albumSlug?: string;
      } | null;
      payloadPreview?: {
        hasTracksArray?: boolean;
        keys?: string[];
      };
    };

    log("[library] qobuz api search", {
      query,
      status: result?.status,
      ok: result?.ok,
      env: result?.env,
      appId: result?.appId,
      hasToken: result?.hasToken,
      count: result?.count,
      firstTrackPreview: result?.firstTrackPreview || null,
      payloadPreview: result?.payloadPreview || null,
    });

    const items = Array.isArray(result?.items)
      ? result.items.filter((item) => item?.title && item?.trackId)
      : [];
    remoteCatalogCache.set("qobuz", items);
    return items;
  }

  function getCachedLibraryItem(
    provider: ActiveLibraryProvider,
    trackId: string,
  ): RemoteCatalogItem | null {
    const items = remoteCatalogCache.get(provider) || [];
    return items.find((item) => item.trackId === trackId) || null;
  }

  return {
    markQobuzPlaySessionObserved,
    resetQobuzPlaySessionObserved,
    waitForDidFinishLoad,
    loadURLAndWait,
    checkLibraryProviderAuthenticated,
    scrapeProviderItems,
    searchQobuzCatalogViaApi,
    getCachedLibraryItem,
  };
}
