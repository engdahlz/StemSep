import { BrowserWindow } from "electron";
import { createHash } from "crypto";

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";

type RemoteCatalogItem = {
  provider: RemoteLibraryProvider;
  trackId: string;
  title: string;
  artist?: string;
  album?: string;
  artworkUrl?: string;
  durationSec?: number;
  canonicalUrl?: string;
  qualityLabel?: string;
  isLossless?: boolean;
  downloadOrigin?: string;
  pageUrl?: string;
  variantId?: string;
  downloadUrlAvailable?: boolean;
  sourceUrl?: string;
};

type RemoteProviderConfig = {
  name: string;
  partition: string;
  loginUrl: string;
  libraryUrl: string;
};

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function makeRemoteTrackId(value: string) {
  return createHash("sha1").update(String(value || "")).digest("hex").slice(0, 24);
}

function isPlaceholderRemoteCatalogItem(
  provider: RemoteLibraryProvider,
  item: Pick<
    RemoteCatalogItem,
    | "title"
    | "artist"
    | "album"
    | "artworkUrl"
    | "durationSec"
    | "variantId"
    | "canonicalUrl"
    | "pageUrl"
    | "sourceUrl"
  >,
) {
  const title = String(item.title || "").trim().toLowerCase();
  const canonical = String(
    item.canonicalUrl || item.pageUrl || item.sourceUrl || "",
  )
    .trim()
    .toLowerCase();
  const hasMetadataSignal = Boolean(
    item.artist ||
      item.album ||
      item.artworkUrl ||
      item.durationSec ||
      item.variantId,
  );

  if (!title) return true;
  if (/^download store$/i.test(title)) return true;
  if (
    provider === "qobuz" &&
    /qobuz\.com\/[^/]+-[a-z]{2}\/music\/download(?:-store|s)?/i.test(canonical)
  ) {
    return true;
  }
  if (
    provider === "bandcamp" &&
    /^music$|^merch$|^collection$/i.test(title) &&
    !hasMetadataSignal
  ) {
    return true;
  }
  return false;
}

function getScrapeCollectionScript(provider: RemoteLibraryProvider) {
  const common = String.raw`
    const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim();
    const textOf = (el) => normalizeText(el && (el.innerText || el.textContent));
    const idOf = (value) => {
      const raw = String(value || "");
      let hash = 0;
      for (let i = 0; i < raw.length; i += 1) hash = ((hash << 5) - hash) + raw.charCodeAt(i);
      return Math.abs(hash).toString(16);
    };
    const resolveDuration = (text) => {
      const match = String(text || "").match(/(\d+):(\d{2})/);
      if (!match) return undefined;
      return Number(match[1]) * 60 + Number(match[2]);
    };
  `;

  if (provider === "qobuz") {
    return `(() => {
      ${common}
      const seen = new Set();
      const items = [];
      const nodes = Array.from(document.querySelectorAll("article, li, .item, .product, .track, .album, [data-testid], .purchase"));
      for (const node of nodes) {
        const text = textOf(node);
        if (!text || !/(download|flac|wav|lossless)/i.test(text)) continue;
        const anchors = Array.from(node.querySelectorAll("a[href]"));
        const pageAnchor = anchors.find((anchor) => /qobuz\\.com/i.test(anchor.href) && !/login/i.test(anchor.href));
        const downloadAnchor = anchors.find((anchor) => /(download|\\.flac|\\.wav|\\.zip)/i.test(anchor.href));
        const title = textOf(
          node.querySelector("h1, h2, h3, h4, [class*='title'], [data-testid*='title']")
        ) || normalizeText(pageAnchor && pageAnchor.textContent);
        const artist = textOf(
          node.querySelector("[class*='artist'], [data-testid*='artist'], [class*='subtitle'], [class*='meta']")
        );
        const album = textOf(
          node.querySelector("[class*='album'], [data-testid*='album']")
        );
        const qualityLabel =
          /wav/i.test(text) ? "WAV" : /flac/i.test(text) ? "FLAC" : "Lossless";
        const canonicalUrl = (pageAnchor && pageAnchor.href) || (downloadAnchor && downloadAnchor.href) || "";
        const key = canonicalUrl || [title, artist, album].filter(Boolean).join("|");
        if (!title || !key || seen.has(key)) continue;
        seen.add(key);
        const img = node.querySelector("img");
        items.push({
          provider: "qobuz",
          trackId: idOf(key),
          title,
          artist: artist || undefined,
          album: album || undefined,
          artworkUrl: img && img.src ? img.src : undefined,
          durationSec: resolveDuration(text),
          canonicalUrl: canonicalUrl || undefined,
          qualityLabel,
          isLossless: true,
          downloadOrigin: "purchase",
          pageUrl: (pageAnchor && pageAnchor.href) || undefined,
          downloadUrlAvailable: !!downloadAnchor,
          sourceUrl: (downloadAnchor && downloadAnchor.href) || (pageAnchor && pageAnchor.href) || undefined,
        });
      }
      return items.slice(0, 250);
    })()`;
  }

  return `(() => {
    ${common}
    const seen = new Set();
    const items = [];
    const attr = document.querySelector("[data-client-items]")?.getAttribute("data-client-items");
    if (attr) {
      try {
        const parsed = JSON.parse(attr);
        for (const item of Array.isArray(parsed) ? parsed : []) {
          const key = item.item_url || item.tralbum_url || item.url || item.title;
          if (!key || seen.has(key)) continue;
          seen.add(key);
          items.push({
            provider: "bandcamp",
            trackId: idOf(key),
            title: normalizeText(item.item_title || item.title),
            artist: normalizeText(item.band_name || item.artist),
            album: normalizeText(item.album_title || item.package_title),
            artworkUrl: item.item_art_url || item.art_url || undefined,
            canonicalUrl: item.item_url || item.tralbum_url || item.url || undefined,
            qualityLabel: "Lossless Download",
            isLossless: true,
            downloadOrigin: "purchase",
            pageUrl: item.item_url || item.tralbum_url || item.url || undefined,
            downloadUrlAvailable: false,
            sourceUrl: item.item_url || item.tralbum_url || item.url || undefined,
          });
        }
      } catch {}
    }

    const nodes = Array.from(document.querySelectorAll("li, article, .collection-item, [data-itemid], .trackTitle"));
    for (const node of nodes) {
      const text = textOf(node);
      if (!text) continue;
      const anchors = Array.from(node.querySelectorAll("a[href]"));
      const pageAnchor = anchors.find((anchor) => /bandcamp\\.com/i.test(anchor.href));
      const title = textOf(
        node.querySelector(".collection-item-title, .trackTitle, h1, h2, h3, [class*='title']")
      ) || normalizeText(pageAnchor && pageAnchor.textContent);
      const artist = textOf(
        node.querySelector(".collection-item-artist, [class*='artist'], .subhead")
      );
      const album = textOf(
        node.querySelector(".collection-item-album, [class*='album']")
      );
      const key = (pageAnchor && pageAnchor.href) || [title, artist, album].filter(Boolean).join("|");
      if (!title || !key || seen.has(key)) continue;
      seen.add(key);
      const img = node.querySelector("img");
      items.push({
        provider: "bandcamp",
        trackId: idOf(key),
        title,
        artist: artist || undefined,
        album: album || undefined,
        artworkUrl: img && img.src ? img.src : undefined,
        canonicalUrl: (pageAnchor && pageAnchor.href) || undefined,
        qualityLabel: "Lossless Download",
        isLossless: true,
        downloadOrigin: "purchase",
        pageUrl: (pageAnchor && pageAnchor.href) || undefined,
        downloadUrlAvailable: false,
        sourceUrl: (pageAnchor && pageAnchor.href) || undefined,
      });
    }
    return items.slice(0, 250);
  })()`;
}

function getResolveTrackScript(provider: RemoteLibraryProvider) {
  if (provider === "qobuz") {
    return `(() => {
      const normalizeText = (value) => String(value || "").replace(/\\s+/g, " ").trim();
      const anchors = Array.from(document.querySelectorAll("a[href]"));
      const downloadAnchor = anchors.find((anchor) => /(download|\\.flac|\\.wav|\\.zip)/i.test(anchor.href));
      const title = normalizeText(
        document.querySelector("h1, h2, [class*='title'], [data-testid*='title']")?.textContent
      );
      const artist = normalizeText(
        document.querySelector("[class*='artist'], [data-testid*='artist'], [class*='subtitle']")?.textContent
      );
      const album = normalizeText(
        document.querySelector("[class*='album'], [data-testid*='album']")?.textContent
      );
      const artwork = document.querySelector("img")?.src;
      const text = normalizeText(document.body?.innerText);
      const qualityLabel = /wav/i.test(text) ? "WAV" : /flac/i.test(text) ? "FLAC" : "Lossless";
      return {
        downloadUrl: downloadAnchor?.href,
        title: title || undefined,
        artist: artist || undefined,
        album: album || undefined,
        artworkUrl: artwork || undefined,
        qualityLabel,
        canonicalUrl: location.href,
      };
    })()`;
  }

  return `(() => {
    const normalizeText = (value) => String(value || "").replace(/\\s+/g, " ").trim();
    const anchors = Array.from(document.querySelectorAll("a[href]"));
    const downloadAnchor = anchors.find((anchor) => /download/i.test(anchor.href) || /download/i.test(anchor.textContent || ""));
    const title = normalizeText(
      document.querySelector("h1, h2, [class*='title'], .trackTitle")?.textContent
    );
    const artist = normalizeText(
      document.querySelector("[class*='artist'], .subhead, .fromAlbum")?.textContent
    );
    const artwork = document.querySelector("img")?.src;
    return {
      downloadUrl: downloadAnchor?.href,
      title: title || undefined,
      artist: artist || undefined,
      artworkUrl: artwork || undefined,
      qualityLabel: "Lossless Download",
      canonicalUrl: location.href,
    };
  })()`;
}

export function createRemoteBrowserLibraryController({
  getRemoteProviderConfig,
  getExistingAuthWindow,
  getMainWindow,
  log,
  remoteCatalogCache,
}: {
  getRemoteProviderConfig: (provider: RemoteLibraryProvider) => RemoteProviderConfig;
  getExistingAuthWindow: (provider: RemoteLibraryProvider) => BrowserWindow | null;
  getMainWindow: () => BrowserWindow | null;
  log: (message: string, ...args: any[]) => void;
  remoteCatalogCache: Map<string, any[]>;
}) {
  async function loadRemoteBrowserWindow(
    provider: RemoteLibraryProvider,
    url: string,
    options: { visible?: boolean; reuseAuthWindow?: boolean } = {},
  ): Promise<{ win: BrowserWindow; disposable: boolean }> {
    const { visible = false, reuseAuthWindow = false } = options;
    const existing = reuseAuthWindow ? getExistingAuthWindow(provider) : null;
    if (existing && !existing.isDestroyed()) {
      await existing.loadURL(url);
      await sleep(1800);
      return { win: existing, disposable: false };
    }

    const win = new BrowserWindow({
      width: visible ? 1180 : 1024,
      height: visible ? 860 : 768,
      show: visible,
      title: `${getRemoteProviderConfig(provider).name} Browser`,
      parent: visible ? getMainWindow() || undefined : undefined,
      modal: false,
      webPreferences: {
        partition: getRemoteProviderConfig(provider).partition,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
      },
    });

    await win.loadURL(url);
    await sleep(1800);
    return { win, disposable: !visible };
  }

  async function scrapeRemoteCollection(
    provider: RemoteLibraryProvider,
  ): Promise<RemoteCatalogItem[]> {
    const { win, disposable } = await loadRemoteBrowserWindow(
      provider,
      getRemoteProviderConfig(provider).libraryUrl,
    );
    try {
      const rawItems = (await win.webContents.executeJavaScript(
        getScrapeCollectionScript(provider),
        true,
      )) as RemoteCatalogItem[];

      const normalized = (Array.isArray(rawItems) ? rawItems : [])
        .map((item) => ({
          provider,
          trackId:
            typeof item?.trackId === "string" && item.trackId.trim()
              ? item.trackId
              : makeRemoteTrackId(
                  `${item?.canonicalUrl || item?.pageUrl || item?.title || ""}|${item?.artist || ""}|${item?.album || ""}`,
                ),
          title: String(item?.title || "").trim(),
          artist: String(item?.artist || "").trim() || undefined,
          album: String(item?.album || "").trim() || undefined,
          artworkUrl: String(item?.artworkUrl || "").trim() || undefined,
          durationSec:
            typeof item?.durationSec === "number" && Number.isFinite(item.durationSec)
              ? item.durationSec
              : undefined,
          canonicalUrl: String(item?.canonicalUrl || "").trim() || undefined,
          qualityLabel: String(item?.qualityLabel || "").trim() || undefined,
          isLossless: item?.isLossless !== false,
          downloadOrigin: String(item?.downloadOrigin || "purchase").trim(),
          pageUrl: String(item?.pageUrl || "").trim() || undefined,
          variantId: String(item?.variantId || "").trim() || undefined,
          downloadUrlAvailable: Boolean(item?.downloadUrlAvailable),
          sourceUrl: String(item?.sourceUrl || "").trim() || undefined,
        }))
        .filter(
          (item) => item.title && !isPlaceholderRemoteCatalogItem(provider, item),
        );

      remoteCatalogCache.set(provider, normalized);
      return normalized;
    } finally {
      if (disposable) {
        try {
          win.destroy();
        } catch {
          // ignore
        }
      }
    }
  }

  async function resolveRemoteDownloadInfo(
    provider: RemoteLibraryProvider,
    pageUrl: string,
  ): Promise<{
    downloadUrl?: string;
    title?: string;
    artist?: string;
    album?: string;
    artworkUrl?: string;
    qualityLabel?: string;
    canonicalUrl?: string;
  }> {
    const { win, disposable } = await loadRemoteBrowserWindow(provider, pageUrl);
    try {
      const result = (await win.webContents.executeJavaScript(
        getResolveTrackScript(provider),
        true,
      )) as Record<string, any>;
      return {
        downloadUrl:
          typeof result?.downloadUrl === "string" && result.downloadUrl.trim()
            ? result.downloadUrl
            : undefined,
        title:
          typeof result?.title === "string" && result.title.trim()
            ? result.title.trim()
            : undefined,
        artist:
          typeof result?.artist === "string" && result.artist.trim()
            ? result.artist.trim()
            : undefined,
        album:
          typeof result?.album === "string" && result.album.trim()
            ? result.album.trim()
            : undefined,
        artworkUrl:
          typeof result?.artworkUrl === "string" && result.artworkUrl.trim()
            ? result.artworkUrl.trim()
            : undefined,
        qualityLabel:
          typeof result?.qualityLabel === "string" && result.qualityLabel.trim()
            ? result.qualityLabel.trim()
            : undefined,
        canonicalUrl:
          typeof result?.canonicalUrl === "string" && result.canonicalUrl.trim()
            ? result.canonicalUrl.trim()
            : undefined,
      };
    } finally {
      if (disposable) {
        try {
          win.destroy();
        } catch (error) {
          log("[remote] failed to destroy temporary browser window", {
            provider,
            error: (error as any)?.message || String(error),
          });
        }
      }
    }
  }

  return {
    scrapeRemoteCollection,
    resolveRemoteDownloadInfo,
  };
}
