type ActiveLibraryProvider = "spotify" | "qobuz";
type PlaybackSurface = "desktop_app" | "browser" | "none";
type CaptureQualityMode = "best_available" | "verified_lossless";

type RemoteCatalogItemLike = {
  playbackSurface?: PlaybackSurface;
  playbackUrl?: string;
  canonicalUrl?: string;
  pageUrl?: string;
  playbackUri?: string;
  qualityLabel?: string;
  isLossless?: boolean;
};

export function inferPlaybackSurface(
  item: RemoteCatalogItemLike,
  provider: ActiveLibraryProvider,
): PlaybackSurface {
  if (item.playbackSurface) return item.playbackSurface;
  return provider === "spotify" ? "desktop_app" : "browser";
}

export function computeCaptureQualityMode(
  provider: ActiveLibraryProvider,
  item: RemoteCatalogItemLike,
  sampleRate?: number,
  channels?: number,
) {
  const qualityText = String(item.qualityLabel || "").toLowerCase();
  const isHiRes =
    /24\s*bit|48\s*k|88\.?2\s*k|96\s*k|176\.?4\s*k|192\s*k|hi-res/i.test(
      qualityText,
    );
  const isCdLossless = provider === "qobuz" && item.isLossless !== false && !isHiRes;
  const verified =
    isCdLossless &&
    Number(sampleRate) === 44100 &&
    Number(channels || 0) === 2;
  return {
    verified,
    qualityMode: verified
      ? ("verified_lossless" as const)
      : ("best_available" as const),
  };
}

export function resolveCaptureLaunchTarget(
  provider: ActiveLibraryProvider,
  item: RemoteCatalogItemLike,
) {
  if (provider === "spotify" && item.playbackUri) {
    return {
      launchTarget: item.playbackUri,
      launchUrl: item.playbackUrl || item.canonicalUrl,
      playbackSurface: "desktop_app" as const,
    };
  }

  return {
    launchTarget: item.playbackUrl || item.canonicalUrl || item.pageUrl,
    launchUrl: item.playbackUrl || item.canonicalUrl || item.pageUrl,
    playbackSurface: inferPlaybackSurface(item, provider),
  };
}

export type { CaptureQualityMode, PlaybackSurface };
