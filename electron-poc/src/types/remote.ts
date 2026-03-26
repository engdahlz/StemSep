export type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";
export type ActiveLibraryProvider = "spotify" | "qobuz";
export type RemoteSourceProvider = "youtube" | RemoteLibraryProvider;
export type SourceType = "local_file" | RemoteSourceProvider;

export type IngestMode =
  | "local_file"
  | "remote_download"
  | "desktop_capture";

export type PlaybackSurface = "desktop_app" | "browser" | "none";
export type CaptureQualityMode = "best_available" | "verified_lossless";

export interface PlaybackDevice {
  id: string;
  label: string;
  kind: "render_endpoint" | "system_loopback" | "display_loopback";
  isDefault?: boolean;
}

export interface CaptureEnvironmentStatus {
  windowsSupported: boolean;
  provider: "qobuz";
  authenticated: boolean;
  selectedDeviceId: string | null;
  selectedDeviceLabel: string | null;
  selectedDeviceReady: boolean;
  speakerSelectionAvailable: boolean;
  message?: string;
}

export interface RemoteCatalogItem {
  provider: RemoteLibraryProvider;
  trackId: string;
  albumId?: string;
  title: string;
  artist?: string;
  album?: string;
  albumArtist?: string;
  trackNumber?: number;
  discNumber?: number;
  releaseYear?: number;
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
  playbackUrl?: string;
  playbackUri?: string;
  playbackSurface?: PlaybackSurface;
  ingestMode?: IngestMode;
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
}

export type RemoteAuthResult =
  | {
      success: true;
      provider: RemoteLibraryProvider;
      authenticated?: boolean;
      message?: string;
    }
  | {
      success: false;
      provider: RemoteLibraryProvider;
      authenticated?: boolean;
      error: string;
      hint?: string;
    };

export type RemoteCatalogResult =
  | {
      success: true;
      provider: RemoteLibraryProvider;
      authenticated?: boolean;
      items: RemoteCatalogItem[];
    }
  | {
      success: false;
      provider: RemoteLibraryProvider;
      authenticated?: boolean;
      error: string;
      hint?: string;
    };

export type CaptureSourceMeta = {
  provider: ActiveLibraryProvider;
  providerTrackId: string;
  title: string;
  artist?: string;
  album?: string;
  artworkUrl?: string;
  durationSec?: number;
  canonicalUrl?: string;
  qualityLabel?: string;
  isLossless?: boolean;
  ingestMode: "desktop_capture";
  playbackSurface: PlaybackSurface;
  qualityMode: CaptureQualityMode;
  verifiedLossless: boolean;
};

export type RemoteResolveResult =
  | {
      success: true;
      provider: RemoteLibraryProvider;
      file_path: string;
      display_name: string;
      source_url?: string;
      canonical_url?: string;
      artist?: string;
      album?: string;
      artwork_url?: string;
      duration_sec?: number;
      quality_label?: string;
      is_lossless?: boolean;
      provider_track_id?: string;
      download_origin?: string;
    }
  | {
      success: false;
      provider: RemoteLibraryProvider;
      error: string;
      code?: string;
      hint?: string;
    };

export type PlaybackCapturePrepareResult =
  | {
      success: true;
      provider: ActiveLibraryProvider;
      trackId: string;
      displayName: string;
      launchUrl?: string;
      launchUri?: string;
      playbackSurface: PlaybackSurface;
      deviceRequired: boolean;
      qualityLabel?: string;
      qualityMode: CaptureQualityMode;
      verifiedLossless: boolean;
      estimatedDurationSec?: number;
      sourceMeta: CaptureSourceMeta;
    }
  | {
      success: false;
      provider: ActiveLibraryProvider;
      error: string;
      code?: string;
      hint?: string;
    };

export type PlaybackCaptureCompleteResult =
  | {
      success: true;
      provider: ActiveLibraryProvider;
      file_path: string;
      display_name: string;
      source_url?: string;
      canonical_url?: string;
      artist?: string;
      album?: string;
      artwork_url?: string;
      duration_sec?: number;
      quality_label?: string;
      is_lossless?: boolean;
      provider_track_id?: string;
      ingest_mode: "desktop_capture";
      playback_surface: PlaybackSurface;
      quality_mode: CaptureQualityMode;
      verified_lossless: boolean;
      capture_device_id?: string;
      capture_sample_rate?: number;
      capture_channels?: number;
      capture_bits_per_sample?: number;
      capture_sample_format?: string;
      capture_start_at?: string;
      capture_end_at?: string;
    }
  | {
      success: false;
      provider: ActiveLibraryProvider;
      error: string;
      code?: string;
      hint?: string;
    };

export interface PlaybackCaptureProgressPayload {
  provider: ActiveLibraryProvider;
  captureId?: string;
  status: string;
  detail?: string;
  progress?: number;
  percent?: string;
  elapsedSec?: number;
  remainingSec?: number;
  error?: string;
  code?: string;
  failureKind?: string;
  hint?: string;
  attempt?: number;
  maxAttempts?: number;
}

export interface RemoteResolveProgressPayload {
  provider: RemoteSourceProvider;
  status: string;
  detail?: string;
  percent?: string;
  progress?: number;
  speed?: string;
  eta?: string;
  error?: string;
}
