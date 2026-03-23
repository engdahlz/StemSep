import { app, session } from "electron";
import path from "path";

export type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";
export type ActiveLibraryProvider = "spotify" | "qobuz";
export type RemoteSourceProvider = "youtube" | RemoteLibraryProvider;

export type RemoteProviderConfig = {
  name: string;
  partition: string;
  loginUrl: string;
  libraryUrl: string;
  searchUrl?: (query: string) => string;
};

const REMOTE_PROVIDER_CONFIG: Record<RemoteLibraryProvider, RemoteProviderConfig> = {
  spotify: {
    name: "Spotify",
    partition: "persist:stemsep-spotify",
    loginUrl: "https://accounts.spotify.com/en/login",
    libraryUrl: "https://open.spotify.com/collection/tracks",
    searchUrl: (query) =>
      `https://open.spotify.com/search/${encodeURIComponent(query)}/tracks`,
  },
  qobuz: {
    name: "Qobuz",
    partition: "persist:stemsep-qobuz",
    loginUrl: "https://play.qobuz.com/login",
    libraryUrl: "https://play.qobuz.com/user/library/favorites/tracks",
    searchUrl: (query) =>
      `https://play.qobuz.com/search/tracks?searchEntry=${encodeURIComponent(query)}`,
  },
  bandcamp: {
    name: "Bandcamp",
    partition: "persist:stemsep-bandcamp",
    loginUrl: "https://bandcamp.com/login",
    libraryUrl: "https://bandcamp.com/collection",
  },
};

export function getRemoteProviderConfig(
  provider: RemoteLibraryProvider,
): RemoteProviderConfig {
  return REMOTE_PROVIDER_CONFIG[provider];
}

export function getRemoteSession(provider: RemoteLibraryProvider) {
  return session.fromPartition(getRemoteProviderConfig(provider).partition);
}

export function getRemoteCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "remote_sources");
}

export function getRemoteProviderCacheDir(provider: RemoteLibraryProvider) {
  return path.join(getRemoteCacheBaseDir(), provider);
}
