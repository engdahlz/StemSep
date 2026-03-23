import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import type { Session } from "electron";
import { runProcessCapture } from "../media/ffmpeg";

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";

const REMOTE_AUDIO_EXTENSIONS = new Set([
  ".wav",
  ".flac",
  ".aif",
  ".aiff",
  ".mp3",
  ".m4a",
  ".aac",
  ".ogg",
  ".opus",
  ".alac",
]);

function safeMkdir(dirPath: string) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
  } catch {
    // ignore
  }
}

function sanitizeForPathSegment(name: string) {
  return (name || "")
    .replace(/[<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80);
}

function inferRemoteFileExtension(url: string, fallback = ".flac") {
  const normalized = String(url || "").split("?")[0];
  const ext = path.extname(normalized).toLowerCase();
  if (REMOTE_AUDIO_EXTENSIONS.has(ext) || ext === ".zip") return ext;
  return fallback;
}

function findAudioFilesInDir(baseDir: string): string[] {
  const files: string[] = [];
  const walk = (dirPath: string) => {
    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(dirPath, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (REMOTE_AUDIO_EXTENSIONS.has(ext)) {
          files.push(full);
        }
      }
    }
  };
  walk(baseDir);
  return files;
}

function scoreExtractedAudioCandidate(filePath: string, title: string) {
  const base = path.basename(filePath, path.extname(filePath)).toLowerCase();
  const target = sanitizeForPathSegment(title).toLowerCase();
  if (!target) return 0;
  if (base === target) return 100;
  if (base.includes(target)) return 70;
  if (target.includes(base)) return 50;
  const targetWords = target.split(/\s+/).filter(Boolean);
  return targetWords.reduce(
    (score, word) => (base.includes(word) ? score + 10 : score),
    0,
  );
}

export function createRemoteDownloadController({
  log,
  getRemoteSession,
  getRemoteProviderCacheDir,
  getUserAgent,
  emitRemoteResolveProgress,
}: {
  log: (message: string, ...args: any[]) => void;
  getRemoteSession: (provider: RemoteLibraryProvider) => Session;
  getRemoteProviderCacheDir: (provider: RemoteLibraryProvider) => string;
  getUserAgent: () => string;
  emitRemoteResolveProgress: (payload: {
    provider: RemoteLibraryProvider;
    status: string;
    detail?: string;
    progress?: number;
    percent?: string;
  }) => void;
}) {
  async function extractArchiveForRemoteTrack(
    provider: RemoteLibraryProvider,
    archivePath: string,
    title: string,
  ): Promise<string | null> {
    const outDir = path.join(
      getRemoteProviderCacheDir(provider),
      `${path.basename(archivePath, path.extname(archivePath))}_unzipped`,
    );
    safeMkdir(outDir);

    try {
      if (process.platform === "win32") {
        await runProcessCapture("powershell", [
          "-NoProfile",
          "-Command",
          `Expand-Archive -Path '${archivePath.replace(/'/g, "''")}' -DestinationPath '${outDir.replace(/'/g, "''")}' -Force`,
        ]);
      } else {
        await runProcessCapture("unzip", ["-o", archivePath, "-d", outDir]);
      }
    } catch (error) {
      log("[remote] archive extraction failed", {
        provider,
        archivePath,
        error: (error as any)?.message || String(error),
      });
      return null;
    }

    const candidates = findAudioFilesInDir(outDir);
    if (candidates.length === 0) return null;
    candidates.sort(
      (a, b) =>
        scoreExtractedAudioCandidate(b, title) -
        scoreExtractedAudioCandidate(a, title),
    );
    return candidates[0] || null;
  }

  async function getCookieHeaderForUrl(
    provider: RemoteLibraryProvider,
    url: string,
  ): Promise<string> {
    const ses = getRemoteSession(provider);
    const cookies = await ses.cookies.get({ url });
    return cookies.map((cookie) => `${cookie.name}=${cookie.value}`).join("; ");
  }

  async function downloadRemoteFile(
    provider: RemoteLibraryProvider,
    url: string,
    title: string,
  ): Promise<string> {
    const cookieHeader = await getCookieHeaderForUrl(provider, url);
    const headers: Record<string, string> = {
      "user-agent": getUserAgent() || "Mozilla/5.0 StemSep Desktop",
    };
    if (cookieHeader) headers.cookie = cookieHeader;

    const response = await fetch(url, {
      method: "GET",
      redirect: "follow",
      headers,
    });
    if (!response.ok || !response.body) {
      throw new Error(`Download failed (${response.status} ${response.statusText})`);
    }
    const contentType = String(
      response.headers.get("content-type") || "",
    ).toLowerCase();
    if (contentType.includes("text/html")) {
      throw new Error(
        "Provider returned an HTML page instead of a downloadable audio file.",
      );
    }

    const responseUrl = response.url || url;
    const ext = inferRemoteFileExtension(responseUrl);
    const baseName =
      sanitizeForPathSegment(title || `remote_${provider}`) || `remote_${provider}`;
    const filePath = path.join(
      getRemoteProviderCacheDir(provider),
      `${baseName}_${randomUUID().slice(0, 6)}${ext}`,
    );
    safeMkdir(path.dirname(filePath));

    const totalBytes = Number(response.headers.get("content-length") || NaN);
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let receivedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        receivedBytes += value.byteLength;
        const progress =
          Number.isFinite(totalBytes) && totalBytes > 0
            ? receivedBytes / totalBytes
            : NaN;
        emitRemoteResolveProgress({
          provider,
          status: "downloading",
          detail: `Downloading ${title}...`,
          progress: Number.isFinite(progress) ? progress : undefined,
          percent: Number.isFinite(progress)
            ? `${Math.max(0, Math.min(100, Math.round(progress * 100)))}%`
            : undefined,
        });
      }
    }

    const buffer = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
    fs.writeFileSync(filePath, buffer);
    return filePath;
  }

  return {
    downloadRemoteFile,
    extractArchiveForRemoteTrack,
  };
}
