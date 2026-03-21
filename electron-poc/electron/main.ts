import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
  protocol,
  Menu,
  screen,
  session,
  desktopCapturer,
} from "electron";
import path from "path";
import { spawn } from "child_process";
import fs from "fs";
import { randomUUID, createHash } from "crypto";
import chokidar, { FSWatcher } from "chokidar";
// const __filename = fileURLToPath(import.meta.url)
// const __dirname = path.dirname(__filename)

// Simple file logger
const getLogPath = () => path.join(app.getPath("userData"), "app.log");
const getBackendStdioLogPath = () =>
  path.join(app.getPath("userData"), "backend-stdio.log");

function rotateLogIfLarge(filePath: string, maxBytes: number) {
  try {
    if (!fs.existsSync(filePath)) return;
    const st = fs.statSync(filePath);
    if (!st.isFile()) return;
    if (st.size <= maxBytes) return;

    const dir = path.dirname(filePath);
    const base = path.basename(filePath, path.extname(filePath));
    const ext = path.extname(filePath) || ".log";
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const rotated = path.join(dir, `${base}.${stamp}${ext}`);
    fs.renameSync(filePath, rotated);
  } catch {
    // ignore
  }
}

function appendBackendStdio(prefix: string, chunk: any) {
  try {
    rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
    const ts = new Date().toISOString();
    const text = typeof chunk === "string" ? chunk : chunk?.toString?.("utf-8") ?? String(chunk);
    fs.appendFileSync(getBackendStdioLogPath(), `${ts} ${prefix} ${text}${text.endsWith("\n") ? "" : "\n"}`);
  } catch {
    // ignore
  }
}

function log(message: string, ...args: any[]) {
  const timestamp = new Date().toISOString();
  const formattedMessage = `${timestamp} - ${message} ${args.length > 0 ? JSON.stringify(args) : ""}\n`;

  // Console log
  console.log(message, ...args);

  // File log
  try {
    fs.appendFileSync(getLogPath(), formattedMessage);
  } catch (e) {
    console.error("Failed to write to log file:", e);
  }
}

function attachBackendStdioLogging(label: string) {
  if (!pythonBridge) return;
  // Make sure the file exists/rotated early.
  try {
    safeMkdir(path.dirname(getBackendStdioLogPath()));
    rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
  } catch {
    // ignore
  }

  pythonBridge.stdout?.on("data", (data) => {
    appendBackendStdio(`[${label} stdout]`, data);
  });

  pythonBridge.stderr?.on("data", (data) => {
    appendBackendStdio(`[${label} stderr]`, data);
  });
}

function safeMkdir(dirPath: string) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
  } catch {
    // ignore
  }
}

const PREVIEW_CACHE_KEEP_LAST = 20;
const PREVIEW_CACHE_MAX_AGE_DAYS = 7;
const SYSTEM_RUNTIME_TTL_MS = 30_000;

function sanitizeForPathSegment(name: string) {
  return (name || "")
    .replace(/[<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80);
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

type AudioSourceProfile = {
  path: string;
  container: string | null;
  codec: string | null;
  codecLongName: string | null;
  sampleRate: number | null;
  channels: number | null;
  sampleFormat: string | null;
  bitDepth: number | null;
  durationSeconds: number | null;
  isLossless: boolean;
};

type SourceStagingDecision = {
  sourcePath: string;
  workingPath: string;
  sourceExt: string;
  copiedDirectly: boolean;
  workingCodec: "original" | "pcm_s16le" | "pcm_s24le" | "pcm_f32le";
  reason: string;
};

type StagedInputInfo = {
  effectiveInputFile: string;
  sourceAudioProfile: AudioSourceProfile;
  stagingDecision: SourceStagingDecision;
};

type ExportMode = "copy" | "transcode";

type ExportTask = {
  stemRaw: string;
  stem: string;
  sourceFile: string;
  sourceProfile: AudioSourceProfile;
  outFile: string;
  format: "wav" | "flac" | "mp3";
  bitrate: string;
  mode: ExportMode;
  estimatedOutputBytes: number;
};

type ExportProgressPayload = {
  requestId: string;
  status: "preflight" | "copying" | "transcoding" | "completed" | "failed";
  stem?: string;
  fileIndex?: number;
  fileCount?: number;
  fileProgress?: number;
  totalProgress?: number;
  detail?: string;
  format?: string;
  outputPath?: string;
  error?: string;
};

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";
type ActiveLibraryProvider = "spotify" | "qobuz";
type RemoteSourceProvider = "youtube" | RemoteLibraryProvider;
type IngestMode = "local_file" | "remote_download" | "desktop_capture";
type PlaybackSurface = "desktop_app" | "browser" | "none";
type CaptureQualityMode = "best_available" | "verified_lossless";

type RemoteResolveProgressPayload = {
  provider: RemoteSourceProvider;
  status: string;
  detail?: string;
  percent?: string;
  progress?: number;
  speed?: string;
  eta?: string;
  error?: string;
};

type RemoteCatalogItem = {
  provider: RemoteLibraryProvider;
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
  variantId?: string;
  downloadUrlAvailable?: boolean;
  sourceUrl?: string;
  playbackUrl?: string;
  playbackUri?: string;
  playbackSurface?: PlaybackSurface;
  ingestMode?: IngestMode;
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
};

type PlaybackDevice = {
  id: string;
  label: string;
  kind: "render_endpoint" | "system_loopback" | "display_loopback";
  isDefault?: boolean;
};

type CaptureEnvironmentStatus = {
  windowsSupported: boolean;
  provider: "qobuz";
  authenticated: boolean;
  selectedDeviceId: string | null;
  selectedDeviceLabel: string | null;
  selectedDeviceReady: boolean;
  speakerSelectionAvailable: boolean;
  message?: string;
};

type PlaybackCaptureProgressPayload = {
  provider: ActiveLibraryProvider;
  captureId?: string;
  status: string;
  detail?: string;
  progress?: number;
  percent?: string;
  elapsedSec?: number;
  remainingSec?: number;
  error?: string;
};

type RemoteProviderConfig = {
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

function getFfmpegExe(): string {
  // Prefer a bundled ffmpeg binary when available.
  // 1) build-electron.mjs copies ffmpeg next to dist-electron/main.js
  // 2) ffmpeg-static (node module)
  // 3) ffmpeg on PATH
  try {
    const local = path.join(
      __dirname,
      process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg",
    );
    if (fs.existsSync(local)) return local;
  } catch {
    // ignore
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ffmpegStaticPath = require("ffmpeg-static");
    if (typeof ffmpegStaticPath === "string" && ffmpegStaticPath) {
      return ffmpegStaticPath;
    }
  } catch {
    // ignore
  }

  return "ffmpeg";
}

function getFfprobeExe(): string {
  try {
    const local = path.join(
      __dirname,
      process.platform === "win32" ? "ffprobe.exe" : "ffprobe",
    );
    if (fs.existsSync(local)) return local;
  } catch {
    // ignore
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ffmpegStaticPath = require("ffmpeg-static");
    if (typeof ffmpegStaticPath === "string" && ffmpegStaticPath) {
      const probePath = ffmpegStaticPath.replace(/ffmpeg(?:\.exe)?$/i, process.platform === "win32" ? "ffprobe.exe" : "ffprobe");
      if (fs.existsSync(probePath)) return probePath;
    }
  } catch {
    // ignore
  }

  return "ffprobe";
}

async function runProcessCapture(
  exe: string,
  args: string[],
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, { windowsHide: true });
    let stdout = "";
    let stderr = "";

    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (err) => {
      reject(
        new Error(
          `Failed to start ${exe}: ${err?.message || String(err)}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${exe} failed (exit=${code}). ${stderr}`.trim()));
    });
  });
}

async function ensureBinaryAvailable(exe: string, label: string): Promise<void> {
  try {
    await runProcessCapture(exe, ["-version"]);
  } catch (error: any) {
    throw new Error(
      `${label} is required but unavailable: ${error?.message || String(error)}`,
    );
  }
}

async function runFfmpeg(args: string[]): Promise<void> {
  const exe = getFfmpegExe();

  await new Promise<void>((resolve, reject) => {
    const child = spawn(exe, args, { windowsHide: true });
    let stderr = "";
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      reject(
        new Error(
          `Failed to start ffmpeg (${exe}): ${err?.message || String(err)}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (exit=${code}). ${stderr}`.trim()));
    });
  });
}

async function runFfmpegWithProgress(
  args: string[],
  durationSeconds: number | null,
  onProgress?: (progress: number) => void,
): Promise<void> {
  const exe = getFfmpegExe();

  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      exe,
      [
        ...args,
        "-progress",
        "pipe:1",
        "-nostats",
      ],
      { windowsHide: true },
    );
    let stderr = "";
    let progressBuffer = "";

    const reportProgress = (raw: number) => {
      if (!onProgress) return;
      const clamped = Math.max(0, Math.min(1, raw));
      onProgress(clamped);
    };

    child.stdout?.on("data", (chunk) => {
      progressBuffer += String(chunk);
      const lines = progressBuffer.split(/\r?\n/);
      progressBuffer = lines.pop() || "";

      for (const line of lines) {
        const [key, value] = line.split("=");
        if (!key || value == null) continue;
        if (key === "out_time_ms" && durationSeconds && durationSeconds > 0) {
          const outTimeMs = Number(value);
          if (Number.isFinite(outTimeMs)) {
            reportProgress(outTimeMs / (durationSeconds * 1_000_000));
          }
        } else if (key === "progress" && value === "end") {
          reportProgress(1);
        }
      }
    });
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      reject(
        new Error(
          `Failed to start ffmpeg (${exe}): ${err?.message || String(err)}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (exit=${code}). ${stderr}`.trim()));
    });
  });
}

function extractAudioBitDepth(stream: Record<string, any>): number | null {
  const direct = Number(
    stream.bits_per_raw_sample ?? stream.bits_per_sample ?? NaN,
  );
  if (Number.isFinite(direct) && direct > 0) return direct;

  const sampleFmt = String(stream.sample_fmt || "");
  const match = sampleFmt.match(/(\d+)/);
  if (match) {
    const parsed = Number(match[1]);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }

  return null;
}

function inferLossless(codec: string | null, container: string | null): boolean {
  const normalizedCodec = String(codec || "").toLowerCase();
  const normalizedContainer = String(container || "").toLowerCase();
  if (!normalizedCodec && !normalizedContainer) return false;
  if (normalizedCodec.startsWith("pcm_")) return true;

  const knownLossless = new Set([
    "flac",
    "alac",
    "wavpack",
    "ape",
    "tta",
    "mlp",
    "truehd",
  ]);
  if (knownLossless.has(normalizedCodec)) return true;

  if (normalizedContainer.includes("wav") && normalizedCodec.startsWith("pcm")) {
    return true;
  }

  return false;
}

async function probeAudioFile(filePath: string): Promise<AudioSourceProfile> {
  const exe = getFfprobeExe();
  const { stdout } = await runProcessCapture(exe, [
    "-v",
    "error",
    "-show_streams",
    "-show_format",
    "-of",
    "json",
    filePath,
  ]);

  const parsed = JSON.parse(stdout || "{}") as {
    streams?: Array<Record<string, any>>;
    format?: Record<string, any>;
  };
  const audioStream =
    parsed.streams?.find((stream) => String(stream.codec_type) === "audio") || {};
  const format = parsed.format || {};

  const sampleRate = Number(audioStream.sample_rate ?? NaN);
  const channels = Number(audioStream.channels ?? NaN);
  const durationSeconds = Number(audioStream.duration ?? format.duration ?? NaN);
  const bitDepth = extractAudioBitDepth(audioStream);
  const container = typeof format.format_name === "string" ? format.format_name : null;
  const codec = typeof audioStream.codec_name === "string" ? audioStream.codec_name : null;
  const codecLongName =
    typeof audioStream.codec_long_name === "string"
      ? audioStream.codec_long_name
      : null;
  const sampleFormat =
    typeof audioStream.sample_fmt === "string" ? audioStream.sample_fmt : null;

  return {
    path: filePath,
    container,
    codec,
    codecLongName,
    sampleRate: Number.isFinite(sampleRate) ? sampleRate : null,
    channels: Number.isFinite(channels) ? channels : null,
    sampleFormat,
    bitDepth,
    durationSeconds: Number.isFinite(durationSeconds) ? durationSeconds : null,
    isLossless: inferLossless(codec, container),
  };
}

function chooseLosslessWorkingCodec(
  profile: AudioSourceProfile,
): "pcm_s16le" | "pcm_s24le" | "pcm_f32le" {
  const sampleFormat = String(profile.sampleFormat || "").toLowerCase();
  const bitDepth = profile.bitDepth ?? 16;

  if (sampleFormat.includes("flt") || sampleFormat.includes("dbl")) {
    return "pcm_f32le";
  }
  if (bitDepth > 24) {
    return "pcm_f32le";
  }
  if (bitDepth > 16) {
    return "pcm_s24le";
  }
  return "pcm_s16le";
}

async function ensureWavInput(
  inputFile: string,
  previewDir: string,
): Promise<StagedInputInfo> {
  const ext = path.extname(inputFile || "").toLowerCase();
  const sourceAudioProfile = await probeAudioFile(inputFile);

  if (ext === ".wav") {
    return {
      effectiveInputFile: inputFile,
      sourceAudioProfile,
      stagingDecision: {
        sourcePath: inputFile,
        workingPath: inputFile,
        sourceExt: ext,
        copiedDirectly: true,
        workingCodec: "original",
        reason: "Source is already WAV; no staging conversion was required.",
      },
    };
  }

  const workingCodec = chooseLosslessWorkingCodec(sourceAudioProfile);
  const staged = path.join(previewDir, `input_${shortHash(inputFile)}.wav`);
  if (!fs.existsSync(staged)) {
    await runFfmpeg([
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      inputFile,
      "-vn",
      "-map_metadata",
      "0",
      "-c:a",
      workingCodec,
      staged,
    ]);
  }

  if (!fs.existsSync(staged)) {
    throw new Error("Decoded WAV was not created.");
  }

  return {
    effectiveInputFile: staged,
    sourceAudioProfile,
    stagingDecision: {
      sourcePath: inputFile,
      workingPath: staged,
      sourceExt: ext,
      copiedDirectly: false,
      workingCodec,
      reason:
        sourceAudioProfile.isLossless && workingCodec !== "pcm_s16le"
          ? `Lossless source preserved with ${workingCodec} staging instead of 16-bit truncation.`
          : `Decoded to WAV with ${workingCodec} for backend compatibility.`,
    },
  };
}

function emitExportProgress(payload: ExportProgressPayload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("export-progress", payload);
}

const remoteAuthWindows = new Map<RemoteLibraryProvider, BrowserWindow>();
let qobuzAutomationWindow: BrowserWindow | null = null;
const remoteCatalogCache = new Map<RemoteLibraryProvider, RemoteCatalogItem[]>();
const qobuzSinkIdByPlaybackDeviceId = new Map<string, string>();
let qobuzObservedPlaySession = false;
const cancelledPlaybackCaptureIds = new Set<string>();
const lastPlaybackCaptureProgressById = new Map<
  string,
  PlaybackCaptureProgressPayload
>();
let cachedNativePlaybackDevices: PlaybackDevice[] = [];
let cachedQobuzSpeakerSelectionAvailable = false;
let cachedQobuzSpeakerSelectionError: string | null = null;
type QobuzPlaybackVerification = {
  targetTrackId?: string;
  targetAlbumId?: string;
  targetTrackNumber?: number | null;
  initialPath?: string;
  finalPath?: string;
  clickStrategy?: string;
  clickedLabel?: string;
  clickedHtml?: string;
  clickedScopeText?: string;
  candidateLabels?: string[];
  candidateTrackRows?: Array<{
    title?: string;
    artist?: string;
    trackNumber?: number | null;
    trackId?: string;
    albumId?: string;
    href?: string;
    rowState?: string;
  }>;
  verificationMatched?: boolean;
  verificationReason?: string;
  currentTitle?: string;
  currentArtist?: string;
  currentTrackHref?: string;
  currentAlbumHref?: string;
  currentTrackId?: string;
  currentAlbumId?: string;
  rowMatched?: boolean;
  rowState?: string;
  activeMedia?: boolean;
  playButtonClicked?: boolean;
  mediaCount?: number;
  sinkApplied?: number;
  sinkError?: string;
  speakerSelectionSupported?: boolean;
};

type PlaybackCaptureSession = {
  captureId: string;
  provider: ActiveLibraryProvider;
  trackId: string;
  deviceId: string;
  item: RemoteCatalogItem;
  startedAt: string;
  outputPath: string;
  diagnosticsPath: string;
  backendCaptureStarted: boolean;
  verification?: QobuzPlaybackVerification | null;
};

const playbackCaptureSessions = new Map<string, PlaybackCaptureSession>();

function emitRemoteResolveProgress(payload: RemoteResolveProgressPayload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("remote-resolve-progress", payload);
}

function emitPlaybackCaptureProgress(payload: PlaybackCaptureProgressPayload) {
  if (payload.captureId) {
    lastPlaybackCaptureProgressById.set(payload.captureId, payload);
    if (
      /completed|cancelled|error|failed/i.test(String(payload.status || ""))
    ) {
      setTimeout(() => {
        lastPlaybackCaptureProgressById.delete(payload.captureId!);
      }, 60_000);
    }
  }
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("playback-capture-progress", payload);
}

function isPlaybackCaptureActive() {
  return playbackCaptureSessions.size > 0;
}

function isQobuzPlaybackCaptureActive() {
  return Array.from(playbackCaptureSessions.values()).some(
    (session) => session.provider === "qobuz",
  );
}

async function abortPlaybackCaptureSession(
  captureId?: string,
  detail = "Capture cancelled.",
) {
  const targetedSessions = captureId
    ? Array.from(playbackCaptureSessions.values()).filter(
        (session) => session.captureId === captureId,
      )
    : Array.from(playbackCaptureSessions.values());

  for (const session of targetedSessions) {
    cancelledPlaybackCaptureIds.add(session.captureId);
    emitPlaybackCaptureProgress({
      provider: session.provider,
      captureId: session.captureId,
      status: "cancelling",
      detail,
    });
  }

  await stopHiddenQobuzPlayback();

  if (targetedSessions.length > 0) {
    for (const session of targetedSessions) {
      if (!session.backendCaptureStarted) {
        try {
          if (fs.existsSync(session.outputPath)) {
            fs.unlinkSync(session.outputPath);
          }
        } catch {
          // ignore partial cleanup failures before backend capture begins
        }
      }
    }
    await sendPythonCommandWithRetry(
      "cancel_playback_capture",
      captureId ? { capture_id: captureId } : {},
      10_000,
      0,
    ).catch(() => null);
    return { success: true };
  }

  await sendPythonCommandWithRetry(
    "cancel_playback_capture",
    captureId ? { capture_id: captureId } : {},
    10_000,
    0,
  ).catch(() => null);
  return { success: true };
}

function getRemoteProviderConfig(provider: RemoteLibraryProvider): RemoteProviderConfig {
  return REMOTE_PROVIDER_CONFIG[provider];
}

function getRemoteSession(provider: RemoteLibraryProvider) {
  return session.fromPartition(getRemoteProviderConfig(provider).partition);
}

function getRemoteCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "remote_sources");
}

function getRemoteProviderCacheDir(provider: RemoteLibraryProvider) {
  return path.join(getRemoteCacheBaseDir(), provider);
}

function getPlaybackCaptureCacheDir() {
  return path.join(app.getPath("userData"), "cache", "playback_capture");
}

function getStoredCaptureOutputDeviceId(): string | null {
  const value = readAppConfig().captureOutputDeviceId;
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function normalizeDeviceLabel(label: string | undefined | null) {
  return String(label || "")
    .toLowerCase()
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function getNativePlaybackDevices(): Promise<PlaybackDevice[]> {
  if (isPlaybackCaptureActive() && cachedNativePlaybackDevices.length > 0) {
    return cachedNativePlaybackDevices;
  }
  const devices = await sendPythonCommandWithRetry(
    "detect_playback_devices",
    {},
    20_000,
    1,
  );
  const normalized = Array.isArray(devices)
    ? devices.filter((device) => device?.id && device?.label)
    : [];
  cachedNativePlaybackDevices = normalized;
  return normalized;
}

async function getQobuzBrowserOutputDevices() {
  const config = getRemoteProviderConfig("qobuz");
  const win = getQobuzAutomationWindow();
  const currentUrl = win.webContents.getURL();
  if (!currentUrl || !/qobuz\.com/i.test(currentUrl)) {
    await loadURLAndWait(win, config.libraryUrl, 20_000);
    await sleep(900);
  }

  const result = (await win.webContents.executeJavaScript(
    `(() => {
      const supportsSetSinkId =
        typeof HTMLMediaElement !== "undefined" &&
        typeof HTMLMediaElement.prototype.setSinkId === "function";
      const enumerate = navigator.mediaDevices?.enumerateDevices;
      if (!enumerate) {
        return {
          supportsSetSinkId,
          devices: [],
        };
      }
      return navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => ({
          supportsSetSinkId,
          devices: devices
            .filter((device) => device.kind === "audiooutput")
            .map((device) => ({
              sinkId: device.deviceId,
              label: device.label || "",
            })),
        }))
        .catch((error) => ({
          supportsSetSinkId,
          devices: [],
          error: String(error?.message || error || "Failed to enumerate audio outputs."),
        }));
    })()`,
    true,
  )) as {
    supportsSetSinkId?: boolean;
    devices?: Array<{ sinkId: string; label: string }>;
    error?: string;
  };

  return {
    supportsSetSinkId: !!result?.supportsSetSinkId,
    devices: Array.isArray(result?.devices) ? result.devices : [],
    error: result?.error || null,
  };
}

async function refreshQobuzPlaybackDeviceMappings() {
  const nativeDevices = await getNativePlaybackDevices();
  qobuzSinkIdByPlaybackDeviceId.clear();

  try {
    const browserOutput = await getQobuzBrowserOutputDevices();
    const browserDevices = browserOutput.devices || [];
    const defaultBrowser = browserDevices.find(
      (device) => device.sinkId === "default",
    );

    for (const nativeDevice of nativeDevices) {
      const normalizedNative = normalizeDeviceLabel(nativeDevice.label);
      let match =
        browserDevices.find(
          (device) =>
            !!device.label &&
            normalizeDeviceLabel(device.label) === normalizedNative,
        ) ||
        browserDevices.find((device) => {
          const normalizedBrowser = normalizeDeviceLabel(device.label);
          return (
            !!normalizedNative &&
            !!normalizedBrowser &&
            (normalizedBrowser.includes(normalizedNative) ||
              normalizedNative.includes(normalizedBrowser))
          );
        });

      if (!match && nativeDevice.isDefault && defaultBrowser) {
        match = defaultBrowser;
      }

      if (match?.sinkId) {
        qobuzSinkIdByPlaybackDeviceId.set(nativeDevice.id, match.sinkId);
      }
    }

    cachedQobuzSpeakerSelectionAvailable =
      browserOutput.supportsSetSinkId &&
      qobuzSinkIdByPlaybackDeviceId.size > 0;
    cachedQobuzSpeakerSelectionError = browserOutput.error || null;
    return {
      devices: nativeDevices,
      speakerSelectionAvailable: cachedQobuzSpeakerSelectionAvailable,
      speakerSelectionError: cachedQobuzSpeakerSelectionError,
    };
  } catch (error: any) {
    cachedQobuzSpeakerSelectionAvailable = false;
    cachedQobuzSpeakerSelectionError = error?.message || String(error);
    return {
      devices: nativeDevices,
      speakerSelectionAvailable: false,
      speakerSelectionError: cachedQobuzSpeakerSelectionError,
    };
  }
}

async function stopHiddenQobuzPlayback() {
  const win = qobuzAutomationWindow;
  if (!win || win.isDestroyed()) return;

  try {
    await win.webContents.executeJavaScript(
      `(() => {
        const normalize = (value) =>
          String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
        const isVisible = (element) => {
          if (!(element instanceof HTMLElement)) return false;
          const style = window.getComputedStyle(element);
          return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            Number(style.opacity || "1") > 0 &&
            !element.hasAttribute("disabled")
          );
        };

        for (const media of Array.from(document.querySelectorAll("audio,video"))) {
          try {
            media.pause();
          } catch {
            // ignore
          }
        }

        const pauseButton = Array.from(
          document.querySelectorAll("button, [role='button']"),
        ).find((element) => {
          const label = normalize(
            element?.getAttribute?.("aria-label") ||
              element?.getAttribute?.("title") ||
              element?.textContent,
          );
          return (
            isVisible(element) &&
            /\\b(pause|stop|paus|anhalten|arrêter|pause playback)\\b/i.test(label)
          );
        });

        if (pauseButton instanceof HTMLElement) {
          pauseButton.click();
          return true;
        }

        return false;
      })()`,
      true,
    );
  } catch {
    // ignore best-effort pause failures
  }
}

async function getCaptureEnvironmentStatusForQobuz(): Promise<CaptureEnvironmentStatus> {
  const windowsSupported = process.platform === "win32";
  const authenticated = isQobuzPlaybackCaptureActive()
    ? true
    : await checkLibraryProviderAuthenticated("qobuz");
  const selectedDeviceId = getStoredCaptureOutputDeviceId();

  if (!windowsSupported) {
    return {
      windowsSupported,
      provider: "qobuz",
      authenticated,
      selectedDeviceId,
      selectedDeviceLabel: null,
      selectedDeviceReady: false,
      speakerSelectionAvailable: false,
      message: "Hidden Qobuz lossless capture is only available on Windows.",
    };
  }

  const deviceState = isPlaybackCaptureActive()
    ? {
        devices: cachedNativePlaybackDevices,
        speakerSelectionAvailable: cachedQobuzSpeakerSelectionAvailable,
        speakerSelectionError: cachedQobuzSpeakerSelectionError,
      }
    : await refreshQobuzPlaybackDeviceMappings();
  const selectedDevice =
    deviceState.devices.find((device) => device.id === selectedDeviceId) || null;
  const selectedDeviceReady = !!(
    selectedDeviceId &&
    selectedDevice &&
    qobuzSinkIdByPlaybackDeviceId.has(selectedDeviceId)
  );

  return {
    windowsSupported,
    provider: "qobuz",
    authenticated,
    selectedDeviceId,
    selectedDeviceLabel: selectedDevice?.label || null,
    selectedDeviceReady,
    speakerSelectionAvailable: deviceState.speakerSelectionAvailable,
    message:
      deviceState.speakerSelectionError ||
      (selectedDeviceId && !selectedDeviceReady
        ? "The saved silent output could not be routed from the hidden Qobuz player."
        : undefined),
  };
}

function makeRemoteTrackId(value: string) {
  return createHash("sha1").update(String(value || "")).digest("hex").slice(0, 24);
}

function toPercent(progress: number) {
  if (!Number.isFinite(progress)) return undefined;
  return `${Math.max(0, Math.min(100, Math.round(progress)))}%`;
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
  return targetWords.reduce((score, word) => (base.includes(word) ? score + 10 : score), 0);
}

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

function isPlaceholderRemoteCatalogItem(
  provider: RemoteLibraryProvider,
  item: Pick<
    RemoteCatalogItem,
    "title" | "artist" | "album" | "artworkUrl" | "durationSec" | "variantId" | "canonicalUrl" | "pageUrl" | "sourceUrl"
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

  if (provider === "qobuz") {
    if (title === "download store" || title === "my account") return true;
    if (
      !hasMetadataSignal &&
      /\/(signin|login|shop|profile\/downloads|account)(\/|$)/i.test(canonical)
    ) {
      return true;
    }
    if (
      !hasMetadataSignal &&
      !/\.(flac|wav|aif|aiff|zip)(\?|$)/i.test(canonical)
    ) {
      return true;
    }
  }

  if (provider === "bandcamp") {
    if (title === "collection" || title === "discover") return true;
  }

  return false;
}

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

function getOpenProviderProbeWindows(provider: ActiveLibraryProvider) {
  const windows: BrowserWindow[] = [];
  const authWindow = remoteAuthWindows.get(provider);
  if (authWindow && !authWindow.isDestroyed()) {
    windows.push(authWindow);
  }
  if (
    provider === "qobuz" &&
    qobuzAutomationWindow &&
    !qobuzAutomationWindow.isDestroyed() &&
    qobuzAutomationWindow !== authWindow
  ) {
    windows.push(qobuzAutomationWindow);
  }
  return windows;
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
            } catch {
              // Ignore non-JSON storage entries.
            }
          }
        } catch {
          // Ignore storage access failures.
        }
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
          new URLSearchParams({
            query,
            offset: "0",
            limit: "25",
          }).toString(),
        {
          method: "GET",
          headers: token
            ? {
                "X-App-Id": appId,
                "X-User-Auth-Token": token,
              }
            : {
                "X-App-Id": appId,
              },
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

function inferPlaybackSurface(
  item: RemoteCatalogItem,
  provider: ActiveLibraryProvider,
): PlaybackSurface {
  if (item.playbackSurface) return item.playbackSurface;
  return provider === "spotify" ? "desktop_app" : "browser";
}

function computeCaptureQualityMode(
  provider: ActiveLibraryProvider,
  item: RemoteCatalogItem,
  sampleRate?: number,
  channels?: number,
) {
  const qualityText = String(item.qualityLabel || "").toLowerCase();
  const isHiRes = /24\s*bit|48\s*k|88\.?2\s*k|96\s*k|176\.?4\s*k|192\s*k|hi-res/i.test(
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

function resolveCaptureLaunchTarget(
  provider: ActiveLibraryProvider,
  item: RemoteCatalogItem,
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

function buildPlaybackCaptureOutputPath(captureId: string, item: RemoteCatalogItem) {
  const providerDir = path.join(getPlaybackCaptureCacheDir(), item.provider);
  safeMkdir(providerDir);
  const baseName = sanitizeForPathSegment(
    [item.artist, item.title].filter(Boolean).join(" - ") ||
      item.title ||
      captureId,
  );
  return path.join(providerDir, `${baseName || captureId}_${captureId}.wav`);
}

function buildPlaybackCaptureDiagnosticsPath(outputPath: string) {
  const parsed = path.parse(outputPath);
  return path.join(parsed.dir, `${parsed.name}.diagnostics.json`);
}

function normalizeCompareText(value: string | undefined | null) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function writePlaybackCaptureDiagnostics(
  session: PlaybackCaptureSession,
  payload: Record<string, any>,
) {
  try {
    safeMkdir(path.dirname(session.diagnosticsPath));
    fs.writeFileSync(
      session.diagnosticsPath,
      JSON.stringify(
        {
          captureId: session.captureId,
          provider: session.provider,
          deviceId: session.deviceId,
          outputPath: session.outputPath,
          selectedItem: session.item,
          startedAt: session.startedAt,
          ...payload,
        },
        null,
        2,
      ),
      "utf-8",
    );
  } catch (error: any) {
    log("[capture] failed to write diagnostics", {
      captureId: session.captureId,
      diagnosticsPath: session.diagnosticsPath,
      error: error?.message || String(error),
    });
  }
}

function getCaptureDurationCapSec() {
  const raw = String(process.env.STEMSEP_CAPTURE_DURATION_CAP_SEC || "").trim();
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function getExpectedCaptureDurationSec(item: RemoteCatalogItem) {
  const capSec = getCaptureDurationCapSec();
  if (typeof item.durationSec !== "number") {
    return capSec ?? undefined;
  }
  return capSec ? Math.min(item.durationSec, capSec) : item.durationSec;
}

async function prepareHiddenQobuzPlayback(
  item: RemoteCatalogItem,
  playbackDeviceId: string,
) {
  const sinkId = qobuzSinkIdByPlaybackDeviceId.get(playbackDeviceId);
  if (!sinkId) {
    throw new Error(
      "The selected silent output is not routable from the hidden Qobuz player yet. Re-run Capture Setup after authenticating.",
    );
  }

  const targetUrl = item.playbackUrl || item.canonicalUrl || item.pageUrl;
  if (!targetUrl) {
    throw new Error("No Qobuz playback URL was found for this track.");
  }

  const win = getQobuzAutomationWindow();
  const currentUrl = win.webContents.getURL();
  if (currentUrl !== targetUrl) {
    await loadURLAndWait(win, targetUrl, 20_000);
  } else {
    await win.webContents.reloadIgnoringCache();
    await waitForDidFinishLoad(win, 20_000);
  }
  await sleep(1200);

  const initialPath = (() => {
    try {
      return new URL(win.webContents.getURL()).pathname;
    } catch {
      return "";
    }
  })();
  if (
    initialPath === "/error/404" &&
    item.pageUrl &&
    item.pageUrl !== targetUrl
  ) {
      await loadURLAndWait(win, item.pageUrl, 20_000);
      await sleep(1200);
  }

  const playbackResult = (await win.webContents.executeJavaScript(
    `(async () => {
      const desiredSinkId = ${JSON.stringify(sinkId)};
      const targetTrackId = ${JSON.stringify(item.trackId)};
      const targetTitle = ${JSON.stringify(
        String(item.title || "")
          .replace(/\s+/g, " ")
          .trim()
          .toLowerCase(),
      )};
      const targetArtist = ${JSON.stringify(
        String(item.artist || "")
          .replace(/\s+/g, " ")
          .trim()
          .toLowerCase(),
      )};
      const targetAlbum = ${JSON.stringify(
        String(item.album || "")
          .replace(/\s+/g, " ")
          .trim()
          .toLowerCase(),
      )};
      const targetAlbumId = ${JSON.stringify(
        item.albumId ? String(item.albumId) : "",
      )};
      const targetTrackNumber = ${
        typeof item.trackNumber === "number" && Number.isFinite(item.trackNumber)
          ? item.trackNumber
          : "null"
      };
      const targetDiscNumber = ${
        typeof item.discNumber === "number" && Number.isFinite(item.discNumber)
          ? item.discNumber
          : "null"
      };
      const ensureSink = async (media) => {
        if (!media) return true;
        if (typeof media.setSinkId !== "function") return false;
        try {
          if (media.sinkId !== desiredSinkId) {
            await media.setSinkId(desiredSinkId);
          }
          return true;
        } catch (error) {
          window.__stemsepLastSinkError = String(error?.message || error || "Failed to set sink.");
          return false;
        }
      };

      window.__stemsepDesiredSinkId = desiredSinkId;
      if (!window.__stemsepSinkPatchInstalled) {
        const originalPlay = HTMLMediaElement.prototype.play;
        HTMLMediaElement.prototype.play = async function(...args) {
          await ensureSink(this);
          return originalPlay.apply(this, args);
        };
        const observer = new MutationObserver(() => {
          for (const media of Array.from(document.querySelectorAll("audio,video"))) {
            void ensureSink(media);
          }
        });
        observer.observe(document.documentElement || document.body, {
          subtree: true,
          childList: true,
        });
        window.__stemsepSinkPatchInstalled = true;
      }

      const mediaNodes = Array.from(document.querySelectorAll("audio,video"));
      let sinkApplied = 0;
      for (const media of mediaNodes) {
        if (await ensureSink(media)) sinkApplied += 1;
      }

      const normalize = (value) =>
        String(value || "")
          .replace(/\s+/g, " ")
          .trim()
          .toLowerCase();
      const labelOf = (element) =>
        normalize(
          element?.getAttribute?.("aria-label") ||
            element?.getAttribute?.("title") ||
            element?.textContent ||
            "",
        );
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) return false;
        const style = window.getComputedStyle(element);
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || "1") > 0 &&
          !element.hasAttribute("disabled")
        );
      };
      const isPlaybackNavigationLabel = (label) =>
        /\bplaylist(s)?\b|\bplayer\b|\bplay queue\b/.test(label);
      const exactPlayLabel = (label) =>
        /^(play|play track|lecture|reproducir|riproduci|spielen|ouvir|lyssna)$/i.test(label);
      const isPlayLikeButton = (element) => {
        const label = labelOf(element);
        const className = String(element?.className || "");
        return (
          (!!label &&
            !isPlaybackNavigationLabel(label) &&
            (exactPlayLabel(label) ||
              /\b(play|lecture|reproducir|riproduci|spielen|ouvir|lyssna)\b/i.test(
                label,
              ))) ||
          /icon-play-arrow|play/i.test(className)
        );
      };
      const extractIdFromHref = (href, kind) => {
        const match = String(href || "").match(
          new RegExp("/" + kind + "/([^/?#]+)", "i"),
        );
        return match?.[1] || "";
      };
      const getTrackRowSummary = (row) => {
        if (!(row instanceof HTMLElement) || !isVisible(row)) return null;
        const text = normalize(row.innerText || row.textContent || "");
        if (!text) return null;

        const titleAnchor =
          row.querySelector('a[href*="/track/"]') ||
          row.querySelector('[class*="ListItem__title"] a[href]');
        const artistAnchor =
          row.querySelector('a[href*="/artist/"]') ||
          row.querySelector('[class*="ListItem__artist"] a[href]');
        const albumAnchor = row.querySelector('a[href*="/album/"]');
        const numberNode = row.querySelector(
          '.ListItem__number, [class*="ListItem__number"]',
        );
        const playerButton = row.querySelector(
          '.ListItem__player[role="button"], [class*="ListItem__player"][role="button"]',
        );
        const title = normalize(
          titleAnchor?.textContent ||
            row.querySelector('[class*="ListItem__title"]')?.textContent ||
            "",
        );
        const artist = normalize(
          artistAnchor?.textContent ||
            row.querySelector('[class*="ListItem__artist"]')?.textContent ||
            "",
        );
        const trackHref = String(titleAnchor?.getAttribute?.("href") || "");
        const albumHref = String(albumAnchor?.getAttribute?.("href") || "");
        const trackId =
          row.getAttribute("data-track-id") ||
          row.getAttribute("data-id") ||
          extractIdFromHref(trackHref, "track") ||
          "";
        const albumId =
          row.getAttribute("data-album-id") ||
          extractIdFromHref(albumHref, "album") ||
          "";
        const parsedTrackNumber = Number.parseInt(
          normalize(numberNode?.textContent || "").replace(/[^\d]/g, ""),
          10,
        );
        const rowState = normalize(
          playerButton?.getAttribute?.("aria-label") ||
            row.getAttribute("data-state") ||
            row.className ||
            "",
        );
        let score = 0;
        if (targetTrackId && trackId && trackId === targetTrackId) score += 220;
        if (targetAlbumId && albumId && albumId === targetAlbumId) score += 32;
        if (targetTitle && title && (title === targetTitle || title.includes(targetTitle)))
          score += 100;
        if (
          targetArtist &&
          artist &&
          (artist === targetArtist || artist.includes(targetArtist) || targetArtist.includes(artist))
        ) {
          score += 40;
        }
        if (
          Number.isFinite(targetTrackNumber) &&
          Number.isFinite(parsedTrackNumber) &&
          targetTrackNumber === parsedTrackNumber
        ) {
          score += 70;
        }
        if (playerButton) score += 12;
        if (/pause|playing|active/.test(rowState)) score += 8;
        return {
          row,
          playerButton:
            playerButton instanceof HTMLElement ? playerButton : null,
          title,
          artist,
          trackId,
          albumId,
          trackHref,
          albumHref,
          trackNumber: Number.isFinite(parsedTrackNumber)
            ? parsedTrackNumber
            : null,
          rowState,
          score,
        };
      };
      const collectTrackRows = (limit = 12) => {
        const rows = Array.from(
          document.querySelectorAll(
            '[class*="ListItem"], [data-testid*="track"], tr, li',
          ),
        )
          .map((row) => getTrackRowSummary(row))
          .filter(Boolean)
          .sort((left, right) => right.score - left.score)
          .slice(0, limit);
        return rows;
      };
      const findExactTrackRowButton = () => {
        const rows = collectTrackRows(16);
        return rows.find((row) => row.playerButton && row.score >= 120) || null;
      };
      const readPlayerSnapshot = () => {
        const playerScopes = [
          ...document.querySelectorAll(
            'footer, [class*="Player"], [class*="NowPlaying"], [data-testid*="player"]',
          ),
        ].filter((scope) => scope instanceof HTMLElement && isVisible(scope));
        for (const scope of playerScopes) {
          const trackAnchor =
            scope.querySelector('a[href*="/track/"]') ||
            scope.querySelector('[class*="title"] a[href]');
          const albumAnchor = scope.querySelector('a[href*="/album/"]');
          const artistAnchor =
            scope.querySelector('a[href*="/artist/"]') ||
            scope.querySelector('[class*="artist"] a[href]');
          const currentTitle = normalize(
            trackAnchor?.textContent ||
              scope.querySelector('[class*="title"]')?.textContent ||
              "",
          );
          const currentArtist = normalize(
            artistAnchor?.textContent ||
              scope.querySelector('[class*="artist"]')?.textContent ||
              "",
          );
          const currentTrackHref = String(trackAnchor?.getAttribute?.("href") || "");
          const currentAlbumHref = String(albumAnchor?.getAttribute?.("href") || "");
          const currentTrackId = extractIdFromHref(currentTrackHref, "track");
          const currentAlbumId = extractIdFromHref(currentAlbumHref, "album");
          if (
            currentTitle ||
            currentArtist ||
            currentTrackHref ||
            currentAlbumHref ||
            currentTrackId ||
            currentAlbumId
          ) {
            return {
              currentTitle,
              currentArtist,
              currentTrackHref,
              currentAlbumHref,
              currentTrackId,
              currentAlbumId,
            };
          }
        }
        return {
          currentTitle: "",
          currentArtist: "",
          currentTrackHref: "",
          currentAlbumHref: "",
          currentTrackId: "",
          currentAlbumId: "",
        };
      };
      const verifyPlayerSnapshot = (snapshot) => {
        const trackIdMatched =
          !!targetTrackId &&
          !!snapshot.currentTrackId &&
          snapshot.currentTrackId === targetTrackId;
        const hrefMatched =
          !!targetTrackId &&
          String(snapshot.currentTrackHref || "").includes("/track/" + targetTrackId);
        const titleMatched =
          !!targetTitle &&
          !!snapshot.currentTitle &&
          (snapshot.currentTitle === targetTitle ||
            snapshot.currentTitle.includes(targetTitle) ||
            targetTitle.includes(snapshot.currentTitle));
        const artistMatched =
          !targetArtist ||
          !snapshot.currentArtist ||
          snapshot.currentArtist === targetArtist ||
          snapshot.currentArtist.includes(targetArtist) ||
          targetArtist.includes(snapshot.currentArtist);
        if (trackIdMatched) return { matched: true, reason: "track_id" };
        if (hrefMatched) return { matched: true, reason: "track_href" };
        if (titleMatched && artistMatched) {
          return { matched: true, reason: "title_artist" };
        }
        return { matched: false, reason: "failed" };
      };
      const waitForPlaybackVerification = async (timeoutMs) => {
        const startedAt = Date.now();
        let snapshot = readPlayerSnapshot();
        let verification = verifyPlayerSnapshot(snapshot);
        while (Date.now() - startedAt < timeoutMs) {
          if (verification.matched) {
            return {
              ...snapshot,
              verificationMatched: true,
              verificationReason: verification.reason,
            };
          }
          await wait(350);
          snapshot = readPlayerSnapshot();
          verification = verifyPlayerSnapshot(snapshot);
        }
        const targetRow = findExactTrackRowButton();
        return {
          ...snapshot,
          verificationMatched: false,
          verificationReason: verification.reason,
          rowMatched: !!targetRow,
          rowState: targetRow?.rowState || "",
        };
      };
      const findTransportButton = (matcher, classPattern) =>
        Array.from(document.querySelectorAll("button, [role='button']")).find(
          (element) => {
            if (!isVisible(element)) return false;
            const label = labelOf(element);
            const className = String(element?.className || "");
            return (
              (!!label && matcher.test(label)) ||
              (!!classPattern && classPattern.test(className))
            );
          },
        ) || null;
      const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const waitForTargetText = async (timeoutMs) => {
        const startedAt = Date.now();
        while (Date.now() - startedAt < timeoutMs) {
          const bodyText = normalize(document.body?.innerText || "");
          if (targetTitle && bodyText.includes(targetTitle)) {
            return true;
          }
          await wait(250);
        }
        return false;
      };
      const findActiveMedia = () =>
        Array.from(document.querySelectorAll("audio,video")).find(
          (media) => !media.paused && !media.ended,
        ) || null;
      const waitForActiveMedia = async (timeoutMs) => {
        const startedAt = Date.now();
        while (Date.now() - startedAt < timeoutMs) {
          const active = findActiveMedia();
          if (active) return active;
          await wait(350);
        }
        return null;
      };
      const collectScopeCandidates = (seed) => {
        const scopes = [];
        let current = seed instanceof Element ? seed : null;
        for (let depth = 0; current && depth < 14; depth += 1) {
          if (
            current instanceof HTMLElement &&
            !scopes.includes(current) &&
            isVisible(current)
          ) {
            scopes.push(current);
          }
          current = current.parentElement;
        }
        return scopes;
      };
      const scoreScope = (scope) => {
        if (!(scope instanceof HTMLElement)) return -1_000_000;
        const text = normalize(scope.innerText || scope.textContent || "");
        if (!text) return -1_000_000;

        let score = 0;
        if (targetTitle && text.includes(targetTitle)) score += 90;
        if (targetArtist && text.includes(targetArtist)) score += 28;
        if (targetAlbum && text.includes(targetAlbum)) score += 10;
        if (
          Number.isFinite(targetTrackNumber) &&
          targetTrackNumber > 0 &&
          new RegExp("(^|\\\\s)" + targetTrackNumber + "(\\\\s|\\\\.|-)").test(
            text.slice(0, 48),
          )
        ) {
          score += 26;
        }
        if (
          Number.isFinite(targetDiscNumber) &&
          targetDiscNumber > 0 &&
          /\bdisc\b|\bcd\b|\bmedia\b/.test(text)
        ) {
          score += 4;
        }

        const hrefs = Array.from(scope.querySelectorAll("a[href]"))
          .map((anchor) => String(anchor.getAttribute("href") || ""))
          .filter(Boolean)
          .join(" ");
        const ids = [
          scope.getAttribute("data-track-id"),
          scope.getAttribute("data-id"),
          hrefs,
        ]
          .filter(Boolean)
          .join(" ");
        if (targetTrackId && ids.includes(targetTrackId)) score += 120;
        if (targetAlbumId && ids.includes(targetAlbumId)) score += 12;

        const playButtons = Array.from(
          scope.querySelectorAll("button, [role='button']"),
        ).filter((element) => isVisible(element) && isPlayLikeButton(element));

        score += Math.min(playButtons.length, 2) * 8;
        score -= Math.min(text.length, 480) / 18;
        if (playButtons.length > 4) score -= 42;
        if (scope === document.body || scope === document.documentElement) score -= 220;
        return score;
      };
      const chooseBestPlayTarget = () => {
        const exactTitleNodes = Array.from(
          document.querySelectorAll("a, button, span, div, td, p"),
        ).filter((element) => {
          if (!(element instanceof HTMLElement) || !isVisible(element)) return false;
          const text = normalize(element.textContent || "");
          return (
            !!targetTitle &&
            text.includes(targetTitle) &&
            text.length <= Math.max(targetTitle.length + 96, 180) &&
            (!targetArtist || text.includes(targetArtist) || text.length <= targetTitle.length + 32)
          );
        });
        const directMatches = [
          ...document.querySelectorAll(
            '[data-track-id="' +
              targetTrackId +
              '"], [data-id="' +
              targetTrackId +
              '"], [href*="/track/' +
              targetTrackId +
              '"]',
          ),
        ].filter((element) => element instanceof HTMLElement && isVisible(element));
        const seedNodes = [...directMatches, ...exactTitleNodes];
        const candidates = [];

        for (const seed of seedNodes) {
          for (const scope of collectScopeCandidates(seed)) {
            const playButtons = Array.from(
              scope.querySelectorAll("button, [role='button']"),
            ).filter((element) => isVisible(element) && isPlayLikeButton(element));
            const titleClickTarget =
              seed instanceof HTMLElement && isVisible(seed) ? seed : null;
            if (playButtons.length === 0 && !titleClickTarget) continue;

            const exact = playButtons.find((element) => exactPlayLabel(labelOf(element)));
            const chosen = exact || playButtons[0] || titleClickTarget;
            if (!chosen) continue;
            candidates.push({
              scope,
              target: chosen,
              score: scoreScope(scope),
              scopeText: normalize(scope.innerText || scope.textContent || "").slice(0, 220),
              targetLabel: labelOf(chosen),
            });
          }
        }

        candidates.sort((left, right) => right.score - left.score);
        return candidates[0] || null;
      };
      const findTrackNumberButton = () => {
        if (!Number.isFinite(targetTrackNumber) || targetTrackNumber <= 0) {
          return null;
        }
        const numberCells = Array.from(
          document.querySelectorAll(
            '.ListItem__number[role="button"], [class*="ListItem__number"][role="button"]',
          ),
        ).filter(isVisible);

        const matchedCell =
          numberCells.find((element) => {
            const text = normalize(element.textContent || "");
            return text === String(targetTrackNumber);
          }) || null;
        if (!(matchedCell instanceof HTMLElement)) {
          return null;
        }
        return (
          matchedCell.querySelector(
            '.ListItem__player[role="button"], [class*="ListItem__player"][role="button"]',
          ) || matchedCell
        );
      };
      const findTrackScopedPlayButton = () => {
        return chooseBestPlayTarget();
      };

      const findPlayButton = () => {
        const scopes = [
          document.querySelector("main"),
          document.querySelector('[data-testid="track-page"]'),
          document.body,
        ].filter(Boolean);

        for (const scope of scopes) {
          const candidates = Array.from(
            scope.querySelectorAll("button, [role='button']"),
          )
            .filter(isVisible)
            .map((element) => ({
              element,
              label: labelOf(element),
            }))
            .filter(({ label }) => !!label && !isPlaybackNavigationLabel(label));

          const exact = candidates.find(({ label }) => exactPlayLabel(label));
          if (exact?.element) return exact.element;

          const fuzzy = candidates.find(({ label }) =>
            /\b(play|lecture|reproducir|riproduci|spielen|ouvir|lyssna)\b/i.test(label),
          );
          if (fuzzy?.element) return fuzzy.element;
        }

        return null;
      };

      let playButtonClicked = false;
      let clickStrategy = "";
      let clickedLabel = "";
      let clickedHtml = "";
      let clickedScopeText = "";
      let candidateLabels = [];
      let candidateTrackRows = [];
      await waitForTargetText(6_000);
      const exactTrackRow = findExactTrackRowButton();
      const scopedTarget = findTrackScopedPlayButton();
      const trackNumberTarget = findTrackNumberButton();
      const playButton =
        exactTrackRow?.playerButton ||
        scopedTarget?.target ||
        trackNumberTarget ||
        findPlayButton();
      if (playButton instanceof HTMLElement) {
        clickStrategy = exactTrackRow?.playerButton
          ? "exact_track_row_button"
          : scopedTarget?.target
          ? scopedTarget.target.tagName === "BUTTON" ||
            scopedTarget.target.getAttribute("role") === "button"
            ? "track_scoped_play"
            : "track_title_click"
          : trackNumberTarget
            ? "track_number_button"
          : "generic_play";
        clickedLabel = labelOf(playButton);
        clickedHtml = String(playButton.outerHTML || "").slice(0, 240);
        clickedScopeText = scopedTarget?.scopeText || "";
        playButton.click();
        playButtonClicked = true;
      }
      candidateLabels = Array.from(
        document.querySelectorAll("button, [role='button']"),
      )
        .filter(isVisible)
        .map((element) => labelOf(element))
        .filter(Boolean)
        .slice(0, 18);
      candidateTrackRows = collectTrackRows(8).map((row) => ({
        title: row.title,
        artist: row.artist,
        trackNumber: row.trackNumber,
        trackId: row.trackId,
        albumId: row.albumId,
        href: row.trackHref,
        rowState: row.rowState,
      }));

      await wait(1200);

      if (
        playButtonClicked &&
        clickStrategy === "generic_play" &&
        Number.isFinite(targetTrackNumber) &&
        targetTrackNumber > 1 &&
        /\\/album\\//.test(String(window.location?.pathname || ""))
      ) {
        const skipsNeeded = Math.max(0, targetTrackNumber - 1);
        for (let index = 0; index < skipsNeeded; index += 1) {
          const nextButton = findTransportButton(
            /\b(next|skip|forward|suivant|nästa|weiter|próximo|seguinte)\b/i,
            /next|skip-forward|forward/i,
          );
          if (!(nextButton instanceof HTMLElement)) break;
          nextButton.click();
          clickStrategy = "album_play_then_skip_" + targetTrackNumber;
          await wait(900);
        }
      }

      let verificationSnapshot = await waitForPlaybackVerification(6_500);
      if (!verificationSnapshot.verificationMatched && !exactTrackRow?.playerButton && trackNumberTarget instanceof HTMLElement) {
        trackNumberTarget.click();
        clickStrategy = "track_number_retry";
        await wait(1200);
        verificationSnapshot = await waitForPlaybackVerification(4_000);
      }

      let activeMedia = await waitForActiveMedia(6_000);

      if (!activeMedia) {
        for (const media of Array.from(document.querySelectorAll("audio,video"))) {
          try {
            await ensureSink(media);
            media.muted = false;
            media.volume = 1;
            await media.play();
            activeMedia = await waitForActiveMedia(2_500);
            if (activeMedia) break;
          } catch {
            // ignore individual play failures
          }
        }
      }

      return {
        activeMedia: !!activeMedia,
        playButtonClicked,
        clickStrategy,
        clickedLabel,
        clickedHtml,
        clickedScopeText,
        candidateLabels,
        candidateTrackRows,
        targetTrackId,
        targetAlbumId,
        targetTrackNumber,
        initialPath: String(window.location?.pathname || ""),
        finalPath: String(window.location?.pathname || ""),
        verificationMatched: !!verificationSnapshot?.verificationMatched,
        verificationReason: String(verificationSnapshot?.verificationReason || ""),
        currentTitle: String(verificationSnapshot?.currentTitle || ""),
        currentArtist: String(verificationSnapshot?.currentArtist || ""),
        currentTrackHref: String(verificationSnapshot?.currentTrackHref || ""),
        currentAlbumHref: String(verificationSnapshot?.currentAlbumHref || ""),
        currentTrackId: String(verificationSnapshot?.currentTrackId || ""),
        currentAlbumId: String(verificationSnapshot?.currentAlbumId || ""),
        rowMatched: !!verificationSnapshot?.rowMatched,
        rowState: String(verificationSnapshot?.rowState || ""),
        mediaCount: document.querySelectorAll("audio,video").length,
        sinkApplied,
        sinkError: String(window.__stemsepLastSinkError || ""),
        currentPath: String(window.location?.pathname || ""),
        speakerSelectionSupported:
          typeof HTMLMediaElement !== "undefined" &&
          typeof HTMLMediaElement.prototype.setSinkId === "function",
      };
    })()`,
    true,
  )) as {
    activeMedia?: boolean;
    playButtonClicked?: boolean;
    clickStrategy?: string;
    clickedLabel?: string;
    clickedHtml?: string;
    clickedScopeText?: string;
    candidateLabels?: string[];
    candidateTrackRows?: Array<{
      title?: string;
      artist?: string;
      trackNumber?: number | null;
      trackId?: string;
      albumId?: string;
      href?: string;
      rowState?: string;
    }>;
    targetTrackId?: string;
    targetAlbumId?: string;
    targetTrackNumber?: number | null;
    initialPath?: string;
    finalPath?: string;
    verificationMatched?: boolean;
    verificationReason?: string;
    currentTitle?: string;
    currentArtist?: string;
    currentTrackHref?: string;
    currentAlbumHref?: string;
    currentTrackId?: string;
    currentAlbumId?: string;
    rowMatched?: boolean;
    rowState?: string;
    mediaCount?: number;
    sinkApplied?: number;
    sinkError?: string;
    currentPath?: string;
    speakerSelectionSupported?: boolean;
  };

  if (!playbackResult?.speakerSelectionSupported) {
    log("[capture] qobuz hidden playback failed", {
      targetUrl,
      currentUrl: win.webContents.getURL(),
      reason: "speaker-selection-unsupported",
      playbackResult: playbackResult || null,
    });
    const error: any = new Error(
      "This Electron/Chromium build does not expose speaker selection for the hidden Qobuz player.",
    );
    error.details = { targetUrl, playbackResult };
    throw error;
  }
  if (playbackResult?.sinkError) {
    log("[capture] qobuz hidden playback failed", {
      targetUrl,
      currentUrl: win.webContents.getURL(),
      reason: "sink-error",
      playbackResult,
    });
    const error: any = new Error(playbackResult.sinkError);
    error.details = { targetUrl, playbackResult };
    throw error;
  }
  if (!playbackResult?.verificationMatched) {
    log("[capture] qobuz hidden playback failed", {
      targetUrl,
      currentUrl: win.webContents.getURL(),
      reason: "playback-verification-failed",
      playbackResult: {
        ...playbackResult,
        initialPath,
      },
    });
    const expectedTrack = [item.artist, item.title].filter(Boolean).join(" - ");
    const detectedTrack = [
      playbackResult?.currentArtist,
      playbackResult?.currentTitle,
    ]
      .filter(Boolean)
      .join(" - ");
    const error: any = new Error(
      detectedTrack
        ? `Hidden Qobuz playback did not lock onto the selected track. Expected ${expectedTrack || item.title}, but detected ${detectedTrack}.`
        : "Hidden Qobuz playback could not be verified against the selected track. Try again or refresh the Qobuz search results.",
    );
    error.details = {
      targetUrl,
      playbackResult: {
        ...playbackResult,
        initialPath,
      },
    };
    throw error;
  }
  if (!playbackResult?.activeMedia) {
    if (!playbackResult?.playButtonClicked) {
      log("[capture] qobuz hidden playback failed", {
        targetUrl,
        currentUrl: win.webContents.getURL(),
        reason: "no-active-media",
        playbackResult: playbackResult || null,
      });
      const error: any = new Error(
        "Hidden Qobuz playback did not start. The provider page layout or player state may have changed.",
      );
      error.details = { targetUrl, playbackResult };
      throw error;
    }
    log("[capture] qobuz hidden playback awaiting backend audio detection", {
      targetUrl,
      currentUrl: win.webContents.getURL(),
      playbackResult: playbackResult || null,
    });
  }

  return {
    sinkId,
    playbackResult: {
      ...playbackResult,
      initialPath,
    },
  };
}

async function startQobuzPlaybackCaptureForItem(
  item: RemoteCatalogItem,
  deviceId: string,
) {
  const captureId = randomUUID();
  const startedAt = new Date().toISOString();
  const outputPath = buildPlaybackCaptureOutputPath(captureId, item);
  const diagnosticsPath = buildPlaybackCaptureDiagnosticsPath(outputPath);
  playbackCaptureSessions.set(captureId, {
    captureId,
    provider: "qobuz",
    trackId: item.trackId,
    deviceId,
    item,
    startedAt,
    outputPath,
    diagnosticsPath,
    backendCaptureStarted: false,
    verification: null,
  });
  const autoCancelAfterMs = Number(
    process.env.STEMSEP_CAPTURE_CANCEL_AFTER_MS || "",
  );
  const autoCancelTimer =
    Number.isFinite(autoCancelAfterMs) && autoCancelAfterMs > 0
      ? setTimeout(() => {
          void abortPlaybackCaptureSession(
            captureId,
            "Capture cancelled by smoketest.",
          );
        }, autoCancelAfterMs)
      : null;

  emitPlaybackCaptureProgress({
    provider: "qobuz",
    captureId,
    status: "launching",
    detail: "Preparing hidden Qobuz playback...",
  });

  try {
    await stopHiddenQobuzPlayback();

    emitPlaybackCaptureProgress({
      provider: "qobuz",
      captureId,
      status: "verifying",
      detail: "Verifying hidden Qobuz playback...",
    });
    const playbackReady = await prepareHiddenQobuzPlayback(item, deviceId);
    const session = playbackCaptureSessions.get(captureId);
    if (session) {
      session.verification = playbackReady.playbackResult || null;
      writePlaybackCaptureDiagnostics(session, {
        stage: "playback_verified",
        verification: playbackReady.playbackResult || null,
        sinkId: playbackReady.sinkId || null,
      });
    }

    if (cancelledPlaybackCaptureIds.has(captureId)) {
      throw new Error("Capture cancelled");
    }

    emitPlaybackCaptureProgress({
      provider: "qobuz",
      captureId,
      status: "awaiting_audio",
      detail: "Verified target track. Starting capture...",
    });
    if (session) {
      session.backendCaptureStarted = true;
    }
    const capturePromise: Promise<
      | { ok: true; value: any }
      | { ok: false; error: any }
    > = sendPythonCommand(
      "capture_playback_loopback",
      {
        capture_id: captureId,
        device_id: deviceId,
        output_path: outputPath,
        expected_duration_sec: getExpectedCaptureDurationSec(item),
        start_timeout_ms: 15_000,
        trailing_silence_ms: 2_500,
        min_active_rms: 0.003,
      },
      Math.max(300_000, Math.round(((item.durationSec || 180) + 30) * 1000)),
    ).then(
      (value) => ({ ok: true as const, value }),
      (error) => ({ ok: false as const, error }),
    );

    const captureOutcome = await capturePromise;
    if (!captureOutcome.ok) {
      throw (captureOutcome as { ok: false; error: any }).error;
    }
    const captureResult = captureOutcome.value;
    const profile = await probeAudioFile(captureResult.file_path);
    const quality = computeCaptureQualityMode(
      "qobuz",
      item,
      captureResult.capture_sample_rate || profile.sampleRate || undefined,
      captureResult.capture_channels || profile.channels || undefined,
    );
    if (session) {
      writePlaybackCaptureDiagnostics(session, {
        stage: "completed",
        verification: session.verification || null,
        captureResult,
        probedFile: profile,
        quality,
      });
    }

    return {
      success: true as const,
      provider: "qobuz" as const,
      file_path: captureResult.file_path,
      display_name:
        [item.artist, item.title].filter(Boolean).join(" - ") || item.title,
      source_url:
        item.playbackUrl || item.canonicalUrl || item.pageUrl || undefined,
      canonical_url:
        item.canonicalUrl || item.playbackUrl || item.pageUrl || undefined,
      artist: item.artist,
      album: item.album,
      artwork_url: item.artworkUrl,
      duration_sec:
        captureResult.duration_sec ||
        item.durationSec ||
        profile.durationSeconds ||
        undefined,
      quality_label:
        quality.verified
          ? "Verified Lossless"
          : item.qualityLabel || "Lossless Source Capture",
      is_lossless: quality.verified,
      provider_track_id: item.trackId,
      ingest_mode: "desktop_capture" as const,
      playback_surface: "browser" as const,
      quality_mode: quality.qualityMode,
      verified_lossless: quality.verified,
      capture_device_id: deviceId,
      capture_sample_rate:
        captureResult.capture_sample_rate || profile.sampleRate || undefined,
      capture_channels:
        captureResult.capture_channels || profile.channels || undefined,
      capture_bits_per_sample:
        captureResult.capture_bits_per_sample || undefined,
      capture_sample_format:
        captureResult.capture_sample_format || undefined,
      capture_start_at: captureResult.capture_start_at || startedAt,
      capture_end_at: captureResult.capture_end_at || new Date().toISOString(),
    };
  } catch (error) {
    const session = playbackCaptureSessions.get(captureId);
    if (session?.backendCaptureStarted) {
      await sendPythonCommandWithRetry(
        "cancel_playback_capture",
        { capture_id: captureId },
        10_000,
        0,
      ).catch(() => null);
    }
    if (session) {
      writePlaybackCaptureDiagnostics(session, {
        stage: cancelledPlaybackCaptureIds.has(captureId) ? "cancelled" : "failed",
        verification: session.verification || null,
        error:
          error instanceof Error ? error.message : String(error || "Playback capture failed"),
        details:
          error && typeof error === "object" && "details" in error
            ? (error as any).details
            : null,
      });
    }
    if (cancelledPlaybackCaptureIds.has(captureId)) {
      throw new Error("Capture cancelled");
    }
    throw error;
  } finally {
    if (autoCancelTimer) {
      clearTimeout(autoCancelTimer);
    }
    await stopHiddenQobuzPlayback();
    playbackCaptureSessions.delete(captureId);
    cancelledPlaybackCaptureIds.delete(captureId);
  }
}

async function maybeRunQobuzCaptureSmokeTest() {
  const query = String(process.env.STEMSEP_QOBUZ_CAPTURE_SMOKETEST_QUERY || "").trim();
  if (!query) return;

  log("[smoketest] qobuz capture requested", {
    query,
    durationCapSec: getCaptureDurationCapSec(),
  });

  try {
    const authenticated = await checkLibraryProviderAuthenticated("qobuz");
    if (!authenticated) {
      throw new Error("Qobuz is not authenticated in the persisted StemSep session.");
    }

    const deviceId = getStoredCaptureOutputDeviceId();
    if (!deviceId) {
      throw new Error("No saved silent output device is configured.");
    }

    await refreshQobuzPlaybackDeviceMappings();
    if (!qobuzSinkIdByPlaybackDeviceId.has(deviceId)) {
      throw new Error(
        "The saved silent output could not be routed from the hidden Qobuz player.",
      );
    }

    const items = await searchQobuzCatalogViaApi(query);
    const item = items.find((entry) => !!entry.trackId) || null;
    if (!item) {
      throw new Error(`No Qobuz tracks were returned for query '${query}'.`);
    }

    log("[smoketest] qobuz capture target", {
      trackId: item.trackId,
      title: item.title,
      artist: item.artist,
      trackNumber: item.trackNumber,
      discNumber: item.discNumber,
      durationSec: item.durationSec,
      qualityLabel: item.qualityLabel,
    });

    const result = await startQobuzPlaybackCaptureForItem(item, deviceId);
    log("[smoketest] qobuz capture success", result);
  } catch (error: any) {
    log("[smoketest] qobuz capture failed", {
      error: error?.message || String(error),
    });
  } finally {
    setTimeout(() => {
      try {
        app.quit();
      } catch {
        // ignore
      }
    }, 1000);
  }
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

async function loadRemoteBrowserWindow(
  provider: RemoteLibraryProvider,
  url: string,
  options: { visible?: boolean; reuseAuthWindow?: boolean } = {},
): Promise<{ win: BrowserWindow; disposable: boolean }> {
  const { visible = false, reuseAuthWindow = false } = options;
  const existing = reuseAuthWindow ? remoteAuthWindows.get(provider) : null;
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
    parent: visible ? mainWindow || undefined : undefined,
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
      } catch {
        // ignore
      }
    }
  }
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
    "user-agent":
      mainWindow?.webContents.getUserAgent() || "Mozilla/5.0 StemSep Desktop",
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
  const contentType = String(response.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes("text/html")) {
    throw new Error("Provider returned an HTML page instead of a downloadable audio file.");
  }

  const responseUrl = response.url || url;
  const ext = inferRemoteFileExtension(responseUrl);
  const baseName = sanitizeForPathSegment(title || `remote_${provider}`) || `remote_${provider}`;
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
      const progress = Number.isFinite(totalBytes) && totalBytes > 0
        ? receivedBytes / totalBytes
        : NaN;
      emitRemoteResolveProgress({
        provider,
        status: "downloading",
        detail: `Downloading ${title}...`,
        progress: Number.isFinite(progress) ? progress : undefined,
        percent: Number.isFinite(progress) ? toPercent(progress * 100) : undefined,
      });
    }
  }

  const output = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
  fs.writeFileSync(filePath, output);
  return filePath;
}

function ensureWritableDirectory(dirPath: string) {
  safeMkdir(dirPath);
  const probeFile = path.join(dirPath, `.stemsep_write_test_${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(probeFile, "ok");
    fs.unlinkSync(probeFile);
  } catch (error: any) {
    throw new Error(
      `Export destination is not writable: ${error?.message || String(error)}`,
    );
  }
}

function getAvailableDiskBytes(dirPath: string): number | null {
  try {
    const statfs = (fs as any).statfsSync?.(dirPath);
    if (!statfs) return null;
    const available = Number(statfs.bavail ?? statfs.blocks ?? NaN);
    const blockSize = Number(statfs.bsize ?? statfs.frsize ?? NaN);
    if (!Number.isFinite(available) || !Number.isFinite(blockSize)) return null;
    return available * blockSize;
  } catch {
    return null;
  }
}

function estimateMp3Bytes(durationSeconds: number | null, bitrate: string): number {
  const match = String(bitrate || "320k").toLowerCase().match(/(\d+)/);
  const kbps = match ? Number(match[1]) : 320;
  if (!durationSeconds || !Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return kbps * 1000;
  }
  return Math.ceil(durationSeconds * (kbps * 1000) / 8);
}

async function resolveExistingAudioSource(filePath: string): Promise<string> {
  let resolvedInputFile = filePath;
  if (!fs.existsSync(resolvedInputFile)) {
    const fallback = resolveMissingPreviewAudioPath(resolvedInputFile);
    if (fallback && fs.existsSync(fallback)) {
      log("[audio-source] resolved missing source path", {
        from: filePath,
        to: fallback,
      });
      resolvedInputFile = fallback;
    }
  }

  if (!fs.existsSync(resolvedInputFile)) {
    const missing = classifyMissingAudioPath(filePath);
    throw createCodedError(
      missing.code,
      `Source file missing: ${filePath}. ${missing.hint}`,
      missing.hint,
      {
        filePath,
      },
    );
  }

  return resolvedInputFile;
}

function uniqueOutputFile(exportPath: string, baseName: string, fmt: string): string {
  let outFile = path.join(exportPath, `${baseName}.${fmt}`);
  for (let i = 2; fs.existsSync(outFile); i++) {
    outFile = path.join(exportPath, `${baseName}_${i}.${fmt}`);
  }
  return outFile;
}

async function buildExportTasks({
  sourceFiles,
  exportPath,
  format,
  bitrate,
}: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
}): Promise<ExportTask[]> {
  const fmt = String(format || "wav").toLowerCase();
  if (!new Set(["wav", "flac", "mp3"]).has(fmt)) {
    throw new Error(`Unsupported export format: ${format}`);
  }

  const tasks: ExportTask[] = [];
  for (const [stemRaw, inputFile] of Object.entries(sourceFiles || {})) {
    const stem = sanitizeForPathSegment(stemRaw || "stem") || "stem";
    if (!inputFile || typeof inputFile !== "string") continue;
    const sourceFile = await resolveExistingAudioSource(inputFile);
    const sourceProfile = await probeAudioFile(sourceFile);
    const sourceExt = path.extname(sourceFile || "").toLowerCase();
    const outFile = uniqueOutputFile(exportPath, stem, fmt);
    const mode: ExportMode =
      (fmt === "wav" && sourceExt === ".wav") ||
      (fmt === "flac" && sourceExt === ".flac")
        ? "copy"
        : "transcode";

    let estimatedOutputBytes = 0;
    try {
      estimatedOutputBytes =
        mode === "copy"
          ? fs.statSync(sourceFile).size
          : fmt === "mp3"
            ? estimateMp3Bytes(sourceProfile.durationSeconds, bitrate)
            : Math.ceil(fs.statSync(sourceFile).size * 0.8);
    } catch {
      estimatedOutputBytes = 50 * 1024 * 1024;
    }

    tasks.push({
      stemRaw,
      stem,
      sourceFile,
      sourceProfile,
      outFile,
      format: fmt as ExportTask["format"],
      bitrate,
      mode,
      estimatedOutputBytes,
    });
  }

  if (tasks.length === 0) {
    throw new Error("No files were exported (no valid source files)");
  }

  return tasks;
}

async function exportFilesLocal({
  sourceFiles,
  exportPath,
  format,
  bitrate,
  requestId,
}: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
  requestId: string;
}): Promise<{ exported: Record<string, string> }> {
  if (!exportPath || typeof exportPath !== "string") {
    throw new Error("Missing exportPath");
  }

  ensureWritableDirectory(exportPath);
  await ensureBinaryAvailable(getFfmpegExe(), "ffmpeg");
  await ensureBinaryAvailable(getFfprobeExe(), "ffprobe");

  emitExportProgress({
    requestId,
    status: "preflight",
    totalProgress: 0,
    detail: "Validating sources and export destination...",
    format: String(format || "wav").toLowerCase(),
  });

  const tasks = await buildExportTasks({ sourceFiles, exportPath, format, bitrate });
  const estimatedTotalBytes = tasks.reduce(
    (sum, task) => sum + task.estimatedOutputBytes,
    0,
  );
  const availableDiskBytes = getAvailableDiskBytes(exportPath);
  if (
    availableDiskBytes !== null &&
    estimatedTotalBytes > 0 &&
    availableDiskBytes < estimatedTotalBytes
  ) {
    throw new Error(
      `Not enough free disk space for export. Required ~${Math.ceil(
        estimatedTotalBytes / (1024 * 1024),
      )} MB, available ~${Math.ceil(availableDiskBytes / (1024 * 1024))} MB.`,
    );
  }

  const exported: Record<string, string> = {};
  let completedWeight = 0;
  const totalWeight = tasks.reduce(
    (sum, task) => sum + Math.max(task.estimatedOutputBytes, 1),
    0,
  );

  for (let index = 0; index < tasks.length; index++) {
    const task = tasks[index];
    const taskWeight = Math.max(task.estimatedOutputBytes, 1);

    emitExportProgress({
      requestId,
      status: task.mode === "copy" ? "copying" : "transcoding",
      stem: task.stemRaw,
      fileIndex: index + 1,
      fileCount: tasks.length,
      fileProgress: 0,
      totalProgress:
        totalWeight > 0 ? (completedWeight / totalWeight) * 100 : 0,
      detail:
        task.mode === "copy"
          ? `Copying ${task.stemRaw}...`
          : `Transcoding ${task.stemRaw} to ${task.format.toUpperCase()}...`,
      format: task.format,
      outputPath: task.outFile,
    });

    if (task.mode === "copy") {
      fs.copyFileSync(task.sourceFile, task.outFile);
      emitExportProgress({
        requestId,
        status: "copying",
        stem: task.stemRaw,
        fileIndex: index + 1,
        fileCount: tasks.length,
        fileProgress: 100,
        totalProgress:
          totalWeight > 0
            ? ((completedWeight + taskWeight) / totalWeight) * 100
            : 100,
        detail: `Copied ${task.stemRaw}`,
        format: task.format,
        outputPath: task.outFile,
      });
    } else if (task.format === "flac") {
      await runFfmpegWithProgress(
        [
          "-y",
          "-hide_banner",
          "-loglevel",
          "error",
          "-i",
          task.sourceFile,
          "-vn",
          "-map_metadata",
          "0",
          "-c:a",
          "flac",
          "-compression_level",
          "5",
          task.outFile,
        ],
        task.sourceProfile.durationSeconds,
        (fileProgress) => {
          emitExportProgress({
            requestId,
            status: "transcoding",
            stem: task.stemRaw,
            fileIndex: index + 1,
            fileCount: tasks.length,
            fileProgress: fileProgress * 100,
            totalProgress:
              totalWeight > 0
                ? ((completedWeight + taskWeight * fileProgress) / totalWeight) *
                  100
                : fileProgress * 100,
            detail: `Transcoding ${task.stemRaw} to FLAC...`,
            format: task.format,
            outputPath: task.outFile,
          });
        },
      );
    } else if (task.format === "mp3") {
      const br = String(task.bitrate || "320k");
      await runFfmpegWithProgress(
        [
          "-y",
          "-hide_banner",
          "-loglevel",
          "error",
          "-i",
          task.sourceFile,
          "-vn",
          "-map_metadata",
          "0",
          "-c:a",
          "libmp3lame",
          "-b:a",
          br,
          task.outFile,
        ],
        task.sourceProfile.durationSeconds,
        (fileProgress) => {
          emitExportProgress({
            requestId,
            status: "transcoding",
            stem: task.stemRaw,
            fileIndex: index + 1,
            fileCount: tasks.length,
            fileProgress: fileProgress * 100,
            totalProgress:
              totalWeight > 0
                ? ((completedWeight + taskWeight * fileProgress) / totalWeight) *
                  100
                : fileProgress * 100,
            detail: `Transcoding ${task.stemRaw} to MP3...`,
            format: task.format,
            outputPath: task.outFile,
          });
        },
      );
    } else {
      throw new Error(`Unsupported export path for format ${task.format}`);
    }

    completedWeight += taskWeight;
    exported[task.stemRaw] = task.outFile;
  }

  emitExportProgress({
    requestId,
    status: "completed",
    fileCount: tasks.length,
    totalProgress: 100,
    detail: "Export complete",
    format: String(format || "wav").toLowerCase(),
  });

  return { exported };
}

function getPreviewCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "previews");
}

function createPreviewDirForInput(inputFile: string) {
  const base = getPreviewCacheBaseDir();
  safeMkdir(base);

  const stamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace("T", "_")
    .replace("Z", "");

  const baseName = sanitizeForPathSegment(
    path.basename(inputFile, path.extname(inputFile)),
  );

  const dirName = `${stamp}__${baseName || "audio"}__${randomUUID().slice(0, 8)}`;
  const full = path.join(base, dirName);
  safeMkdir(full);
  return full;
}

function resolveEffectiveModelId(modelId: any, ensembleConfig: any): string {
  // If ensembleConfig is present, the backend expects model_id="ensemble".
  if (ensembleConfig && Array.isArray(ensembleConfig.models) && ensembleConfig.models.length > 0) {
    return "ensemble";
  }
  if (typeof modelId === "string" && modelId.trim()) return modelId.trim();
  throw new Error(
    "Missing modelId. This is a preset/config bug (modelId must be a non-empty string).",
  );
}

function shortHash(input: string): string {
  return createHash("sha1").update(input).digest("hex").slice(0, 12);
}

type MissingAudioCode =
  | "MISSING_CACHE_FILE"
  | "STALE_SESSION"
  | "MISSING_SOURCE_FILE";

function isPathInsideDir(targetPath: string, parentDir: string): boolean {
  const target = path.resolve(targetPath).toLowerCase();
  const parent = path.resolve(parentDir).toLowerCase();
  return target === parent || target.startsWith(`${parent}${path.sep}`);
}

function classifyMissingAudioPath(missingPath: string): {
  code: MissingAudioCode;
  hint: string;
} {
  const previewBaseDir = getPreviewCacheBaseDir();
  const insidePreviewCache = isPathInsideDir(missingPath, previewBaseDir);
  if (insidePreviewCache) {
    const parentDir = path.dirname(missingPath);
    if (!fs.existsSync(parentDir)) {
      return {
        code: "STALE_SESSION",
        hint: "Historical session cache is no longer available. Run a new separation to regenerate stems.",
      };
    }
    return {
      code: "MISSING_CACHE_FILE",
      hint: "Cached preview file is missing. Run a new separation to regenerate this stem.",
    };
  }
  return {
    code: "MISSING_SOURCE_FILE",
    hint: "Source file is missing or moved. Verify the path and try again.",
  };
}

function createCodedError(
  code: MissingAudioCode,
  message: string,
  hint: string,
  extra: Record<string, any> = {},
): Error & { code: MissingAudioCode; hint: string; [k: string]: any } {
  const error = new Error(message) as Error & {
    code: MissingAudioCode;
    hint: string;
    [k: string]: any;
  };
  error.code = code;
  error.hint = hint;
  Object.assign(error, extra);
  return error;
}

function resolveMissingPreviewAudioPath(missingPath: string): string | null {
  try {
    const parent = path.dirname(missingPath);
    if (!fs.existsSync(parent)) return null;

    const requestedName = path.basename(missingPath).toLowerCase();
    const requestedStem = requestedName.replace(path.extname(requestedName), "");

    const wavs: string[] = [];
    const maxFiles = 200;
    const walk = (dir: string) => {
      if (wavs.length >= maxFiles) return;
      let entries: fs.Dirent[] = [];
      try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const e of entries) {
        if (wavs.length >= maxFiles) return;
        const full = path.join(dir, e.name);
        if (e.isDirectory()) {
          walk(full);
        } else if (e.isFile() && e.name.toLowerCase().endsWith(".wav")) {
          wavs.push(full);
        }
      }
    };

    walk(parent);
    if (wavs.length === 0) return null;

    const score = (p: string): number => {
      const b = path.basename(p).toLowerCase();
      let s = 0;
      if (b === `${requestedStem}.wav`) s += 100;
      if (requestedStem === "instrumental") {
        if (
          b.includes("(instrumental)") ||
          b.includes("_instrumental_") ||
          b.includes(" instrumental ")
        ) {
          s += 50;
        }
        if (b.includes("(vocals)")) s -= 50;
      } else if (requestedStem === "vocals") {
        if (
          b.includes("(vocals)") ||
          b.includes("_vocals_") ||
          b.includes(" vocals ")
        ) {
          s += 50;
        }
        if (b.includes("(instrumental)")) s -= 50;
      } else if (b.includes(requestedStem)) {
        s += 20;
      }

      const rel = path.relative(parent, p);
      const depth = rel.split(path.sep).length;
      s += Math.max(0, 10 - depth);
      return s;
    };

    let best: string | null = null;
    let bestScore = -Infinity;
    for (const p of wavs) {
      const sc = score(p);
      if (sc > bestScore) {
        bestScore = sc;
        best = p;
      }
    }
    return best;
  } catch {
    return null;
  }
}

const PLAYBACK_AUDIO_EXTENSIONS = new Set([
  ".wav",
  ".flac",
  ".mp3",
  ".ogg",
  ".m4a",
  ".aac",
  ".wma",
  ".aiff",
]);

type PlaybackResolveIssue = {
  code: MissingAudioCode;
  hint: string;
  originalPath?: string;
};

type PlaybackMetadataLike = {
  sourceKind?: string;
  previewDir?: string;
  savedDir?: string;
};

function listPlaybackAudioFiles(baseDir: string, maxFiles = 200): string[] {
  const files: string[] = [];
  const walk = (dir: string) => {
    if (files.length >= maxFiles) return;

    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      if (files.length >= maxFiles) return;

      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
        continue;
      }

      if (
        entry.isFile() &&
        PLAYBACK_AUDIO_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
      ) {
        files.push(full);
      }
    }
  };

  walk(baseDir);
  return files;
}

function inferStemKeyFromPlaybackPath(filePath: string): string {
  const base = path.basename(filePath, path.extname(filePath)).toLowerCase();
  if (base.includes("instrumental") || base.includes("inst")) return "instrumental";
  if (base.includes("vocal")) return "vocals";
  if (base.includes("drum")) return "drums";
  if (base.includes("bass")) return "bass";
  if (base.includes("guitar")) return "guitar";
  if (base.includes("piano") || base.includes("keys")) return "piano";
  if (base.includes("other")) return "other";
  return sanitizeForPathSegment(base) || "stem";
}

function scorePlaybackCandidate(
  candidatePath: string,
  {
    stemName,
    requestedName,
    searchRoot,
  }: { stemName?: string; requestedName?: string; searchRoot: string },
): number {
  const base = path.basename(candidatePath, path.extname(candidatePath)).toLowerCase();
  const stem = String(stemName || "").trim().toLowerCase();
  const requested = String(requestedName || "").trim().toLowerCase();

  let score = 0;
  if (requested && `${base}${path.extname(candidatePath).toLowerCase()}` === requested) {
    score += 100;
  }
  if (requested && base === requested.replace(path.extname(requested), "")) {
    score += 80;
  }
  if (stem && base === stem) {
    score += 70;
  }
  if (stem && base.includes(stem)) {
    score += 35;
  }

  if (stem === "instrumental" && base.includes("vocal")) score -= 40;
  if (stem === "vocals" && base.includes("inst")) score -= 40;

  const rel = path.relative(searchRoot, candidatePath);
  const depth = rel.split(path.sep).length;
  score += Math.max(0, 10 - depth);

  return score;
}

function resolvePlaybackFilePath(
  filePath: string | undefined,
  options: { previewDir?: string; savedDir?: string; stemName?: string } = {},
): string | null {
  const requestedPath = String(filePath || "").trim();
  if (requestedPath && fs.existsSync(requestedPath)) {
    return requestedPath;
  }

  if (requestedPath) {
    const fallback = resolveMissingPreviewAudioPath(requestedPath);
    if (fallback && fs.existsSync(fallback)) {
      return fallback;
    }
  }

  const searchRoots = [options.previewDir, options.savedDir]
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .filter(Boolean)
    .filter((value, index, self) => self.indexOf(value) === index)
    .filter((value) => fs.existsSync(value));

  const requestedName = requestedPath ? path.basename(requestedPath) : "";
  let best: string | null = null;
  let bestScore = -Infinity;

  for (const searchRoot of searchRoots) {
    const candidates = listPlaybackAudioFiles(searchRoot);
    for (const candidatePath of candidates) {
      const score = scorePlaybackCandidate(candidatePath, {
        stemName: options.stemName,
        requestedName,
        searchRoot,
      });
      if (score > bestScore) {
        bestScore = score;
        best = candidatePath;
      }
    }
  }

  return best;
}

function resolvePlaybackStems(
  outputFiles: Record<string, string> | undefined,
  playback?: PlaybackMetadataLike,
): {
  stems: Record<string, string>;
  issues: Record<string, PlaybackResolveIssue>;
} {
  const stems: Record<string, string> = {};
  const issues: Record<string, PlaybackResolveIssue> = {};
  const sourceEntries = Object.entries(outputFiles || {});

  for (const [stemName, originalPath] of sourceEntries) {
    const resolvedPath = resolvePlaybackFilePath(originalPath, {
      previewDir: playback?.previewDir,
      savedDir: playback?.savedDir,
      stemName,
    });

    if (resolvedPath) {
      stems[stemName] = resolvedPath;
      continue;
    }

    const missing = classifyMissingAudioPath(
      originalPath ||
        path.join(playback?.previewDir || playback?.savedDir || "", `${stemName}.wav`),
    );
    issues[stemName] = {
      code: missing.code,
      hint: missing.hint,
      originalPath,
    };
  }

  if (
    sourceEntries.length === 0 &&
    typeof playback?.previewDir === "string" &&
    playback.previewDir.trim() &&
    fs.existsSync(playback.previewDir)
  ) {
    for (const candidatePath of listPlaybackAudioFiles(playback.previewDir)) {
      const stemName = inferStemKeyFromPlaybackPath(candidatePath);
      if (!stems[stemName]) {
        stems[stemName] = candidatePath;
      }
    }
  }

  return { stems, issues };
}

function cleanupPreviewCache() {
  try {
    const base = getPreviewCacheBaseDir();
    if (!fs.existsSync(base)) return;

    const entries = fs
      .readdirSync(base, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => {
        const full = path.join(base, e.name);
        let mtime = 0;
        try {
          mtime = fs.statSync(full).mtimeMs;
        } catch {
          // ignore
        }
        return { full, mtime };
      })
      .sort((a, b) => b.mtime - a.mtime);

    const cutoff = Date.now() - PREVIEW_CACHE_MAX_AGE_DAYS * 24 * 60 * 60 * 1000;

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (i < PREVIEW_CACHE_KEEP_LAST) continue;
      if (entry.mtime && entry.mtime > cutoff) continue;
      try {
        fs.rmSync(entry.full, { recursive: true, force: true });
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
}

// Global error handlers
process.on("uncaughtException", (error) => {
  log("CRITICAL: Uncaught Exception:", error);
  // Optionally show a dialog to the user
  dialog.showErrorBox(
    "Application Error",
    `An unexpected error occurred: ${error.message}\n\nPlease check the logs for more details.`,
  );
});

process.on("unhandledRejection", (reason) => {
  log("CRITICAL: Unhandled Rejection:", reason);
});

let backendProcess: ReturnType<typeof spawn> | null = null;
// Historically this code called the backend process `pythonBridge`.
// Keep the name for compatibility with existing helper functions.
let pythonBridge: ReturnType<typeof spawn> | null = null;

let hfAuthWindow: BrowserWindow | null = null;

function getStoredModelsDir(): string | null {
  try {
    // Read zustand persist storage from localStorage file
    // On Windows: %APPDATA%/[appName]/Local Storage/leveldb
    // Simpler: read from a config file we create
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      return config.modelsDir || null;
    }
  } catch (e) {
    log("Could not read modelsDir from config:", e);
  }
  return null;
}

function readAppConfig(): Record<string, any> {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    }
  } catch (e) {
    log("Could not read app-config.json:", e);
  }
  return {};
}

function writeAppConfig(partial: Record<string, any>): boolean {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    const existingConfig = readAppConfig();
    const newConfig = { ...existingConfig, ...partial };
    fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
    return true;
  } catch (e) {
    log("Could not write app-config.json:", e);
    return false;
  }
}

function getStoredHuggingFaceToken(): string | null {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      const token = config.hfToken;
      if (typeof token === "string" && token.trim()) return token.trim();
    }
  } catch (e) {
    log("Could not read hfToken from config:", e);
  }
  return null;
}

function setStoredHuggingFaceToken(token: string | null): { success: boolean; error?: string } {
  try {
    const trimmed = typeof token === "string" ? token.trim() : "";
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    const existingConfig = readAppConfig();

    if (!trimmed) {
      if (Object.prototype.hasOwnProperty.call(existingConfig, "hfToken")) {
        delete existingConfig.hfToken;
        fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
      }
      return { success: true };
    }

    // Basic sanity check (HF tokens are typically fairly long). Avoid rejecting valid tokens.
    if (trimmed.length < 20) {
      return { success: false, error: "Token looks too short." };
    }

    existingConfig.hfToken = trimmed;
    fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
    return { success: true };
  } catch (e: any) {
    log("Failed to set hfToken:", e);
    return { success: false, error: e?.message || "Failed to save token." };
  }
}

// Bridge auto-restart state
const MAX_BRIDGE_RESTARTS = 3;
let bridgeRestartCount = 0;
let lastBridgeRestart = 0;
let isAppQuitting = false;
let manualBridgeRestartPending = false;

function requestBridgeRestart(reason: string) {
  log("Manual backend restart requested:", reason);
  if (!backendProcess) {
    ensureBackend();
    return;
  }
  manualBridgeRestartPending = true;
  try {
    backendProcess.kill();
  } catch (e) {
    log("Failed to kill backend bridge:", e);
    backendProcess = null;
    ensureBackend();
  }
}

function shouldUseRustBackend(): boolean {
  const v = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
  if (v === "python") return false;
  if (v === "rust") return true;
  // Default: prefer Rust backend.
  return true;
}

function resolveBundledPythonForRustBackend(): string | null {
  // The Rust backend spawns `scripts/inference.py`. On Windows, it defaults to using the `py` launcher,
  // which may select a Python without our dependencies. Provide an explicit interpreter when possible.
  try {
    const venv = (process.env.VIRTUAL_ENV || "").trim();
    if (venv) {
      if (process.platform === "win32") {
        const p = path.join(venv, "Scripts", "python.exe");
        try {
          if (fs.existsSync(p)) return p;
        } catch {
          // ignore
        }
      } else {
        const p = path.join(venv, "bin", "python");
        try {
          if (fs.existsSync(p)) return p;
        } catch {
          // ignore
        }
      }
    }

    if (process.platform === "win32") {
      const candidates = [
        // Packaged app: extraResources ships ".venv" next to the app
        path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
        // Dev repo: root venv (preferred)
        path.join(process.cwd(), ".venv", "Scripts", "python.exe"),
        // Dev repo: prefer StemSepApp venv if present
        path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
        // Dev repo: prefer StemSepApp venv from current working directory (more robust than __dirname)
        path.join(process.cwd(), "StemSepApp", ".venv", "Scripts", "python.exe"),
        // Dev repo: root venv (legacy)
        path.join(__dirname, "../../.venv/Scripts/python.exe"),
      ];

      for (const c of candidates) {
        try {
          if (fs.existsSync(c)) return c;
        } catch {
          // ignore
        }
      }
      return null;
    }

    const candidates = [
      path.join(process.resourcesPath, ".venv", "bin", "python"),
      path.join(process.cwd(), ".venv", "bin", "python"),
      path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
      path.join(process.cwd(), "StemSepApp", ".venv", "bin", "python"),
      path.join(__dirname, "../../.venv/bin/python"),
    ];

    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) return c;
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
  return null;
}

function ensureBackend() {
  return ensurePythonBridge();
}

function configureProviderSessionPermissions(provider: RemoteLibraryProvider) {
  const ses = getRemoteSession(provider);
  if ((ses as any).__stemsepPermissionsConfigured) return;

  ses.setPermissionCheckHandler((_webContents, permission) => {
    if (String(permission) === "speaker-selection") return true;
    return false;
  });

  ses.setPermissionRequestHandler(
    (_webContents, permission, callback) => {
      if (String(permission) === "speaker-selection") {
        callback(true);
        return;
      }
      callback(false);
    },
  );

  (ses as any).__stemsepPermissionsConfigured = true;
}

function getQobuzAutomationWindow() {
  if (qobuzAutomationWindow && !qobuzAutomationWindow.isDestroyed()) {
    return qobuzAutomationWindow;
  }

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
    qobuzSinkIdByPlaybackDeviceId.clear();
  });

  return qobuzAutomationWindow;
}

function openRemoteSourceAuthWindow(provider: RemoteLibraryProvider) {
  const existing = remoteAuthWindows.get(provider);
  if (existing && !existing.isDestroyed()) {
    existing.focus();
    return existing;
  }

  const config = getRemoteProviderConfig(provider);
  configureProviderSessionPermissions(provider);
  const authWindow = new BrowserWindow({
    width: 1180,
    height: 860,
    parent: mainWindow || undefined,
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
      qobuzObservedPlaySession = true;
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

  authWindow
    .loadURL(config.loginUrl)
    .catch(async () => {
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

function openHuggingFaceAuthWindow() {
  if (hfAuthWindow && !hfAuthWindow.isDestroyed()) {
    hfAuthWindow.focus();
    return;
  }

  hfAuthWindow = new BrowserWindow({
    width: 520,
    height: 360,
    resizable: false,
    minimizable: false,
    maximizable: false,
    modal: !!mainWindow,
    parent: mainWindow || undefined,
    title: "Authorize Hugging Face",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  hfAuthWindow.on("closed", () => {
    hfAuthWindow = null;
  });

  const html = `<!doctype html>
  <html>
    <head>
      <meta charset="utf-8" />
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
      <title>Authorize Hugging Face</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color: #111; }
        h1 { font-size: 18px; margin: 0 0 8px; }
        p { margin: 8px 0; line-height: 1.35; color: #333; }
        .box { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }
        label { display: block; font-weight: 600; margin: 12px 0 6px; }
        input { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 13px; }
        .row { display: flex; gap: 10px; margin-top: 12px; }
        button { padding: 10px 12px; border-radius: 8px; border: 1px solid #bbb; background: white; cursor: pointer; font-weight: 600; }
        button.primary { background: #1677ff; border-color: #1677ff; color: white; }
        button.danger { background: #fff; border-color: #d33; color: #d33; }
        .status { margin-top: 10px; font-size: 12px; color: #444; }
        .hint { font-size: 12px; color: #555; }
        a { color: #1677ff; text-decoration: none; }
      </style>
    </head>
    <body>
      <h1>Authorize Hugging Face (optional)</h1>
      <p class="hint">Only needed for gated/private model downloads. Public models work without this.</p>
      <div class="box">
        <div id="current" class="status">Checking status…</div>
        <label for="token">Token</label>
        <input id="token" type="password" placeholder="Paste your Hugging Face access token" autocomplete="off" />
        <div class="row">
          <button class="primary" id="save">Save token</button>
          <button class="danger" id="clear">Clear token</button>
          <button id="close">Close</button>
        </div>
        <p class="hint">Create a token in <a href="#" id="openTokens">huggingface.co/settings/tokens</a>.</p>
        <div id="msg" class="status"></div>
      </div>
      <script>
        const current = document.getElementById('current');
        const msg = document.getElementById('msg');
        const tokenEl = document.getElementById('token');

        async function refresh() {
          try {
            const st = await window.electronAPI.getHuggingFaceAuthStatus();
            current.textContent = st && st.configured ? 'Status: Token configured' : 'Status: Not configured';
          } catch (e) {
            current.textContent = 'Status: Unknown';
          }
        }

        document.getElementById('openTokens').addEventListener('click', async (e) => {
          e.preventDefault();
          await window.electronAPI.openExternalUrl('https://huggingface.co/settings/tokens');
        });

        document.getElementById('save').addEventListener('click', async () => {
          msg.textContent = '';
          const t = tokenEl.value || '';
          const res = await window.electronAPI.setHuggingFaceToken(t);
          if (res && res.success) {
            tokenEl.value = '';
            msg.textContent = 'Saved. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to save token.';
          }
        });

        document.getElementById('clear').addEventListener('click', async () => {
          msg.textContent = '';
          const res = await window.electronAPI.clearHuggingFaceToken();
          if (res && res.success) {
            msg.textContent = 'Cleared. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to clear token.';
          }
        });

        document.getElementById('close').addEventListener('click', () => window.close());
        refresh();
      </script>
    </body>
  </html>`;

  hfAuthWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(html)}`,
  );
}

function ensurePythonBridge() {
  if (pythonBridge) return pythonBridge;

  const hfToken = getStoredHuggingFaceToken();
  const modelsDir = getStoredModelsDir();

  // Optional: spawn Rust backend instead of Python.
  // This keeps renderer/UI unchanged while we migrate backend functionality.
  if (shouldUseRustBackend()) {
    const backendPref = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
    const rustExe = (() => {
      const candidates: string[] = [];
      if (process.platform === "win32") {
        candidates.push(
          path.join(process.resourcesPath, "stemsep-backend.exe"),
          path.join(__dirname, "../../stemsep-backend/target/release/stemsep-backend.exe"),
          path.join(__dirname, "../../stemsep-backend/target/debug/stemsep-backend.exe"),
        );
      } else {
        candidates.push(
          path.join(process.resourcesPath, "stemsep-backend"),
          path.join(__dirname, "../../stemsep-backend/target/release/stemsep-backend"),
          path.join(__dirname, "../../stemsep-backend/target/debug/stemsep-backend"),
        );
      }

      for (const c of candidates) {
        try {
          if (fs.existsSync(c)) return c;
        } catch {
          // ignore
        }
      }
      return null;
    })();

    if (rustExe) {
      const rustArgs: string[] = [];

      const assetsDir = (() => {
        const candidates: string[] = [
          path.join(process.resourcesPath, "StemSepApp", "assets"),
          path.join(__dirname, "../../StemSepApp/assets"),
        ];

        for (const c of candidates) {
          try {
            if (fs.existsSync(c)) return c;
          } catch {
            // ignore
          }
        }
        return null;
      })();

      if (assetsDir) {
        rustArgs.push("--assets-dir", assetsDir);
      }
      if (modelsDir) {
        rustArgs.push("--models-dir", modelsDir);
      }

      const explicitPython = resolveBundledPythonForRustBackend();

      if (explicitPython) {
        log("Pinning Rust backend Python via STEMSEP_PYTHON:", explicitPython);
      } else {
        log(
          "WARNING: Could not resolve explicit Python interpreter for Rust backend; it may fall back to 'py' launcher.",
        );
      }

      log("Spawning Rust backend:", rustExe, rustArgs);
      pythonBridge = spawn(rustExe, rustArgs, {
        cwd: path.dirname(rustExe),
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          // IMPORTANT: default to Rust-native separation/preflight.
          // This avoids requiring the python-bridge proxy for `separation_preflight`, which can
          // break dev runs when the bridge script isn't present.
          STEMSEP_PREFER_RUST_SEPARATION:
            (process.env.STEMSEP_PREFER_RUST_SEPARATION || "").trim() || "1",
          // Ensure inference uses the same Python environment as the app.
          ...(explicitPython ? { STEMSEP_PYTHON: explicitPython } : {}),
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
          ...(hfToken
            ? {
                HF_TOKEN: hfToken,
                HUGGINGFACE_HUB_TOKEN: hfToken,
                STEMSEP_HF_TOKEN: hfToken,
              }
            : {}),
        },
      });

      attachBackendStdioLogging("rust");

      pythonBridge.stderr?.on("data", (data) => {
        console.error("Rust backend stderr:", data.toString());
      });

      backendProcess = pythonBridge;

      pythonBridge.stdout?.setMaxListeners(50);
      pythonBridge.stderr?.setMaxListeners(50);

      // Continue with shared stdout handling + restart logic below.
    } else {
      const msg = "Rust backend not found.";
      if (backendPref === "rust") {
        log("CRITICAL: STEMSEP_BACKEND=rust but Rust binary not found.");
        dialog.showErrorBox(
          "Startup Error",
          "STEMSEP_BACKEND is set to 'rust' but stemsep-backend binary was not found.\n\nRebuild it (cargo build --release) or adjust STEMSEP_BACKEND.",
        );
        return null;
      }
      log("WARNING:", msg, "Falling back to Python backend.");
    }
  }

  // If Rust backend was spawned above, skip Python-specific interpreter/script resolution.
  if (pythonBridge) {
    log("Backend bridge spawned (rust mode)");
  } else {

  // Resolve a usable Python interpreter.
  // Prefer packaged/bundled venv if present, otherwise fall back to system Python on PATH.
  //
  // Why: hardcoding repo-root ".venv" breaks dev setups where the venv is located elsewhere
  // (e.g. StemSepApp/.venv) or not present at all.
  const pythonPath = (() => {
    if (process.platform === "win32") {
      const candidates = [
        // Packaged app: extraResources ships ".venv" next to the app
        path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
        // Dev repo: prefer backend venv if present
        path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
        // Dev repo: root venv (legacy)
        path.join(__dirname, "../../.venv/Scripts/python.exe"),
        // Fallback: system installs (common locations)
        "python",
        "python3",
        "py",
        "py.exe",
      ];

      for (const c of candidates) {
        try {
          if (c.includes("\\") || c.includes("/")) {
            if (fs.existsSync(c)) return c;
          } else {
            // command on PATH
            return c;
          }
        } catch {
          // ignore and continue
        }
      }
      return "python";
    }

    // macOS/Linux
    const candidates = [
      path.join(process.resourcesPath, ".venv", "bin", "python"),
      path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
      path.join(__dirname, "../../.venv/bin/python"),
      "python3",
      "python",
    ];
    for (const c of candidates) {
      try {
        if (c.includes("/")) {
          if (fs.existsSync(c)) return c;
        } else {
          return c;
        }
      } catch {
        // ignore and continue
      }
    }
    return "python3";
  })();

  // Resolve python-bridge.py location.
  // In dev: it lives next to the project root (electron-poc/python-bridge.py).
  // In packaged builds: electron-builder `extraResources` places it under `process.resourcesPath`.
  let scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, "python-bridge.py")
    : path.join(__dirname, "../python-bridge.py");

  if (!fs.existsSync(scriptPath)) {
    const candidates = [
      // Dev fallback
      path.join(__dirname, "../python-bridge.py"),
      // Packaged (normal)
      path.join(process.resourcesPath, "python-bridge.py"),
      // Packaged (some builders place extraResources under app.asar.unpacked)
      path.join(process.resourcesPath, "app.asar.unpacked", "python-bridge.py"),
    ];
    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) {
          scriptPath = c;
          break;
        }
      } catch {
        // ignore and continue
      }
    }
  }

  if (!fs.existsSync(scriptPath)) {
    log("CRITICAL: python-bridge.py not found at:", scriptPath);
    dialog.showErrorBox(
      "Startup Error",
      "Python backend was selected/fell back, but python-bridge.py is missing.\n\nSet STEMSEP_BACKEND=rust or restore the python bridge file.",
    );
    return null;
  }

  // Build arguments - include modelsDir if set
  const args = [scriptPath];
  const modelsDir = getStoredModelsDir();
  if (modelsDir) {
    args.push("--models-dir", modelsDir);
  }

  log("Spawning Python bridge:", pythonPath, args.join(" "));

  pythonBridge = spawn(pythonPath, args, {
    cwd: path.dirname(scriptPath),
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      ...(typeof modelsDir === "string" && modelsDir.trim()
        ? { STEMSEP_MODELS_DIR: modelsDir }
        : {}),
      ...(() => {
        const candidates: string[] = [
          path.join(process.resourcesPath, "StemSepApp", "assets"),
          path.join(__dirname, "../../StemSepApp/assets"),
        ];

        for (const c of candidates) {
          try {
            if (fs.existsSync(c)) return { STEMSEP_ASSETS_DIR: c };
          } catch {
            // ignore
          }
        }

        return {};
      })(),
      PYTHONIOENCODING: "utf-8",
      ...(hfToken
        ? {
            HF_TOKEN: hfToken,
            HUGGINGFACE_HUB_TOKEN: hfToken,
            STEMSEP_HF_TOKEN: hfToken,
          }
        : {}),
    },
  });

  attachBackendStdioLogging("python");

  pythonBridge.stderr?.on("data", (data) => {
    console.error("Python stderr:", data.toString());
  });

  backendProcess = pythonBridge;

  pythonBridge.stdout?.setMaxListeners(50);
  pythonBridge.stderr?.setMaxListeners(50);
  }

  // Attach a single stdout message router for responses + events.
  attachBackendMessageRouter(pythonBridge);

  pythonBridge.on("exit", (code) => {
    log("Python bridge exited with code:", code);
    rejectAllPendingBackendCommands(
      new Error(`Backend bridge exited with code ${code ?? "unknown"}`),
    );
    rejectBridgeReadyWaiters(
      new Error(`Backend bridge exited with code ${code ?? "unknown"}`),
    );
    detachBackendMessageRouter();
    pythonBridge = null;
    backendProcess = null;

    // Manual restart: don't count towards crash restart budget.
    if (manualBridgeRestartPending) {
      manualBridgeRestartPending = false;
      bridgeRestartCount = 0;
      lastBridgeRestart = Date.now();

      if (!isAppQuitting) {
        setTimeout(() => {
          ensurePythonBridge();
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("bridge-reconnected");
          }
        }, 200);
      }
      return;
    }

    // Skip restart logic if app is quitting
    if (isAppQuitting) {
      return;
    }

    // Reset restart count if it's been more than 60 seconds since last restart
    const now = Date.now();
    if (now - lastBridgeRestart > 60000) {
      bridgeRestartCount = 0;
    }

    // Attempt restart if we haven't exceeded the limit
    if (bridgeRestartCount < MAX_BRIDGE_RESTARTS) {
      bridgeRestartCount++;
      lastBridgeRestart = now;
      log(
        `Attempting to restart Python bridge (attempt ${bridgeRestartCount}/${MAX_BRIDGE_RESTARTS})`,
      );

      // Delay restart slightly to avoid rapid restart loops
      setTimeout(() => {
        if (!isAppQuitting) {
          ensurePythonBridge();
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("bridge-reconnected");
          }
        }
      }, 1000 * bridgeRestartCount); // Exponential backoff: 1s, 2s, 3s
    } else {
      log("CRITICAL: Python bridge failed to restart after maximum attempts");
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("bridge-connection-failed");
        dialog.showErrorBox(
          "Backend Connection Failed",
          "The audio processing backend has stopped responding and could not be restarted. Please restart the application.",
        );
      }
    }
  });

  return pythonBridge;
}

function resolveAssetsDirForLocalOps(): string | null {
  const candidates: string[] = [
    path.join(process.resourcesPath, "StemSepApp", "assets"),
    path.join(__dirname, "../../StemSepApp/assets"),
    path.join(process.cwd(), "StemSepApp", "assets"),
  ];
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch {
      // ignore
    }
  }
  return null;
}

function urlBasename(url: unknown): string | null {
  if (typeof url !== "string") return null;
  const u = url.split("?")[0].replace(/\/+$/, "");
  const parts = u.split("/");
  const last = parts[parts.length - 1];
  return last ? last : null;
}

function removeModelLocal(modelId: string): { success: true; removedFiles: string[] } {
  const modelsDir = getStoredModelsDir();
  if (!modelsDir) {
    throw new Error("Models directory is not configured.");
  }

  const assetsDir = resolveAssetsDirForLocalOps();
  if (!assetsDir) {
    throw new Error("Assets directory not found (cannot resolve model registry).");
  }

  const registryPath = path.join(assetsDir, "models.json.bak");
  let model: any = null;
  try {
    const raw = fs.readFileSync(registryPath, "utf-8");
    const json = JSON.parse(raw);
    const models = Array.isArray(json?.models) ? json.models : [];
    model = models.find((m: any) => m && m.id === modelId) || null;
  } catch (e: any) {
    throw new Error(
      `Failed to read model registry at ${registryPath}: ${e?.message || String(e)}`,
    );
  }

  const removed: string[] = [];

  const links = model?.links && typeof model.links === "object" ? model.links : null;
  const ckptBase = urlBasename(links?.checkpoint);
  const cfgBase = urlBasename(links?.config);

  // NOTE: Avoid deleting generic config.yaml/config.yml because it may be shared between models.
  const genericConfig = new Set(["config.yaml", "config.yml"]);

  const candidateNames = new Set<string>();

  if (ckptBase) candidateNames.add(ckptBase);
  if (cfgBase && !genericConfig.has(cfgBase.toLowerCase())) candidateNames.add(cfgBase);

  // Per-model aliases we create / accept
  for (const ext of [
    ".ckpt",
    ".chpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".onnx",
    ".yaml",
    ".yml",
  ]) {
    candidateNames.add(`${modelId}${ext}`);
  }

  // Known special-case aliases
  if (modelId === "mel-band-roformer-kim") {
    candidateNames.add("MelBandRoformer.ckpt");
    candidateNames.add("vocals_mel_band_roformer.ckpt");
    candidateNames.add("vocals_mel_band_roformer.yaml");
  }

  for (const name of Array.from(candidateNames)) {
    const p = path.join(modelsDir, name);
    try {
      if (fs.existsSync(p) && fs.statSync(p).isFile()) {
        fs.unlinkSync(p);
        removed.push(p);
      }
    } catch {
      // ignore individual failures to keep batch delete resilient
    }
  }

  return { success: true, removedFiles: removed };
}

// Health check state
let healthCheckInterval: NodeJS.Timeout | null = null;
let consecutiveHealthCheckFailures = 0;
const HEALTH_CHECK_INTERVAL_MS = 30000; // 30 seconds
const MAX_HEALTH_CHECK_FAILURES = 2;

function startHealthChecks() {
  if (healthCheckInterval) return;

  healthCheckInterval = setInterval(async () => {
    if (isAppQuitting || !backendProcess) return;
    if (playbackCaptureSessions.size > 0) {
      consecutiveHealthCheckFailures = 0;
      return;
    }

    try {
      // Send ping with short timeout
      await sendPythonCommand("ping", {}, 5000);
      consecutiveHealthCheckFailures = 0;
    } catch (error) {
      consecutiveHealthCheckFailures++;
      log(
        `Health check failed (${consecutiveHealthCheckFailures}/${MAX_HEALTH_CHECK_FAILURES}): ${error}`,
      );

      if (consecutiveHealthCheckFailures >= MAX_HEALTH_CHECK_FAILURES) {
        log("Bridge unresponsive, forcing restart...");
        if (backendProcess) {
          backendProcess.kill("SIGKILL");
          backendProcess = null;
        }
        // Auto-restart will be triggered by the exit handler
        consecutiveHealthCheckFailures = 0;
      }
    }
  }, HEALTH_CHECK_INTERVAL_MS);
}

function stopHealthChecks() {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
  }
}

let mainWindow: InstanceType<typeof BrowserWindow> | null = null;
let isCreatingMainWindow = false;

const lastProgressByJobId = new Map<string, number>();
const modelInfoByJobId = new Map<
  string,
  {
    requestedModelId: string;
    effectiveModelId: string;
    inputFile: string;
    finalOutputDir: string;
    previewDir: string;
    sourceAudioProfile?: AudioSourceProfile;
    stagingDecision?: SourceStagingDecision;
    outputFiles?: Record<string, string>;
  }
>();

// Avoid multiple Electron instances (common in dev when scripts rerun), which otherwise
// results in multiple app windows.
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
  // IMPORTANT: Return immediately to prevent the rest of the script (e.g., app.whenReady)
  // from running and opening a window in this secondary process.
  process.exit(0);
} else {
  app.on("second-instance", () => {
    // In dev it’s common to accidentally launch Electron twice (e.g. script reruns).
    // If this happens before `whenReady` has created the window, calling createWindow()
    // here can race and produce two windows. Always wait for readiness and rely on
    // createWindow()’s internal guard.
    app
      .whenReady()
      .then(() => {
        try {
          if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.show();
            mainWindow.focus();
          } else {
            createWindow();
          }
        } catch {
          // ignore
        }
      })
      .catch(() => {
        // ignore
      });
  });
}

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

function createWindow() {
  // Hard guard to avoid duplicate windows (e.g. second-instance race in dev).
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
      return;
    }
  } catch {
    // ignore
  }

  if (isCreatingMainWindow) return;
  isCreatingMainWindow = true;

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Make sure the window is visible (helps if it was spawned off-screen/minimized).
  try {
    ensureWindowOnScreen(mainWindow);
    mainWindow.center();
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  } catch {
    // ignore
  }

  mainWindow.once("ready-to-show", () => {
    try {
      if (mainWindow) ensureWindowOnScreen(mainWindow);
      if (mainWindow?.isMinimized()) mainWindow.restore();
      mainWindow?.show();
      mainWindow?.focus();
    } catch {
      // ignore
    }
  });

  const isDev = !app.isPackaged;

  if (isDev) {
    // Briefly bring the window above others so it's obvious it launched.
    try {
      mainWindow.setAlwaysOnTop(true);
      setTimeout(() => {
        try {
          mainWindow?.setAlwaysOnTop(false);
        } catch {
          // ignore
        }
      }, 1500);
    } catch {
      // ignore
    }

    const loadURLWithRetry = (url: string, retries = 10) => {
      mainWindow?.loadURL(url)
        .then(() => {
          try {
            if (mainWindow) ensureWindowOnScreen(mainWindow);
            if (mainWindow?.isMinimized()) mainWindow.restore();
            mainWindow?.show();
            mainWindow?.focus();
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

    // On Windows, `localhost` often resolves to IPv6 (::1) first, while Vite typically
    // listens on IPv4 only (0.0.0.0/127.0.0.1). Prefer 127.0.0.1 to avoid
    // intermittent `ERR_CONNECTION_REFUSED` during dev startup.
    const envDevUrl = (process.env.VITE_DEV_SERVER_URL || "").trim();
    const rawDevUrl = envDevUrl || "http://127.0.0.1:5173/";
    const normalizedDevUrl = rawDevUrl
      .replace(/^http:\/\/localhost(?=[:/]|$)/i, "http://127.0.0.1")
      .replace(/^https:\/\/localhost(?=[:/]|$)/i, "https://127.0.0.1");

    log("Dev renderer URL", { envDevUrl: envDevUrl || null, normalizedDevUrl });

    loadURLWithRetry(normalizedDevUrl);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow
      // Renderer output is built by Vite into dist-renderer/ (kept separate from electron-builder output dir).
      .loadFile(path.join(__dirname, "../dist-renderer/index.html"))
      .catch((e) => {
        log("CRITICAL: Failed to load index.html:", e);
        dialog.showErrorBox(
          "Startup Error",
          "Failed to load application resources. Please reinstall the application.",
        );
      });
  }

  mainWindow.on("unresponsive", () => {
    log("WARNING: Main window became unresponsive");
    dialog
      .showMessageBox(mainWindow!, {
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
          mainWindow?.reload();
        }
      });
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
    isCreatingMainWindow = false;
  });

  mainWindow.webContents.once("did-fail-load", () => {
    // Avoid getting stuck “creating” if load fails and the user tries again.
    isCreatingMainWindow = false;
  });

  mainWindow.once("ready-to-show", () => {
    isCreatingMainWindow = false;
  });
}

// Suppress Autofill warnings
app.commandLine.appendSwitch("disable-features", "AutofillServer");
app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");

app.whenReady().then(() => {
  cleanupPreviewCache();
  session.defaultSession.setDisplayMediaRequestHandler(
    async (_request, callback) => {
      try {
        const sources = await desktopCapturer.getSources({
          types: ["screen"],
          thumbnailSize: { width: 1, height: 1 },
          fetchWindowIcons: false,
        });
        callback({
          video: sources[0],
          audio: "loopback",
        });
      } catch (error: any) {
        log("[capture] failed to set display media handler", {
          error: error?.message || String(error),
        });
        callback({ video: undefined, audio: undefined });
      }
    },
    { useSystemPicker: false },
  );

  // Register 'media' protocol to serve local files
  protocol.registerFileProtocol("media", (request, callback) => {
    // URL format: media:///C:/Users/... or media:///path/to/file
    let filePath = request.url.replace("media://", "");
    // Remove leading slash for Windows absolute paths (e.g., /C:/Users -> C:/Users)
    if (
      filePath.startsWith("/") &&
      filePath.length > 2 &&
      filePath[2] === ":"
    ) {
      filePath = filePath.slice(1);
    }
    try {
      const decodedPath = decodeURIComponent(filePath);
      console.log("[media protocol] Serving file:", decodedPath);
      return callback(decodedPath);
    } catch (error) {
      console.error("Failed to serve media file:", error);
    }
  });

  log("App is ready, creating window.");
  createWindow();
  if (process.env.STEMSEP_QOBUZ_CAPTURE_SMOKETEST_QUERY) {
    setTimeout(() => {
      void maybeRunQobuzCaptureSmokeTest();
    }, 2500);
  }

  // Minimal app menu with optional Hugging Face authorization (no React UI changes).
  try {
    const template: Electron.MenuItemConstructorOptions[] = [
      ...(process.platform === "darwin"
        ? ([
            {
              label: app.name,
              submenu: [
                { role: "about" },
                { type: "separator" },
                { role: "quit" },
              ],
            },
          ] as Electron.MenuItemConstructorOptions[])
        : []),
      {
        label: "File",
        submenu: [
          process.platform === "darwin" ? { role: "close" } : { role: "quit" },
        ],
      },
      {
        label: "Hugging Face",
        submenu: [
          {
            label: "Authorize Hugging Face…",
            click: () => openHuggingFaceAuthWindow(),
          },
          {
            label: "Clear Hugging Face Token",
            click: () => {
              const res = setStoredHuggingFaceToken(null);
              if (res.success) requestBridgeRestart("cleared huggingface token");
            },
          },
          { type: "separator" },
          {
            label: "Open Token Settings…",
            click: async () => {
              await shell.openExternal("https://huggingface.co/settings/tokens");
            },
          },
        ],
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  } catch (e) {
    log("Failed to set application menu:", e);
  }

  startHealthChecks();

  // Optional: run an automated smoke separation to validate end-to-end behavior
  // without relying on manual UI clicks.
  void maybeRunSmokeSeparation();
});

async function maybeRunSmokeSeparation(): Promise<void> {
  const enabled = (process.env.STEMSEP_SMOKE_SEPARATION || "").trim();
  if (!enabled || enabled === "0" || enabled.toLowerCase() === "false") {
    return;
  }

  const inputFile =
    (process.env.STEMSEP_SMOKE_INPUT_FILE || "").trim() ||
    path.join(__dirname, "../strip mall.wav");
  const modelId =
    (process.env.STEMSEP_SMOKE_MODEL_ID || "").trim() || "bs-roformer-viperx-1297";
  const device = (process.env.STEMSEP_SMOKE_DEVICE || "").trim() || undefined;

  const outRoot = path.join(app.getPath("userData"), "smoke_runs");
  const runDir = path.join(outRoot, `run-${Date.now()}`);
  try {
    fs.mkdirSync(runDir, { recursive: true });
  } catch {
    // ignore
  }

  log("[smoke] Starting smoke separation", { inputFile, modelId, runDir, device });

  try {
    // 1) Preflight (same command as UI)
    const pre = await sendPythonCommandWithRetry(
      "separation_preflight",
      {
        file_path: inputFile,
        model_id: modelId,
        output_dir: runDir,
        device,
      },
      60000,
    );
    log("[smoke] preflight ok", pre);

    // 2) Start separation
    const start = await sendPythonCommand(
      "separate_audio",
      {
        file_path: inputFile,
        model_id: modelId,
        output_dir: runDir,
        device,
      },
      60000,
    );

    const jobId = start?.job_id;
    if (!jobId) throw new Error("Smoke separation: backend did not return job_id");
    log("[smoke] job started", { jobId, start });

    // 3) Wait for completion event
    const process = ensureBackend();
    if (!process) throw new Error("Smoke separation: backend unavailable");

    let completeMsg: any = null;

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error("Smoke separation timeout (60 minutes)"));
      }, 60 * 60 * 1000);

      const handler = createLineBuffer((line) => {
        try {
          const msg = JSON.parse(line);
          if (msg?.type === "separation_progress" && msg.job_id === jobId) {
            log(`[smoke] progress ${msg.progress}%`, msg.message || "");
          }
          if (msg?.job_id !== jobId) return;

          if (msg.type === "separation_complete") {
            completeMsg = msg;
            log("[smoke] complete", msg);
            cleanup();
            resolve();
          } else if (msg.type === "separation_error") {
            cleanup();
            reject(new Error(msg.error || "Smoke separation failed"));
          } else if (msg.type === "separation_cancelled") {
            cleanup();
            reject(new Error("Smoke separation cancelled"));
          }
        } catch {
          // ignore
        }
      });

      const cleanup = () => {
        clearTimeout(timeout);
        process.stdout?.removeListener("data", handler);
      };

      process.stdout?.on("data", handler);
    });

    // 4) Validate output exists
    const outputFiles: string[] = (() => {
      const byEvent = completeMsg?.output_files;
      if (byEvent && typeof byEvent === "object") {
        return Object.values(byEvent)
          .filter((v) => typeof v === "string")
          .map((v) => v as string);
      }
      return [];
    })();

    const existingFromEvent = outputFiles.filter((p) => {
      try {
        return fs.existsSync(p);
      } catch {
        return false;
      }
    });

    const listWavsRecursively = (baseDir: string, maxFiles = 50): string[] => {
      const out: string[] = [];
      const walk = (dir: string) => {
        if (out.length >= maxFiles) return;
        let entries: fs.Dirent[] = [];
        try {
          entries = fs.readdirSync(dir, { withFileTypes: true });
        } catch {
          return;
        }
        for (const e of entries) {
          if (out.length >= maxFiles) return;
          const full = path.join(dir, e.name);
          if (e.isDirectory()) {
            walk(full);
          } else if (e.isFile() && e.name.toLowerCase().endsWith(".wav")) {
            out.push(full);
          }
        }
      };
      walk(baseDir);
      return out;
    };

    const fallbackWavs = listWavsRecursively(runDir);

    log("[smoke] output_files (event)", completeMsg?.output_files || null);
    log("[smoke] output files (existing)", existingFromEvent);
    if (existingFromEvent.length === 0 && fallbackWavs.length === 0) {
      throw new Error(`Smoke separation completed but no wav outputs found under ${runDir}`);
    }

    log("[smoke] SUCCESS", { runDir });
  } catch (e: any) {
    log("[smoke] FAILED", e?.message || String(e));
  } finally {
    if ((process.env.STEMSEP_SMOKE_QUIT || "").trim() === "1") {
      log("[smoke] Quitting app (STEMSEP_SMOKE_QUIT=1)");
      app.quit();
    }
  }
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    log("All windows closed, quitting app.");
    app.quit();
  }
});

app.on("before-quit", () => {
  isAppQuitting = true;
  stopHealthChecks();
  for (const win of Array.from(remoteAuthWindows.values())) {
    try {
      win.close();
    } catch {
      // ignore
    }
  }
  if (qobuzAutomationWindow && !qobuzAutomationWindow.isDestroyed()) {
    try {
      qobuzAutomationWindow.close();
    } catch {
      // ignore
    }
    qobuzAutomationWindow = null;
  }
  if (backendProcess) {
    log("Killing backend process before quit");
    backendProcess.kill();
    backendProcess = null;
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Helper to create a line buffer for processing stream data
const createLineBuffer = (onLine: (line: string) => void) => {
  let buffer = "";
  return (data: Buffer) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    // Keep the last incomplete line in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.trim()) {
        onLine(line);
      }
    }
  };
};

type PendingBackendCommand = {
  command: string;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
  timeout: NodeJS.Timeout;
};

const pendingBackendCommands = new Map<string, PendingBackendCommand>();
const backendEventSubscribers = new Map<string, Set<(msg: any) => void>>();
let backendMessageRouter: ((data: Buffer) => void) | null = null;
let bridgeReadyWaiters: Array<{
  resolve: () => void;
  reject: (reason?: any) => void;
  timeout: NodeJS.Timeout;
}> = [];

function subscribeBackendEvent(
  eventType: string,
  handler: (msg: any) => void,
): () => void {
  const set = backendEventSubscribers.get(eventType) || new Set();
  set.add(handler);
  backendEventSubscribers.set(eventType, set);
  return () => {
    const existing = backendEventSubscribers.get(eventType);
    if (!existing) return;
    existing.delete(handler);
    if (existing.size === 0) {
      backendEventSubscribers.delete(eventType);
    }
  };
}

function emitBackendEvent(msg: any) {
  const eventType = typeof msg?.type === "string" ? msg.type : "";
  if (!eventType) return;
  const listeners = backendEventSubscribers.get(eventType);
  if (!listeners || listeners.size === 0) return;
  for (const cb of Array.from(listeners)) {
    try {
      cb(msg);
    } catch (e) {
      log(`backend event subscriber failed (${eventType})`, e);
    }
  }
}

function extractChunkCounters(message: string | undefined): {
  chunksDone?: number;
  chunksTotal?: number;
} {
  const text = String(message || "");
  const match = text.match(
    /\b(?:chunk|chunks|segment|segments)\D+(\d+)\D+(\d+)\b/i,
  );
  if (!match) return {};

  const chunksDone = Number(match[1]);
  const chunksTotal = Number(match[2]);
  if (!Number.isFinite(chunksDone) || !Number.isFinite(chunksTotal)) {
    return {};
  }
  return { chunksDone, chunksTotal };
}

function getStepLabelFromMeta(
  meta: Record<string, any> | undefined,
): string | undefined {
  if (!meta || typeof meta !== "object") return undefined;
  const step =
    meta.step && typeof meta.step === "object"
      ? (meta.step as Record<string, any>)
      : undefined;
  const phase = typeof meta.phase === "string" ? meta.phase : undefined;
  const stepName =
    step && typeof step.name === "string" && step.name.trim()
      ? step.name.trim()
      : undefined;
  const modelId =
    step && typeof step.model_id === "string" && step.model_id.trim()
      ? step.model_id.trim()
      : undefined;

  if (stepName && modelId) return `${stepName} (${modelId})`;
  if (stepName) return stepName;
  if (modelId) return modelId;
  if (!phase) return undefined;
  return phase
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function emitNormalizedSeparationEvent(msg: any) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const type = String(msg?.type || "");
  const jobId = typeof msg?.job_id === "string" ? msg.job_id : undefined;
  if (!jobId) return;

  const meta =
    msg?.meta && typeof msg.meta === "object"
      ? (msg.meta as Record<string, any>)
      : undefined;
  const step =
    meta?.step && typeof meta.step === "object"
      ? (meta.step as Record<string, any>)
      : undefined;
  const progress = Number(msg?.progress);
  const durationSeconds = Number(msg?.duration_seconds);
  const { chunksDone, chunksTotal } = extractChunkCounters(
    typeof msg?.message === "string" ? msg.message : undefined,
  );

  mainWindow.webContents.send("separation-progress-event", {
    jobId,
    kind:
      type === "separation_step_started"
        ? "step_started"
        : type === "separation_step_completed"
          ? "step_completed"
          : type === "separation_complete"
            ? "completed"
            : type === "separation_cancelled"
              ? "cancelled"
            : type === "separation_error"
              ? "error"
              : type === "separation_started"
                ? "job_started"
                : "progress",
    progress: Number.isFinite(progress) ? progress : undefined,
    message: typeof msg?.message === "string" ? msg.message : undefined,
    phase: typeof meta?.phase === "string" ? meta.phase : undefined,
    stepId:
      step && (typeof step.index === "number" || typeof step.index === "string")
        ? `${meta?.phase || "step"}:${String(step.index)}:${String(step.model_id || step.name || "")}`
        : undefined,
    stepLabel: getStepLabelFromMeta(meta),
    stepIndex:
      step && Number.isFinite(Number(step.index)) ? Number(step.index) : undefined,
    stepCount:
      step && Number.isFinite(Number(step.total)) ? Number(step.total) : undefined,
    modelId:
      step && typeof step.model_id === "string" ? step.model_id : undefined,
    chunksDone,
    chunksTotal,
    elapsedMs:
      Number.isFinite(durationSeconds) && durationSeconds >= 0
        ? Math.round(durationSeconds * 1000)
        : undefined,
    ts:
      Number.isFinite(Number(msg?.ts)) && Number(msg.ts) > 0
        ? Number(msg.ts)
        : undefined,
    meta,
    error: typeof msg?.error === "string" ? msg.error : undefined,
  });
}

function rejectAllPendingBackendCommands(error: Error) {
  for (const [key, pending] of Array.from(pendingBackendCommands.entries())) {
    clearTimeout(pending.timeout);
    pendingBackendCommands.delete(key);
    pending.reject(error);
  }
}

function resolveBridgeReadyWaiters() {
  for (const waiter of bridgeReadyWaiters.splice(0)) {
    clearTimeout(waiter.timeout);
    waiter.resolve();
  }
}

function rejectBridgeReadyWaiters(error: Error) {
  for (const waiter of bridgeReadyWaiters.splice(0)) {
    clearTimeout(waiter.timeout);
    waiter.reject(error);
  }
}

function waitForBridgeReady(timeoutMs = 30000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      bridgeReadyWaiters = bridgeReadyWaiters.filter((entry) => entry.resolve !== resolve);
      reject(new Error(`Backend did not become ready within ${timeoutMs}ms`));
    }, timeoutMs);
    bridgeReadyWaiters.push({ resolve, reject, timeout });
  });
}

async function restartBackendAndWait(
  reason: string,
  timeoutMs = 45000,
): Promise<void> {
  const readyPromise = waitForBridgeReady(timeoutMs);
  requestBridgeRestart(reason);
  try {
    await readyPromise;
    return;
  } catch (eventError) {
    let lastError: unknown = eventError;
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        await sendPythonCommand("ping", {}, 4000);
        return;
      } catch (error) {
        lastError = error;
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }
}

function routeBackendMessage(msg: any) {
  const hasResponseShape =
    msg && Object.prototype.hasOwnProperty.call(msg, "id") &&
    typeof msg.success === "boolean";

  if (hasResponseShape) {
    const key = String(msg.id);
    const pending = pendingBackendCommands.get(key);
    if (pending) {
      clearTimeout(pending.timeout);
      pendingBackendCommands.delete(key);
      if (msg.success) {
        pending.resolve(msg.data);
      } else {
        pending.reject(new Error(msg.error || "Unknown error from backend"));
      }
    }
  }

  if (msg?.type === "bridge_ready") {
    log(
      `Bridge ready! Capabilities: ${msg.capabilities?.join(", ")}, Models: ${msg.models_count}, Recipes: ${msg.recipes_count}`,
    );
    resolveBridgeReadyWaiters();
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("bridge-ready", {
        capabilities: msg.capabilities,
        modelsCount: msg.models_count,
        recipesCount: msg.recipes_count,
      });
    }
  } else if (
    msg?.type === "progress" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-progress", {
      modelId: msg.model_id,
      progress: msg.progress,
      stage: msg.stage,
      artifactIndex: msg.artifactIndex,
      artifactCount: msg.artifactCount,
      currentFile: msg.currentFile,
      currentRelativePath: msg.currentRelativePath,
      currentSource: msg.currentSource,
      verified: msg.verified,
      message: msg.message,
    });
  } else if (
    msg?.type === "complete" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-complete", {
      modelId: msg.model_id,
      artifactCount: msg.artifactCount,
      stage: msg.stage,
      verified: msg.verified,
    });
  } else if (
    msg?.type === "error" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-error", {
      modelId: msg.model_id,
      error: msg.error,
    });
  } else if (
    msg?.type === "paused" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-paused", {
      modelId: msg.model_id,
      artifactIndex: msg.artifactIndex,
      artifactCount: msg.artifactCount,
      currentFile: msg.currentFile,
      currentRelativePath: msg.currentRelativePath,
      currentSource: msg.currentSource,
      stage: msg.stage,
      verified: msg.verified,
      progress: msg.progress,
    });
  } else if (
    msg?.type === "separation_progress" &&
    msg?.job_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    const jobId = msg.job_id;
    const next = Number(msg.progress);
    const prev = lastProgressByJobId.get(jobId) ?? 0;
    const clamped = Number.isFinite(next)
      ? Math.max(prev, Math.max(0, Math.min(100, next)))
      : prev;
    lastProgressByJobId.set(jobId, clamped);
    mainWindow.webContents.send("separation-progress", {
      jobId,
      progress: clamped,
      message: msg.message,
      device: msg.device,
      meta: msg.meta,
    });
    emitNormalizedSeparationEvent({
      ...msg,
      progress: clamped,
    });
  } else if (
    (msg?.type === "separation_step_started" ||
      msg?.type === "separation_step_completed" ||
      msg?.type === "separation_started") &&
    msg?.job_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    emitNormalizedSeparationEvent(msg);
  } else if (
    msg?.type === "separation_error" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    emitNormalizedSeparationEvent(msg);
    if (msg?.job_id) {
      lastProgressByJobId.delete(msg.job_id);
      modelInfoByJobId.delete(msg.job_id);
    }
    mainWindow.webContents.send("separation-error", {
      jobId: msg.job_id,
      error: msg.error || "Separation failed",
    });
  } else if (
    msg?.type === "separation_cancelled" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    emitNormalizedSeparationEvent({
      ...msg,
      type: "separation_cancelled",
    });
    if (msg?.job_id) {
      lastProgressByJobId.delete(msg.job_id);
      modelInfoByJobId.delete(msg.job_id);
    }
  } else if (
    msg?.type === "separation_complete" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    let normalizedOutputFiles = msg.output_files || {};
    if (msg?.job_id) {
      const state = modelInfoByJobId.get(msg.job_id);
      if (state) {
        const resolved = resolvePlaybackStems(msg.output_files || {}, {
          sourceKind: "preview_cache",
          previewDir: state.previewDir,
          savedDir: state.finalOutputDir,
        });
        normalizedOutputFiles =
          Object.keys(resolved.stems).length > 0 ? resolved.stems : normalizedOutputFiles;
        state.outputFiles = normalizedOutputFiles;
      }
    }
    emitNormalizedSeparationEvent(msg);
    if (msg?.job_id) {
      lastProgressByJobId.delete(msg.job_id);
    }
    mainWindow.webContents.send("separation-complete", {
      outputFiles: normalizedOutputFiles,
      jobId: msg.job_id,
    });
  } else if (
    msg?.type === "playback_capture_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    const sessionInfo =
      typeof msg?.capture_id === "string"
        ? playbackCaptureSessions.get(msg.capture_id)
        : null;
    emitPlaybackCaptureProgress({
      provider: sessionInfo?.provider || "qobuz",
      captureId: msg.capture_id,
      status: msg.status,
      detail: msg.detail,
      progress:
        typeof msg.progress === "number" ? msg.progress : undefined,
      percent:
        typeof msg.progress === "number"
          ? `${Math.round(msg.progress * 100)}%`
          : undefined,
      elapsedSec:
        typeof msg.elapsed_sec === "number" ? msg.elapsed_sec : undefined,
      error: typeof msg.error === "string" ? msg.error : undefined,
    });
  } else if (
    msg?.type === "youtube_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("youtube-progress", msg);
    mainWindow.webContents.send("remote-resolve-progress", {
      provider: "youtube",
      ...msg,
    });
  } else if (
    msg?.type === "quality_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("quality-progress", msg);
  } else if (
    msg?.type === "quality_complete" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("quality-complete", msg);
  }

  emitBackendEvent(msg);
}

function attachBackendMessageRouter(process: ReturnType<typeof spawn>) {
  if (backendMessageRouter) return;
  backendMessageRouter = createLineBuffer((line) => {
    try {
      const msg = JSON.parse(line);
      routeBackendMessage(msg);
    } catch {
      // Ignore non-JSON lines on stdout.
    }
  });
  process.stdout?.on("data", backendMessageRouter);
}

function detachBackendMessageRouter() {
  if (!backendMessageRouter) return;
  if (pythonBridge?.stdout) {
    pythonBridge.stdout.removeListener("data", backendMessageRouter);
  }
  backendMessageRouter = null;
}

// Helper to send a command to backend and await a single JSON response.
let commandIdCounter = 0;

async function sendPythonCommand(
  command: string,
  payload: Record<string, any> = {},
  timeoutMs: number = 60000,
): Promise<any> {
  const process = ensureBackend();
  if (!process) throw new Error("Backend not available");

  const cmdId = ++commandIdCounter;
  const cmdKey = String(cmdId);

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingBackendCommands.delete(cmdKey);
      reject(
        new Error(`Backend command '${command}' timed out after ${timeoutMs}ms`),
      );
    }, timeoutMs);

    pendingBackendCommands.set(cmdKey, {
      command,
      resolve,
      reject,
      timeout,
    });

    try {
      process.stdin?.write(
        JSON.stringify({ command, id: cmdId, ...payload }) + "\n",
      );
    } catch (e) {
      clearTimeout(timeout);
      pendingBackendCommands.delete(cmdKey);
      reject(
        new Error(
          `Failed to write command '${command}' to backend: ${e}`,
        ),
      );
    }
  });
}

/**
 * Wrapper for sendPythonCommand with automatic retry and exponential backoff.
 * Only retries transient failures (timeouts, bridge errors).
 * User cancellations and "not found" errors are not retried.
 */
async function sendPythonCommandWithRetry(
  command: string,
  payload: Record<string, any> = {},
  timeoutMs: number = 60000,
  maxRetries: number = 2,
): Promise<any> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Exponential backoff: 0ms, 1000ms, 2000ms
      if (attempt > 0) {
        const delay = Math.min(1000 * attempt, 5000);
        log(
          `Retrying command '${command}' after ${delay}ms (attempt ${attempt + 1}/${maxRetries + 1})`,
        );
        await new Promise((r) => setTimeout(r, delay));
      }

      return await sendPythonCommand(command, payload, timeoutMs);
    } catch (error) {
      lastError = error as Error;

      // Don't retry user cancellations or definitive failures
      const errorMsg = lastError.message.toLowerCase();
      if (
        errorMsg.includes("cancelled") ||
        errorMsg.includes("not found") ||
        errorMsg.includes("missing") ||
        errorMsg.includes("invalid")
      ) {
        throw lastError;
      }

      // Log the retry attempt
      if (attempt < maxRetries) {
        log(
          `Command '${command}' failed (attempt ${attempt + 1}): ${lastError.message}`,
        );
      }
    }
  }

  throw lastError!;
}

let getGpuDevicesInflight: Promise<any> | null = null;
let getRuntimeFingerprintInflight: Promise<any> | null = null;
let getSystemRuntimeInfoInflight: Promise<any> | null = null;

let gpuDevicesCache: { value: any; expiresAt: number } | null = null;
let runtimeFingerprintCache: { value: any; expiresAt: number } | null = null;

function getGpuDevicesDeduped(): Promise<any> {
  if (getGpuDevicesInflight) return getGpuDevicesInflight;
  getGpuDevicesInflight = sendPythonCommandWithRetry(
    "get-gpu-devices",
    {},
    30000,
  ).finally(() => {
    getGpuDevicesInflight = null;
  });
  return getGpuDevicesInflight;
}

function getRuntimeFingerprintDeduped(): Promise<any> {
  if (getRuntimeFingerprintInflight) return getRuntimeFingerprintInflight;
  getRuntimeFingerprintInflight = sendPythonCommandWithRetry(
    "get_runtime_fingerprint",
    {},
    20000,
    1,
  ).finally(() => {
    getRuntimeFingerprintInflight = null;
  });
  return getRuntimeFingerprintInflight;
}

async function getGpuDevicesCached(): Promise<{ data: any; fromCache: boolean }> {
  if (gpuDevicesCache && gpuDevicesCache.expiresAt > Date.now()) {
    return { data: gpuDevicesCache.value, fromCache: true };
  }
  if (isPlaybackCaptureActive()) {
    return { data: gpuDevicesCache?.value || [], fromCache: true };
  }
  const data = await getGpuDevicesDeduped();
  gpuDevicesCache = {
    value: data,
    expiresAt: Date.now() + SYSTEM_RUNTIME_TTL_MS,
  };
  return { data, fromCache: false };
}

async function getRuntimeFingerprintCached(): Promise<{
  data: any | null;
  fromCache: boolean;
  error: string | null;
}> {
  if (runtimeFingerprintCache && runtimeFingerprintCache.expiresAt > Date.now()) {
    return { data: runtimeFingerprintCache.value, fromCache: true, error: null };
  }
  if (isPlaybackCaptureActive()) {
    return {
      data: runtimeFingerprintCache?.value || null,
      fromCache: true,
      error: runtimeFingerprintCache ? null : "Runtime fingerprint paused during playback capture.",
    };
  }

  try {
    const data = await getRuntimeFingerprintDeduped();
    runtimeFingerprintCache = {
      value: data,
      expiresAt: Date.now() + SYSTEM_RUNTIME_TTL_MS,
    };
    return { data, fromCache: false, error: null };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error || "unknown error");
    return { data: null, fromCache: false, error: message };
  }
}

async function getSystemRuntimeInfoCached(): Promise<any> {
  if (getSystemRuntimeInfoInflight) return getSystemRuntimeInfoInflight;

  getSystemRuntimeInfoInflight = (async () => {
    const [gpu, runtimeFingerprint] = await Promise.all([
      getGpuDevicesCached(),
      getRuntimeFingerprintCached(),
    ]);

    return {
      fetchedAt: new Date().toISOString(),
      cache: {
        ttlMs: SYSTEM_RUNTIME_TTL_MS,
        gpuSource: gpu.fromCache ? "cache" : "fresh",
        runtimeFingerprintSource: runtimeFingerprint.fromCache
          ? "cache"
          : runtimeFingerprint.error
            ? "error"
            : "fresh",
      },
      gpu: gpu.data,
      runtimeFingerprint: runtimeFingerprint.data,
      runtimeFingerprintError: runtimeFingerprint.error,
      previewCachePolicy: {
        baseDir: getPreviewCacheBaseDir(),
        keepLast: PREVIEW_CACHE_KEEP_LAST,
        maxAgeDays: PREVIEW_CACHE_MAX_AGE_DAYS,
        ephemeral: true,
      },
    };
  })().finally(() => {
    getSystemRuntimeInfoInflight = null;
  });

  return getSystemRuntimeInfoInflight;
}

// IPC handler for audio separation
ipcMain.handle(
  "separate-audio",
  async (
    event,
    {
      inputFile,
      modelId,
      outputDir,
      stems,
      device,
      overlap,
      segmentSize,
      tta,
      outputFormat,
      exportMixes,
      shifts,
      bitrate,
      ensembleConfig,
      ensembleAlgorithm,
      invert,
      splitFreq,
      phaseParams,
      postProcessingSteps,
      volumeCompensation,
      pipelineConfig,
      workflow,
      runtimePolicy,
      exportPolicy,
    }: {
      inputFile: string;
      modelId: string;
      outputDir: string;
      stems?: string[];
      device?: string;
      overlap?: number;
      segmentSize?: number;
      tta?: boolean;
      outputFormat?: string;
      exportMixes?: string[];
      shifts?: number;
      bitrate?: string;
      ensembleConfig?: any;
      ensembleAlgorithm?: string;
      invert?: boolean;
      splitFreq?: number;
      phaseParams?: {
        enabled: boolean;
        lowHz: number;
        highHz: number;
        highFreqWeight: number;
      };
      postProcessingSteps?: any[];
      volumeCompensation?: { enabled: boolean; stage?: "export" | "blend" | "both"; dbPerExtraModel?: number };
      pipelineConfig?: any[];
      workflow?: Record<string, any>;
      runtimePolicy?: Record<string, any>;
      exportPolicy?: Record<string, any>;
    },
  ) => {
    const requestId = randomUUID().slice(0, 8);
    log("[separate-audio] request", {
      requestId,
      inputFile,
      modelId,
      splitFreq,
      hasEnsemble: Boolean(
        ensembleConfig &&
          Array.isArray(ensembleConfig.models) &&
          ensembleConfig.models.length > 0,
      ),
    });
    const process = ensureBackend();
    if (!process)
      return Promise.reject(new Error("Backend not available."));

    // Always stage outputs into a stable preview cache for playback/preview.
    // Export is handled separately (Results -> Export).
    const previewDir = createPreviewDirForInput(inputFile);

    // Ensure model id is always valid (never null/undefined), and normalize ensembles.
    const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);

    // Ensure backend receives WAV input even if user dropped MP3/FLAC/M4A,
    // while preserving lossless precision when possible.
    const stagedInput = await ensureWavInput(inputFile, previewDir);
    const effectiveInputFile = stagedInput.effectiveInputFile;
    log("[separate-audio] staging-ready", {
      requestId,
      effectiveModelId,
      previewDir,
      effectiveInputFile,
      sourceAudioProfile: stagedInput.sourceAudioProfile,
      stagingDecision: stagedInput.stagingDecision,
    });

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error("Separation timeout (60 minutes)"));
      }, 60 * 60 * 1000);

      let myJobId: string | null = null;
      const payload = {
        file_path: effectiveInputFile,
        model_id: effectiveModelId,
        output_dir: previewDir,
        stems,
        device,
        shifts: shifts || 0,
        overlap,
        segment_size: segmentSize,
        tta,
        output_format: outputFormat,
        bitrate,
        ensemble_config: ensembleConfig,
        ensemble_algorithm: ensembleAlgorithm,
        invert,
        split_freq: splitFreq,
        phase_params: phaseParams,
        post_processing_steps: postProcessingSteps,
        export_mixes: exportMixes,
        volume_compensation: volumeCompensation,
        pipeline_config: pipelineConfig,
        workflow,
        runtime_policy: runtimePolicy,
        export_policy: exportPolicy,
      };

      const onDone = (msg: any) => {
        if (!myJobId || msg?.job_id !== myJobId) return;
        log("[separate-audio] complete", { requestId, jobId: myJobId });
        cleanup();
        const resolvedPlayback = resolvePlaybackStems(msg.output_files || {}, {
          sourceKind: "preview_cache",
          previewDir,
          savedDir: outputDir,
        });
        const normalizedOutputFiles =
          Object.keys(resolvedPlayback.stems).length > 0
            ? resolvedPlayback.stems
            : msg.output_files || {};
        resolve({
          success: true,
          outputFiles: normalizedOutputFiles,
          jobId: msg.job_id,
          outputDir: previewDir,
          sourceAudioProfile: stagedInput.sourceAudioProfile,
          stagingDecision: stagedInput.stagingDecision,
          playbackSourceKind: "preview_cache",
        });
      };

      const onError = (msg: any) => {
        if (!myJobId || msg?.job_id !== myJobId) return;
        log("[separate-audio] backend-error", {
          requestId,
          jobId: myJobId,
          error: msg?.error || "Separation failed",
        });
        cleanup();
        reject(new Error(msg.error || "Separation failed"));
      };

      const onCancelled = (msg: any) => {
        if (!myJobId || msg?.job_id !== myJobId) return;
        log("[separate-audio] cancelled", { requestId, jobId: myJobId });
        cleanup();
        reject(new Error("Separation cancelled"));
      };

      const unsubComplete = subscribeBackendEvent("separation_complete", onDone);
      const unsubError = subscribeBackendEvent("separation_error", onError);
      const unsubCancelled = subscribeBackendEvent(
        "separation_cancelled",
        onCancelled,
      );

      const cleanup = () => {
        clearTimeout(timeout);
        unsubComplete();
        unsubError();
        unsubCancelled();
      };

      sendPythonCommand("separate_audio", payload)
        .then((response) => {
          if (!response?.job_id) {
            cleanup();
            reject(new Error("Backend did not return a job ID"));
            return;
          }
          myJobId = String(response.job_id);
          log("[separate-audio] job-started", {
            requestId,
            jobId: myJobId,
            effectiveModelId,
          });
          modelInfoByJobId.set(myJobId, {
            requestedModelId: String(modelId || ""),
            effectiveModelId: String(effectiveModelId || ""),
            inputFile: String(inputFile || ""),
            finalOutputDir: String(outputDir || ""),
            previewDir,
            sourceAudioProfile: stagedInput.sourceAudioProfile,
            stagingDecision: stagedInput.stagingDecision,
          });
          lastProgressByJobId.set(myJobId, 0);
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("separation-started", {
              jobId: myJobId,
            });
            mainWindow.webContents.send("separation-progress", {
              jobId: myJobId,
              progress: 1,
              message: "Starting separation...",
            });
          }
        })
        .catch((err) => {
          log("[separate-audio] dispatch-failed", {
            requestId,
            error: err?.message || String(err),
          });
          cleanup();
          reject(err);
        });
    });
  },
);

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
    const device = state.devices.find((entry) => entry.id === trimmed);
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
      const device = deviceState.devices.find((entry) => entry.id === deviceId);
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
      const cancelled =
        /cancelled/i.test(message) ||
        Array.from(cancelledPlaybackCaptureIds).length > 0;
      return {
        success: false,
        provider,
        error: cancelled ? "Capture cancelled" : message,
        code: cancelled ? "CAPTURE_CANCELLED" : "CAPTURE_START_FAILED",
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
      const backendStatus = await sendPythonCommandWithRetry(
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
      if (path.extname(downloadedPath).toLowerCase() === ".zip") {
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

// Resolve YouTube URL to a local temp audio file (WAV)
ipcMain.handle("resolve-youtube-url", async (event, { url }: { url: string }) => {
  try {
    const result = await sendPythonCommand("resolve_youtube", { url }, 10 * 60 * 1000);
    return {
      success: true,
      file_path: (result as any)?.file_path,
      title: (result as any)?.title,
      source_url: (result as any)?.source_url,
      channel: (result as any)?.channel,
      duration_sec: (result as any)?.duration_sec,
      thumbnail_url: (result as any)?.thumbnail_url,
      canonical_url: (result as any)?.canonical_url,
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

// Cancel separation
ipcMain.handle("cancel-separation", async (event, jobId: string) => {
  return sendPythonCommand("cancel_job", { job_id: jobId });
});

// Save/Discard output
ipcMain.handle("save-job-output", async (event, jobId: string) => {
  const jobState = modelInfoByJobId.get(jobId);
  if (!jobState?.outputFiles || Object.keys(jobState.outputFiles).length === 0) {
    return sendPythonCommand("save_output", { job_id: jobId });
  }
  if (!jobState.finalOutputDir) {
    return {
      success: false,
      error: "No final output directory is configured for this separation.",
    };
  }

  const requestId = `save-${jobId}`;
  try {
    const res = await exportFilesLocal({
      sourceFiles: jobState.outputFiles,
      exportPath: jobState.finalOutputDir,
      format: "wav",
      bitrate: "320k",
      requestId,
    });
    jobState.outputFiles = res.exported;
    return {
      success: true,
      outputFiles: res.exported,
      sourceAudioProfile: jobState.sourceAudioProfile,
      stagingDecision: jobState.stagingDecision,
    };
  } catch (error: any) {
    emitExportProgress({
      requestId,
      status: "failed",
      error: error?.message || String(error),
      totalProgress: 0,
    });
    return {
      success: false,
      error: error?.message || String(error),
    };
  }
});

ipcMain.handle(
  "export-output",
  async (
    event,
    {
      jobId,
      exportPath,
      format,
      bitrate,
      requestId,
    }: {
      jobId: string;
      exportPath: string;
      format: string;
      bitrate: string;
      requestId?: string;
    },
  ) => {
    const jobState = modelInfoByJobId.get(jobId);
    const sourceFiles = jobState?.outputFiles;
    if (!sourceFiles || Object.keys(sourceFiles).length === 0) {
      return sendPythonCommand("export_output", {
        job_id: jobId,
        export_path: exportPath,
        format,
        bitrate,
      });
    }

    const resolvedRequestId = requestId || randomUUID().slice(0, 8);
    try {
      const res = await exportFilesLocal({
        sourceFiles,
        exportPath,
        format,
        bitrate,
        requestId: resolvedRequestId,
      });
      return {
        success: true,
        exported: res.exported,
        requestId: resolvedRequestId,
        sourceAudioProfile: jobState.sourceAudioProfile,
        stagingDecision: jobState.stagingDecision,
      };
    } catch (error: any) {
      emitExportProgress({
        requestId: resolvedRequestId,
        status: "failed",
        error: error?.message || String(error),
      });
      return {
        success: false,
        error: error?.message || String(error),
        requestId: resolvedRequestId,
      };
    }
  },
);

ipcMain.handle("discard-job-output", async (event, jobId: string) => {
  try {
    const result = await sendPythonCommand("discard_output", { job_id: jobId });
    if (result?.success !== false) {
      modelInfoByJobId.delete(jobId);
      lastProgressByJobId.delete(jobId);
    }
    return result;
  } catch (error) {
    modelInfoByJobId.delete(jobId);
    lastProgressByJobId.delete(jobId);
    throw error;
  }
});

// Export files directly from paths (bypasses job registry - for historical exports)
ipcMain.handle(
  "export-files",
  async (
    event,
    {
      sourceFiles,
      exportPath,
      format,
      bitrate,
      requestId,
    }: {
      sourceFiles: Record<string, string>;
      exportPath: string;
      format: string;
      bitrate: string;
      requestId?: string;
    },
  ) => {
    const resolvedRequestId = requestId || randomUUID().slice(0, 8);
    // Export is a pure file operation (copy/transcode) and should not require a backend.
    // This also avoids failures when the legacy python-bridge proxy is not present.
    log("[export-files] local export requested", {
      requestId: resolvedRequestId,
      stems: Object.keys(sourceFiles || {}),
      exportPath,
      format,
      bitrate,
    });
    try {
      const res = await exportFilesLocal({
        sourceFiles,
        exportPath,
        format,
        bitrate,
        requestId: resolvedRequestId,
      });
      log("[export-files] local export complete", {
        requestId: resolvedRequestId,
        exported: Object.keys(res?.exported || {}),
      });
      return {
        success: true,
        exported: res.exported,
        requestId: resolvedRequestId,
      };
    } catch (e: any) {
      emitExportProgress({
        requestId: resolvedRequestId,
        status: "failed",
        error: e?.message || String(e),
      });
      log("[export-files] local export FAILED", {
        requestId: resolvedRequestId,
        error: e?.message || String(e),
      });
      return {
        success: false,
        error: e?.message || String(e),
        code: e?.code || "MISSING_SOURCE_FILE",
        hint:
          e?.hint ||
          "Export source is unavailable. Run a new separation to refresh cache files.",
        requestId: resolvedRequestId,
      };
    }
  },
);

// Queue Management
ipcMain.handle("pause-queue", async () => sendPythonCommand("pause_queue"));
ipcMain.handle("resume-queue", async () => sendPythonCommand("resume_queue"));
ipcMain.handle("reorder-queue", async (event, jobIds: string[]) =>
  sendPythonCommand("reorder_queue", { job_ids: jobIds }),
);

ipcMain.handle("open-audio-file-dialog", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile", "multiSelections"],
      filters: [
        {
          name: "Audio Files",
          extensions: [
            "mp3",
            "wav",
            "flac",
            "m4a",
            "ogg",
            "aac",
            "wma",
            "aiff",
          ],
        },
      ],
    });
    return result.filePaths || [];
  } catch (error) {
    log("Error opening file dialog:", error);
    throw new Error("Failed to open file dialog");
  }
});

ipcMain.handle("open-model-file-dialog", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile", "multiSelections"],
      filters: [
        {
          name: "Model Files",
          extensions: ["ckpt", "pth", "pt", "onnx", "safetensors"],
        },
      ],
    });
    return result.filePaths || [];
  } catch (error) {
    log("Error opening model file dialog:", error);
    throw new Error("Failed to open model file dialog");
  }
});

ipcMain.handle("select-output-directory", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"],
    });
    return result.filePaths[0] || null;
  } catch (error) {
    log("Error selecting output directory:", error);
    throw new Error("Failed to open directory dialog");
  }
});

ipcMain.handle("select-models-directory", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory", "createDirectory"],
      title: "Choose Models Directory",
    });
    return result.filePaths[0] || null;
  } catch (error) {
    log("Error selecting models directory:", error);
    throw new Error("Failed to open models directory dialog");
  }
});

// Scan directory for audio files
ipcMain.handle("scan-directory", async (event, folderPath: string) => {
  const audioExtensions = new Set([".mp3", ".wav", ".flac", ".ogg", ".m4a"]);
  const audioFiles: string[] = [];

  async function scan(dir: string) {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await scan(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (audioExtensions.has(ext)) {
          audioFiles.push(fullPath);
        }
      }
    }
  }

  try {
    await scan(folderPath);
    return audioFiles;
  } catch (error) {
    console.error("Error scanning directory:", error);
    log("Error scanning directory:", error);
    throw new Error(
      `Failed to scan directory: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
});

// Open folder
ipcMain.handle("open-folder", async (event, folderPath: string) => {
  await shell.openPath(folderPath);
});

ipcMain.handle("check-file-exists", async (_event, filePath: string) => {
  try {
    if (!filePath || typeof filePath !== "string") return false;
    const resolvedPath = resolvePlaybackFilePath(filePath) || filePath;
    return fs.existsSync(resolvedPath);
  } catch {
    return false;
  }
});

// Read audio file as base64 for use with Blob URLs
// This works around browser security restrictions that block file:// URLs
ipcMain.handle("read-audio-file", async (_event, filePath: string) => {
  try {
    // Security: block arbitrary file reads if renderer is compromised.
    // Only allow a strict list of audio extensions.
    const ext = filePath.split(".").pop()?.toLowerCase();
    const allowedExtensions = new Set([
      "mp3",
      "wav",
      "flac",
      "ogg",
      "m4a",
      "aac",
      "wma",
      "aiff",
    ]);
    if (!ext || !allowedExtensions.has(ext)) {
      throw new Error(
        `Security violation: Attempted to read non-audio file extension '.${ext || ""}'`,
      );
    }

    const resolvedPath = resolvePlaybackFilePath(filePath) || filePath;

    if (!fs.existsSync(resolvedPath)) {
      const missing = classifyMissingAudioPath(filePath);
      return {
        success: false,
        error: "Audio file not found",
        code: missing.code,
        hint: missing.hint,
      };
    }

    const data = fs.readFileSync(resolvedPath);
    const base64 = data.toString("base64");
    // Determine MIME type from extension
    const mimeTypes: Record<string, string> = {
      mp3: "audio/mpeg",
      wav: "audio/wav",
      flac: "audio/flac",
      ogg: "audio/ogg",
      m4a: "audio/mp4",
      aac: "audio/aac",
      wma: "audio/x-ms-wma",
      aiff: "audio/aiff",
    };
    const mimeType = mimeTypes[ext] || "audio/mpeg";
    return { success: true, data: base64, mimeType, resolvedPath };
  } catch (error: any) {
    log(`Failed to read audio file: ${filePath}`, error);
    if (error?.code === "ENOENT") {
      const missing = classifyMissingAudioPath(filePath);
      return {
        success: false,
        error: "Audio file not found",
        code: missing.code,
        hint: missing.hint,
      };
    }
    return {
      success: false,
      error: error?.message || String(error),
      code: "MISSING_SOURCE_FILE",
      hint: "Unable to read the audio file. Verify the file exists and retry.",
    };
  }
});

ipcMain.handle(
  "resolve-playback-stems",
  async (
    _event,
    {
      outputFiles,
      playback,
    }: {
      outputFiles?: Record<string, string>;
      playback?: PlaybackMetadataLike;
    },
  ) => {
    try {
      const resolved = resolvePlaybackStems(outputFiles, playback);
      return {
        success: true,
        stems: resolved.stems,
        issues: resolved.issues,
      };
    } catch (error: any) {
      return {
        success: false,
        stems: outputFiles || {},
        issues: {},
        error: error?.message || String(error),
      };
    }
  },
);

// Check preset models
ipcMain.handle(
  "check-preset-models",
  async (event, presetMappings: Record<string, string>) => {
    return sendPythonCommandWithRetry("check-preset-models", {
      preset_mappings: presetMappings,
    });
  },
);

// Get GPU devices
ipcMain.handle("get-gpu-devices", async () => {
  const gpu = await getGpuDevicesCached();
  return gpu.data;
});

ipcMain.handle("get-system-runtime-info", async () => {
  return getSystemRuntimeInfoCached();
});

// Get workflow types (Live vs Studio)
ipcMain.handle("get-workflows", async () => {
  return sendPythonCommandWithRetry("get_workflows", {}, 10000);
});

// Get all available models
ipcMain.handle("get-models", async () => {
  return sendPythonCommandWithRetry("get_models", {}, 120000);
});

ipcMain.handle("get-model-tech", async (_event, modelId: string) => {
  return sendPythonCommandWithRetry("get_model_tech", { model_id: modelId }, 20000);
});

ipcMain.handle("resolve-model-download", async (_event, modelId: string) => {
  return sendPythonCommandWithRetry(
    "resolve_model_download",
    { model_id: modelId },
    20000,
  );
});

ipcMain.handle("get-model-installation", async (_event, modelId: string) => {
  return sendPythonCommandWithRetry(
    "get_model_installation",
    { model_id: modelId },
    20000,
  );
});

ipcMain.handle(
  "separation-preflight",
  async (
    _event,
    {
      inputFile,
      modelId,
      outputDir,
      stems,
      device,
      overlap,
      segmentSize,
      tta,
      outputFormat,
      exportMixes,
      shifts,
      bitrate,
      ensembleConfig,
      ensembleAlgorithm,
      invert,
      splitFreq,
      phaseParams,
      postProcessingSteps,
      volumeCompensation,
      pipelineConfig,
      workflow,
      runtimePolicy,
      exportPolicy,
    }: {
      inputFile: string;
      modelId: string;
      outputDir: string;
      stems?: string[];
      device?: string;
      overlap?: number;
      segmentSize?: number;
      tta?: boolean;
      outputFormat?: string;
      exportMixes?: string[];
      shifts?: number;
      bitrate?: string;
      ensembleConfig?: any;
      ensembleAlgorithm?: string;
      invert?: boolean;
      splitFreq?: number;
      phaseParams?: {
        enabled: boolean;
        lowHz: number;
        highHz: number;
        highFreqWeight: number;
      };
      postProcessingSteps?: any[];
      volumeCompensation?: { enabled: boolean; stage?: "export" | "blend" | "both"; dbPerExtraModel?: number };
      pipelineConfig?: any[];
      workflow?: Record<string, any>;
      runtimePolicy?: Record<string, any>;
      exportPolicy?: Record<string, any>;
    },
  ) => {
    // Preflight should mirror the real run: normalize model id and stage non-WAV inputs to WAV.
    const previewDir = createPreviewDirForInput(inputFile);
    const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);
    const stagedInput = await ensureWavInput(inputFile, previewDir);
    const effectiveInputFile = stagedInput.effectiveInputFile;

    const result = await sendPythonCommandWithRetry(
      "separation_preflight",
      {
        file_path: effectiveInputFile,
        model_id: effectiveModelId,
        output_dir: outputDir,
        stems,
        device,
        shifts: shifts || 0,
        overlap,
        segment_size: segmentSize,
        tta,
        output_format: outputFormat,
        bitrate,
        ensemble_config: ensembleConfig,
        ensemble_algorithm: ensembleAlgorithm,
        invert,
        split_freq: splitFreq,
        phase_params: phaseParams,
        post_processing_steps: postProcessingSteps,
        export_mixes: exportMixes,
        volume_compensation: volumeCompensation,
        pipeline_config: pipelineConfig,
        workflow,
        runtime_policy: runtimePolicy,
        export_policy: exportPolicy,
      },
      30000,
    );

    return {
      ...result,
      sourceAudioProfile: stagedInput.sourceAudioProfile,
      stagingDecision: stagedInput.stagingDecision,
    };
  },
);

// Get recipes
ipcMain.handle("get-recipes", async () => {
  return sendPythonCommandWithRetry("get_recipes", {}, 10000);
});

ipcMain.handle("quality-baseline-create", async (_event, payload: Record<string, any>) => {
  return sendPythonCommandWithRetry(
    "quality_baseline_create",
    payload || {},
    5 * 60 * 1000,
  );
});

ipcMain.handle("quality-compare", async (_event, payload: Record<string, any>) => {
  return sendPythonCommandWithRetry(
    "quality_compare",
    payload || {},
    5 * 60 * 1000,
  );
});

// Download model
ipcMain.handle("download-model", async (event, modelId: string) => {
  const process = ensureBackend();
  if (!process) throw new Error("Backend not available");

  // Similar to YouTube: Rust backend handles download in background and emits events.
  // We trigger it with a command that returns immediately (scheduled: true).
  
  // Note: Rust backend emits global events for progress. We don't need a specific listener attached here
  // because `globalDownloadHandler` in `ensureBackend` already forwards them to the window!
  
  // We just need to send the command.
  return sendPythonCommand("download_model", { model_id: modelId });
});

// Remove model
ipcMain.handle("remove-model", async (event, modelId: string) => {
  try {
    const res = await sendPythonCommandWithRetry("remove_model", { model_id: modelId }, 60000);
    log("remove-model completed", { modelId, res });
    return res;
  } catch (e: any) {
    const msg = (e?.message || String(e) || "").toLowerCase();
    log("remove-model failed", { modelId, error: e?.message || String(e) });

    // Fallback: allow uninstall even if python proxy is missing.
    if (
      msg.includes("python proxy unavailable") ||
      msg.includes("python-bridge.py") ||
      msg.includes("backend_unavailable") ||
      msg.includes("backend not available")
    ) {
      try {
        const local = removeModelLocal(modelId);
        log("remove-model local fallback succeeded", {
          modelId,
          removed: local.removedFiles.length,
        });
        return local;
      } catch (localErr: any) {
        log("remove-model local fallback FAILED", {
          modelId,
          error: localErr?.message || String(localErr),
        });
        throw new Error(
          `BACKEND_UNAVAILABLE: ${e?.message || String(e)}\n\nLocal delete failed: ${localErr?.message || String(localErr)}`,
        );
      }
    }

    throw e;
  }
});

// Pause download
ipcMain.handle("pause-download", async (event, modelId: string) => {
  // Simple command, events handled globally
  return sendPythonCommand("pause_download", { model_id: modelId });
});

// Resume download
ipcMain.handle("resume-download", async (event, modelId: string) => {
  // Simple command, events handled globally
  return sendPythonCommand("resume_download", { model_id: modelId });
});

ipcMain.handle(
  "import-model-files",
  async (
    _event,
    {
      modelId,
      files,
      allowCopy,
    }: { modelId: string; files: Array<{ kind?: string; path: string }>; allowCopy?: boolean },
  ) => {
    return sendPythonCommandWithRetry(
      "import_model_files",
      { model_id: modelId, files, allow_copy: allowCopy !== false },
      120000,
    );
  },
);

// Import custom model
ipcMain.handle(
  "import-custom-model",
  async (
    event,
    {
      filePath,
      modelName,
      architecture,
    }: { filePath: string; modelName: string; architecture?: string },
  ) => {
    return sendPythonCommand("import_custom_model", {
      file_path: filePath,
      model_name: modelName,
      architecture: architecture || "Custom",
    });
  },
);

// Queue Persistence
const getQueuePath = () =>
  path.join(app.getPath("userData"), "queue_state.json");

ipcMain.handle("save-queue", async (event, queueData: any) => {
  try {
    await fs.promises.writeFile(
      getQueuePath(),
      JSON.stringify(queueData),
      "utf-8",
    );
    return { success: true };
  } catch (error) {
    console.error("Failed to save queue:", error);
    return { success: false, error: String(error) };
  }
});

ipcMain.handle("load-queue", async () => {
  try {
    const queuePath = getQueuePath();
    if (!fs.existsSync(queuePath)) {
      return null;
    }
    const data = await fs.promises.readFile(queuePath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    console.error("Failed to load queue:", error);
    return null;
  }
});

// Watch Folder Logic
let watcher: FSWatcher | null = null;

ipcMain.handle("start-watch-mode", async (event, folderPath: string) => {
  if (watcher) {
    await watcher.close();
  }

  log("Starting watch mode on:", folderPath);

  watcher = chokidar.watch(folderPath, {
    ignored: /(^|[\/\\])\../, // ignore dotfiles
    persistent: true,
    ignoreInitial: true, // Don't process existing files immediately
    awaitWriteFinish: {
      stabilityThreshold: 2000, // Wait 2s for file write to finish
      pollInterval: 100,
    },
  });

  watcher.on("add", (filePath) => {
    const ext = path.extname(filePath).toLowerCase();
    if ([".wav", ".mp3", ".flac", ".m4a", ".ogg"].includes(ext)) {
      log("New file detected:", filePath);
      mainWindow?.webContents.send("watch-file-detected", filePath);
    }
  });

  return true;
});

ipcMain.handle("stop-watch-mode", async () => {
  if (watcher) {
    await watcher.close();
    watcher = null;
    log("Watch mode stopped");
  }
  return true;
});

// App config persistence for main process settings (like modelsDir)
ipcMain.handle("set-models-dir", async (_event, modelsDir: string) => {
  const normalized = typeof modelsDir === "string" ? modelsDir.trim() : "";
  if (!normalized) {
    throw new Error("Models directory is required.");
  }

  fs.mkdirSync(normalized, { recursive: true });
  const saved = writeAppConfig({ modelsDir: normalized });
  if (!saved) {
    throw new Error("Failed to persist models directory.");
  }

  await restartBackendAndWait("updated models directory");
  const models = await sendPythonCommandWithRetry("get_models", {}, 120000);

  return {
    success: true,
    modelsDir: normalized,
    models,
  };
});

ipcMain.handle(
  "save-app-config",
  async (_event, config: Record<string, any>) => {
    try {
      const configPath = path.join(app.getPath("userData"), "app-config.json");
      let existingConfig: Record<string, any> = {};

      if (fs.existsSync(configPath)) {
        existingConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      }

      // Never allow saving secrets through the generic config endpoint.
      const { hfToken: _ignoredHfToken, ...safeConfig } = config || {};
      const newConfig = { ...existingConfig, ...safeConfig };
      fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
      log("Saved app config:", Object.keys(safeConfig).join(", "));
      return true;
    } catch (error) {
      log("Failed to save app config:", error);
      return false;
    }
  },
);

ipcMain.handle("get-app-config", async () => {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      if (config && typeof config === "object") {
        delete config.hfToken;
      }
      return config;
    }
  } catch (error) {
    log("Failed to read app config:", error);
  }
  return {};
});

// Hugging Face auth (optional): store token in app-config.json and restart backend
ipcMain.handle("get-huggingface-auth-status", async () => {
  return { configured: !!getStoredHuggingFaceToken() };
});

ipcMain.handle("set-huggingface-token", async (_event, token: string) => {
  const res = setStoredHuggingFaceToken(token);
  if (res.success) {
    requestBridgeRestart("updated huggingface token");
  }
  return res;
});

ipcMain.handle("clear-huggingface-token", async () => {
  const res = setStoredHuggingFaceToken(null);
  if (res.success) {
    requestBridgeRestart("cleared huggingface token");
  }
  return res;
});

ipcMain.handle("open-external-url", async (_event, url: string) => {
  try {
    if (typeof url !== "string" || !url.trim()) return false;
    const u = new URL(url);
    if (u.protocol !== "https:" && u.protocol !== "http:") return false;
    await shell.openExternal(url);
    return true;
  } catch (e) {
    log("Failed to open external url:", e);
    return false;
  }
});
