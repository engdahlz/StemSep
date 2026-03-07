import { app, BrowserWindow, ipcMain, dialog, shell, protocol, Menu, screen } from "electron";
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

app.whenReady().then(() => {
  cleanupPreviewCache();

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

  if (msg?.type === "bridge_ready" && mainWindow && !mainWindow.isDestroyed()) {
    log(
      `Bridge ready! Capabilities: ${msg.capabilities?.join(", ")}, Models: ${msg.models_count}, Recipes: ${msg.recipes_count}`,
    );
    mainWindow.webContents.send("bridge-ready", {
      capabilities: msg.capabilities,
      modelsCount: msg.models_count,
      recipesCount: msg.recipes_count,
    });
  } else if (
    msg?.type === "progress" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-progress", {
      modelId: msg.model_id,
      progress: msg.progress,
    });
  } else if (
    msg?.type === "complete" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-complete", {
      modelId: msg.model_id,
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
    if (msg?.job_id) {
      const state = modelInfoByJobId.get(msg.job_id);
      if (state) {
        state.outputFiles = msg.output_files || {};
      }
    }
    emitNormalizedSeparationEvent(msg);
    if (msg?.job_id) {
      lastProgressByJobId.delete(msg.job_id);
    }
    mainWindow.webContents.send("separation-complete", {
      outputFiles: msg.output_files,
      jobId: msg.job_id,
    });
  } else if (
    msg?.type === "youtube_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("youtube-progress", msg);
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
      };

      const onDone = (msg: any) => {
        if (!myJobId || msg?.job_id !== myJobId) return;
        log("[separate-audio] complete", { requestId, jobId: myJobId });
        cleanup();
        resolve({
          success: true,
          outputFiles: msg.output_files,
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

// Resolve YouTube URL to a local temp audio file (WAV)
ipcMain.handle("resolve-youtube-url", async (event, { url }: { url: string }) => {
  try {
    const result = await sendPythonCommand("resolve_youtube", { url }, 10 * 60 * 1000);
    return {
      success: true,
      file_path: (result as any)?.file_path,
      title: (result as any)?.title,
      source_url: (result as any)?.source_url,
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
      properties: ["openFile"],
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

    let resolvedPath = filePath;
    if (!fs.existsSync(resolvedPath)) {
      const fallback = resolveMissingPreviewAudioPath(resolvedPath);
      if (fallback && fs.existsSync(fallback)) {
        log("Resolved missing preview audio path", { from: filePath, to: fallback });
        resolvedPath = fallback;
      }
    }

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
