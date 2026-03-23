import { app } from "electron";
import { createHash, randomUUID } from "crypto";
import fs from "fs";
import path from "path";
import {
  ensureBinaryAvailable,
  getFfmpegExe,
  getFfprobeExe,
  runFfmpeg,
  runFfmpegWithProgress,
  runProcessCapture,
} from "./ffmpeg";

export const PREVIEW_CACHE_KEEP_LAST = 20;
export const PREVIEW_CACHE_MAX_AGE_DAYS = 7;

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

export type AudioSourceProfile = {
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

export type SourceStagingDecision = {
  sourcePath: string;
  workingPath: string;
  sourceExt: string;
  copiedDirectly: boolean;
  workingCodec: "original" | "pcm_s16le" | "pcm_s24le" | "pcm_f32le";
  reason: string;
};

export type StagedInputInfo = {
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

export type ExportProgressPayload = {
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

export type MissingAudioCode =
  | "MISSING_CACHE_FILE"
  | "STALE_SESSION"
  | "MISSING_SOURCE_FILE";

export type PlaybackResolveIssue = {
  code: MissingAudioCode;
  hint: string;
  originalPath?: string;
};

export type PlaybackMetadataLike = {
  sourceKind?: string;
  previewDir?: string;
  savedDir?: string;
};

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

export async function probeAudioFile(
  filePath: string,
): Promise<AudioSourceProfile> {
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
  const container =
    typeof format.format_name === "string" ? format.format_name : null;
  const codec =
    typeof audioStream.codec_name === "string" ? audioStream.codec_name : null;
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

function shortHash(input: string): string {
  return createHash("sha1").update(input).digest("hex").slice(0, 12);
}

export async function ensureWavInput(
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

function ensureWritableDirectory(dirPath: string) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
    const testFile = path.join(dirPath, `.write_test_${Date.now()}`);
    fs.writeFileSync(testFile, "ok");
    fs.unlinkSync(testFile);
  } catch (error: any) {
    throw new Error(
      `Export directory is not writable: ${dirPath}. ${error?.message || String(error)}`,
    );
  }
}

function getAvailableDiskBytes(dirPath: string): number | null {
  try {
    const parsed = path.parse(path.resolve(dirPath));
    const total = fs.statfsSync(parsed.root);
    const bavail = typeof total.bavail === "number" ? total.bavail : Number(total.bavail);
    const bsize = typeof total.bsize === "number" ? total.bsize : Number(total.bsize);
    if (Number.isFinite(bavail) && Number.isFinite(bsize)) {
      return Math.max(0, Math.floor(bavail * bsize));
    }
  } catch {
    // ignore lack of statfs support
  }
  return null;
}

function estimateMp3Bytes(durationSeconds: number | null, bitrate: string): number {
  const match = String(bitrate || "320k").toLowerCase().match(/(\d+)/);
  const kbps = match ? Number(match[1]) : 320;
  if (!durationSeconds || !Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return kbps * 1000;
  }
  return Math.ceil(durationSeconds * (kbps * 1000) / 8);
}

export function getPreviewCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "previews");
}

function isPathInsideDir(targetPath: string, parentDir: string): boolean {
  const target = path.resolve(targetPath).toLowerCase();
  const parent = path.resolve(parentDir).toLowerCase();
  return target === parent || target.startsWith(`${parent}${path.sep}`);
}

export function createPreviewDirForInput(inputFile: string) {
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

export function resolveEffectiveModelId(modelId: any, ensembleConfig: any): string {
  if (ensembleConfig && Array.isArray(ensembleConfig.models) && ensembleConfig.models.length > 0) {
    return "ensemble";
  }
  if (typeof modelId === "string" && modelId.trim()) return modelId.trim();
  throw new Error(
    "Missing modelId. This is a preset/config bug (modelId must be a non-empty string).",
  );
}

export function classifyMissingAudioPath(missingPath: string): {
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

export function resolvePlaybackFilePath(
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

export function resolvePlaybackStems(
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

function resolveExistingAudioSource(
  filePath: string,
  log?: (message: string, ...args: any[]) => void,
): string {
  let resolvedInputFile = filePath;
  if (!fs.existsSync(resolvedInputFile)) {
    const fallback = resolveMissingPreviewAudioPath(resolvedInputFile);
    if (fallback && fs.existsSync(fallback)) {
      log?.("[audio-source] resolved missing source path", {
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
  log,
}: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
  log?: (message: string, ...args: any[]) => void;
}): Promise<ExportTask[]> {
  const fmt = String(format || "wav").toLowerCase();
  if (!new Set(["wav", "flac", "mp3"]).has(fmt)) {
    throw new Error(`Unsupported export format: ${format}`);
  }

  const tasks: ExportTask[] = [];
  for (const [stemRaw, inputFile] of Object.entries(sourceFiles || {})) {
    const stem = sanitizeForPathSegment(stemRaw || "stem") || "stem";
    if (!inputFile || typeof inputFile !== "string") continue;
    const sourceFile = resolveExistingAudioSource(inputFile, log);
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

export async function exportFilesLocal({
  sourceFiles,
  exportPath,
  format,
  bitrate,
  requestId,
  onProgress,
  log,
}: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
  requestId: string;
  onProgress: (payload: ExportProgressPayload) => void;
  log?: (message: string, ...args: any[]) => void;
}): Promise<{ exported: Record<string, string> }> {
  if (!exportPath || typeof exportPath !== "string") {
    throw new Error("Missing exportPath");
  }

  ensureWritableDirectory(exportPath);
  await ensureBinaryAvailable(getFfmpegExe(), "ffmpeg");
  await ensureBinaryAvailable(getFfprobeExe(), "ffprobe");

  onProgress({
    requestId,
    status: "preflight",
    totalProgress: 0,
    detail: "Validating sources and export destination...",
    format: String(format || "wav").toLowerCase(),
  });

  const tasks = await buildExportTasks({
    sourceFiles,
    exportPath,
    format,
    bitrate,
    log,
  });
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

    onProgress({
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
      onProgress({
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
          onProgress({
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
          onProgress({
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

  onProgress({
    requestId,
    status: "completed",
    fileCount: tasks.length,
    totalProgress: 100,
    detail: "Export complete",
    format: String(format || "wav").toLowerCase(),
  });

  return { exported };
}

export function cleanupPreviewCache() {
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
