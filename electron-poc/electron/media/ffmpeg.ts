import { spawn } from "child_process";
import fs from "fs";
import path from "path";

export function getFfmpegExe(): string {
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

export function getFfprobeExe(): string {
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
      const probePath = ffmpegStaticPath.replace(
        /ffmpeg(?:\.exe)?$/i,
        process.platform === "win32" ? "ffprobe.exe" : "ffprobe",
      );
      if (fs.existsSync(probePath)) return probePath;
    }
  } catch {
    // ignore
  }

  return "ffprobe";
}

export async function runProcessCapture(
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
      reject(new Error(`Failed to start ${exe}: ${err?.message || String(err)}`));
    });
    child.on("close", (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${exe} failed (exit=${code}). ${stderr}`.trim()));
    });
  });
}

export async function ensureBinaryAvailable(
  exe: string,
  label: string,
): Promise<void> {
  try {
    await runProcessCapture(exe, ["-version"]);
  } catch (error: any) {
    throw new Error(
      `${label} is required but unavailable: ${error?.message || String(error)}`,
    );
  }
}

export async function runFfmpeg(args: string[]): Promise<void> {
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

export async function runFfmpegWithProgress(
  args: string[],
  durationSeconds: number | null,
  onProgress?: (progress: number) => void,
): Promise<void> {
  const exe = getFfmpegExe();

  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      exe,
      [...args, "-progress", "pipe:1", "-nostats"],
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
