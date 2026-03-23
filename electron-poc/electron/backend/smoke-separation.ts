import { app } from "electron";
import fs from "fs";
import path from "path";
import type { LogFn } from "../system/logger";

type SendBackendCommand = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
) => Promise<any>;

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

type SelectionJobSnapshot = {
  job_id?: string;
  jobId?: string;
  status?: string;
  output_files?: Record<string, string>;
  outputFiles?: Record<string, string>;
  output_dir?: string;
  outputDir?: string;
  message?: string;
  error?: string;
  progress?: number;
};

function extractJobId(value: any): string | null {
  const candidate =
    value?.job_id ??
    value?.jobId ??
    value?.selection_job_id ??
    value?.selectionJobId ??
    value?.job?.job_id ??
    value?.job?.jobId ??
    value?.selection_job?.job_id ??
    value?.selection_job?.jobId;
  return typeof candidate === "string" && candidate.trim()
    ? candidate.trim()
    : null;
}

function normalizeSelectionJobSnapshot(
  value: any,
): SelectionJobSnapshot | null {
  if (!value || typeof value !== "object") return null;
  return {
    job_id:
      typeof value.job_id === "string"
        ? value.job_id
        : typeof value.jobId === "string"
          ? value.jobId
          : undefined,
    status: typeof value.status === "string" ? value.status : undefined,
    output_files:
      value.output_files && typeof value.output_files === "object"
        ? (value.output_files as Record<string, string>)
        : value.outputFiles && typeof value.outputFiles === "object"
          ? (value.outputFiles as Record<string, string>)
          : undefined,
    output_dir:
      typeof value.output_dir === "string"
        ? value.output_dir
        : typeof value.outputDir === "string"
          ? value.outputDir
          : undefined,
    message: typeof value.message === "string" ? value.message : undefined,
    error: typeof value.error === "string" ? value.error : undefined,
    progress:
      typeof value.progress === "number" ? value.progress : undefined,
  };
}

function isFinalSelectionJobStatus(status?: string) {
  return ["completed", "failed", "cancelled", "discarded"].includes(
    String(status || "").toLowerCase(),
  );
}

async function waitForSelectionJobCompletion({
  sendBackendCommand,
  log,
  jobId,
}: {
  sendBackendCommand: SendBackendCommand;
  log: LogFn;
  jobId: string;
}) {
  const startedAt = Date.now();
  const timeoutMs = 60 * 60 * 1000;
  let lastLoggedProgress = -1;

  while (Date.now() - startedAt < timeoutMs) {
    const snapshot = normalizeSelectionJobSnapshot(
      await sendBackendCommand(
        "get_selection_job",
        {
          job_id: jobId,
        },
        15000,
      ),
    );

    if (!snapshot) {
      throw new Error("Selection job not found");
    }

    if (
      typeof snapshot.progress === "number" &&
      snapshot.progress !== lastLoggedProgress
    ) {
      lastLoggedProgress = snapshot.progress;
      log(`[smoke] progress ${snapshot.progress}%`, snapshot.message || "");
    }

    const outputFiles = snapshot.output_files || {};
    if (
      isFinalSelectionJobStatus(snapshot.status) &&
      snapshot.status === "completed" &&
      Object.keys(outputFiles).length > 0
    ) {
      return snapshot;
    }

    if (isFinalSelectionJobStatus(snapshot.status) && snapshot.status !== "completed") {
      throw new Error(
        snapshot.error || snapshot.message || `Selection job ${snapshot.status}`,
      );
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  throw new Error("Smoke separation timeout (60 minutes)");
}

export function createSmokeSeparationRunner({
  log,
  sendBackendCommand,
  sendBackendCommandWithRetry,
}: {
  log: LogFn;
  sendBackendCommand: SendBackendCommand;
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
}) {
  async function maybeRunSmokeSeparation(): Promise<void> {
    const enabled = (process.env.STEMSEP_SMOKE_SEPARATION || "").trim();
    if (!enabled || enabled === "0" || enabled.toLowerCase() === "false") {
      return;
    }

    const inputFile =
      (process.env.STEMSEP_SMOKE_INPUT_FILE || "").trim() ||
      path.join(__dirname, "../strip mall.wav");
    const modelId =
      (process.env.STEMSEP_SMOKE_MODEL_ID || "").trim() ||
      "bs-roformer-viperx-1297";
    const device = (process.env.STEMSEP_SMOKE_DEVICE || "").trim() || undefined;

    const outRoot = path.join(app.getPath("userData"), "smoke_runs");
    const runDir = path.join(outRoot, `run-${Date.now()}`);
    try {
      fs.mkdirSync(runDir, { recursive: true });
    } catch {
      // ignore
    }

    log("[smoke] Starting smoke separation", {
      inputFile,
      modelId,
      runDir,
      device,
    });

    try {
      const preflight = await sendBackendCommandWithRetry(
        "separation_preflight",
        {
          file_path: inputFile,
          model_id: modelId,
          selection_type: "model",
          selection_id: modelId,
          selection_envelope: {
            selectionType: "model",
            selectionId: modelId,
            selection_type: "model",
            selection_id: modelId,
          },
          output_dir: runDir,
          device,
        },
        60000,
      );
      log("[smoke] preflight ok", preflight);

      const start = await sendBackendCommand(
        "run_selection_job",
        {
          inputFile,
          modelId,
          outputDir: runDir,
          selectionType: "model",
          selectionId: modelId,
          selectionEnvelope: {
            selectionType: "model",
            selectionId: modelId,
            selection_type: "model",
            selection_id: modelId,
          },
          device,
        },
        60000,
      );

      const jobId = extractJobId(start);
      if (!jobId) {
        throw new Error("Smoke separation: backend did not return job id");
      }
      log("[smoke] job started", { jobId, start });

      const snapshot = await waitForSelectionJobCompletion({
        sendBackendCommand,
        log,
        jobId,
      });

      const outputFiles = Object.values(snapshot.output_files || {}).filter(
        (value): value is string => typeof value === "string" && value.trim().length > 0,
      );
      const existingOutputs = outputFiles.filter((filePath) => {
        try {
          return fs.existsSync(filePath);
        } catch {
          return false;
        }
      });

      if (existingOutputs.length === 0) {
        throw new Error(
          `Smoke separation completed but no output files were found under ${runDir}`,
        );
      }

      log("[smoke] SUCCESS", {
        jobId,
        runDir,
        outputFiles: existingOutputs,
      });
    } catch (error: any) {
      log("[smoke] FAILED", error?.message || String(error));
    } finally {
      if ((process.env.STEMSEP_SMOKE_QUIT || "").trim() === "1") {
        log("[smoke] Quitting app (STEMSEP_SMOKE_QUIT=1)");
        app.quit();
      }
    }
  }

  return {
    maybeRunSmokeSeparation,
  };
}
