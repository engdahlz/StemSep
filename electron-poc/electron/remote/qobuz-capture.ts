import { app, type BrowserWindow } from "electron";
import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import { computeCaptureQualityMode } from "../playback/capture-metadata";

type ActiveLibraryProvider = "spotify" | "qobuz";

type PlaybackSurface = "desktop_app" | "browser" | "none";
type CaptureQualityMode = "best_available" | "verified_lossless";

type RemoteCatalogItem = {
  provider: "qobuz" | "spotify" | "bandcamp";
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
  playbackUrl?: string;
  playbackUri?: string;
  playbackSurface?: PlaybackSurface;
  ingestMode?: "local_file" | "remote_download" | "desktop_capture";
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
};

export type PlaybackDevice = {
  id: string;
  label: string;
  kind: "render_endpoint" | "system_loopback" | "display_loopback";
  isDefault?: boolean;
};

export type CaptureEnvironmentStatus = {
  windowsSupported: boolean;
  provider: "qobuz";
  authenticated: boolean;
  selectedDeviceId: string | null;
  selectedDeviceLabel: string | null;
  selectedDeviceReady: boolean;
  speakerSelectionAvailable: boolean;
  message?: string;
};

export type PlaybackCaptureProgressPayload = {
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

export type QobuzPlaybackVerification = {
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
  rowTitle?: string;
  rowArtist?: string;
  rowTrackId?: string;
  rowAlbumId?: string;
  activeMedia?: boolean;
  playButtonClicked?: boolean;
  mediaCount?: number;
  sinkApplied?: number;
  sinkError?: string;
  speakerSelectionSupported?: boolean;
};

export type PlaybackCaptureSession = {
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

function normalizeDeviceLabel(label: string | undefined | null) {
  return String(label || "")
    .toLowerCase()
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function getPlaybackCaptureCacheDir() {
  return path.join(app.getPath("userData"), "cache", "playback_capture");
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

function getQobuzTrailingSilenceMs() {
  const raw = String(
    process.env.STEMSEP_QOBUZ_CAPTURE_TRAILING_SILENCE_MS ||
      process.env.STEMSEP_CAPTURE_TRAILING_SILENCE_MS ||
      "",
  ).trim();
  if (!raw) return 10_000;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 2_500) {
    return 10_000;
  }
  return Math.min(parsed, 30_000);
}

function getMinimumAcceptableCaptureDurationSec(expectedDurationSec: number) {
  if (!Number.isFinite(expectedDurationSec) || expectedDurationSec <= 0) {
    return 0;
  }
  if (expectedDurationSec <= 12) {
    return Math.max(3, expectedDurationSec - 2);
  }
  if (expectedDurationSec <= 45) {
    return expectedDurationSec * 0.7;
  }
  if (expectedDurationSec <= 120) {
    return expectedDurationSec * 0.75;
  }
  return expectedDurationSec * 0.8;
}

async function readQobuzPlaybackHealth(win: BrowserWindow) {
  try {
    return await win.webContents.executeJavaScript(
      `(() => {
        const bodyText = document.body?.innerText || "";
        const media = globalThis.navigator?.mediaSession?.metadata
          ? {
              title: globalThis.navigator.mediaSession.metadata.title || "",
              artist: globalThis.navigator.mediaSession.metadata.artist || "",
              album: globalThis.navigator.mediaSession.metadata.album || "",
            }
          : null;
        const timeMatches = Array.from(bodyText.matchAll(/\\b\\d{2}:\\d{2}\\b/g)).map((match) => match[0]);
        return {
          hasPlaybackError: /an error occurred while playing the current track/i.test(bodyText),
          mediaSession: media,
          visibleTimes: timeMatches.slice(0, 4),
          bodySnippet: bodyText.slice(0, 400),
        };
      })()`,
      true,
    );
  } catch {
    return null;
  }
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

export function createQobuzCaptureController({
  log,
  getMainWindow,
  getStoredCaptureOutputDeviceId,
  getQobuzLibraryUrl,
  getExistingQobuzAutomationWindow,
  getQobuzAutomationWindow,
  loadURLAndWait,
  waitForDidFinishLoad,
  checkLibraryProviderAuthenticated,
  searchQobuzCatalogViaApi,
  probeAudioFile,
  sendBackendCommand,
  sendBackendCommandWithRetry,
}: {
  log: (message: string, ...args: any[]) => void;
  getMainWindow: () => BrowserWindow | null;
  getStoredCaptureOutputDeviceId: () => string | null;
  getQobuzLibraryUrl: () => string;
  getExistingQobuzAutomationWindow: () => BrowserWindow | null;
  getQobuzAutomationWindow: () => BrowserWindow;
  loadURLAndWait: (win: BrowserWindow, url: string, timeoutMs?: number) => Promise<void>;
  waitForDidFinishLoad: (win: BrowserWindow, timeoutMs?: number) => Promise<void>;
  checkLibraryProviderAuthenticated: (provider: ActiveLibraryProvider) => Promise<boolean>;
  searchQobuzCatalogViaApi: (query: string) => Promise<RemoteCatalogItem[]>;
  probeAudioFile: (filePath: string) => Promise<{
    sampleRate?: number;
    channels?: number;
    durationSeconds?: number;
  }>;
  sendBackendCommand: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
  ) => Promise<any>;
  sendBackendCommandWithRetry: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
    maxRetries?: number,
  ) => Promise<any>;
}) {
  const qobuzSinkIdByPlaybackDeviceId = new Map<string, string>();
  const cancelledPlaybackCaptureIds = new Set<string>();
  const lastPlaybackCaptureProgressById = new Map<
    string,
    PlaybackCaptureProgressPayload
  >();
  let cachedNativePlaybackDevices: PlaybackDevice[] = [];
  let cachedQobuzSpeakerSelectionAvailable = false;
  let cachedQobuzSpeakerSelectionError: string | null = null;
  const playbackCaptureSessions = new Map<string, PlaybackCaptureSession>();

  function emitPlaybackCaptureProgress(payload: PlaybackCaptureProgressPayload) {
    if (payload.captureId) {
      lastPlaybackCaptureProgressById.set(payload.captureId, payload);
      if (/completed|cancelled|error|failed/i.test(String(payload.status || ""))) {
        setTimeout(() => {
          lastPlaybackCaptureProgressById.delete(payload.captureId!);
        }, 60_000);
      }
    }
    const mainWindow = getMainWindow();
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

  async function getNativePlaybackDevices(): Promise<PlaybackDevice[]> {
    if (isPlaybackCaptureActive() && cachedNativePlaybackDevices.length > 0) {
      return cachedNativePlaybackDevices;
    }
    const devices = await sendBackendCommandWithRetry(
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

  async function resolveEffectiveCaptureDevice(
    preferredDeviceId: string,
    playbackResult: QobuzPlaybackVerification | null | undefined,
  ) {
    const sinkApplied = Number(playbackResult?.sinkApplied || 0);
    const mediaCount = Number(playbackResult?.mediaCount || 0);
    const activeMedia = !!playbackResult?.activeMedia;
    if (sinkApplied > 0 || mediaCount > 0 || activeMedia) {
      return {
        deviceId: preferredDeviceId,
        reason: null as string | null,
        probes: [] as Array<Record<string, any>>,
      };
    }

    const devices = await getNativePlaybackDevices();
    const preferredDevice =
      devices.find((device) => device.id === preferredDeviceId) || null;
    const defaultDevice = devices.find((device) => device.isDefault) || null;
    const orderedDevices = [
      preferredDevice,
      defaultDevice && defaultDevice.id !== preferredDeviceId ? defaultDevice : null,
      ...devices.filter(
        (device) =>
          device.id !== preferredDeviceId &&
          (!defaultDevice || device.id !== defaultDevice.id),
      ),
    ].filter((device): device is PlaybackDevice => !!device);

    const probes: Array<Record<string, any>> = [];
    for (const device of orderedDevices) {
      try {
        const probe = await sendBackendCommand(
          "probe_playback_device_activity",
          {
            device_id: device.id,
            timeout_ms: 2_500,
            min_active_rms: 0.0006,
          },
          12_000,
        );
        probes.push({
          deviceId: device.id,
          label: device.label,
          isDefault: !!device.isDefault,
          ...(probe && typeof probe === "object" ? probe : {}),
        });
        if (probe?.detected) {
          if (device.id === preferredDeviceId) {
            return {
              deviceId: device.id,
              reason: null as string | null,
              probes,
            };
          }
          return {
            deviceId: device.id,
            reason: `Qobuz did not expose a routable media element, so capture is using the playback device that actually showed audio activity: ${device.label}.`,
            probes,
          };
        }
      } catch (error: any) {
        probes.push({
          deviceId: device.id,
          label: device.label,
          isDefault: !!device.isDefault,
          error: error?.message || String(error),
        });
      }
    }

    if (!defaultDevice || defaultDevice.id === preferredDeviceId) {
      return {
        deviceId: preferredDeviceId,
        reason: null as string | null,
        probes,
      };
    }

    return {
      deviceId: defaultDevice.id,
      reason: `Qobuz did not expose a routable media element, so capture is falling back from ${preferredDevice?.label || "the saved silent output"} to the current default playback device ${defaultDevice.label}.`,
      probes,
    };
  }

  async function prepareQobuzPlaybackForCapture(
    item: RemoteCatalogItem,
    preferredDeviceId: string,
    captureId: string,
  ) {
    const attempts: Array<Record<string, any>> = [];
    let lastError: any = null;

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      if (attempt > 1) {
        emitPlaybackCaptureProgress({
          provider: "qobuz",
          captureId,
          status: "verifying",
          detail: "Qobuz did not start cleanly. Retrying hidden playback once...",
        });
        await stopHiddenQobuzPlayback();
        await sleep(1_100);
      }

      try {
        const playbackReady = await prepareHiddenQobuzPlayback(
          item,
          preferredDeviceId,
        );
        const effectiveCaptureDevice = await resolveEffectiveCaptureDevice(
          preferredDeviceId,
          playbackReady.playbackResult || null,
        );
        const playbackWindow =
          getExistingQobuzAutomationWindow() || getQobuzAutomationWindow();
        const playbackHealth = await readQobuzPlaybackHealth(playbackWindow);
        const detectedAudio =
          Number(playbackReady.playbackResult?.sinkApplied || 0) > 0 ||
          Number(playbackReady.playbackResult?.mediaCount || 0) > 0 ||
          !!playbackReady.playbackResult?.activeMedia ||
          effectiveCaptureDevice.probes.some((probe) => !!probe?.detected);

        const attemptSummary = {
          attempt,
          verificationMatched:
            !!playbackReady.playbackResult?.verificationMatched,
          verificationReason:
            playbackReady.playbackResult?.verificationReason || null,
          activeMedia: !!playbackReady.playbackResult?.activeMedia,
          mediaCount: Number(playbackReady.playbackResult?.mediaCount || 0),
          sinkApplied: Number(playbackReady.playbackResult?.sinkApplied || 0),
          detectedAudio,
          effectiveDeviceId: effectiveCaptureDevice.deviceId,
          captureFallbackReason: effectiveCaptureDevice.reason,
          playbackHealth,
          deviceProbes: effectiveCaptureDevice.probes,
        };
        attempts.push(attemptSummary);

        if (detectedAudio) {
          return {
            playbackReady,
            effectiveCaptureDevice,
            playbackHealth,
            attempts,
          };
        }

        if (playbackHealth?.hasPlaybackError) {
          const error: any = new Error(
            "Qobuz reported a playback error before audio reached the capture device.",
          );
          error.details = {
            playbackHealth,
            playbackResult: playbackReady.playbackResult || null,
            attempts,
          };
          throw error;
        }
      } catch (error: any) {
        lastError = error;
        attempts.push({
          attempt,
          error: error?.message || String(error),
          details:
            error && typeof error === "object" && "details" in error
              ? (error as any).details
              : null,
        });
        if (cancelledPlaybackCaptureIds.has(captureId)) {
          throw error;
        }
      }
    }

    if (lastError instanceof Error) {
      const typedLastError = lastError as Error & { details?: Record<string, any> };
      if (!typedLastError.details || typeof typedLastError.details !== "object") {
        typedLastError.details = {};
      }
      typedLastError.details.attempts = attempts;
      throw typedLastError;
    }

    const error: any = new Error(
      "Qobuz accepted the track selection but no playback activity reached any capture device. The web player appears stalled.",
    );
    error.details = { attempts };
    throw error;
  }

  async function getQobuzBrowserOutputDevices() {
    const win = getQobuzAutomationWindow();
    const currentUrl = win.webContents.getURL();
    if (!currentUrl || !/qobuz\.com/i.test(currentUrl)) {
      await loadURLAndWait(win, getQobuzLibraryUrl(), 20_000);
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
    const win = getExistingQobuzAutomationWindow();
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
      await sendBackendCommandWithRetry(
        "cancel_playback_capture",
        captureId ? { capture_id: captureId } : {},
        10_000,
        0,
      ).catch(() => null);
      return { success: true };
    }

    await sendBackendCommandWithRetry(
      "cancel_playback_capture",
      captureId ? { capture_id: captureId } : {},
      10_000,
      0,
    ).catch(() => null);
    return { success: true };
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

  function writePlaybackCaptureDiagnostics(
    session: PlaybackCaptureSession,
    payload: Record<string, any>,
  ) {
    try {
      safeMkdir(path.dirname(session.diagnosticsPath));
      let existing: Record<string, any> = {};
      if (fs.existsSync(session.diagnosticsPath)) {
        try {
          existing = JSON.parse(fs.readFileSync(session.diagnosticsPath, "utf-8"));
        } catch {
          existing = {};
        }
      }
      fs.writeFileSync(
        session.diagnosticsPath,
        JSON.stringify(
          {
            ...existing,
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
    if (initialPath === "/error/404" && item.pageUrl && item.pageUrl !== targetUrl) {
      await loadURLAndWait(win, item.pageUrl, 20_000);
      await sleep(1200);
    }

    const playbackResult = (await win.webContents.executeJavaScript(
      `(async () => {
        const desiredSinkId = ${JSON.stringify(sinkId)};
        const targetTrackId = ${JSON.stringify(item.trackId)};
        const targetTitle = ${JSON.stringify(
          String(item.title || "").replace(/\\s+/g, " ").trim().toLowerCase(),
        )};
        const targetArtist = ${JSON.stringify(
          String(item.artist || "").replace(/\\s+/g, " ").trim().toLowerCase(),
        )};
        const targetAlbum = ${JSON.stringify(
          String(item.album || "").replace(/\\s+/g, " ").trim().toLowerCase(),
        )};
        const targetAlbumId = ${JSON.stringify(item.albumId ? String(item.albumId) : "")};
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
            .replace(/\\s+/g, " ")
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
          /\\bplaylist(s)?\\b|\\bplayer\\b|\\bplay queue\\b/.test(label);
        const exactPlayLabel = (label) =>
          /^(play|play track|lecture|reproducir|riproduci|spielen|ouvir|lyssna)$/i.test(label);
        const isPlayLikeButton = (element) => {
          const label = labelOf(element);
          const className = String(element?.className || "");
          return (
            (!!label &&
              !isPlaybackNavigationLabel(label) &&
              (exactPlayLabel(label) ||
                /\\b(play|lecture|reproducir|riproduci|spielen|ouvir|lyssna)\\b/i.test(
                  label,
                ))) ||
            /icon-play-arrow|play/i.test(className)
          );
        };
        const extractIdFromHref = (href, kind) => {
          const match = String(href || "").match(new RegExp("/" + kind + "/([^/?#]+)", "i"));
          return match?.[1] || "";
        };
        const getTrackRowSummary = (row) => {
          if (!(row instanceof HTMLElement) || !isVisible(row)) return null;
          const text = normalize(row.innerText || row.textContent || "");
          if (!text) return null;

          const titleAnchor = row.querySelector('a[href*="/track/"]');
          const titleNode = row.querySelector('[class*="ListItem__title"]');
          const artistAnchor =
            row.querySelector('a[href*="/artist/"]') ||
            row.querySelector('[class*="ListItem__artist"] a[href]');
          const artistNode = row.querySelector('[class*="ListItem__artist"]');
          const albumAnchor = row.querySelector('a[href*="/album/"]');
          const numberNode = row.querySelector('.ListItem__number, [class*="ListItem__number"]');
          const playerButton = row.querySelector(
            '.ListItem__player[role="button"], [class*="ListItem__player"][role="button"]',
          );
          const title = normalize(
            titleAnchor?.textContent ||
              titleNode?.textContent ||
              "",
          );
          const artist = normalize(
            artistAnchor?.textContent ||
              artistNode?.textContent ||
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
            normalize(numberNode?.textContent || "").replace(/[^\\d]/g, ""),
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
          if (targetTitle && title && (title === targetTitle || title.includes(targetTitle))) score += 100;
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
            playerButton: playerButton instanceof HTMLElement ? playerButton : null,
            title,
            artist,
            trackId,
            albumId,
            trackHref,
            albumHref,
            trackNumber: Number.isFinite(parsedTrackNumber) ? parsedTrackNumber : null,
            rowState,
            score,
          };
        };
        const collectTrackRows = (limit = 12) =>
          Array.from(document.querySelectorAll('[class*="ListItem"], [data-testid*="track"], tr, li'))
            .map((row) => getTrackRowSummary(row))
            .filter(Boolean)
            .sort((left, right) => right.score - left.score)
            .slice(0, limit);
        const findExactTrackRowButton = () => {
          const rows = collectTrackRows(16);
          return rows.find((row) => row.playerButton && row.score >= 120) || null;
        };
        const readPlayerSnapshot = () => {
          const mediaSessionTitle = normalize(
            navigator.mediaSession?.metadata?.title || "",
          );
          const mediaSessionArtist = normalize(
            navigator.mediaSession?.metadata?.artist || "",
          );
          const playerScopes = [
            ...document.querySelectorAll('footer, [class*="Player"], [class*="NowPlaying"], [data-testid*="player"]'),
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
                mediaSessionTitle ||
                "",
            );
            const currentArtist = normalize(
              artistAnchor?.textContent ||
                scope.querySelector('[class*="artist"]')?.textContent ||
                mediaSessionArtist ||
                "",
            );
            const currentTrackHref = String(trackAnchor?.getAttribute?.("href") || "");
            const currentAlbumHref = String(albumAnchor?.getAttribute?.("href") || "");
            const currentTrackId = extractIdFromHref(currentTrackHref, "track");
            const currentAlbumId = extractIdFromHref(currentAlbumHref, "album");
            if (currentTitle || currentArtist || currentTrackHref || currentAlbumHref || currentTrackId || currentAlbumId) {
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
            currentTitle: mediaSessionTitle || "",
            currentArtist: mediaSessionArtist || "",
            currentTrackHref: "",
            currentAlbumHref: "",
            currentTrackId: "",
            currentAlbumId: "",
          };
        };
        const looselyMatches = (left, right) =>
          !!left &&
          !!right &&
          (left === right || left.includes(right) || right.includes(left));
        const verifyPlayerSnapshot = (snapshot, rowSnapshot = null, rowScopeText = "") => {
          const trackIdMatched =
            !!targetTrackId &&
            !!snapshot.currentTrackId &&
            snapshot.currentTrackId === targetTrackId;
          const hrefMatched =
            !!targetTrackId &&
            String(snapshot.currentTrackHref || "").includes("/track/" + targetTrackId);
          const titleMatched =
            !!targetTitle && looselyMatches(snapshot.currentTitle, targetTitle);
          const artistMatched =
            !targetArtist ||
            !snapshot.currentArtist ||
            looselyMatches(snapshot.currentArtist, targetArtist);
          if (trackIdMatched) return { matched: true, reason: "track_id" };
          if (hrefMatched) return { matched: true, reason: "track_href" };
          if (titleMatched && artistMatched) return { matched: true, reason: "title_artist" };
          const rowState = normalize(rowSnapshot?.rowState || "");
          const rowTitle = normalize(rowSnapshot?.title || "");
          const rowArtist = normalize(rowSnapshot?.artist || "");
          const rowTrackId = String(rowSnapshot?.trackId || "");
          const rowAlbumId = String(rowSnapshot?.albumId || "");
          const normalizedScopeText = normalize(rowScopeText || "");
          const rowActive = /pause|playing|active/.test(rowState);
          const rowTrackIdMatched =
            !!targetTrackId && !!rowTrackId && rowTrackId === targetTrackId;
          const rowAlbumMatched =
            !targetAlbumId || !rowAlbumId || rowAlbumId === targetAlbumId;
          const rowTitleMatched =
            (!!targetTitle && looselyMatches(rowTitle, targetTitle)) ||
            (!!targetTitle && !!normalizedScopeText && normalizedScopeText.includes(targetTitle));
          const rowArtistMatched =
            !targetArtist ||
            looselyMatches(rowArtist, targetArtist) ||
            (!!normalizedScopeText && normalizedScopeText.includes(targetArtist));
          const rowTrackNumberMatched =
            !Number.isFinite(targetTrackNumber) ||
            !Number.isFinite(rowSnapshot?.trackNumber) ||
            rowSnapshot?.trackNumber === targetTrackNumber;
          if (rowActive && rowTrackIdMatched) {
            return { matched: true, reason: "row_track_id" };
          }
          if (rowActive && rowAlbumMatched && rowTitleMatched && rowArtistMatched && rowTrackNumberMatched) {
            return { matched: true, reason: "row_state" };
          }
          return { matched: false, reason: "failed" };
        };
        const waitForPlaybackVerification = async (timeoutMs) => {
          const startedAt = Date.now();
          let snapshot = readPlayerSnapshot();
          let targetRow = findExactTrackRowButton() || exactTrackRow || null;
          let verification = verifyPlayerSnapshot(snapshot, targetRow, clickedScopeText);
          while (Date.now() - startedAt < timeoutMs) {
            if (verification.matched) {
              return {
                ...snapshot,
                verificationMatched: true,
                verificationReason: verification.reason,
                rowMatched: !!targetRow,
                rowState: targetRow?.rowState || "",
                rowTitle: targetRow?.title || "",
                rowArtist: targetRow?.artist || "",
                rowTrackId: targetRow?.trackId || "",
                rowAlbumId: targetRow?.albumId || "",
              };
            }
            await wait(350);
            snapshot = readPlayerSnapshot();
            targetRow = findExactTrackRowButton() || exactTrackRow || null;
            verification = verifyPlayerSnapshot(snapshot, targetRow, clickedScopeText);
          }
          targetRow = findExactTrackRowButton() || exactTrackRow || null;
          return {
            ...snapshot,
            verificationMatched: false,
            verificationReason: verification.reason,
            rowMatched: !!targetRow,
            rowState: targetRow?.rowState || "",
            rowTitle: targetRow?.title || "",
            rowArtist: targetRow?.artist || "",
            rowTrackId: targetRow?.trackId || "",
            rowAlbumId: targetRow?.albumId || "",
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
            /\\bdisc\\b|\\bcd\\b|\\bmedia\\b/.test(text)
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
              (!targetArtist ||
                text.includes(targetArtist) ||
                text.length <= targetTitle.length + 32)
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
        const findTrackScopedPlayButton = () => chooseBestPlayTarget();

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
              /\\b(play|lecture|reproducir|riproduci|spielen|ouvir|lyssna)\\b/i.test(label),
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
              /\\b(next|skip|forward|suivant|nästa|weiter|próximo|seguinte)\\b/i,
              /next|skip-forward|forward/i,
            );
            if (!(nextButton instanceof HTMLElement)) break;
            nextButton.click();
            clickStrategy = "album_play_then_skip_" + targetTrackNumber;
            await wait(900);
          }
        }

        let verificationSnapshot = await waitForPlaybackVerification(6_500);
        if (
          !verificationSnapshot.verificationMatched &&
          !exactTrackRow?.playerButton &&
          trackNumberTarget instanceof HTMLElement
        ) {
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
          rowTitle: String(verificationSnapshot?.rowTitle || ""),
          rowArtist: String(verificationSnapshot?.rowArtist || ""),
          rowTrackId: String(verificationSnapshot?.rowTrackId || ""),
          rowAlbumId: String(verificationSnapshot?.rowAlbumId || ""),
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
    )) as QobuzPlaybackVerification & { currentPath?: string };

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
        playbackResult?.currentArtist || playbackResult?.rowArtist,
        playbackResult?.currentTitle || playbackResult?.rowTitle,
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
    let partialCaptureResult: any = null;
    let partialProbeProfile: any = null;
    let partialQuality: any = null;
    let playbackHealth: any = null;
    let playbackWarmupAttempts: Array<Record<string, any>> = [];

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
      const preparedPlayback = await prepareQobuzPlaybackForCapture(
        item,
        deviceId,
        captureId,
      );
      const playbackReady = preparedPlayback.playbackReady;
      const session = playbackCaptureSessions.get(captureId);
      const effectiveCaptureDevice = preparedPlayback.effectiveCaptureDevice;
      playbackWarmupAttempts = preparedPlayback.attempts;
      if (session) {
        session.deviceId = effectiveCaptureDevice.deviceId;
        session.verification = playbackReady.playbackResult || null;
        writePlaybackCaptureDiagnostics(session, {
          stage: "playback_verified",
          verification: playbackReady.playbackResult || null,
          sinkId: playbackReady.sinkId || null,
          requestedDeviceId: deviceId,
          effectiveDeviceId: effectiveCaptureDevice.deviceId,
          captureFallbackReason: effectiveCaptureDevice.reason,
          deviceProbes: effectiveCaptureDevice.probes,
          playbackWarmupAttempts,
        });
      }

      if (cancelledPlaybackCaptureIds.has(captureId)) {
        throw new Error("Capture cancelled");
      }

      if (effectiveCaptureDevice.reason) {
        emitPlaybackCaptureProgress({
          provider: "qobuz",
          captureId,
          status: "awaiting_audio",
          detail: effectiveCaptureDevice.reason,
        });
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
      > = sendBackendCommand(
        "capture_playback_loopback",
        {
          capture_id: captureId,
          device_id: effectiveCaptureDevice.deviceId,
          output_path: outputPath,
          expected_duration_sec: getExpectedCaptureDurationSec(item),
          start_timeout_ms: 22_000,
          trailing_silence_ms: getQobuzTrailingSilenceMs(),
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
      partialCaptureResult = captureResult;
      const profile = await probeAudioFile(captureResult.file_path);
      partialProbeProfile = profile;
      const quality = computeCaptureQualityMode(
        "qobuz",
        item,
        captureResult.capture_sample_rate || profile.sampleRate || undefined,
        captureResult.capture_channels || profile.channels || undefined,
      );
      partialQuality = quality;
      const playbackWindow =
        getExistingQobuzAutomationWindow() || getQobuzAutomationWindow();
      const expectedDurationSec = getExpectedCaptureDurationSec(item);
      const actualDurationSec =
        captureResult.duration_sec || profile.durationSeconds || 0;
      playbackHealth = await readQobuzPlaybackHealth(playbackWindow);
      const minimumAcceptableDurationSec =
        typeof expectedDurationSec === "number"
          ? getMinimumAcceptableCaptureDurationSec(expectedDurationSec)
          : 0;
      const endedTooEarly =
        typeof expectedDurationSec === "number" &&
        expectedDurationSec > 0 &&
        actualDurationSec > 0 &&
        actualDurationSec < minimumAcceptableDurationSec;
      if (session) {
        writePlaybackCaptureDiagnostics(session, {
          stage: "completed",
          verification: session.verification || null,
          captureResult,
          probedFile: profile,
          quality,
          playbackHealth,
        });
      }
      if (
        playbackHealth?.hasPlaybackError &&
        typeof expectedDurationSec === "number" &&
        expectedDurationSec > 0 &&
        actualDurationSec + 5 < expectedDurationSec
      ) {
        throw new Error(
          `Qobuz playback ended early after ${actualDurationSec.toFixed(1)}s of expected ${expectedDurationSec.toFixed(1)}s.`,
        );
      }
      if (endedTooEarly) {
        throw new Error(
          `Captured audio ended too early after ${actualDurationSec.toFixed(1)}s of expected ${expectedDurationSec!.toFixed(1)}s.`,
        );
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
        capture_device_id: effectiveCaptureDevice.deviceId,
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
      const errorMessage =
        error instanceof Error
          ? error.message
          : String(error || "Playback capture failed");
      const session = playbackCaptureSessions.get(captureId);
      if (session?.backendCaptureStarted) {
        await sendBackendCommandWithRetry(
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
          playbackWarmupAttempts,
          captureResult: partialCaptureResult,
          probedFile: partialProbeProfile,
          quality: partialQuality,
          playbackHealth,
          error: errorMessage,
          details:
            error && typeof error === "object" && "details" in error
              ? (error as any).details
              : null,
        });
      }
      const partialFilePath =
        partialCaptureResult?.file_path && typeof partialCaptureResult.file_path === "string"
          ? partialCaptureResult.file_path
          : outputPath;
      try {
        if (
          partialFilePath &&
          fs.existsSync(partialFilePath) &&
          !cancelledPlaybackCaptureIds.has(captureId)
        ) {
          fs.unlinkSync(partialFilePath);
        }
      } catch {
        // ignore cleanup failures for partial captures
      }
      if (cancelledPlaybackCaptureIds.has(captureId)) {
        emitPlaybackCaptureProgress({
          provider: "qobuz",
          captureId,
          status: "cancelled",
          detail: "Capture cancelled.",
          error: "Capture cancelled",
        });
        throw new Error("Capture cancelled");
      }
      emitPlaybackCaptureProgress({
        provider: "qobuz",
        captureId,
        status: "failed",
        detail: errorMessage,
        error: errorMessage,
      });
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
    const query = String(
      process.env.STEMSEP_QOBUZ_CAPTURE_SMOKETEST_QUERY || "",
    ).trim();
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

  return {
    qobuzSinkIdByPlaybackDeviceId,
    lastPlaybackCaptureProgressById,
    playbackCaptureSessions,
    emitPlaybackCaptureProgress,
    isPlaybackCaptureActive,
    isQobuzPlaybackCaptureActive,
    getNativePlaybackDevices,
    refreshQobuzPlaybackDeviceMappings,
    getCaptureEnvironmentStatusForQobuz,
    stopHiddenQobuzPlayback,
    startQobuzPlaybackCaptureForItem,
    abortPlaybackCaptureSession,
    maybeRunQobuzCaptureSmokeTest,
  };
}
