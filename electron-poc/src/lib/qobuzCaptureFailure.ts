export type QobuzCaptureFailureKind =
  | "cancelled"
  | "verification_failed"
  | "device_probe_failed"
  | "capture_silence"
  | "playback_stopped_upstream"
  | "segment_fetch_failed"
  | "unknown";

export type QobuzCaptureFailureInfo = {
  kind: QobuzCaptureFailureKind;
  code: string;
  retryable: boolean;
  hint?: string;
};

export type QobuzCaptureFailurePlan = QobuzCaptureFailureInfo & {
  willRetry: boolean;
  cleanupPartialCapture: boolean;
  status: "retrying" | "failed" | "cancelled";
};

function joinedFailureText(message: string | undefined, details: Record<string, any> | null | undefined) {
  return [
    message || "",
    details?.message || "",
    details?.playbackHealth?.bodySnippet || "",
    JSON.stringify(details?.deviceProbes || []),
    JSON.stringify(details?.attempts || []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function classifyQobuzCaptureFailure(
  message: string | undefined,
  details?: Record<string, any> | null,
): QobuzCaptureFailureInfo {
  const text = joinedFailureText(message, details);

  if (/504|segment|akamai/.test(text)) {
    return {
      kind: "segment_fetch_failed",
      code: "QOBUZ_SEGMENT_FETCH_FAILED",
      retryable: true,
      hint: "Qobuz lost one of the encrypted stream segments mid-playback. StemSep can retry once, but repeated failures usually mean an upstream playback problem.",
    };
  }

  if (
    /ended early|an error occurred while playing the current track|playback error before audio reached the capture device|playback ended/.test(
      text,
    )
  ) {
    return {
      kind: "playback_stopped_upstream",
      code: "QOBUZ_PLAYBACK_STOPPED_UPSTREAM",
      retryable: true,
      hint: "Qobuz started the track but the web player lost playback before capture finished.",
    };
  }

  if (
    /no playback activity reached any capture device|routable media element|speaker selection|capture device/.test(
      text,
    )
  ) {
    return {
      kind: "device_probe_failed",
      code: "QOBUZ_DEVICE_PROBE_FAILED",
      retryable: true,
      hint: "The selected Qobuz track was verified, but no stable playback endpoint carried audio into capture.",
    };
  }

  if (/captured audio ended too early|capture timeout|silence|timeout elapsed/.test(text)) {
    return {
      kind: "capture_silence",
      code: "QOBUZ_CAPTURE_SILENCE",
      retryable: true,
      hint: "Capture started but the recorded audio never stayed active long enough to finish cleanly.",
    };
  }

  if (
    /could not be verified|did not lock onto|playback could not be verified|selected track/i.test(
      text,
    )
  ) {
    return {
      kind: "verification_failed",
      code: "QOBUZ_VERIFICATION_FAILED",
      retryable: true,
      hint: "StemSep could not prove that the hidden Qobuz player locked onto the requested track.",
    };
  }

  return {
    kind: "unknown",
    code: "CAPTURE_START_FAILED",
    retryable: false,
  };
}

export function shouldRetryQobuzCaptureFailure(
  failure: Pick<QobuzCaptureFailureInfo, "retryable">,
  attempt: number,
  maxAttempts: number,
) {
  return failure.retryable && attempt < maxAttempts;
}

export function buildQobuzCaptureFailurePlan({
  message,
  details,
  cancelled,
  attempt,
  maxAttempts,
}: {
  message: string | undefined;
  details?: Record<string, any> | null;
  cancelled?: boolean;
  attempt: number;
  maxAttempts: number;
}): QobuzCaptureFailurePlan {
  if (cancelled) {
    return {
      kind: "cancelled",
      code: "CAPTURE_CANCELLED",
      retryable: false,
      willRetry: false,
      cleanupPartialCapture: true,
      status: "cancelled",
    };
  }

  const failure = classifyQobuzCaptureFailure(message, details);
  const willRetry = shouldRetryQobuzCaptureFailure(failure, attempt, maxAttempts);
  return {
    ...failure,
    willRetry,
    cleanupPartialCapture: true,
    status: willRetry ? "retrying" : "failed",
  };
}
