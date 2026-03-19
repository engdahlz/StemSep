import type { HistoryItem, ResultAvailabilityStatus } from "@/types/store";

export function getResultAvailabilityStatus(
  item: Pick<HistoryItem, "status" | "outputFiles" | "playback" | "exportSummary">,
  overrides?: {
    hasPlaybackIssue?: boolean;
    hasMissingSource?: boolean;
  },
): ResultAvailabilityStatus {
  if (item.status === "failed") return "failed";
  if (overrides?.hasMissingSource) return "missing_source";
  if (item.exportSummary?.status === "exported") {
    const exportedFiles = Object.keys(item.exportSummary.exportedFiles || {});
    if (exportedFiles.length > 0) return "exported";
  }
  if (item.playback?.sourceKind === "missing_source") return "missing_source";
  if (overrides?.hasPlaybackIssue) return "playback_issue";

  const outputCount = Object.keys(item.outputFiles || {}).length;
  if (outputCount === 0) return "playback_issue";

  if (
    item.playback?.sourceKind === "preview_cache" ||
    item.playback?.sourceKind === "saved_output"
  ) {
    return "preview_ready";
  }

  return "preview_only";
}

export function getResultAvailabilityLabel(status: ResultAvailabilityStatus) {
  switch (status) {
    case "exported":
      return "Exported";
    case "preview_ready":
      return "Preview Ready";
    case "preview_only":
      return "Preview Only";
    case "playback_issue":
      return "Playback Issue";
    case "missing_source":
      return "Missing Source";
    case "failed":
      return "Failed";
    default:
      return "Unknown";
  }
}

export function getResultAvailabilityClasses(status: ResultAvailabilityStatus) {
  switch (status) {
    case "exported":
      return "bg-emerald-50 text-emerald-700";
    case "preview_ready":
      return "bg-sky-50 text-sky-700";
    case "preview_only":
      return "bg-amber-50 text-amber-700";
    case "playback_issue":
      return "bg-rose-50 text-rose-700";
    case "missing_source":
      return "bg-orange-50 text-orange-700";
    case "failed":
      return "bg-red-50 text-red-700";
    default:
      return "bg-slate-100 text-slate-600";
  }
}
