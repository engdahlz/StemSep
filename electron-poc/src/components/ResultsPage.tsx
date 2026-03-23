import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  Calendar,
  ChevronDown,
  Clock,
  Download,
  Headphones,
  Layers,
  Search,
} from "lucide-react";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import ExportDialog from "./ExportDialog";
import { MultiTrackPlayer } from "./MultiTrackPlayer";
import { PageShell } from "./PageShell";
import { useStore, type HistoryItem } from "../stores/useStore";
import { previewLoadMessage } from "../lib/previewErrors";
import type { Model, ResultAvailabilityStatus } from "../types/store";
import type { Recipe } from "../types/recipes";
import {
  getResultAvailabilityClasses,
  getResultAvailabilityLabel,
  getResultAvailabilityStatus,
} from "../lib/results/availability";

interface ResultsPageProps {
  onBack?: () => void;
}

type SortKey = "newest" | "oldest" | "stems" | "model";
type DateFilterKey = "all" | "today" | "last7" | "last30" | "thisYear";
type StatusFilterKey = "all" | ResultAvailabilityStatus;

type ResultSession = HistoryItem & {
  timestamp: number;
  architectureLabels: string[];
  primaryArchitecture: string;
  recipeType?: Recipe["type"];
};

const SORT_LABELS: Record<SortKey, string> = {
  newest: "Newest First",
  oldest: "Oldest First",
  stems: "Most Stems",
  model: "Model",
};

const DATE_FILTER_LABELS: Record<DateFilterKey, string> = {
  all: "Any Date",
  today: "Today",
  last7: "Last 7 Days",
  last30: "Last 30 Days",
  thisYear: "This Year",
};

const STATUS_FILTER_LABELS: Record<StatusFilterKey, string> = {
  all: "All Statuses",
  preview_ready: "Preview Ready",
  preview_only: "Preview Only",
  exported: "Exported",
  playback_issue: "Playback Issue",
  missing_source: "Missing Source",
  failed: "Failed",
};

const normalizeLabel = (value: string | undefined | null) =>
  String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();

const unique = <T,>(values: T[]) => Array.from(new Set(values));

const getTimestamp = (value: string) => {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
};

const formatDuration = (duration?: number) => {
  if (!duration || duration <= 0) return null;
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

const getSourceArtworkUrl = (item: HistoryItem) =>
  item.sourceMeta?.artworkUrl || item.sourceMeta?.thumbnailUrl || null;

const getSourceProviderLabel = (item: HistoryItem) => {
  const provider = item.sourceMeta?.provider || item.sourceType;
  if (!provider || provider === "local_file") return null;
  return String(provider)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

const getIngestModeLabel = (item: HistoryItem) => {
  const ingestMode = item.sourceMeta?.ingestMode;
  if (!ingestMode || ingestMode === "local_file") return null;
  if (ingestMode === "desktop_capture") return "Desktop Capture";
  if (ingestMode === "remote_download") return "Remote Download";
  return String(ingestMode).replace(/_/g, " ");
};

const getCaptureQualityLabel = (item: HistoryItem) => {
  if (item.sourceMeta?.verifiedLossless) return "Verified Lossless";
  if (item.sourceMeta?.qualityMode === "best_available") {
    return "Best Available Capture";
  }
  return null;
};

const buildSourceMetaLine = (item: HistoryItem) =>
  [
    getSourceProviderLabel(item),
    getIngestModeLabel(item),
    getCaptureQualityLabel(item),
    item.sourceMeta?.artist,
    item.sourceMeta?.album,
    item.sourceMeta?.channel,
    item.sourceMeta?.qualityLabel,
    typeof item.sourceMeta?.durationSec === "number"
      ? formatDuration(item.sourceMeta.durationSec)
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

const getRecipeModelIds = (recipe?: Recipe | null) => {
  if (!recipe) return [];
  return unique(
    recipe.steps.flatMap((step) => {
      const candidates = [
        "model_id" in step && typeof step.model_id === "string"
          ? step.model_id
          : undefined,
        "source_model" in step && typeof step.source_model === "string"
          ? step.source_model
          : undefined,
      ];
      return candidates.filter((value): value is string => !!value?.trim());
    }),
  );
};

const getArchitectureLabels = (
  item: HistoryItem,
  modelsById: Map<string, Model>,
  recipesById: Map<string, Recipe>,
) => {
  const directArchitecture = modelsById.get(item.modelId)?.architecture;
  if (directArchitecture) {
    return [normalizeLabel(directArchitecture)];
  }

  const recipe = recipesById.get(item.modelId);
  const recipeArchitectures = getRecipeModelIds(recipe)
    .map((modelId) => modelsById.get(modelId)?.architecture)
    .filter((value): value is string => !!value)
    .map(normalizeLabel)
    .filter(Boolean);

  if (recipeArchitectures.length > 0) {
    return unique(recipeArchitectures);
  }

  if (recipe?.type) {
    return [normalizeLabel(recipe.type)];
  }

  return [];
};

const getPrimaryArchitectureLabel = (
  labels: string[],
  recipeType?: Recipe["type"],
) => {
  if (labels.length === 1) return labels[0];
  if (labels.length > 1) {
    if (recipeType === "ensemble") return "Ensemble";
    if (recipeType === "pipeline" || recipeType === "chained") return "Workflow";
    return "Hybrid";
  }
  if (recipeType) return normalizeLabel(recipeType);
  return "Unknown";
};

const matchesDateFilter = (
  timestamp: number,
  filter: DateFilterKey,
  now: Date,
) => {
  if (!timestamp) return filter === "all";
  if (filter === "all") return true;

  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);

  switch (filter) {
    case "today":
      return timestamp >= startOfToday.getTime();
    case "last7":
      return timestamp >= startOfToday.getTime() - 6 * 24 * 60 * 60 * 1000;
    case "last30":
      return timestamp >= startOfToday.getTime() - 29 * 24 * 60 * 60 * 1000;
    case "thisYear":
      return new Date(timestamp).getFullYear() === now.getFullYear();
    default:
      return true;
  }
};

export function ResultsPage({ onBack }: ResultsPageProps) {
  const history = useStore((state) => state.history);
  const models = useStore((state) => state.models);
  const recipes = useStore((state) => state.recipes);
  const settings = useStore((state) => state.settings);
  const sessionToLoad = useStore((state) => state.sessionToLoad);
  const loadSession = useStore((state) => state.loadSession);
  const clearLoadedSession = useStore((state) => state.clearLoadedSession);
  const removeFromHistory = useStore((state) => state.removeFromHistory);
  const updateHistoryItem = useStore((state) => state.updateHistoryItem);
  const defaultExportDir = useStore((state) => state.settings.defaultExportDir);
  const setDefaultExportDir = useStore((state) => state.setDefaultExportDir);

  const [showExportDialog, setShowExportDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("newest");
  const [sortOpen, setSortOpen] = useState(false);
  const [dateFilter, setDateFilter] = useState<DateFilterKey>("all");
  const [dateOpen, setDateOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilterKey>("all");
  const [filterModel, setFilterModel] = useState("All");
  const [filterArch, setFilterArch] = useState("All");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null,
  );
  const [resolvedPlaybackFiles, setResolvedPlaybackFiles] = useState<
    Record<string, string>
  >({});
  const [playbackIssues, setPlaybackIssues] = useState<
    Record<string, { code?: string; hint?: string; originalPath?: string }>
  >({});
  const [verifiedPlaybackFiles, setVerifiedPlaybackFiles] = useState<
    Record<string, string>
  >({});
  const [isResolvingPlayback, setIsResolvingPlayback] = useState(false);

  const modelsById = useMemo(
    () => new Map(models.map((model) => [model.id, model])),
    [models],
  );

  const recipesById = useMemo(
    () => new Map(recipes.map((recipe) => [recipe.id, recipe])),
    [recipes],
  );

  const resultSessions = useMemo(() => {
    return history
      .map((item) => {
        const architectureLabels = getArchitectureLabels(
          item,
          modelsById,
          recipesById,
        );
        const recipeType = recipesById.get(item.modelId)?.type;
        return {
          ...item,
          timestamp: getTimestamp(item.date),
          architectureLabels,
          primaryArchitecture: getPrimaryArchitectureLabel(
            architectureLabels,
            recipeType,
          ),
          recipeType,
        } satisfies ResultSession;
      });
  }, [history, modelsById, recipesById]);

  const sessionMetaById = useMemo(
    () => new Map(resultSessions.map((item) => [item.id, item])),
    [resultSessions],
  );

  const allModels = useMemo(() => {
    const values = unique(
      resultSessions
        .map((item) => item.modelName)
        .filter((value) => !!value?.trim()),
    ).sort((a, b) => a.localeCompare(b));
    return ["All", ...values];
  }, [resultSessions]);

  const allArchitectures = useMemo(() => {
    const values = unique(
      resultSessions.flatMap((item) => item.architectureLabels).filter(Boolean),
    ).sort((a, b) => a.localeCompare(b));
    return ["All", ...values];
  }, [resultSessions]);

  const allStatuses = useMemo(
    () =>
      [
        "all",
        "preview_ready",
        "preview_only",
        "exported",
        "playback_issue",
        "missing_source",
        "failed",
      ] satisfies StatusFilterKey[],
    [],
  );

  const filteredSessions = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const now = new Date();
    const filtered = resultSessions.filter((item) => {
      const fileName =
        item.displayName || item.inputFile.split(/[\\/]/).pop() || item.inputFile;
      const availabilityStatus = getResultAvailabilityStatus(item);
      return (
        (!query ||
          fileName.toLowerCase().includes(query) ||
          item.modelName.toLowerCase().includes(query) ||
          item.modelId.toLowerCase().includes(query) ||
          item.primaryArchitecture.toLowerCase().includes(query) ||
          item.architectureLabels.some((arch) =>
            arch.toLowerCase().includes(query),
          )) &&
        (statusFilter === "all" || availabilityStatus === statusFilter) &&
        (filterModel === "All" || item.modelName === filterModel) &&
        (filterArch === "All" || item.architectureLabels.includes(filterArch)) &&
        matchesDateFilter(item.timestamp, dateFilter, now)
      );
    });

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case "oldest":
          return a.timestamp - b.timestamp;
        case "stems":
          return (
            Object.keys(b.outputFiles || {}).length -
            Object.keys(a.outputFiles || {}).length
          );
        case "model":
          return a.modelName.localeCompare(b.modelName);
        case "newest":
        default:
          return b.timestamp - a.timestamp;
      }
    });
  }, [dateFilter, filterArch, filterModel, resultSessions, searchQuery, sortBy, statusFilter]);

  useEffect(() => {
    if (sessionToLoad?.id) {
      setSelectedSessionId(sessionToLoad.id);
      return;
    }
    setSelectedSessionId(null);
  }, [sessionToLoad]);

  const activeSession = useMemo(() => {
    if (!selectedSessionId) return null;
    return (
      filteredSessions.find((item) => item.id === selectedSessionId) ||
      resultSessions.find((item) => item.id === selectedSessionId) ||
      history.find((item) => item.id === selectedSessionId) ||
      null
    );
  }, [filteredSessions, history, resultSessions, selectedSessionId]);

  useEffect(() => {
    let cancelled = false;

    const resolvePlayback = async () => {
      if (!activeSession?.outputFiles) {
        setResolvedPlaybackFiles({});
        setVerifiedPlaybackFiles({});
        setPlaybackIssues({});
        setIsResolvingPlayback(false);
        return;
      }

      if (!window.electronAPI?.resolvePlaybackStems) {
        setResolvedPlaybackFiles(activeSession.outputFiles);
        setVerifiedPlaybackFiles(activeSession.outputFiles);
        setPlaybackIssues({});
        setIsResolvingPlayback(false);
        return;
      }

      setIsResolvingPlayback(true);
      try {
        const result = await window.electronAPI.resolvePlaybackStems(
          activeSession.outputFiles,
          activeSession.playback,
        );
        if (cancelled) return;
        const resolvedStems = result.stems || {};
        const nextIssues = { ...(result.issues || {}) };
        setResolvedPlaybackFiles(resolvedStems);

        if (window.electronAPI?.checkFileExists) {
          const verifiedEntries = await Promise.all(
            Object.entries(resolvedStems).map(async ([stem, filePath]) => {
              const exists = await window.electronAPI?.checkFileExists?.(filePath);
              return [stem, filePath, !!exists] as const;
            }),
          );

          if (cancelled) return;

          const verified: Record<string, string> = {};
          for (const [stem, filePath, exists] of verifiedEntries) {
            if (exists) {
              verified[stem] = filePath;
            } else if (!nextIssues[stem]) {
              nextIssues[stem] = {
                code: "MISSING_SOURCE_FILE",
                hint: `Playback file is missing: ${filePath}`,
                originalPath: filePath,
              };
            }
          }
          setVerifiedPlaybackFiles(verified);
        } else {
          setVerifiedPlaybackFiles(resolvedStems);
        }

        setPlaybackIssues(nextIssues);
      } catch {
        if (cancelled) return;
        setResolvedPlaybackFiles(activeSession.outputFiles);
        setVerifiedPlaybackFiles(activeSession.outputFiles);
        setPlaybackIssues({});
      } finally {
        if (!cancelled) {
          setIsResolvingPlayback(false);
        }
      }
    };

    void resolvePlayback();

    return () => {
      cancelled = true;
    };
  }, [activeSession]);

  const playableOutputFiles = useMemo(() => {
    if (!activeSession?.outputFiles) return {};
    if (Object.keys(verifiedPlaybackFiles).length > 0) return verifiedPlaybackFiles;
    if (Object.keys(resolvedPlaybackFiles).length > 0) return resolvedPlaybackFiles;
    if (Object.keys(playbackIssues).length > 0) return {};
    return isResolvingPlayback ? {} : activeSession.outputFiles;
  }, [
    activeSession,
    isResolvingPlayback,
    playbackIssues,
    resolvedPlaybackFiles,
    verifiedPlaybackFiles,
  ]);

  const playbackIssueMessages = useMemo(
    () =>
      Object.entries(playbackIssues).map(([stem, issue]) => ({
        stem,
        message: issue.hint || previewLoadMessage(issue.code),
      })),
    [playbackIssues],
  );

  const activeAvailabilityStatus = useMemo(
    () =>
      activeSession
        ? getResultAvailabilityStatus(activeSession, {
            hasMissingSource:
              activeSession.playback?.sourceKind === "missing_source" ||
              Object.values(playbackIssues).some(
                (issue) => issue.code === "MISSING_SOURCE_FILE",
              ),
            hasPlaybackIssue: Object.keys(playbackIssues).length > 0,
          })
        : null,
    [activeSession, playbackIssues],
  );

  const openSession = (item: HistoryItem) => {
    if (item.status !== "completed" || !item.outputFiles) {
      return;
    }
    setSelectedSessionId(item.id);
    loadSession(item);
  };

  const handleBackToList = () => {
    clearLoadedSession();
    setSelectedSessionId(null);
  };

  if (activeSession?.outputFiles) {
    const fileName =
      activeSession.displayName ||
      activeSession.inputFile.split(/[\\/]/).pop() ||
      activeSession.inputFile;
    const activeSessionMeta = sessionMetaById.get(activeSession.id);

    return (
      <PageShell>
        <div className="mx-auto min-h-full w-full max-w-[960px] px-6 pb-32 pt-20">
          <button
            type="button"
            onClick={handleBackToList}
            className="stemsep-config-secondary mb-4 inline-flex items-center gap-2 rounded-[999px] border border-white/65 bg-white/66 px-4 py-2.5 text-[13px] text-slate-700 transition-colors hover:bg-white/84 hover:text-slate-900"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to separations
          </button>

          <div className="mb-8">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="stemsep-config-chip">Results</span>
              <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
                {Object.keys(playableOutputFiles).length || Object.keys(activeSession.outputFiles).length} stems
              </span>
              {activeAvailabilityStatus && (
                <span
                  className={`rounded-full px-3 py-1 text-[11px] tracking-[-0.15px] ${getResultAvailabilityClasses(
                    activeAvailabilityStatus,
                  )}`}
                >
                  {getResultAvailabilityLabel(activeAvailabilityStatus)}
                </span>
              )}
            </div>
            <h1 className="truncate text-[28px] font-normal tracking-[-1px] text-white">
              {fileName}
            </h1>
            <p className="mt-1 text-[13px] text-white/58">
              {activeSession.modelName} ·{" "}
              {activeSessionMeta?.primaryArchitecture ?? "Unknown"}
              {formatDuration(activeSession.duration)
                ? ` · ${formatDuration(activeSession.duration)}`
                : ""}{" "}
              · {Object.keys(playableOutputFiles).length || Object.keys(activeSession.outputFiles).length} stems
            </p>
            {buildSourceMetaLine(activeSession) && (
              <p className="mt-2 text-[12px] text-white/48">
                {buildSourceMetaLine(activeSession)}
              </p>
            )}
          </div>

          <div className="space-y-6">
            <MultiTrackPlayer
              stems={playableOutputFiles}
              jobId={activeSession.backendJobId || activeSession.id}
              outputDir={activeSession.outputDir}
              isResolvingPlayback={isResolvingPlayback}
              onDiscard={() => {
                removeFromHistory(activeSession.id);
                handleBackToList();
              }}
            />

            {playbackIssueMessages.length > 0 && (
              <div className="rounded-[1.5rem] border border-amber-200/75 bg-[rgba(255,248,235,0.78)] p-4 text-[13px] text-amber-900 shadow-[0_20px_48px_rgba(0,0,0,0.08)] backdrop-blur-xl">
                <div className="mb-1 text-[12px] font-medium uppercase tracking-[0.18em] text-amber-700/80">
                  Playback Notes
                </div>
                <div className="space-y-1.5">
                  {playbackIssueMessages.map((issue) => (
                    <p key={issue.stem}>
                      <span className="font-medium capitalize">{issue.stem}:</span>{" "}
                      {issue.message}
                    </p>
                  ))}
                </div>
                {Object.keys(playableOutputFiles).length > 0 && (
                  <p className="pt-1 text-[12px] text-amber-800/80">
                    Verified playable stems: {Object.keys(playableOutputFiles).length}/
                    {Object.keys(activeSession.outputFiles || {}).length}
                  </p>
                )}
              </div>
            )}

            <div className="flex justify-center">
              <Button
                onClick={() => setShowExportDialog(true)}
                disabled={isResolvingPlayback || Object.keys(playableOutputFiles).length === 0}
                className="stemsep-config-action relative overflow-hidden rounded-[38px] border border-white/72 bg-white/82 px-6 py-3 text-[18px] font-normal tracking-[-0.45px] text-[#23324c] transition-all duration-300 hover:scale-[1.02] hover:bg-white"
              >
                <Download className="mr-2 h-4 w-4" />
                Export All Stems
              </Button>
            </div>
          </div>

          <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-white/40 bg-[rgba(250,248,252,0.72)] backdrop-blur-2xl">
            <div className="mx-auto flex max-w-[960px] items-center justify-between px-6 py-4">
              <div className="text-xs text-slate-600">
                {fileName} · {activeSession.modelName}
              </div>
              <div className="flex items-center gap-1">
                {Object.keys(playableOutputFiles).map((stem) => (
                  <span
                    key={stem}
                    title={stem}
                    className="h-2 w-2 rounded-full bg-slate-400/70"
                  />
                ))}
              </div>
            </div>
          </div>

          <ExportDialog
            isOpen={showExportDialog}
            onClose={() => setShowExportDialog(false)}
            outputFiles={playableOutputFiles}
            defaultExportDir={defaultExportDir}
            onDefaultDirChange={setDefaultExportDir}
            defaultExportFormat={settings?.advancedSettings?.outputFormat}
            defaultExportBitrate={settings?.advancedSettings?.bitrate}
            onExportComplete={({ exportedFiles, exportDir, format }) => {
              updateHistoryItem(activeSession.id, {
                exportSummary: {
                  status: "exported",
                  exportedAt: new Date().toISOString(),
                  exportDir,
                  format,
                  exportedFiles,
                },
              });
            }}
          />
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <div className="mx-auto min-h-full w-full max-w-[900px] px-6 pb-12 pt-20">
        <div className="mb-8">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="stemsep-config-chip">Results</span>
            <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
              {filteredSessions.length} sessions
            </span>
          </div>
          <h1 className="text-[32px] font-normal tracking-[-1.2px] text-white/96">
            Results
          </h1>
          <p className="mt-2 text-[14px] text-white/58">
            Browse previous separations, then open one to preview and export.
          </p>
        </div>

        <div className="mb-6 flex flex-col gap-4">
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
              <input
                type="text"
                placeholder="Search files, models or architectures..."
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                className="w-full rounded-[1.45rem] border border-white/22 bg-white/14 py-3 pl-11 pr-4 text-[14px] text-white placeholder:text-white/38 outline-none backdrop-blur-xl transition-colors focus:border-white/36 focus:bg-white/20"
              />
            </div>

            <div className="relative">
              <button
                type="button"
                onClick={() => {
                  setSortOpen((value) => !value);
                  setDateOpen(false);
                }}
                className="inline-flex items-center gap-1.5 rounded-[1.45rem] border border-white/22 bg-white/14 px-4 py-3 text-[14px] text-white/72 backdrop-blur-xl transition-colors hover:bg-white/20"
              >
                {SORT_LABELS[sortBy]}
                <ChevronDown className="h-3.5 w-3.5" />
              </button>
              {sortOpen && (
                <div className="absolute right-0 top-full z-50 mt-2 overflow-hidden rounded-[1.15rem] border border-white/25 bg-white/18 shadow-[0_24px_60px_rgba(0,0,0,0.18)] backdrop-blur-2xl">
                  {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => {
                        setSortBy(key);
                        setSortOpen(false);
                      }}
                      className={
                        sortBy === key
                          ? "block w-full whitespace-nowrap bg-white/16 px-4 py-2 text-left text-[13px] text-white"
                          : "block w-full whitespace-nowrap px-4 py-2 text-left text-[13px] text-white/68 transition-colors hover:bg-white/10 hover:text-white"
                      }
                    >
                      {SORT_LABELS[key]}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="relative">
              <button
                type="button"
                onClick={() => {
                  setDateOpen((value) => !value);
                  setSortOpen(false);
                }}
                className="inline-flex items-center gap-1.5 rounded-[1.45rem] border border-white/22 bg-white/14 px-4 py-3 text-[14px] text-white/72 backdrop-blur-xl transition-colors hover:bg-white/20"
              >
                <Calendar className="h-3.5 w-3.5" />
                {DATE_FILTER_LABELS[dateFilter]}
                <ChevronDown className="h-3.5 w-3.5" />
              </button>
              {dateOpen && (
                <div className="absolute right-0 top-full z-50 mt-2 overflow-hidden rounded-[1.15rem] border border-white/25 bg-white/18 shadow-[0_24px_60px_rgba(0,0,0,0.18)] backdrop-blur-2xl">
                  {(Object.keys(DATE_FILTER_LABELS) as DateFilterKey[]).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => {
                        setDateFilter(key);
                        setDateOpen(false);
                      }}
                      className={
                        dateFilter === key
                          ? "block w-full whitespace-nowrap bg-white/16 px-4 py-2 text-left text-[13px] text-white"
                          : "block w-full whitespace-nowrap px-4 py-2 text-left text-[13px] text-white/68 transition-colors hover:bg-white/10 hover:text-white"
                      }
                    >
                      {DATE_FILTER_LABELS[key]}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {allStatuses.map((statusKey) => (
              <button
                key={statusKey}
                type="button"
                onClick={() => setStatusFilter(statusKey)}
                className={`rounded-full border px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                  statusFilter === statusKey
                    ? "border-white/75 bg-white text-gray-900 shadow-[0_10px_24px_rgba(0,0,0,0.12)]"
                    : "border-white/18 bg-white/10 text-white/62 hover:bg-white/15 hover:text-white/84"
                }`}
              >
                {STATUS_FILTER_LABELS[statusKey]}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            {allModels.map((model) => (
              <button
                key={model}
                type="button"
                onClick={() => setFilterModel(model)}
                className={`rounded-full border px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                  filterModel === model
                    ? "border-white/75 bg-white text-gray-900 shadow-[0_10px_24px_rgba(0,0,0,0.12)]"
                    : "border-white/18 bg-white/10 text-white/62 hover:bg-white/15 hover:text-white/84"
                }`}
              >
                {model}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Layers className="h-3.5 w-3.5 shrink-0 text-white/30" />
            {allArchitectures.map((arch) => (
              <button
                key={arch}
                type="button"
                onClick={() =>
                  setFilterArch((current) =>
                    arch === "All" ? "All" : current === arch ? "All" : arch,
                  )
                }
                className={`rounded-full border px-3 py-1 text-[12px] tracking-[-0.2px] transition-all ${
                  filterArch === arch
                    ? "border-white/34 bg-white/20 text-white"
                    : "border-white/12 bg-white/6 text-white/48 hover:border-white/20 hover:bg-white/12 hover:text-white/74"
                }`}
              >
                {arch === "All" ? "All Architectures" : arch}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          {filteredSessions.map((item) => {
            const fileName =
              item.displayName || item.inputFile.split(/[\\/]/).pop() || item.inputFile;
            const duration = formatDuration(item.duration);
            const availabilityStatus = getResultAvailabilityStatus(item);
            const isPlayable = item.status === "completed" && !!item.outputFiles;
            const sourceMetaLine = buildSourceMetaLine(item);
            const artworkUrl = getSourceArtworkUrl(item);

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => openSession(item)}
                disabled={!isPlayable}
                className={`flex w-full items-center gap-4 rounded-[1.45rem] border border-white/85 bg-[rgba(255,255,255,0.92)] p-5 text-left text-[#111111] shadow-[0_24px_70px_rgba(0,0,0,0.12)] backdrop-blur-xl transition-all duration-300 ${
                  isPlayable
                    ? "hover:-translate-y-[1px] hover:bg-white"
                    : "cursor-default opacity-88"
                }`}
              >
                {artworkUrl ? (
                  <div className="h-11 w-11 shrink-0 overflow-hidden rounded-[1rem] border border-white/70 bg-white/80 shadow-[0_10px_24px_rgba(141,150,179,0.1)]">
                    <img
                      src={artworkUrl}
                      alt=""
                      className="h-full w-full object-cover"
                      loading="lazy"
                    />
                  </div>
                ) : (
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[1rem] border border-white/70 bg-white/80 text-[#60656f] shadow-[0_10px_24px_rgba(141,150,179,0.1)]">
                    <Headphones className="h-5 w-5" />
                  </div>
                )}

                <div className="min-w-0 flex-1">
                  <div className="mb-1 flex items-center gap-2.5">
                    <h3 className="truncate text-[14px] tracking-[-0.025em] text-[#111111]">
                      {fileName}
                    </h3>
                    <span className="rounded-full border border-white/70 bg-white/76 px-2.5 py-0.5 text-[11px] text-[#5d6169] shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                      {item.primaryArchitecture}
                    </span>
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[11px] shadow-[inset_0_1px_0_rgba(255,255,255,0.4)] ${getResultAvailabilityClasses(
                        availabilityStatus,
                      )}`}
                    >
                      {getResultAvailabilityLabel(availabilityStatus)}
                    </span>
                  </div>
                  <p className="mb-2 text-[12px] text-[#7a7f87]">
                    {item.modelName}
                    {duration ? ` · ${duration}` : ""} ·{" "}
                    {new Date(item.date).toLocaleString(undefined, {
                      dateStyle: "medium",
                      timeStyle: "short",
                    })}
                  </p>
                  {sourceMetaLine && (
                    <p className="mb-2 text-[12px] text-[#8a8f98]">
                      {sourceMetaLine}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-1.5">
                    {Object.keys(item.outputFiles || {}).map((stem) => (
                      <Badge
                        key={stem}
                        variant="secondary"
                        className="rounded-full border border-white/70 bg-white/76 px-2.5 py-0.5 text-[11px] text-[#5d6169]"
                      >
                        {stem}
                      </Badge>
                    ))}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {filteredSessions.length === 0 && (
          <div className="rounded-[1.6rem] border border-white/20 bg-white/10 py-16 text-center backdrop-blur-xl">
            <Clock className="mx-auto mb-3 h-10 w-10 text-white/24" />
            <p className="text-[14px] text-white/48">
              {history.length === 0 ? "No results yet" : "No results found"}
            </p>
            <p className="mt-1 text-[13px] text-white/30">
              {history.length === 0
                ? "Run a separation or import a YouTube track to build your history."
                : "Try a different search, status, model, architecture or date."}
            </p>
            <div className="mt-6">
              <Button
                variant="outline"
                onClick={onBack}
                className="rounded-[999px] border-white/18 bg-white/12 text-white hover:bg-white/16"
              >
                Return Home
              </Button>
            </div>
          </div>
        )}
      </div>
    </PageShell>
  );
}
