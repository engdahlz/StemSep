import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
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

interface ResultsPageProps {
  onBack?: () => void;
}

type SortKey = "newest" | "oldest" | "stems" | "model";

const SORT_LABELS: Record<SortKey, string> = {
  newest: "Newest First",
  oldest: "Oldest First",
  stems: "Most Stems",
  model: "Model",
};

const normalizeLabel = (value: string | undefined | null) =>
  String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();

const formatDuration = (duration?: number) => {
  if (!duration || duration <= 0) return null;
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

const getArchitecture = (item: HistoryItem) => normalizeLabel(item.modelId);

export function ResultsPage({ onBack }: ResultsPageProps) {
  const history = useStore((state) => state.history);
  const settings = useStore((state) => state.settings);
  const sessionToLoad = useStore((state) => state.sessionToLoad);
  const loadSession = useStore((state) => state.loadSession);
  const clearLoadedSession = useStore((state) => state.clearLoadedSession);
  const removeFromHistory = useStore((state) => state.removeFromHistory);
  const defaultExportDir = useStore((state) => state.settings.defaultExportDir);
  const setDefaultExportDir = useStore((state) => state.setDefaultExportDir);

  const [showExportDialog, setShowExportDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("newest");
  const [sortOpen, setSortOpen] = useState(false);
  const [filterModel, setFilterModel] = useState("All");
  const [filterArch, setFilterArch] = useState("All");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null,
  );

  const completedItems = useMemo(() => {
    const items = history.filter((item) => item.status === "completed");
    return [...items];
  }, [history]);

  const allModels = useMemo(
    () => ["All", ...Array.from(new Set(completedItems.map((item) => item.modelName)))],
    [completedItems],
  );

  const allArchitectures = useMemo(() => {
    const values = completedItems.map(getArchitecture).filter(Boolean);
    return ["All", ...Array.from(new Set(values))];
  }, [completedItems]);

  const filteredSessions = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = completedItems.filter((item) => {
      const fileName = item.inputFile.split(/[\\/]/).pop() || item.inputFile;
      const arch = getArchitecture(item);
      return (
        (!query ||
          fileName.toLowerCase().includes(query) ||
          item.modelName.toLowerCase().includes(query) ||
          item.modelId.toLowerCase().includes(query)) &&
        (filterModel === "All" || item.modelName === filterModel) &&
        (filterArch === "All" || arch === filterArch)
      );
    });

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case "oldest":
          return new Date(a.date).getTime() - new Date(b.date).getTime();
        case "stems":
          return (
            Object.keys(b.outputFiles || {}).length -
            Object.keys(a.outputFiles || {}).length
          );
        case "model":
          return a.modelName.localeCompare(b.modelName);
        case "newest":
        default:
          return new Date(b.date).getTime() - new Date(a.date).getTime();
      }
    });
  }, [completedItems, filterArch, filterModel, searchQuery, sortBy]);

  useEffect(() => {
    if (sessionToLoad?.id) {
      setSelectedSessionId(sessionToLoad.id);
      return;
    }
    if (!selectedSessionId && filteredSessions.length > 0) {
      setSelectedSessionId(filteredSessions[0].id);
    }
  }, [filteredSessions, selectedSessionId, sessionToLoad]);

  const activeSession = useMemo(() => {
    if (!selectedSessionId) return null;
    return (
      filteredSessions.find((item) => item.id === selectedSessionId) ||
      history.find((item) => item.id === selectedSessionId) ||
      null
    );
  }, [filteredSessions, history, selectedSessionId]);

  const openSession = (item: HistoryItem) => {
    setSelectedSessionId(item.id);
    loadSession(item);
  };

  const handleBackToList = () => {
    clearLoadedSession();
    setSelectedSessionId(null);
  };

  if (activeSession?.outputFiles) {
    const fileName =
      activeSession.inputFile.split(/[\\/]/).pop() || activeSession.inputFile;

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
              <span className="stemsep-config-chip">Result Studio</span>
              <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
                {Object.keys(activeSession.outputFiles).length} stems
              </span>
            </div>
            <h1 className="truncate text-[28px] font-normal tracking-[-1px] text-white">
              {fileName}
            </h1>
            <p className="mt-1 text-[13px] text-white/58">
              {activeSession.modelName} · {getArchitecture(activeSession)}
              {formatDuration(activeSession.duration)
                ? ` · ${formatDuration(activeSession.duration)}`
                : ""}{" "}
              · {Object.keys(activeSession.outputFiles).length} stems
            </p>
          </div>

          <div className="space-y-6">
            <MultiTrackPlayer
              stems={activeSession.outputFiles}
              jobId={activeSession.backendJobId || activeSession.id}
              onDiscard={() => {
                removeFromHistory(activeSession.id);
                handleBackToList();
              }}
            />

            <div className="flex justify-center">
              <Button
                onClick={() => setShowExportDialog(true)}
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
                {Object.keys(activeSession.outputFiles).map((stem) => (
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
            outputFiles={activeSession.outputFiles}
            defaultExportDir={defaultExportDir}
            onDefaultDirChange={setDefaultExportDir}
            defaultExportFormat={settings?.advancedSettings?.outputFormat}
            defaultExportBitrate={settings?.advancedSettings?.bitrate}
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
            <span className="stemsep-config-chip">Result Studio</span>
            <span className="stemsep-config-chip stemsep-config-chip-subtle normal-case tracking-[-0.1px]">
              {filteredSessions.length} sessions
            </span>
          </div>
          <h1 className="text-[32px] font-normal tracking-[-1.2px] text-white/96">
            Result Studio
          </h1>
          <p className="mt-2 text-[14px] text-white/58">
            Play, inspect, and export your separated stems.
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
                onClick={() => setSortOpen((value) => !value)}
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
              item.inputFile.split(/[\\/]/).pop() || item.inputFile;
            const architecture = getArchitecture(item);
            const duration = formatDuration(item.duration);

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => openSession(item)}
                className="group flex w-full items-center gap-4 rounded-[1.45rem] border border-white/85 bg-[rgba(255,255,255,0.92)] p-5 text-left text-[#111111] shadow-[0_24px_70px_rgba(0,0,0,0.12)] backdrop-blur-xl transition-all duration-300 hover:-translate-y-[1px] hover:bg-white"
              >
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[1rem] border border-white/70 bg-white/80 text-[#60656f] shadow-[0_10px_24px_rgba(141,150,179,0.1)]">
                  <Headphones className="h-5 w-5" />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="mb-1 flex items-center gap-2.5">
                    <h3 className="truncate text-[14px] tracking-[-0.025em] text-[#111111]">
                      {fileName}
                    </h3>
                    <span className="rounded-full border border-white/70 bg-white/76 px-2.5 py-0.5 text-[11px] text-[#5d6169] shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                      {architecture}
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

                <div className="translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100">
                  <div className="inline-flex items-center gap-1.5 rounded-[999px] border border-white/70 bg-white/82 px-4 py-2 text-[13px] text-[#4f5e7a] shadow-[0_10px_24px_rgba(141,150,179,0.12)]">
                    <Headphones className="h-3.5 w-3.5" />
                    Open
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {filteredSessions.length === 0 && (
          <div className="rounded-[1.6rem] border border-white/20 bg-white/10 py-16 text-center backdrop-blur-xl">
            <Clock className="mx-auto mb-3 h-10 w-10 text-white/24" />
            <p className="text-[14px] text-white/48">No results found</p>
            <p className="mt-1 text-[13px] text-white/30">
              Try a different search, model or architecture.
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
