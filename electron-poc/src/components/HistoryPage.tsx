import { useMemo, useState } from "react";
import {
  AlertCircle,
  AudioLines,
  CheckCircle2,
  ChevronDown,
  Clock,
  Download,
  Drum,
  Guitar,
  Loader2,
  Mic2,
  Play,
  RotateCcw,
  Search,
  Trash2,
} from "lucide-react";

import { PageShell } from "./PageShell";
import { useStore, type HistoryItem } from "../stores/useStore";
import type { Page } from "../types/navigation";

interface HistoryPageProps {
  onNavigate: (page: Page) => void;
}

type FilterStatus = "all" | "completed" | "failed" | "processing";

const formatDuration = (duration?: number) => {
  if (!duration || duration <= 0) return null;
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

const getHistoryIcon = (item: HistoryItem) => {
  const stems = Object.keys(item.outputFiles || {}).map((stem) =>
    stem.toLowerCase(),
  );
  if (stems.some((stem) => stem.includes("vocal"))) return Mic2;
  if (stems.some((stem) => stem.includes("drum") || stem.includes("kick"))) {
    return Drum;
  }
  if (
    stems.some((stem) => stem.includes("guitar") || stem.includes("bass"))
  ) {
    return Guitar;
  }
  return AudioLines;
};

const getStatusLabel = (item: HistoryItem) => {
  if (item.status === "failed") return "failed";
  return "completed";
};

export function HistoryPage({ onNavigate }: HistoryPageProps) {
  const history = useStore((state) => state.history);
  const removeFromHistory = useStore((state) => state.removeFromHistory);
  const clearHistory = useStore((state) => state.clearHistory);
  const loadSession = useStore((state) => state.loadSession);

  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [filterOpen, setFilterOpen] = useState(false);

  const sortedHistory = useMemo(
    () =>
      [...history].sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
      ),
    [history],
  );

  const filteredHistory = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return sortedHistory.filter((item) => {
      const status = getStatusLabel(item);
      const matchesStatus =
        filterStatus === "all" ||
        (filterStatus === "processing" ? false : status === filterStatus);
      const matchesSearch =
        !query ||
        item.inputFile.toLowerCase().includes(query) ||
        item.modelName.toLowerCase().includes(query) ||
        item.modelId.toLowerCase().includes(query);
      return matchesStatus && matchesSearch;
    });
  }, [filterStatus, searchQuery, sortedHistory]);

  const stats = useMemo(() => {
    const completed = sortedHistory.filter((item) => item.status === "completed")
      .length;
    const failed = sortedHistory.filter((item) => item.status === "failed")
      .length;
    return { completed, failed, processing: 0 };
  }, [sortedHistory]);

  const handleOpenResults = (item: HistoryItem) => {
    loadSession(item);
    onNavigate("results");
  };

  return (
    <PageShell>
      <div className="mx-auto min-h-full w-full max-w-[900px] px-6 pb-12 pt-20">
        <div className="mb-8">
          <h1 className="text-[32px] font-normal tracking-[-1.2px] text-white/96">
            History
          </h1>
          <p className="mt-2 text-[14px] text-white/50">
            Your past separation jobs and results.
          </p>
        </div>

        <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-3">
          {[
            {
              label: "Completed",
              value: stats.completed,
              icon: CheckCircle2,
              color: "text-emerald-300",
            },
            {
              label: "Failed",
              value: stats.failed,
              icon: AlertCircle,
              color: "text-rose-300",
            },
            {
              label: "Processing",
              value: stats.processing,
              icon: Loader2,
              color: "text-amber-300",
            },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.label}
                className="flex items-center gap-3 rounded-[1.2rem] border border-white/10 bg-white/8 px-4 py-3.5 backdrop-blur-2xl"
              >
                <Icon className={`h-4 w-4 shrink-0 ${item.color}`} />
                <div>
                  <p className={`text-[20px] tracking-[-0.05em] ${item.color}`}>
                    {item.value}
                  </p>
                  <p className="text-[12px] text-white/40">{item.label}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mb-6 flex gap-3">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
            <input
              type="text"
              placeholder="Search files or models..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="w-full rounded-[1.1rem] border border-white/15 bg-white/10 py-3 pl-11 pr-4 text-[14px] text-white placeholder:text-white/30 outline-none backdrop-blur-2xl transition-colors focus:border-white/30 focus:bg-white/14"
            />
          </div>

          <div className="relative">
            <button
              type="button"
              onClick={() => setFilterOpen((value) => !value)}
              className="inline-flex items-center gap-1.5 rounded-2xl border border-white/15 bg-white/10 px-4 py-3 text-[14px] tracking-[-0.2px] text-white/60 backdrop-blur-md transition-all hover:bg-white/15"
            >
              <Clock className="h-4 w-4" />
              {filterStatus === "all"
                ? "All Status"
                : filterStatus.charAt(0).toUpperCase() + filterStatus.slice(1)}
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
            {filterOpen && (
              <div className="absolute right-0 top-full z-50 mt-2 overflow-hidden rounded-xl border border-white/20 bg-white/15 backdrop-blur-xl">
                {(["all", "completed", "failed", "processing"] as FilterStatus[]).map(
                  (value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => {
                        setFilterStatus(value);
                        setFilterOpen(false);
                      }}
                      className={`block w-full whitespace-nowrap px-4 py-2 text-left text-[13px] tracking-[-0.2px] transition-colors ${
                        filterStatus === value
                          ? "bg-white/10 text-white"
                          : "text-white/60 hover:bg-white/5 hover:text-white"
                      }`}
                    >
                      {value === "all"
                        ? "All Status"
                        : value.charAt(0).toUpperCase() + value.slice(1)}
                    </button>
                  ),
                )}
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3">
          {filteredHistory.map((item) => {
            const status = getStatusLabel(item);
            const statusClass =
              status === "completed"
                ? "bg-emerald-100 text-emerald-600"
                : "bg-rose-100 text-rose-500";
            const StatusIcon =
              status === "completed" ? CheckCircle2 : AlertCircle;
            const CardIcon = getHistoryIcon(item);
            const itemDuration = formatDuration(item.duration);

            return (
              <div
                key={item.id}
                className="group rounded-[1.25rem] border border-white/80 bg-white p-5 text-[#111111] shadow-[0_24px_70px_rgba(0,0,0,0.12)] transition-all duration-300 hover:bg-white/92"
              >
                <div className="flex items-start gap-4">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[#f3f4f6] text-[#60656f]">
                    <CardIcon className="h-5 w-5" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="mb-1 flex items-center gap-2.5">
                      <h3 className="truncate text-[14px] tracking-[-0.025em] text-[#111111]">
                        {item.inputFile.split(/[\\/]/).pop()}
                      </h3>
                      <span
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${statusClass}`}
                      >
                        <StatusIcon className="h-3 w-3" />
                        {status === "completed" ? "Completed" : "Failed"}
                      </span>
                    </div>

                    <p className="mb-2.5 text-[12px] text-[#7a7f87]">
                      {item.modelName}
                      {itemDuration ? ` · ${itemDuration}` : ""}
                      {Object.keys(item.outputFiles || {}).length > 0
                        ? ` · ${Object.keys(item.outputFiles || {}).length} stems`
                        : ""}
                    </p>

                    <div className="mb-2.5 flex flex-wrap gap-1.5">
                      {Object.keys(item.outputFiles || {}).map((stem) => (
                        <span
                          key={stem}
                          className="rounded-md bg-[#f3f4f6] px-2 py-0.5 text-[11px] text-[#5d6169]"
                        >
                          {stem}
                        </span>
                      ))}
                    </div>

                    <p className="text-[11px] text-[#9ca3af]">
                      {new Date(item.date).toLocaleString(undefined, {
                        dateStyle: "medium",
                        timeStyle: "short",
                      })}
                    </p>
                  </div>

                  <div className="flex shrink-0 items-center gap-1.5 opacity-0 transition-opacity group-hover:opacity-100">
                    {status === "completed" && (
                      <>
                        <button
                          type="button"
                          onClick={() => handleOpenResults(item)}
                          className="rounded-xl bg-[#f3f4f6] p-2 text-[#5d6169] transition-colors hover:bg-[#e8eaee] hover:text-[#111111]"
                          title="Open in Result Studio"
                        >
                          <Play className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={async () => {
                            await window.electronAPI?.openFolder?.(item.outputDir);
                          }}
                          className="rounded-xl bg-[#f3f4f6] p-2 text-[#5d6169] transition-colors hover:bg-[#e8eaee] hover:text-[#111111]"
                          title="Open output folder"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </button>
                      </>
                    )}
                    {status === "failed" && (
                      <button
                        type="button"
                        onClick={() => {
                          loadSession(item);
                          onNavigate("home");
                        }}
                        className="rounded-xl bg-[#f3f4f6] p-2 text-[#5d6169] transition-colors hover:bg-[#e8eaee] hover:text-[#111111]"
                        title="Retry from Home"
                      >
                        <RotateCcw className="h-3.5 w-3.5" />
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => removeFromHistory(item.id)}
                      className="rounded-xl bg-[#f3f4f6] p-2 text-[#5d6169] transition-colors hover:bg-[#fee2e2] hover:text-[#ef4444]"
                      title="Delete"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {filteredHistory.length === 0 && (
          <div className="py-16 text-center">
            <Clock className="mx-auto mb-3 h-10 w-10 text-white/20" />
            <p className="text-[14px] text-white/40">No results found</p>
            <p className="mt-1 text-[13px] text-white/25">
              Try a different search or filter.
            </p>
          </div>
        )}

        {history.length > 0 && (
          <div className="mt-6 flex justify-end">
            <button
              type="button"
              onClick={() => {
                if (
                  confirm("Clear all separation history? This cannot be undone.")
                ) {
                  clearHistory();
                }
              }}
              className="rounded-[999px] border border-white/16 bg-white/10 px-4 py-2 text-[13px] text-white/72 backdrop-blur-xl transition-colors hover:bg-white/14 hover:text-white"
            >
              Clear History
            </button>
          </div>
        )}
      </div>
    </PageShell>
  );
}
