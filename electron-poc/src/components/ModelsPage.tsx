import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Layers,
  Search,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

import { Model, useStore } from "../stores/useStore";
import { ModelCard } from "./ModelCard";
import { ModelDetails } from "./ModelDetails";
import { PageShell } from "./PageShell";
import { normalizeModel } from "../lib/models/normalizeModel";
import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";
import { useModels } from "../hooks/useModels";
import {
  getModelCatalogTier,
  isManualCatalogModel,
} from "../lib/models/catalog";
import {
  getModelMachineFit,
  type ModelMachineFitStatus,
} from "../lib/systemRuntime/machineAnalysis";

type SortKey =
  | "Popular"
  | "Rating"
  | "Newest"
  | "Installed"
  | "Not Installed";
type StatusFilter =
  | "All"
  | "Installed"
  | "Ready"
  | "Verified"
  | "Advanced"
  | "Manual";
type MachineFitFilter =
  | "All"
  | "Fits This Machine"
  | "Heavy"
  | "Runtime Blocked"
  | "Manual Setup";

const categories = ["All", "Vocals", "Drums", "Guitar", "Keys", "Full Mix"];
const statusFilters: StatusFilter[] = [
  "All",
  "Installed",
  "Ready",
  "Verified",
  "Advanced",
  "Manual",
];
const machineFitFilters: MachineFitFilter[] = [
  "All",
  "Fits This Machine",
  "Heavy",
  "Runtime Blocked",
  "Manual Setup",
];
const SORT_LABELS: Record<SortKey, string> = {
  Popular: "Best Overall",
  Rating: "Highest Quality",
  Newest: "Recently Added",
  Installed: "Installed First",
  "Not Installed": "Need Install",
};

interface ModelsPageProps {
  preSelectedModel?: string;
  onModelDownloadComplete?: () => void;
  onBack?: () => void;
}

const normalizeLabel = (value: string | undefined | null) =>
  String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();

const matchesCategory = (model: Model, category: string) => {
  if (category === "All") return true;
  const stems = Array.isArray(model.stems)
    ? model.stems.map((stem) => String(stem).toLowerCase())
    : [];

  if (category === "Full Mix") return stems.length >= 4;
  if (category === "Keys") {
    return stems.some(
      (stem) =>
        stem.includes("key") || stem.includes("piano") || stem.includes("synth"),
    );
  }

  return stems.some((stem) =>
    stem.includes(category.toLowerCase().replace(/\s+/g, "")),
  );
};

const matchesStatus = (model: Model, status: StatusFilter) => {
  if (status === "All") return true;
  if (status === "Installed") return !!model.installed;
  if (status === "Manual") return isManualCatalogModel(model);
  const catalogTier = getModelCatalogTier(model);
  if (status === "Verified") return catalogTier === "verified";
  if (status === "Advanced") {
    return (
      catalogTier === "advanced" ||
      catalogTier === "advanced_manual" ||
      catalogTier === "online"
    );
  }
  if (status === "Ready") {
    return (
      !model.installed &&
      !model.downloading &&
      !model.downloadPaused &&
      !isManualCatalogModel(model) &&
      model.download?.mode !== "manual" &&
      model.download?.mode !== "unavailable"
    );
  }
  return true;
};

export function ModelsPage({ preSelectedModel }: ModelsPageProps) {
  const {
    models,
    setModels,
    startDownload,
    setDownloadError,
    pauseDownload,
    resumeDownload,
    setModelInstalled,
  } = useStore();
  const { loading } = useModels();
  const { info: runtimeInfo } = useSystemRuntimeInfo();

  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeArch, setActiveArch] = useState("All");
  const [activeStatus, setActiveStatus] = useState<StatusFilter>("All");
  const [activeMachineFit, setActiveMachineFit] =
    useState<MachineFitFilter>("All");
  const [sortBy, setSortBy] = useState<SortKey>("Popular");
  const [sortOpen, setSortOpen] = useState(false);
  const [detailsModel, setDetailsModel] = useState<Model | null>(null);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!preSelectedModel || models.length === 0) return;
    const timer = window.setTimeout(() => {
      document.getElementById(`model-${preSelectedModel}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }, 50);
    return () => window.clearTimeout(timer);
  }, [models, preSelectedModel]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      setShowBackToTop(container.scrollTop > 420);
    };

    handleScroll();
    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const architectures = useMemo(() => {
    const values = Array.from(
      new Set(
        models
          .map((model) => normalizeLabel(model.architecture))
          .filter((value) => value && value !== "Ensemble"),
      ),
    );
    return ["All", ...values];
  }, [models]);

  const machineFitById = useMemo(
    () =>
      new Map(
        models.map((model) => [model.id, getModelMachineFit(model, runtimeInfo)]),
      ),
    [models, runtimeInfo],
  );

  const matchesMachineFit = useMemo(
    () =>
      (model: Model) => {
        if (activeMachineFit === "All") return true;
        const fit = machineFitById.get(model.id);
        const status = fit?.status;
        const matches: Record<Exclude<MachineFitFilter, "All">, ModelMachineFitStatus> =
          {
            "Fits This Machine": "fits_this_machine",
            Heavy: "heavy_for_this_machine",
            "Runtime Blocked": "runtime_blocked",
            "Manual Setup": "manual_setup",
          };
        return status === matches[activeMachineFit];
      },
    [activeMachineFit, machineFitById],
  );

  const filteredModels = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = models.filter((model) => {
      if (normalizeLabel(model.architecture) === "Ensemble") return false;

      const matchesSearch =
        !query ||
        [
          model.name,
          model.description,
          model.architecture,
          model.category,
          model.catalog?.tier,
          model.catalog?.sourceKind,
          model.catalog?.installPolicy,
          model.source_kind,
          model.install_policy,
          model.card_metrics?.source,
          model.card_metrics?.last_verified,
          ...(model.tags || []),
          ...(model.best_for || []),
          ...(model.workflow_groups || []),
        ].some((value) => String(value || "").toLowerCase().includes(query));

      const matchesArch =
        activeArch === "All" ||
        normalizeLabel(model.architecture) === activeArch;

      return (
        matchesSearch &&
        matchesArch &&
        matchesCategory(model, activeCategory) &&
        matchesStatus(model, activeStatus) &&
        matchesMachineFit(model)
      );
    });

    return filtered.sort((a, b) => {
      if (sortBy === "Rating") return (b.sdr || 0) - (a.sdr || 0);
      if (sortBy === "Newest")
        return Number(b.recommended) - Number(a.recommended);
      if (sortBy === "Installed")
        return (
          Number(b.installed) - Number(a.installed) || (b.sdr || 0) - (a.sdr || 0)
        );
      if (sortBy === "Not Installed")
        return (
          Number(a.installed) - Number(b.installed) || (b.sdr || 0) - (a.sdr || 0)
        );
      return Number(b.installed) - Number(a.installed) || (b.sdr || 0) - (a.sdr || 0);
    });
  }, [activeArch, activeCategory, activeMachineFit, activeStatus, matchesMachineFit, models, searchQuery, sortBy]);

  const installedCount = useMemo(
    () => models.filter((model) => model.installed).length,
    [models],
  );

  const verifiedCount = useMemo(
    () => models.filter((model) => getModelCatalogTier(model) === "verified").length,
    [models],
  );

  const advancedCount = useMemo(
    () => models.filter((model) => getModelCatalogTier(model) === "advanced").length,
    [models],
  );

  const manualCount = useMemo(
    () => models.filter((model) => isManualCatalogModel(model)).length,
    [models],
  );

  const handleDownload = async (modelId: string) => {
    try {
      const model = models.find((entry) => entry.id === modelId);
      if (model && isManualCatalogModel(model)) {
        setDetailsModel(model);
        return;
      }
      if (model && getModelCatalogTier(model) === "blocked") {
        setDownloadError(modelId, "This model is blocked in the catalog and cannot be downloaded yet.");
        return;
      }
      if (model?.download?.mode === "unavailable") {
        setDownloadError(modelId, "No verified download source is configured for this model yet.");
        return;
      }
      startDownload(modelId);
      await window.electronAPI?.downloadModel?.(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handlePause = async (modelId: string) => {
    try {
      await window.electronAPI?.pauseDownload?.(modelId);
      pauseDownload(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handleResume = async (modelId: string) => {
    try {
      resumeDownload(modelId);
      await window.electronAPI?.resumeDownload?.(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handleRemove = async (modelId: string) => {
    if (!confirm("Remove this model?")) return;
    try {
      await window.electronAPI?.removeModel?.(modelId);
      const model = models.find((item) => item.id === modelId);
      if (model?.is_custom || modelId.startsWith("custom_")) {
        const backendModels = await window.electronAPI?.getModels?.();
        if (backendModels) setModels(backendModels.map(normalizeModel));
        return;
      }
      setModelInstalled(modelId, false);
    } catch (error) {
      alert(`Failed to remove model: ${String(error)}`);
    }
  };

  const handleBackToTop = () => {
    scrollContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <PageShell contentRef={scrollContainerRef}>
      <div className="relative z-10 mx-auto min-h-full w-full max-w-[1420px] px-6 pb-12 pt-20">
        <div className="mb-8">
          <h1 className="text-[32px] font-normal tracking-[-1.2px] text-white">
            Model Library
          </h1>
          <p className="mt-2 text-[14px] tracking-[-0.2px] text-white/50">
            Browse installed and downloadable models by role, architecture and setup path.
          </p>
        </div>

        <div className="mb-6 space-y-4 rounded-[2rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.08))] p-5 shadow-[0_28px_80px_rgba(12,16,28,0.12)] backdrop-blur-2xl">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-[1.4rem] border border-white/16 bg-white/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                Visible Models
              </div>
              <div className="mt-2 text-[28px] tracking-[-1px] text-white">
                {filteredModels.length}
              </div>
              <div className="mt-1 text-[12px] text-white/40">
                from {models.length} total entries
              </div>
            </div>
            <div className="rounded-[1.4rem] border border-white/16 bg-white/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                Installed
              </div>
              <div className="mt-2 text-[28px] tracking-[-1px] text-white">
                {installedCount}
              </div>
              <div className="mt-1 text-[12px] text-white/40">
                ready on this machine
              </div>
            </div>
            <div className="rounded-[1.4rem] border border-white/16 bg-white/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                Catalog Tiers
              </div>
              <div className="mt-2 text-[28px] tracking-[-1px] text-white">
                {verifiedCount} / {advancedCount} / {manualCount}
              </div>
              <div className="mt-1 text-[12px] text-white/40">
                verified, advanced and manual setup paths
              </div>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
              <input
              type="text"
              placeholder="Search models, architectures, tags, source kind or workflow fit..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="w-full rounded-[1.6rem] border border-white/18 bg-white/10 py-3.5 pl-11 pr-4 text-[14px] tracking-[-0.2px] text-white outline-none transition-all placeholder:text-white/30 focus:border-white/34 focus:bg-white/16"
            />
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-2">
                <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                  Target
                </div>
                <div className="flex flex-wrap gap-2">
                  {categories.map((category) => (
                    <button
                      key={category}
                      type="button"
                      onClick={() => setActiveCategory(category)}
                      className={`rounded-full border px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                        activeCategory === category
                          ? "border-white/80 bg-white text-gray-900 shadow-[0_10px_26px_rgba(0,0,0,0.12)]"
                          : "border-white/14 bg-white/8 text-white/62 hover:border-white/20 hover:bg-white/14 hover:text-white/84"
                      }`}
                    >
                      {category}
                    </button>
                  ))}
                </div>
              </div>

              <div className="relative shrink-0">
                <button
                  type="button"
                  onClick={() => setSortOpen((value) => !value)}
                  className="inline-flex items-center gap-2 rounded-full border border-white/16 bg-white/10 px-3.5 py-2 text-[13px] tracking-[-0.2px] text-white/72 transition-all hover:border-white/24 hover:bg-white/16 hover:text-white"
                >
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                  {SORT_LABELS[sortBy]}
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
                {sortOpen && (
                  <div className="absolute right-0 top-full z-50 mt-2 overflow-hidden rounded-[1.1rem] border border-white/20 bg-white/16 shadow-[0_24px_60px_rgba(0,0,0,0.18)] backdrop-blur-xl">
                    {(
                      [
                        "Popular",
                        "Rating",
                        "Newest",
                        "Installed",
                        "Not Installed",
                      ] as SortKey[]
                    ).map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => {
                          setSortBy(option);
                          setSortOpen(false);
                        }}
                        className={`block w-full whitespace-nowrap px-4 py-2 text-left text-[13px] tracking-[-0.2px] transition-colors ${
                          sortBy === option
                            ? "bg-white/12 text-white"
                            : "text-white/64 hover:bg-white/8 hover:text-white"
                        }`}
                      >
                        {SORT_LABELS[option]}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                Machine Fit
              </div>
              <div className="flex flex-wrap gap-2">
                {machineFitFilters.map((fitLabel) => (
                  <button
                    key={fitLabel}
                    type="button"
                    onClick={() => setActiveMachineFit(fitLabel)}
                    className={`rounded-full border px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                      activeMachineFit === fitLabel
                        ? "border-white/72 bg-white text-slate-900 shadow-[0_10px_24px_rgba(0,0,0,0.12)]"
                        : "border-white/18 bg-white/10 text-white/62 hover:bg-white/15 hover:text-white/84"
                    }`}
                  >
                    {fitLabel}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/42">
                <Sparkles className="h-3.5 w-3.5" />
                Availability
              </div>
              <div className="flex flex-wrap gap-2">
                {statusFilters.map((status) => (
                  <button
                    key={status}
                    type="button"
                    onClick={() => setActiveStatus(status)}
                    className={`rounded-full border px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                      activeStatus === status
                        ? "border-white/30 bg-white/22 text-white"
                        : "border-white/12 bg-white/6 text-white/52 hover:border-white/18 hover:bg-white/10 hover:text-white/78"
                    }`}
                  >
                    {status}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/42">
                <Layers className="h-3.5 w-3.5 shrink-0" />
                Architecture
              </div>
              <div className="flex flex-wrap gap-2">
                {architectures.map((architecture) => (
                  <button
                    key={architecture}
                    type="button"
                    onClick={() =>
                      setActiveArch((current) =>
                        architecture === "All"
                          ? "All"
                          : current === architecture
                            ? "All"
                            : architecture,
                      )
                    }
                    className={`rounded-full border px-3 py-1 text-[12px] tracking-[-0.2px] transition-all ${
                      activeArch === architecture
                        ? "border-white/34 bg-white/20 text-white"
                        : "border-white/10 bg-white/5 text-white/45 hover:border-white/18 hover:bg-white/10 hover:text-white/70"
                    }`}
                  >
                    {architecture === "All" ? "All Architectures" : architecture}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-[250px] rounded-[2rem] border border-white/10 bg-white/10"
              />
            ))}
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="rounded-[2rem] border border-white/12 bg-white/8 py-16 text-center backdrop-blur-xl">
            <p className="text-[14px] tracking-[-0.2px] text-white/46">
              No models found
            </p>
            <p className="mt-1 text-[13px] tracking-[-0.2px] text-white/25">
              Try a different search, target, availability, machine fit or architecture
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            {filteredModels.map((model) => (
              <div key={model.id} id={`model-${model.id}`}>
                <ModelCard
                  model={model}
                  machineFit={machineFitById.get(model.id)}
                  onDownload={handleDownload}
                  onPause={handlePause}
                  onResume={handleResume}
                  onDetails={setDetailsModel}
                  isSelected={false}
                  onToggleSelection={() => {}}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {detailsModel && (
        <ModelDetails
          model={detailsModel}
          onClose={() => setDetailsModel(null)}
          onDownload={handleDownload}
          onRemove={handleRemove}
        />
      )}

      <button
        type="button"
        onClick={handleBackToTop}
        aria-label="Back to top"
        className={`fixed bottom-8 right-8 z-40 flex items-center gap-2 rounded-full border border-white/18 bg-white/12 px-4 py-2 text-[13px] tracking-[-0.2px] text-white/82 shadow-xl shadow-black/20 backdrop-blur-xl transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/18 hover:text-white ${
          showBackToTop
            ? "pointer-events-auto translate-y-0 opacity-100"
            : "pointer-events-none translate-y-4 opacity-0"
        }`}
      >
        <ChevronUp className="h-3.5 w-3.5" />
        <span>Back to top</span>
      </button>
    </PageShell>
  );
}
