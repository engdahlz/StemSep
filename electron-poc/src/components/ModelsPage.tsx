import { useEffect, useMemo, useState } from "react";
import { ChevronDown, Layers, Search } from "lucide-react";

import { logger } from "../utils/logger";
import { Model, useStore } from "../stores/useStore";
import { ModelCard } from "./ModelCard";
import { ModelDetails } from "./ModelDetails";
import { PageShell } from "./PageShell";
import { normalizeModel } from "../lib/models/normalizeModel";

type SortKey = "Popular" | "Rating" | "Newest";

const categories = ["All", "Vocals", "Drums", "Guitar", "Keys", "Full Mix"];

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

  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeArch, setActiveArch] = useState("All");
  const [sortBy, setSortBy] = useState<SortKey>("Popular");
  const [sortOpen, setSortOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailsModel, setDetailsModel] = useState<Model | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      if (!window.electronAPI?.getModels) {
        setLoading(false);
        return;
      }
      try {
        const backendModels = await window.electronAPI.getModels();
        const converted = backendModels.map(normalizeModel);
        setModels(converted);
        logger.info(
          `Loaded ${converted.length} models`,
          { count: converted.length },
          "ModelsPage",
        );
      } catch (error) {
        logger.error("Failed to load models", error, "ModelsPage");
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, [setModels]);

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

  const filteredModels = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = models.filter((model) => {
      if (normalizeLabel(model.architecture) === "Ensemble") return false;

      const matchesSearch =
        !query ||
        [model.name, model.description, model.architecture].some((value) =>
          String(value || "").toLowerCase().includes(query),
        );

      const matchesArch =
        activeArch === "All" ||
        normalizeLabel(model.architecture) === activeArch;

      return (
        matchesSearch &&
        matchesArch &&
        matchesCategory(model, activeCategory)
      );
    });

    return filtered.sort((a, b) => {
      if (sortBy === "Rating") return (b.sdr || 0) - (a.sdr || 0);
      if (sortBy === "Newest")
        return Number(b.recommended) - Number(a.recommended);
      return Number(b.installed) - Number(a.installed) || (b.sdr || 0) - (a.sdr || 0);
    });
  }, [activeArch, activeCategory, models, searchQuery, sortBy]);

  const handleDownload = async (modelId: string) => {
    try {
      const model = models.find((entry) => entry.id === modelId);
      if (model?.download?.mode === "manual") {
        setDetailsModel(model);
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

  return (
    <PageShell>
      <div className="relative z-10 mx-auto min-h-full w-full max-w-[900px] px-6 pb-12 pt-20">
        <div className="mb-8">
          <h1 className="text-[32px] font-normal tracking-[-1.2px] text-white">
            Model Library
          </h1>
          <p className="mt-2 text-[14px] tracking-[-0.2px] text-white/50">
            Choose a separation model that fits your needs
          </p>
        </div>

        <div className="mb-6 flex flex-col gap-4">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
            <input
              type="text"
              placeholder="Search models..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="w-full rounded-2xl border border-white/15 bg-white/10 py-3 pl-11 pr-4 text-[14px] tracking-[-0.2px] text-white outline-none transition-all placeholder:text-white/30 focus:border-white/30 focus:bg-white/15"
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => setActiveCategory(category)}
                  className={`rounded-full px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all ${
                    activeCategory === category
                      ? "bg-white text-gray-900"
                      : "bg-white/10 text-white/60 hover:bg-white/15 hover:text-white/80"
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            <div className="relative">
              <button
                type="button"
                onClick={() => setSortOpen((value) => !value)}
                className="flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1.5 text-[13px] tracking-[-0.2px] text-white/60 transition-all hover:bg-white/15"
              >
                {sortBy}
                <ChevronDown className="h-3.5 w-3.5" />
              </button>
              {sortOpen && (
                <div className="absolute right-0 top-full z-50 mt-2 overflow-hidden rounded-xl border border-white/20 bg-white/15 backdrop-blur-xl">
                  {(["Popular", "Rating", "Newest"] as SortKey[]).map((option) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => {
                        setSortBy(option);
                        setSortOpen(false);
                      }}
                      className={`block w-full whitespace-nowrap px-4 py-2 text-left text-[13px] tracking-[-0.2px] transition-colors ${
                        sortBy === option
                          ? "bg-white/10 text-white"
                          : "text-white/60 hover:bg-white/5 hover:text-white"
                      }`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Layers className="h-3.5 w-3.5 shrink-0 text-white/30" />
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
                    ? "border-white/30 bg-white/20 text-white"
                    : "border-white/10 bg-white/5 text-white/45 hover:border-white/18 hover:bg-white/10 hover:text-white/70"
                }`}
              >
                {architecture === "All" ? "All Architectures" : architecture}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="flex flex-col gap-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-[130px] rounded-2xl border border-white/10 bg-white/10"
              />
            ))}
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="py-16 text-center">
            <p className="text-[14px] tracking-[-0.2px] text-white/40">
              No models found
            </p>
            <p className="mt-1 text-[13px] tracking-[-0.2px] text-white/25">
              Try a different search, category or architecture
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {filteredModels.map((model) => (
              <div key={model.id} id={`model-${model.id}`}>
                <ModelCard
                  model={model}
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
    </PageShell>
  );
}
