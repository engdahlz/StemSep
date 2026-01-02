import { useState, useMemo, useEffect, useRef } from "react";
import {
  Download,
  Upload,
  Search,
  Filter,
  Trash2,
  ArrowLeft,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ModelCard } from "./ModelCard";
import { ModelDetails } from "./ModelDetails";
import { logger } from "../utils/logger";
import { useStore, Model } from "../stores/useStore";

const normalizeTag = (tag: unknown): string | null => {
  if (typeof tag !== "string") return null;
  const t = tag.trim().toLowerCase();
  return t ? t : null;
};

const getModelTags = (model: any): string[] => {
  const raw = model?.tags;
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  for (const t of raw) {
    const norm = normalizeTag(t);
    if (norm) out.push(norm);
  }
  // de-dupe, stable
  return Array.from(new Set(out));
};

interface ModelsPageProps {
  preSelectedModel?: string;
  onModelDownloadComplete?: () => void;
  onBack?: () => void;
}

export function ModelsPage({ preSelectedModel, onBack }: ModelsPageProps) {
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
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [speedFilter, setSpeedFilter] = useState<string>("all");
  const [downloadedFilter, setDownloadedFilter] = useState<string>("all");
  const [archFilter, setArchFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [loading, setLoading] = useState(true);
  const [showBlockedModels, setShowBlockedModels] = useState(false);

  // Tag filtering
  // - "all" shows everything
  // - otherwise the model must have the selected tag
  const [tagFilter, setTagFilter] = useState<string>("all");

  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [batchDownloading, setBatchDownloading] = useState(false);
  const [batchAction, setBatchAction] = useState<"download" | "delete" | null>(
    null,
  );
  const [detailsModel, setDetailsModel] = useState<Model | null>(null);
  const cancelBatchRef = useRef(false);

  // Handle pre-selected model
  useEffect(() => {
    if (preSelectedModel && models.length > 0) {
      // Ensure the preselected model is visible even if user previously set filters.
      setSearchQuery("");
      setCategoryFilter("all");
      setSpeedFilter("all");
      setDownloadedFilter("all");
      setArchFilter("all");
      setSourceFilter("all");
      setShowBlockedModels(true);
      setTagFilter("all");

      setSelectedModels(new Set([preSelectedModel]));

      const t = window.setTimeout(() => {
        const element = document.getElementById(`model-${preSelectedModel}`);
        if (element)
          element.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 50);

      return () => window.clearTimeout(t);
    }
  }, [preSelectedModel, models]);

  // Support navigation from global "missing models" events (batch -> Model Browser).
  // Payload shape is expected from App.tsx/useSeparation.ts: detail.missingModels = [{ modelId, ... }]
  useEffect(() => {
    const handler = (evt: Event) => {
      const e = evt as CustomEvent<any>;
      const missing = Array.isArray(e?.detail?.missingModels)
        ? e.detail.missingModels
        : [];
      const ids = missing
        .map((m: any) => (typeof m?.modelId === "string" ? m.modelId : null))
        .filter((x: any) => typeof x === "string" && x.trim().length > 0);

      if (ids.length === 0) return;

      // Ensure highlighted models are visible even if user previously set filters.
      setSearchQuery("");
      setCategoryFilter("all");
      setSpeedFilter("all");
      setDownloadedFilter("all");
      setArchFilter("all");
      setSourceFilter("all");
      setShowBlockedModels(true);
      setTagFilter("all");

      setSelectedModels(new Set(ids));

      const firstId = ids[0];
      const t = window.setTimeout(() => {
        const element = document.getElementById(`model-${firstId}`);
        if (element)
          element.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 50);

      return () => window.clearTimeout(t);
    };

    window.addEventListener("stemsep:missing-models", handler as EventListener);
    return () => {
      window.removeEventListener(
        "stemsep:missing-models",
        handler as EventListener,
      );
    };
  }, [models]);

  // Load models from backend on mount
  useEffect(() => {
    const loadModels = async () => {
      if (!window.electronAPI?.getModels) {
        console.log("Electron API not available");
        setLoading(false);
        return;
      }

      try {
        const backendModels = await window.electronAPI.getModels();
        const convertedModels = backendModels.map((m: any) => {
          const stems = Array.isArray(m?.stems) ? m.stems : [];
          const name =
            typeof m?.name === "string" && m.name.trim()
              ? m.name
              : (m?.id ?? "Unknown model");
          const description =
            typeof m?.description === "string" ? m.description : "";

          return {
            ...m,
            id: typeof m?.id === "string" ? m.id : String(m?.id ?? ""),
            name,
            description,
            stems,
            architecture:
              typeof m?.architecture === "string" ? m.architecture : "Unknown",
            version: typeof m?.version === "string" ? m.version : "",
            speed: typeof m?.speed === "string" ? m.speed : "unknown",
            vram_required:
              typeof m?.vram_required === "number"
                ? m.vram_required
                : Number(m?.vram_required ?? 0),
            file_size:
              typeof m?.file_size === "number"
                ? m.file_size
                : Number(m?.file_size ?? 0),
            sdr: typeof m?.sdr === "number" ? m.sdr : Number(m?.sdr ?? 0),
            fullness:
              typeof m?.fullness === "number"
                ? m.fullness
                : Number(m?.fullness ?? 0),
            bleedless:
              typeof m?.bleedless === "number"
                ? m.bleedless
                : Number(m?.bleedless ?? 0),
            category: m?.category || "primary",
            installed: !!m?.installed,
            downloading: false,
            downloadPaused: false,
            downloadProgress: 0,
            recommended: !!m?.recommended,
          };
        });

        setModels(convertedModels);
        logger.info(
          `Loaded ${convertedModels.length} models`,
          { count: convertedModels.length },
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

  const refreshModels = async () => {
    if (!window.electronAPI?.getModels) return;
    const backendModels = await window.electronAPI.getModels();
    const convertedModels = backendModels.map((m: any) => {
      const stems = Array.isArray(m?.stems) ? m.stems : [];
      const name =
        typeof m?.name === "string" && m.name.trim()
          ? m.name
          : (m?.id ?? "Unknown model");
      const description =
        typeof m?.description === "string" ? m.description : "";

      return {
        ...m,
        id: typeof m?.id === "string" ? m.id : String(m?.id ?? ""),
        name,
        description,
        stems,
        architecture:
          typeof m?.architecture === "string" ? m.architecture : "Unknown",
        version: typeof m?.version === "string" ? m.version : "",
        speed: typeof m?.speed === "string" ? m.speed : "unknown",
        vram_required:
          typeof m?.vram_required === "number"
            ? m.vram_required
            : Number(m?.vram_required ?? 0),
        file_size:
          typeof m?.file_size === "number"
            ? m.file_size
            : Number(m?.file_size ?? 0),
        sdr: typeof m?.sdr === "number" ? m.sdr : Number(m?.sdr ?? 0),
        fullness:
          typeof m?.fullness === "number"
            ? m.fullness
            : Number(m?.fullness ?? 0),
        bleedless:
          typeof m?.bleedless === "number"
            ? m.bleedless
            : Number(m?.bleedless ?? 0),
        category: m?.category || "primary",
        installed: !!m?.installed,
        downloading: false,
        downloadPaused: false,
        downloadProgress: 0,
        recommended: !!m?.recommended,
      };
    });
    setModels(convertedModels);
  };

  // Filter logic
  const filteredModels = useMemo(() => {
    return models.filter((model) => {
      // Exclude ensembles - they are presets, not actual AI models
      if (model.architecture === "Ensemble") return false;

      // Hide blocked models by default (unless user toggles them on)
      const runtimeAllowed = model.runtime?.allowed;
      const runtimeBlockedReason = model.runtime?.blocking_reason;
      const isBlocked = runtimeAllowed === false || !!runtimeBlockedReason;
      if (isBlocked && !showBlockedModels) return false;

      const matchesSource =
        sourceFilter === "all" ||
        (sourceFilter === "custom" && !!model.is_custom) ||
        (sourceFilter === "builtin" && !model.is_custom);

      const matchesSearch =
        searchQuery === "" ||
        (model.name || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
        (model.description || "")
          .toLowerCase()
          .includes(searchQuery.toLowerCase()) ||
        (model.architecture || "")
          .toLowerCase()
          .includes(searchQuery.toLowerCase());

      const matchesCategory =
        categoryFilter === "all" || model.category === categoryFilter;
      const matchesSpeed = speedFilter === "all" || model.speed === speedFilter;
      const matchesArch =
        archFilter === "all" ||
        (model.architecture || "")
          .toLowerCase()
          .includes(archFilter.toLowerCase().replace("-", "")) ||
        (model.architecture || "")
          .toLowerCase()
          .includes(archFilter.toLowerCase());
      const matchesDownloaded =
        downloadedFilter === "all" ||
        (downloadedFilter === "downloaded" && model.installed) ||
        (downloadedFilter === "not-downloaded" && !model.installed);

      const normalizedTagFilter = normalizeTag(tagFilter) ?? "all";
      const modelTags = getModelTags(model as any);
      const matchesTag =
        normalizedTagFilter === "all" ||
        modelTags.includes(normalizedTagFilter);

      return (
        matchesSearch &&
        matchesCategory &&
        matchesSpeed &&
        matchesDownloaded &&
        matchesArch &&
        matchesSource &&
        matchesTag
      );
    });
  }, [
    models,
    searchQuery,
    categoryFilter,
    speedFilter,
    downloadedFilter,
    archFilter,
    sourceFilter,
    showBlockedModels,
    tagFilter,
  ]);

  // Download handlers
  const handleDownload = async (modelId: string) => {
    startDownload(modelId);
    try {
      if (window.electronAPI?.downloadModel)
        await window.electronAPI.downloadModel(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handlePause = async (modelId: string) => {
    try {
      if (window.electronAPI?.pauseDownload)
        await window.electronAPI.pauseDownload(modelId);
      pauseDownload(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  const handleResume = async (modelId: string) => {
    try {
      resumeDownload(modelId);
      if (window.electronAPI?.resumeDownload)
        await window.electronAPI.resumeDownload(modelId);
    } catch (error) {
      setDownloadError(modelId, String(error));
    }
  };

  // Internal remove function (no confirmation dialog)
  const removeModelWithoutConfirm = async (modelId: string) => {
    try {
      if (window.electronAPI?.removeModel)
        await window.electronAPI.removeModel(modelId);
    } catch (e) {
      console.error(`[ModelsPage] removeModel failed for ${modelId}:`, e);
      alert(`Failed to remove model: ${modelId}\n\n${String(e)}`);
      throw e;
    }

    const m = models.find((model) => model.id === modelId);
    if (m?.is_custom || modelId.startsWith("custom_")) {
      await refreshModels();
      return;
    }

    // Optimistic UI update, then resync from backend so installed state matches disk.
    setModelInstalled(modelId, false);
    await refreshModels();
  };

  const handleRemove = async (modelId: string) => {
    if (!confirm("Remove this model?")) return;
    await removeModelWithoutConfirm(modelId);
  };

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });
  };

  const handleBatchDownload = async () => {
    const toDownload = Array.from(selectedModels).filter((id) => {
      const m = models.find((model) => model.id === id);
      return m && !m.installed && !m.downloading;
    });

    if (!toDownload.length) return alert("No downloadable models selected");
    if (!confirm(`Download ${toDownload.length} models?`)) return;

    cancelBatchRef.current = false;
    setBatchAction("download");
    setBatchDownloading(true);
    for (const id of toDownload) {
      if (cancelBatchRef.current) break;
      await handleDownload(id);
    }
    setBatchDownloading(false);
    setBatchAction(null);
    setSelectedModels(new Set());
  };

  const handleDownloadAll = async () => {
    const toDownload = models.filter((m) => !m.installed && !m.downloading);
    if (toDownload.length === 0)
      return alert("All models are already installed or downloading");

    if (
      !confirm(
        `Download all ${toDownload.length} available models? This may take a while.`,
      )
    )
      return;

    cancelBatchRef.current = false;
    setBatchAction("download");
    setBatchDownloading(true);
    for (const model of toDownload) {
      if (cancelBatchRef.current) break;
      if (model.downloading) continue;
      await handleDownload(model.id);
    }
    setBatchDownloading(false);
    setBatchAction(null);
  };

  const handleCancelBatch = () => {
    cancelBatchRef.current = true;
  };

  const handleDeleteAll = async () => {
    const toDelete = models.filter((m) => m.installed);
    if (toDelete.length === 0) return alert("No installed models to delete");

    if (
      !confirm(
        `Are you sure you want to delete ALL ${toDelete.length} installed models? This cannot be undone.`,
      )
    )
      return;

    setBatchAction("delete");
    setBatchDownloading(true); // Show loading state
    try {
      for (const model of toDelete) {
        try {
          await removeModelWithoutConfirm(model.id);
        } catch (err) {
          console.error(`Failed to remove model ${model.id}:`, err);
          // Continue with other models even if one fails
        }
      }
    } finally {
      setBatchDownloading(false); // Always reset loading state
      setBatchAction(null);
    }
  };

  const handleUploadCustom = async () => {
    if (
      !window.electronAPI?.openModelFileDialog ||
      !window.electronAPI?.importCustomModel
    ) {
      alert("Import not available");
      return;
    }

    try {
      const paths = await window.electronAPI.openModelFileDialog();
      const filePath = paths?.[0];
      if (!filePath) return;

      const defaultName = filePath
        .replace(/^.*[\\/]/, "")
        .replace(/\.(ckpt|pth|pt|onnx|safetensors)$/i, "");
      const modelName = prompt("Model name", defaultName) || "";
      if (!modelName.trim()) return;

      const architecture =
        (prompt("Architecture (optional)", "Custom") || "Custom").trim() ||
        "Custom";

      await window.electronAPI.importCustomModel(
        filePath,
        modelName.trim(),
        architecture,
      );
      await refreshModels();
    } catch (e) {
      console.error(e);
      alert(`Import failed: ${String(e)}`);
    }
  };

  const categories = [
    "all",
    "primary",
    "karaoke",
    "instrumental",
    "vocal",
    "ensemble",
    "post_processing",
    "custom",
  ];
  const speeds = ["all", "fast", "medium", "slow"];
  const architectures = [
    "all",
    "BS-Roformer",
    "Mel-Roformer",
    "MDX-Net",
    "VR",
    "SCNet",
    "HTDemucs",
  ];
  const sources = ["all", "builtin", "custom"];

  return (
    <div className="h-full flex flex-col bg-background/50">
      {/* Header Section */}
      <div className="p-6 pb-0 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-4 mb-2">
              {onBack && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onBack}
                  className="rounded-full hover:bg-primary/10"
                >
                  <ArrowLeft className="h-6 w-6" />
                </Button>
              )}
              <h1 className="text-4xl font-bold tracking-tight text-foreground">
                Model Library
              </h1>
            </div>
            <p className="text-muted-foreground mt-1 ml-14">
              Manage your AI separation models.{" "}
              {models.filter((m) => m.installed).length} installed.
            </p>
          </div>
          <div className="flex gap-3">
            {batchDownloading && batchAction === "download" && (
              <Button variant="destructive" onClick={handleCancelBatch}>
                Cancel Download
              </Button>
            )}
            {selectedModels.size > 0 && (
              <Button
                onClick={handleBatchDownload}
                disabled={batchDownloading}
                className="shadow-lg shadow-primary/20"
              >
                <Download className="mr-2 h-4 w-4" />
                Download Selected ({selectedModels.size})
              </Button>
            )}

            <Button
              variant="outline"
              onClick={() => setShowBlockedModels((v) => !v)}
              className={showBlockedModels ? "border-primary/40" : ""}
              title="Toggle visibility of models blocked by runtime constraints (missing checkpoints, unverified URLs, etc.)"
            >
              {showBlockedModels ? "Hide Blocked" : "Show Blocked"}
            </Button>

            <Button variant="outline" onClick={handleUploadCustom}>
              <Upload className="mr-2 h-4 w-4" />
              Import Custom
            </Button>
            <Button
              variant="outline"
              onClick={handleDownloadAll}
              disabled={batchDownloading}
            >
              <Download className="mr-2 h-4 w-4" />
              Download All
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteAll}
              disabled={batchDownloading}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete All
            </Button>
          </div>
        </div>

        {/* Filters Bar */}
        <div className="flex flex-col md:flex-row gap-4 items-center bg-card/30 p-4 rounded-xl border border-border/40 backdrop-blur-sm">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search models, architectures..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 bg-background/50 border-border/50 focus:bg-background transition-all"
            />
          </div>

          <div className="flex gap-2 w-full md:w-auto overflow-x-auto pb-2 md:pb-0 no-scrollbar">
            <select
              value={archFilter}
              onChange={(e) => setArchFilter(e.target.value)}
              className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              {architectures.map((a) => (
                <option key={a} value={a}>
                  {a === "all" ? "All Archs" : a}
                </option>
              ))}
            </select>

            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              {sources.map((s) => (
                <option key={s} value={s}>
                  {s === "all"
                    ? "All Sources"
                    : s === "builtin"
                      ? "Built-in"
                      : "Custom"}
                </option>
              ))}
            </select>

            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c.charAt(0).toUpperCase() + c.slice(1).replace("_", " ")}
                </option>
              ))}
            </select>

            <select
              value={speedFilter}
              onChange={(e) => setSpeedFilter(e.target.value)}
              className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              {speeds.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)} Speed
                </option>
              ))}
            </select>

            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              title="Filter models by tags from the v2 registry (e.g. restoration, dereverb, denoise, debleed, drumsep)"
            >
              <option value="all">All Tags</option>
              <option value="restoration">Restoration</option>
              <option value="dereverb">DeReverb</option>
              <option value="denoise">DeNoise</option>
              <option value="debleed">DeBleed</option>
              <option value="drumsep">DrumSep</option>
              <option value="karaoke">Karaoke</option>
              <option value="instrumental">Instrumental</option>
              <option value="vocal">Vocal</option>
            </select>

            <div className="flex bg-background/50 rounded-md border border-border/50 p-1">
              <button
                onClick={() => setDownloadedFilter("all")}
                className={`px-3 py-1 text-xs rounded-sm transition-all ${downloadedFilter === "all" ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-muted"}`}
              >
                All
              </button>
              <button
                onClick={() => setDownloadedFilter("downloaded")}
                className={`px-3 py-1 text-xs rounded-sm transition-all ${downloadedFilter === "downloaded" ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-muted"}`}
              >
                Installed
              </button>
              <button
                onClick={() => setDownloadedFilter("not-downloaded")}
                className={`px-3 py-1 text-xs rounded-sm transition-all ${downloadedFilter === "not-downloaded" ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-muted"}`}
              >
                Available
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
            <Filter className="h-12 w-12 mb-4 opacity-20" />
            <p>No models found matching your filters.</p>
            <Button
              variant="link"
              onClick={() => {
                setSearchQuery("");
                setCategoryFilter("all");
                setArchFilter("all");
                setSourceFilter("all");
                setTagFilter("all");
              }}
            >
              Clear Filters
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-10">
            {filteredModels.map((model, idx) => (
              <div
                key={`${model.id}:${idx}`}
                id={`model-${model.id}`}
                data-model-id={model.id}
              >
                <ModelCard
                  model={model}
                  onDownload={handleDownload}
                  onRemove={handleRemove}
                  onPause={handlePause}
                  onResume={handleResume}
                  onDetails={setDetailsModel}
                  isSelected={selectedModels.has(model.id)}
                  onToggleSelection={toggleModelSelection}
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
    </div>
  );
}
