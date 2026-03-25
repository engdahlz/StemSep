import {
  AudioLines,
  Clock,
  Download,
  Drum,
  Guitar,
  HardDrive,
  Mic,
  Piano,
  Sparkles,
} from "lucide-react";

import { Model } from "../stores/useStore";
import type { ModelMachineFit } from "../lib/systemRuntime/machineAnalysis";
import {
  formatCatalogTierLabel,
  formatModelSourceKind,
  getModelCatalogTier,
  isManualCatalogModel,
} from "../lib/models/catalog";

interface ModelCardProps {
  model: Model;
  machineFit?: ModelMachineFit;
  onDownload: (id: string) => void;
  onDetails: (model: Model) => void;
  isSelected: boolean;
  onToggleSelection: (id: string) => void;
}

const normalizeLabel = (value: string | undefined | null) =>
  String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();

const formatFileSize = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "--";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${Math.round(gb * 10) / 10} GB`;
  const mb = bytes / (1024 * 1024);
  return `${Math.round(mb)} MB`;
};

const formatVram = (value: number | undefined) => {
  if (!Number.isFinite(value) || !value || value <= 0) return "Auto";
  return `${value} GB`;
};

const getModelIcon = (model: Model) => {
  const stems = Array.isArray(model.stems)
    ? model.stems.map((stem) => String(stem).toLowerCase())
    : [];

  if (stems.some((stem) => stem.includes("vocal"))) return Mic;
  if (stems.some((stem) => stem.includes("drum") || stem.includes("kick")))
    return Drum;
  if (stems.some((stem) => stem.includes("guitar") || stem.includes("bass")))
    return Guitar;
  if (
    stems.some(
      (stem) =>
        stem.includes("key") || stem.includes("piano") || stem.includes("synth"),
    )
  )
    return Piano;
  if (model.recommended) return Sparkles;
  return AudioLines;
};

const roleBadgeMap: Record<string, string> = {
  primary: "Fullness",
  ensemble_partner: "Blend",
  phase_reference: "Reference",
  post_process: "Restoration",
  special_stem: "Special",
  fullness_source: "Fullness",
  cleanup_reference: "Bleedless",
  karaoke_base: "Karaoke",
  lead_back_splitter: "Lead/Back",
  restoration_step: "Restoration",
  low_band_source: "Low Band",
  high_band_source: "High Band",
}

function getModelBadges(model: Model): string[] {
  const badges = new Set<string>()
  const tierLabel = formatCatalogTierLabel(model)
  if (tierLabel !== "Catalog") badges.add(tierLabel)
  const sourceKind = formatModelSourceKind(model)
  if (sourceKind) badges.add(sourceKind)
  if (isManualCatalogModel(model)) badges.add("Manual")
  const roles = Array.isArray(model.quality_role)
    ? model.quality_role
    : model.quality_role
      ? [model.quality_role]
      : []
  for (const role of roles) {
    const label = roleBadgeMap[String(role)]
    if (label) badges.add(label)
  }
  if (model.workflow_groups?.some((group) => String(group).toLowerCase().includes("karaoke"))) {
    badges.add("Karaoke")
  }
  if (model.workflow_groups?.some((group) => String(group).toLowerCase().includes("drum"))) {
    badges.add("Drumsep")
  }
  if (model.content_fit?.some((fit) => String(fit).toLowerCase().includes("harmon"))) {
    badges.add("Harmonies")
  }
  if (model.status?.support_tier === "supported_advanced") badges.add("Advanced")
  if (model.status?.readiness === "experimental") badges.add("Experimental")
  return Array.from(badges).slice(0, 4)
}

const getPrimaryIntent = (model: Model) => {
  if (model.status?.curated) return "Verified pick";
  const catalogTier = getModelCatalogTier(model);
  if (catalogTier === "verified") return "Verified pick";
  if (catalogTier === "advanced_manual") return "Advanced / Manual";
  if (catalogTier === "advanced") return "Advanced pick";
  if (catalogTier === "manual") return "Manual setup";
  if (model.best_for?.length) return normalizeLabel(model.best_for[0]);
  if (model.quality_role) {
    const firstRole = Array.isArray(model.quality_role)
      ? model.quality_role[0]
      : model.quality_role;
    const mapped = roleBadgeMap[String(firstRole)];
    if (mapped) return mapped;
  }
  if (model.download?.mode === "manual") return "Manual setup";
  if (Array.isArray(model.stems) && model.stems.length > 0) {
    return `${normalizeLabel(model.stems[0])} focused`;
  }
  return "General separation";
};

const getToneClasses = (model: Model) => {
  if (model.downloading || model.downloadPaused) {
    return {
      iconWrap: "bg-sky-50 text-sky-700",
      accent: "from-sky-100/90 via-white to-white",
      badge: "bg-sky-50 text-sky-700",
    };
  }
  if (model.status?.curated || model.recommended) {
    return {
      iconWrap: "bg-emerald-50 text-emerald-700",
      accent: "from-emerald-100/85 via-white to-white",
      badge: "bg-emerald-50 text-emerald-700",
    };
  }
  if (isManualCatalogModel(model)) {
    return {
      iconWrap: "bg-amber-50 text-amber-700",
      accent: "from-amber-100/80 via-white to-white",
      badge: "bg-amber-50 text-amber-700",
    };
  }
  return {
    iconWrap: "bg-slate-100 text-slate-700",
    accent: "from-slate-100/85 via-white to-white",
    badge: "bg-slate-100 text-slate-700",
  };
};

const getAction = (
  model: Model,
  onDownload: (id: string) => void,
  onDetails: (model: Model) => void,
) => {
  const downloadMode = model.download?.mode;
  if (model.installed) {
    return {
      label: "Downloaded",
      icon: Download,
      className:
        "border border-emerald-200 bg-emerald-50 text-emerald-700",
      disabled: true,
      onClick: () => {},
    };
  }
  if (model.downloadPaused) {
    return {
      label: "Queued",
      icon: Download,
      className:
        "border border-sky-200 bg-sky-50 text-sky-700",
      disabled: true,
      onClick: () => {},
    };
  }
  if (model.downloading) {
    return {
      label: "Installing",
      icon: Download,
      className:
        "border border-sky-200 bg-sky-50 text-sky-700",
      disabled: true,
      onClick: () => {},
    };
  }
  if (downloadMode === "manual") {
    return {
      label: "Manual Setup",
      icon: Download,
      className: "bg-amber-50 text-amber-700 hover:bg-amber-100 hover:text-amber-900",
      disabled: false,
      onClick: () => onDetails(model),
    };
  }
  if (downloadMode === "unavailable") {
    return {
      label: "Unavailable",
      icon: Download,
      className: "bg-gray-100 text-gray-400 cursor-not-allowed",
      disabled: false,
      onClick: () => onDetails(model),
    };
  }
  const catalogTier = getModelCatalogTier(model);
  if (isManualCatalogModel(model)) {
    return {
      label: "Manual Setup",
      icon: Download,
      className: "bg-amber-50 text-amber-700 hover:bg-amber-100 hover:text-amber-900",
      disabled: false,
      onClick: () => onDetails(model),
    };
  }
  if (catalogTier === "blocked") {
    return {
      label: "Unavailable",
      icon: Download,
      className: "bg-gray-100 text-gray-400 cursor-not-allowed",
      disabled: false,
      onClick: () => onDetails(model),
    };
  }
  if (catalogTier === "verified") {
    return {
      label: "Verified",
      icon: Download,
      className: "border border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 hover:text-emerald-900",
      disabled: false,
      onClick: () => onDownload(model.id),
    };
  }
  if (catalogTier === "advanced") {
    return {
      label: "Advanced",
      icon: Download,
      className: "border border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 hover:text-amber-900",
      disabled: false,
      onClick: () => onDownload(model.id),
    };
  }
  if (catalogTier === "advanced_manual") {
    return {
      label: "Advanced / Manual",
      icon: Download,
      className:
        "border border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 hover:text-amber-900",
      disabled: false,
      onClick: () =>
        isManualCatalogModel(model) ? onDetails(model) : onDownload(model.id),
    };
  }
  return {
    label:
      model.download?.artifact_count && model.download.artifact_count > 1
        ? "Download Pack"
        : "Download",
    icon: Download,
    className:
      "border border-slate-200 bg-slate-100 text-slate-700 hover:bg-slate-200 hover:text-slate-900",
    disabled: false,
    onClick: () => onDownload(model.id),
  };
};

const getStatusChip = (model: Model) => {
  if (model.downloading) {
    return {
      label: "Downloading",
      className: "bg-sky-100 text-sky-700",
    };
  }
  if (model.downloadPaused) {
    return {
      label: "Paused",
      className: "bg-amber-100 text-amber-700",
    };
  }
  if (model.installed) {
    return {
      label: "Installed",
      className: "bg-emerald-100 text-emerald-700",
    };
  }
  if (model.download?.mode === "manual") {
    return {
      label: "Manual Setup",
      className: "bg-amber-50 text-amber-700",
    };
  }
  if (model.download?.mode === "unavailable") {
    return {
      label: "Unavailable",
      className: "bg-gray-100 text-gray-500",
    };
  }
  const tierLabel = formatCatalogTierLabel(model);
  return {
    label: tierLabel === "Catalog" ? "Not Installed" : tierLabel,
    className: "bg-rose-50 text-rose-600",
  };
};

const getMachineFitChip = (machineFit?: ModelMachineFit) => {
  switch (machineFit?.status) {
    case "fits_this_machine":
      return { label: machineFit.label, className: "bg-sky-50 text-sky-700" };
    case "heavy_for_this_machine":
      return { label: machineFit.label, className: "bg-amber-50 text-amber-700" };
    case "runtime_blocked":
      return { label: machineFit.label, className: "bg-rose-50 text-rose-700" };
    case "manual_setup":
      return { label: machineFit.label, className: "bg-orange-50 text-orange-700" };
    default:
      return null;
  }
};

export function ModelCard({
  model,
  machineFit,
  onDownload,
  onDetails,
}: ModelCardProps) {
  const Icon = getModelIcon(model);
  const stems = Array.isArray(model.stems) ? model.stems.slice(0, 6) : [];
  const architecture = normalizeLabel(model.architecture || "Unknown");
  const quality =
    model.sdr >= 9 ? "Ultra" : model.sdr >= 7 ? "High" : "Standard";
  const speed = model.speed ? normalizeLabel(model.speed) : "Standard";
  const fileSize = formatFileSize(model.file_size);
  const vram = formatVram(model.vram_required);
  const rating = Math.max(
    4.1,
    Math.min(
      5,
      Number.isFinite(model.sdr) && model.sdr > 0 ? 4 + model.sdr / 10 : 4.4,
    ),
  ).toFixed(1);
  const action = getAction(
    model,
    onDownload,
    onDetails,
  );
  const ActionIcon = action.icon;
  const badges = getModelBadges(model);
  const statusChip = getStatusChip(model);
  const machineFitChip = getMachineFitChip(machineFit);
  const primaryIntent = getPrimaryIntent(model);
  const tones = getToneClasses(model);
  const topStems = stems.slice(0, 3);
  const metrics = [
    { label: "Quality", value: quality, accent: "text-violet-700 bg-violet-50" },
    { label: "Speed", value: speed, accent: "text-sky-700 bg-sky-50" },
    { label: "VRAM", value: vram, accent: "text-slate-700 bg-slate-100" },
  ];

  return (
    <div
      className={`group rounded-[2rem] border border-white/80 bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(255,255,255,0.92))] p-5 shadow-[0_26px_72px_rgba(0,0,0,0.12)] backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:border-white hover:shadow-[0_32px_80px_rgba(0,0,0,0.16)]`}
      role="button"
      tabIndex={0}
      onClick={() => onDetails(model)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onDetails(model);
        }
      }}
    >
      <div className="mb-5 flex items-start gap-4">
        <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-[1.15rem] ${tones.iconWrap} transition-colors`}>
          <Icon className="h-5 w-5" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="truncate text-[17px] tracking-[-0.45px] text-gray-950">
                  {model.name}
                </h3>
                {(model.recommended || model.status?.curated) && (
                  <span className={`rounded-full px-2.5 py-1 text-[11px] tracking-[-0.15px] ${tones.badge}`}>
                    {model.status?.curated ? "Verified" : "Recommended"}
                  </span>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-slate-500">
                <span>{primaryIntent}</span>
                <span className="text-slate-300">•</span>
                <span>{architecture}</span>
                <span className="text-slate-300">•</span>
                <span>{quality}</span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span
                  className={`rounded-full px-2.5 py-1 text-[11px] tracking-[-0.1px] ${statusChip.className}`}
                >
                  {statusChip.label}
                </span>
                {machineFitChip && (
                  <span
                    className={`rounded-full px-2.5 py-1 text-[11px] tracking-[-0.1px] ${machineFitChip.className}`}
                    title={machineFit?.reason}
                  >
                    {machineFitChip.label}
                  </span>
                )}
                {model.download?.artifact_count && model.download.artifact_count > 1 && (
                  <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[11px] tracking-[-0.1px] text-sky-700">
                    {model.download.artifact_count} files
                  </span>
                )}
                {model.status?.support_tier === "supported_advanced" && (
                  <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[11px] tracking-[-0.1px] text-amber-700">
                    Advanced
                  </span>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                action.onClick();
              }}
              disabled={action.disabled}
              className={`flex shrink-0 items-center gap-1.5 rounded-[1rem] px-4 py-2 text-[13px] tracking-[-0.2px] transition-all disabled:cursor-default disabled:opacity-100 ${action.className}`}
            >
              <ActionIcon className="h-3.5 w-3.5" />
              {action.label}
            </button>
          </div>

          <p className="line-clamp-2 text-[13px] leading-[1.55] tracking-[-0.2px] text-slate-500">
            {model.description ||
              `${architecture} model for ${stems.length > 0 ? stems.join(", ") : "stem separation"}.`}
          </p>

          <div className={`mt-4 rounded-[1.35rem] border border-slate-200/75 bg-gradient-to-r ${tones.accent} p-3.5`}>
            <div className="grid grid-cols-3 gap-2">
              {metrics.map((metric) => (
                <div
                  key={metric.label}
                  className="rounded-[1rem] border border-white/80 bg-white/72 px-3 py-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]"
                >
                  <div className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
                    {metric.label}
                  </div>
                  <div className={`mt-1 inline-flex rounded-full px-2 py-0.5 text-[11px] ${metric.accent}`}>
                    {metric.value}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-slate-500">
              {badges.slice(0, 3).map((badge) => (
                <span
                  key={badge}
                  className="rounded-full bg-amber-50 px-2.5 py-1 text-[11px] tracking-[-0.1px] text-amber-700"
                >
                  {badge}
                </span>
              ))}
              {topStems.map((stem) => (
                <span
                  key={stem}
                  className="rounded-full bg-white/88 px-2.5 py-1 text-[11px] tracking-[-0.1px] text-slate-600"
                >
                  {normalizeLabel(stem)}
                </span>
              ))}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-4 text-[12px] text-slate-500">
              {model.runtime?.engine && (
                <span className="rounded-full bg-white/84 px-2.5 py-1 text-[11px] text-slate-600">
                  {normalizeLabel(model.runtime.engine)}
                </span>
              )}
              {fileSize !== "--" && (
                <span className="flex items-center gap-1">
                  <HardDrive className="h-3 w-3" />
                  {fileSize}
                </span>
              )}
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {speed}
              </span>
              <span className="flex items-center gap-1">
                <Sparkles className="h-3 w-3 text-amber-400" />
                {rating}
              </span>
            </div>
          </div>
        </div>
      </div>

      {model.downloading && (
        <div className="mt-4 rounded-[1.1rem] border border-sky-100 bg-sky-50/80 p-3">
          <div className="mb-2 flex items-center justify-between text-[12px] text-sky-800">
            <span>Downloading model</span>
            <span>{Math.round(model.downloadProgress || 0)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-sky-100">
            <div
              className="h-full rounded-full bg-sky-500 transition-all duration-300"
              style={{ width: `${model.downloadProgress || 0}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
