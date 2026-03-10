import {
  AudioLines,
  Clock,
  Download,
  Drum,
  Guitar,
  Mic,
  Pause,
  Piano,
  Play,
  Sparkles,
  Zap,
} from "lucide-react";

import { Model } from "../stores/useStore";

interface ModelCardProps {
  model: Model;
  onDownload: (id: string) => void;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
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
  if (model.install?.mode === "manual") badges.add("Manual")
  if (model.status?.curated) badges.add("Curated")
  if (model.status?.support_tier === "supported_advanced") badges.add("Advanced")
  if (model.status?.readiness === "experimental") badges.add("Experimental")
  return Array.from(badges).slice(0, 4)
}

const getAction = (
  model: Model,
  onDownload: (id: string) => void,
  onPause: (id: string) => void,
  onResume: (id: string) => void,
  onDetails: (model: Model) => void,
) => {
  if (model.installed) {
    return {
      label: "Use",
      icon: Zap,
      className: "bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900",
      onClick: () => onDetails(model),
    };
  }
  if (model.downloadPaused) {
    return {
      label: "Resume",
      icon: Play,
      className: "bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900",
      onClick: () => onResume(model.id),
    };
  }
  if (model.downloading) {
    return {
      label: "Pause",
      icon: Pause,
      className: "bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900",
      onClick: () => onPause(model.id),
    };
  }
  return {
    label: "Download",
    icon: Download,
    className: "bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900",
    onClick: () => onDownload(model.id),
  };
};

export function ModelCard({
  model,
  onDownload,
  onPause,
  onResume,
  onDetails,
}: ModelCardProps) {
  const Icon = getModelIcon(model);
  const stems = Array.isArray(model.stems) ? model.stems.slice(0, 6) : [];
  const architecture = normalizeLabel(model.architecture || "Unknown");
  const quality =
    model.sdr >= 9 ? "Ultra" : model.sdr >= 7 ? "High" : "Standard";
  const speed = model.speed ? normalizeLabel(model.speed) : "Standard";
  const fileSize = formatFileSize(model.file_size);
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
    onPause,
    onResume,
    onDetails,
  );
  const ActionIcon = action.icon;
  const badges = getModelBadges(model)

  return (
    <div
      className="group rounded-2xl border border-white/80 bg-white p-5 backdrop-blur-md transition-all duration-300 hover:border-white hover:bg-white/90"
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
      <div className="flex items-start gap-4">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gray-100 transition-colors group-hover:bg-gray-200">
          <Icon className="h-5 w-5 text-gray-600" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-2.5">
            <h3 className="text-[15px] tracking-[-0.3px] text-gray-900">
              {model.name}
            </h3>
            {model.recommended && (
              <span className="rounded-full bg-green-100 px-2 py-0.5 text-[11px] tracking-[-0.2px] text-green-600">
                New
              </span>
            )}
          </div>

          <p className="mb-3 text-[13px] leading-[1.5] tracking-[-0.2px] text-gray-500">
            {model.description ||
              `${architecture} model for ${stems.length > 0 ? stems.join(", ") : "stem separation"}.`}
          </p>

          <div className="mb-3 flex flex-wrap gap-1.5">
            {badges.map((badge) => (
              <span
                key={badge}
                className="rounded-md bg-amber-50 px-2 py-0.5 text-[11px] tracking-[-0.1px] text-amber-700"
              >
                {badge}
              </span>
            ))}
            {stems.map((stem) => (
              <span
                key={stem}
                className="rounded-md bg-gray-100 px-2 py-0.5 text-[11px] tracking-[-0.1px] text-gray-500"
              >
                {normalizeLabel(stem)}
              </span>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-4 text-[12px] text-gray-400">
            <span
              className={`rounded-md px-2 py-0.5 text-[11px] ${
                quality === "Ultra"
                  ? "bg-violet-100 text-violet-600"
                  : quality === "High"
                    ? "bg-blue-50 text-blue-600"
                    : "bg-gray-100 text-gray-600"
              }`}
            >
              {quality}
            </span>
            <span className="rounded-md bg-violet-100 px-2 py-0.5 text-[11px] text-violet-600">
              {architecture}
            </span>
            {model.runtime?.engine && (
              <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                {normalizeLabel(model.runtime.engine)}
              </span>
            )}
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {speed}
            </span>
            {fileSize !== "--" && (
              <span className="flex items-center gap-1">
                <Download className="h-3 w-3" />
                {fileSize}
              </span>
            )}
            <span className="flex items-center gap-1">
              <Sparkles className="h-3 w-3 text-amber-400" />
              {rating}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            action.onClick();
          }}
          className={`flex shrink-0 items-center gap-1.5 rounded-xl px-4 py-2 text-[13px] tracking-[-0.2px] opacity-0 transition-all group-hover:opacity-100 ${action.className}`}
        >
          <ActionIcon className="h-3.5 w-3.5" />
          {action.label}
        </button>
      </div>
    </div>
  );
}
