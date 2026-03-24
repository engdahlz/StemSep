import { useEffect, useMemo, useState } from "react";
import { Cpu, HardDriveDownload, Layers3, ShieldCheck, Sparkles, Wand2 } from "lucide-react";
import { toast } from "sonner";

import { PageShell } from "./PageShell";
import { EnsembleBuilder } from "./EnsembleBuilder";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import type { Preset } from "@/presets";
import { getRequiredModels } from "@/presets";
import type { Model } from "@/types/store";
import type { QualityProfile, SeparationConfig } from "@/types/separation";
import {
  deriveRecommendedHardwareTier,
  deriveRuntimeBurden,
  deriveTargetUseCase,
  deriveWorkflowTier,
  getQualityProfiles,
  resolveQualityProfileAdvancedParams,
} from "@/lib/separation/qualityProfiles";
import { bestVolumeCompensation } from "@/utils/volumeCompensation";

type WorkflowLabSurface = "guided" | "manual" | "restoration";

interface WorkflowLabPageProps {
  presets: Preset[];
  models: Model[];
  onPrepareSeparation: (
    config: SeparationConfig,
    file: { name: string; path: string; presetId?: string },
  ) => void;
}

type GuidedWorkflowSpec = {
  key: string;
  title: string;
  description: string;
  candidates: string[];
  fallback?: (preset: Preset) => boolean;
};

type GuidedWorkflowCard = {
  spec: GuidedWorkflowSpec;
  preset: Preset;
  workflowTier: "verified" | "advanced";
  runtimeBurden: string;
  hardwareTier: string;
  targetUseCase: string;
  installedCount: number;
  requiredCount: number;
};

const GUIDED_WORKFLOWS: GuidedWorkflowSpec[] = [
  {
    key: "best-vocals",
    title: "Best Vocals",
    description: "Targets lead-vocal clarity first, with cleanup-oriented defaults for modern vocal extraction.",
    candidates: ["recipe_vocal_denoise", "best_vocals", "natural_body_vocals"],
    fallback: (preset) =>
      preset.simpleGoal === "vocals" &&
      (preset.workflowFamily?.toLowerCase().includes("vocal") ||
        preset.name.toLowerCase().includes("vocals")),
  },
  {
    key: "best-instrumental",
    title: "Best Instrumental",
    description: "Guide-style instrumental extraction with fuller backing recovery and stronger anti-buzz defaults.",
    candidates: ["workflow_phase_fix_instrumental", "best_instrumental", "high_quality_instrumental"],
    fallback: (preset) =>
      preset.simpleGoal === "instrumental" &&
      (preset.workflowFamily?.toLowerCase().includes("instrument") ||
        preset.name.toLowerCase().includes("instrumental")),
  },
  {
    key: "bleedless-karaoke",
    title: "Bleedless Karaoke",
    description: "Optimized for the cleanest backing track possible, prioritising low vocal bleed over fullness.",
    candidates: ["recipe_karaoke_fusion", "best_karaoke"],
    fallback: (preset) =>
      preset.simpleGoal === "karaoke" || preset.name.toLowerCase().includes("karaoke"),
  },
  {
    key: "vocal-restoration",
    title: "Separated Vocal Restoration",
    description: "Chains separation with repair-oriented cleanup to produce a vocal track that is easier to mix or edit.",
    candidates: ["workflow_live_restoration", "recipe_vocal_denoise"],
    fallback: (preset) =>
      preset.simpleGoal === "cleanup" &&
      (preset.workflowFamily?.toLowerCase().includes("restoration") ||
        preset.name.toLowerCase().includes("restoration")),
  },
  {
    key: "precision-dereverb",
    title: "Precision Dereverb",
    description: "Focused restoration workflow for reverb-heavy material where vocal intelligibility matters more than speed.",
    candidates: ["workflow_dereverb_precision_bs", "restoration_dereverb_denoise", "de_reverb_room"],
    fallback: (preset) =>
      preset.workflowFamily?.toLowerCase().includes("dereverb") ||
      preset.name.toLowerCase().includes("reverb"),
  },
  {
    key: "bass-definition-hybrid",
    title: "Bass Definition Hybrid",
    description: "Hybrid low-end workflow inspired by Colab-style specialist chains for tighter, more defined bass recovery.",
    candidates: ["bass_hybrid_scnet_demucs"],
    fallback: (preset) =>
      preset.workflowFamily?.toLowerCase().includes("bass") || preset.name.toLowerCase().includes("bass"),
  },
];

const RESTORATION_WORKFLOWS: GuidedWorkflowSpec[] = [
  {
    key: "separated-vocal-restoration",
    title: "Separated Vocal Restoration",
    description: "Restoration-first vocal repair chain for already separated or cleanup-heavy vocal material.",
    candidates: ["recipe_vocal_denoise", "workflow_live_restoration", "restoration_dereverb_denoise"],
    fallback: (preset) =>
      Boolean(
        preset.simpleGoal === "cleanup" &&
          (preset.workflowFamily?.toLowerCase().includes("restoration") ||
            preset.name.toLowerCase().includes("restoration")),
      ),
  },
  {
    key: "live-restoration",
    title: "Live Vocal Restoration",
    description: "Use when the source is noisy, roomy or audience-heavy and you want cleanup to be part of the first pass.",
    candidates: ["workflow_live_restoration", "restoration_dereverb_denoise"],
    fallback: (preset) =>
      Boolean(
        preset.simpleGoal === "cleanup" &&
          preset.workflowFamily?.toLowerCase().includes("restoration"),
      ),
  },
  {
    key: "precision-dereverb",
    title: "Precision Dereverb",
    description: "Dedicated dereverb path for speech-like vocals and reverberant rooms.",
    candidates: ["workflow_dereverb_precision_bs", "restoration_dereverb_denoise", "de_reverb_room"],
    fallback: (preset) =>
      preset.workflowFamily?.toLowerCase().includes("dereverb") ||
      preset.name.toLowerCase().includes("reverb"),
  },
  {
    key: "noise-and-bleed-cleanup",
    title: "Noise and Bleed Cleanup",
    description: "Useful as a second-stage cleanup path when separation is already done but the result still needs repair.",
    candidates: ["recipe_vocal_denoise", "de_noise_aggressive", "de_bleed"],
    fallback: (preset) =>
      preset.simpleGoal === "cleanup" &&
      (preset.name.toLowerCase().includes("noise") || preset.name.toLowerCase().includes("bleed")),
  },
];

const basename = (path: string) => path.split(/[\\/]/).pop() || path;

function resolveGuidedWorkflow(
  presets: Preset[],
  models: Model[],
  spec: GuidedWorkflowSpec,
): GuidedWorkflowCard | null {
  const preset =
    spec.candidates.map((id) => presets.find((entry) => entry.id === id)).find(Boolean) ||
    presets.find((entry) => spec.fallback?.(entry));

  if (!preset) return null;

  const requiredIds = getRequiredModels(preset);
  const installedCount = requiredIds.filter((id) =>
    models.some((model) => model.id === id && model.installed),
  ).length;

  return {
    spec,
    preset,
    workflowTier: deriveWorkflowTier(preset),
    runtimeBurden: deriveRuntimeBurden(preset),
    hardwareTier: deriveRecommendedHardwareTier(preset),
    targetUseCase: deriveTargetUseCase(preset),
    installedCount,
    requiredCount: requiredIds.length,
  };
}

function buildGuidedConfig(preset: Preset, qualityProfile: QualityProfile): SeparationConfig {
  return {
    mode: "simple",
    presetId: preset.id,
    modelId: preset.modelId,
    selectionEnvelope: preset.selectionEnvelope,
    workflowId: preset.id,
    device: "auto",
    outputFormat: "wav",
    normalize: true,
    stems: preset.stems,
    qualityProfile,
    advancedParams: resolveQualityProfileAdvancedParams({
      profile: qualityProfile,
      preset,
    }),
  };
}

export default function QualityLabPage({
  presets,
  models,
  onPrepareSeparation,
}: WorkflowLabPageProps) {
  const [surface, setSurface] = useState<WorkflowLabSurface>("guided");
  const [qualityProfile, setQualityProfile] = useState<QualityProfile>("balanced");
  const [manualTarget, setManualTarget] = useState<"vocals" | "instrumental" | "karaoke">("instrumental");
  const [manualModels, setManualModels] = useState<Array<{ model_id: string; weight: number }>>([]);
  const [manualAlgorithm, setManualAlgorithm] = useState<"average" | "max_spec" | "min_spec" | "frequency_split">("average");
  const [manualStemAlgorithms, setManualStemAlgorithms] = useState<{
    vocals?: "average" | "max_spec" | "min_spec";
    instrumental?: "average" | "max_spec" | "min_spec";
  }>();
  const [manualPhaseFixEnabled, setManualPhaseFixEnabled] = useState(false);
  const [manualPhaseFixParams, setManualPhaseFixParams] = useState({
    enabled: true,
    lowHz: 500,
    highHz: 5000,
    highFreqWeight: 2,
  });
  const [manualVolumeComp, setManualVolumeComp] = useState(true);
  const [secondaryCleanupModelId, setSecondaryCleanupModelId] = useState("");
  const [dereverbModelId, setDereverbModelId] = useState("");
  const [denoiseModelId, setDenoiseModelId] = useState("");
  const [isPreparing, setIsPreparing] = useState(false);

  const qualityProfiles = useMemo(() => getQualityProfiles(), []);
  const installedModels = useMemo(() => models.filter((model) => model.installed), [models]);

  useEffect(() => {
    if (manualModels.length > 0 || installedModels.length === 0) return;
    const initialModels = installedModels.slice(0, Math.min(2, installedModels.length));
    setManualModels(initialModels.map((model) => ({ model_id: model.id, weight: 1 })));
  }, [installedModels, manualModels.length]);

  const guidedCards = useMemo(
    () => GUIDED_WORKFLOWS.map((spec) => resolveGuidedWorkflow(presets, models, spec)).filter(Boolean) as GuidedWorkflowCard[],
    [models, presets],
  );
  const restorationCards = useMemo(
    () => RESTORATION_WORKFLOWS.map((spec) => resolveGuidedWorkflow(presets, models, spec)).filter(Boolean) as GuidedWorkflowCard[],
    [models, presets],
  );

  const cleanupCandidates = useMemo(
    () => installedModels.filter((model) => /bleed|clean|noise|denoise|crowd/i.test(`${model.name} ${model.id} ${model.description}`)),
    [installedModels],
  );
  const dereverbCandidates = useMemo(
    () => installedModels.filter((model) => /reverb|room/i.test(`${model.name} ${model.id} ${model.description}`)),
    [installedModels],
  );
  const denoiseCandidates = useMemo(
    () => installedModels.filter((model) => /noise|denoise|clean/i.test(`${model.name} ${model.id} ${model.description}`)),
    [installedModels],
  );

  const manualRequiredIds = useMemo(() => {
    const ids = new Set(manualModels.map((entry) => entry.model_id).filter(Boolean));
    if (secondaryCleanupModelId) ids.add(secondaryCleanupModelId);
    if (dereverbModelId) ids.add(dereverbModelId);
    if (denoiseModelId) ids.add(denoiseModelId);
    return Array.from(ids);
  }, [denoiseModelId, dereverbModelId, manualModels, secondaryCleanupModelId]);

  const manualStepSummary = useMemo(() => {
    const steps: string[] = [];
    steps.push(manualTarget === "karaoke" ? "Separate into no-vocals and vocal reference" : `Separate for ${manualTarget}`);
    if (secondaryCleanupModelId) steps.push("Secondary cleanup");
    if (dereverbModelId) steps.push("Dereverb");
    if (denoiseModelId) steps.push("Denoise");
    if (manualPhaseFixEnabled) steps.push("Phase fix");
    if (manualAlgorithm === "frequency_split") steps.push("Frequency split blend");
    return steps;
  }, [denoiseModelId, dereverbModelId, manualAlgorithm, manualPhaseFixEnabled, manualTarget, secondaryCleanupModelId]);

  const selectedProfile = useMemo(
    () => qualityProfiles.find((profile) => profile.id === qualityProfile) || null,
    [qualityProfile, qualityProfiles],
  );

  const prepareWithAudioFile = async (config: SeparationConfig, presetId?: string) => {
    if (!window.electronAPI?.openAudioFileDialog) {
      toast.error("Audio file picker is not available.");
      return;
    }
    setIsPreparing(true);
    try {
      const filePaths = await window.electronAPI.openAudioFileDialog();
      if (!filePaths || filePaths.length === 0) return;
      const path = filePaths[0];
      onPrepareSeparation(config, { name: basename(path), path, presetId });
      toast.success("Workflow staged on Home. Review the prepared run and start it there.");
    } catch (error: any) {
      toast.error(error?.message || "Failed to prepare workflow.");
    } finally {
      setIsPreparing(false);
    }
  };

  const launchGuidedWorkflow = async (card: GuidedWorkflowCard) => {
    const config = buildGuidedConfig(card.preset, qualityProfile);
    await prepareWithAudioFile(config, card.preset.id);
  };

  const launchManualWorkflow = async () => {
    const normalizedModels = manualModels.filter((entry) => entry.model_id);
    if (normalizedModels.length === 0) {
      toast.error("Add at least one installed model to the ensemble.");
      return;
    }

    const postProcessingSteps = [
      secondaryCleanupModelId
        ? {
            type:
              manualTarget === "instrumental" || manualTarget === "karaoke" ? "de_bleed" : "de_noise",
            modelId: secondaryCleanupModelId,
            description: "Secondary cleanup pass",
            targetStem: manualTarget === "instrumental" ? "instrumental" : "vocals",
          }
        : null,
      dereverbModelId
        ? {
            type: "de_reverb",
            modelId: dereverbModelId,
            description: "Dereverb pass",
            targetStem: "vocals" as const,
          }
        : null,
      denoiseModelId
        ? {
            type: "de_noise",
            modelId: denoiseModelId,
            description: "Denoise pass",
            targetStem: manualTarget === "instrumental" ? ("instrumental" as const) : ("vocals" as const),
          }
        : null,
    ].filter(Boolean) as NonNullable<SeparationConfig["postProcessingSteps"]>;

    const config: SeparationConfig = {
      mode: "advanced",
      workflowId: `workflow_lab_${manualTarget}`,
      device: "auto",
      outputFormat: "wav",
      normalize: true,
      qualityProfile,
      stems:
        manualTarget === "karaoke"
          ? ["no_vocals", "vocals"]
          : manualTarget === "vocals"
            ? ["vocals", "instrumental"]
            : ["instrumental", "vocals"],
      advancedParams: resolveQualityProfileAdvancedParams({
        profile: qualityProfile,
        current: { batchSize: 1 },
      }),
      ensembleConfig: {
        models: normalizedModels,
        algorithm: manualAlgorithm,
        stemAlgorithms: manualStemAlgorithms,
        phaseFixEnabled: manualPhaseFixEnabled,
        phaseFixParams: manualPhaseFixEnabled ? manualPhaseFixParams : undefined,
      },
      postProcessingSteps,
      volumeCompensation: manualVolumeComp ? bestVolumeCompensation() : undefined,
    };

    await prepareWithAudioFile(config);
  };

  const renderWorkflowCard = (card: GuidedWorkflowCard) => {
    const ready = card.requiredCount === 0 || card.installedCount === card.requiredCount;
    return (
      <Card key={card.spec.key} className="border-white/60 bg-white/58 backdrop-blur-xl shadow-[0_18px_50px_rgba(0,0,0,0.10)]">
        <CardHeader className="gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={card.workflowTier === "verified" ? "default" : "secondary"}>
              {card.workflowTier === "verified" ? "Verified Workflow" : "Advanced / Experimental"}
            </Badge>
            <Badge variant="outline">{card.preset.workflowFamily || card.preset.category}</Badge>
            <Badge variant="outline">{card.runtimeBurden}</Badge>
          </div>
          <div>
            <CardTitle className="text-[1.2rem] tracking-[-0.4px] text-slate-900">{card.spec.title}</CardTitle>
            <p className="mt-2 text-sm leading-6 text-slate-600">{card.spec.description}</p>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-[1.1rem] border border-white/60 bg-white/62 p-3">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-400">
                <Sparkles className="h-3.5 w-3.5" />
                Target use
              </div>
              <p className="mt-2 text-sm text-slate-700">{card.targetUseCase}</p>
            </div>
            <div className="rounded-[1.1rem] border border-white/60 bg-white/62 p-3">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-400">
                <Cpu className="h-3.5 w-3.5" />
                Hardware
              </div>
              <p className="mt-2 text-sm text-slate-700">{card.hardwareTier}</p>
            </div>
          </div>
          <div className="rounded-[1.1rem] border border-white/60 bg-white/62 p-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-400">
              <HardDriveDownload className="h-3.5 w-3.5" />
              Model readiness
            </div>
            <p className="mt-2 text-sm text-slate-700">
              {card.requiredCount === 0
                ? "No extra model dependencies surfaced for this workflow."
                : `${card.installedCount} of ${card.requiredCount} required models already installed.`}
            </p>
            {!ready && (
              <p className="mt-1 text-xs text-slate-500">
                You can still prepare the run now. Home will block on missing assets before execution.
              </p>
            )}
          </div>
          <div className="rounded-[1.1rem] border border-white/60 bg-white/50 p-3 text-sm text-slate-600">
            <div className="font-medium text-slate-800">Why this workflow exists</div>
            <p className="mt-1">{card.preset.workflowSummary || card.preset.description}</p>
            {card.preset.contraindications?.length ? (
              <p className="mt-2 text-xs text-slate-500">
                Watch-outs: {card.preset.contraindications.join(", ")}
              </p>
            ) : null}
          </div>
          <Button className="w-full rounded-[1rem]" size="lg" onClick={() => void launchGuidedWorkflow(card)} disabled={isPreparing}>
            {isPreparing ? "Preparing..." : "Choose Audio and Prepare"}
          </Button>
        </CardContent>
      </Card>
    );
  };

  return (
    <PageShell>
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
        <div className="rounded-[1.8rem] border border-white/60 bg-white/50 p-6 shadow-[0_28px_90px_rgba(0,0,0,0.12)] backdrop-blur-2xl">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-3 flex flex-wrap gap-2">
                <Badge variant="secondary">Workflow Lab</Badge>
                <Badge variant="outline">Colab-inspired</Badge>
                <Badge variant="outline">Quality-first</Badge>
              </div>
              <h1 className="text-3xl tracking-[-1px] text-slate-900">
                Curated high-quality separation without notebook chaos
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                Guided workflows expose the best notebook-inspired chains as curated products. Manual Ensemble keeps
                the power-user path available without forcing you to remember community model lore.
              </p>
            </div>
            <div className="min-w-[280px] rounded-[1.3rem] border border-white/60 bg-white/62 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Selected quality profile</div>
              <div className="mt-2 text-lg font-medium text-slate-900">{selectedProfile?.label || "Balanced"}</div>
              <p className="mt-1 text-sm text-slate-600">{selectedProfile?.description}</p>
            </div>
          </div>
        </div>

        <div className="rounded-[1.5rem] border border-white/60 bg-white/55 p-3 backdrop-blur-xl">
          <div className="flex flex-wrap gap-2">
            {qualityProfiles.map((profile) => (
              <button
                key={profile.id}
                type="button"
                onClick={() => setQualityProfile(profile.id)}
                className={`rounded-full border px-4 py-2 text-sm transition-all ${
                  qualityProfile === profile.id
                    ? "border-slate-900/10 bg-slate-900 text-white shadow-[0_10px_25px_rgba(0,0,0,0.18)]"
                    : "border-white/70 bg-white/70 text-slate-600 hover:bg-white/90 hover:text-slate-900"
                }`}
              >
                {profile.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {[
            { id: "guided" as const, label: "Guided", description: "Curated workflow picks" },
            { id: "manual" as const, label: "Manual Ensemble", description: "Build your own chain" },
            { id: "restoration" as const, label: "Restoration", description: "Cleanup and repair" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setSurface(tab.id)}
              className={`rounded-[1.2rem] border px-4 py-3 text-left transition-all ${
                surface === tab.id
                  ? "border-white/90 bg-white/88 text-slate-900 shadow-[0_16px_40px_rgba(0,0,0,0.10)]"
                  : "border-white/60 bg-white/50 text-slate-600 hover:bg-white/72 hover:text-slate-900"
              }`}
            >
              <div className="text-sm font-medium">{tab.label}</div>
              <div className="mt-1 text-xs text-slate-500">{tab.description}</div>
            </button>
          ))}
        </div>

        {surface === "guided" && (
          <div className="grid gap-6 xl:grid-cols-2">
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
                <ShieldCheck className="h-4 w-4 text-emerald-600" />
                Verified Workflows
              </div>
              {guidedCards.filter((card) => card.workflowTier === "verified").map(renderWorkflowCard)}
            </div>
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
                <Layers3 className="h-4 w-4 text-slate-700" />
                Advanced / Experimental
              </div>
              {guidedCards.filter((card) => card.workflowTier === "advanced").map(renderWorkflowCard)}
            </div>
          </div>
        )}

        {surface === "restoration" && <div className="grid gap-6 xl:grid-cols-2">{restorationCards.map(renderWorkflowCard)}</div>}

        {surface === "manual" && (
          <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <Card className="border-white/60 bg-white/58 backdrop-blur-xl shadow-[0_18px_50px_rgba(0,0,0,0.10)]">
              <CardHeader>
                <CardTitle className="text-slate-900">Manual Ensemble Builder</CardTitle>
                <p className="text-sm leading-6 text-slate-600">
                  Combine installed models, phase fix and cleanup passes into a single staged run. This keeps the
                  exploratory notebook power-user path, but still hands execution back to the Rust control plane.
                </p>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-800">Primary goal</label>
                    <select
                      value={manualTarget}
                      onChange={(event) => setManualTarget(event.target.value as typeof manualTarget)}
                      className="flex h-10 w-full rounded-[1rem] border border-white/60 bg-white/76 px-3 text-sm text-slate-800 shadow-sm outline-none"
                    >
                      <option value="instrumental">Instrumental / backing</option>
                      <option value="vocals">Lead vocals</option>
                      <option value="karaoke">Karaoke / no vocals</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-800">Secondary cleanup</label>
                    <select
                      value={secondaryCleanupModelId}
                      onChange={(event) => setSecondaryCleanupModelId(event.target.value)}
                      className="flex h-10 w-full rounded-[1rem] border border-white/60 bg-white/76 px-3 text-sm text-slate-800 shadow-sm outline-none"
                    >
                      <option value="">None</option>
                      {cleanupCandidates.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-800">Dereverb</label>
                    <select
                      value={dereverbModelId}
                      onChange={(event) => setDereverbModelId(event.target.value)}
                      className="flex h-10 w-full rounded-[1rem] border border-white/60 bg-white/76 px-3 text-sm text-slate-800 shadow-sm outline-none"
                    >
                      <option value="">None</option>
                      {dereverbCandidates.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-800">Denoise</label>
                    <select
                      value={denoiseModelId}
                      onChange={(event) => setDenoiseModelId(event.target.value)}
                      className="flex h-10 w-full rounded-[1rem] border border-white/60 bg-white/76 px-3 text-sm text-slate-800 shadow-sm outline-none"
                    >
                      <option value="">None</option>
                      {denoiseCandidates.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <EnsembleBuilder
                  models={models}
                  config={manualModels}
                  algorithm={manualAlgorithm}
                  phaseFixEnabled={manualPhaseFixEnabled}
                  volumeCompEnabled={manualVolumeComp}
                  stemAlgorithms={manualStemAlgorithms}
                  phaseFixParams={manualPhaseFixParams}
                  onVolumeCompEnabledChange={setManualVolumeComp}
                  onChange={(config, algorithm, stemAlgorithms, phaseFixParams, phaseFixEnabled) => {
                    setManualModels(config);
                    setManualAlgorithm(algorithm);
                    setManualStemAlgorithms(stemAlgorithms);
                    if (phaseFixParams) setManualPhaseFixParams(phaseFixParams);
                    if (typeof phaseFixEnabled === "boolean") setManualPhaseFixEnabled(phaseFixEnabled);
                  }}
                />

                <Button size="lg" className="w-full rounded-[1rem]" onClick={() => void launchManualWorkflow()} disabled={isPreparing || manualModels.length === 0}>
                  {isPreparing ? "Preparing..." : "Choose Audio and Prepare Manual Workflow"}
                </Button>
              </CardContent>
            </Card>

            <div className="space-y-4">
              <Card className="border-white/60 bg-white/58 backdrop-blur-xl shadow-[0_18px_50px_rgba(0,0,0,0.10)]">
                <CardHeader>
                  <CardTitle className="text-slate-900">Workflow Preview</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">{manualTarget}</Badge>
                    <Badge variant="outline">{manualAlgorithm}</Badge>
                    {manualPhaseFixEnabled && <Badge variant="outline">phase fix</Badge>}
                    {manualVolumeComp && <Badge variant="outline">volume compensation</Badge>}
                  </div>
                  <div className="rounded-[1.1rem] border border-white/60 bg-white/60 p-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Planned chain</div>
                    <ul className="mt-2 space-y-2 text-sm text-slate-700">
                      {manualStepSummary.map((step) => (
                        <li key={step} className="flex items-start gap-2">
                          <Wand2 className="mt-0.5 h-4 w-4 text-slate-400" />
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="rounded-[1.1rem] border border-white/60 bg-white/60 p-3">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Required installed models</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {manualRequiredIds.length > 0 ? (
                        manualRequiredIds.map((id) => (
                          <Badge key={id} variant="outline">
                            {models.find((model) => model.id === id)?.name || id}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-sm text-slate-500">No models selected yet.</span>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-white/60 bg-white/58 backdrop-blur-xl shadow-[0_18px_50px_rgba(0,0,0,0.10)]">
                <CardHeader>
                  <CardTitle className="text-slate-900">What stays out of scope</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm leading-6 text-slate-600">
                  <p>
                    This pass intentionally keeps transcription, mastering and voice cloning outside the main separation
                    surface.
                  </p>
                  <p>
                    Workflow Lab is focused on separation quality, cleanup and restoration, not on turning the app into
                    a generic audio notebook shell.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </PageShell>
  );
}
