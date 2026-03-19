import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Cpu,
  Loader2,
  Monitor,
  Sparkles,
  X,
} from "lucide-react";

import { Badge } from "./ui/badge";
import { useStore } from "../stores/useStore";
import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";
import { ALL_PRESETS } from "../presets";
import { recipesToPresets } from "../lib/recipePresets";
import { buildMachineAnalysis } from "../lib/systemRuntime/machineAnalysis";

type AnalysisState = "idle" | "scanning" | "done";

const toneClasses = (score: number) => {
  if (score >= 75) {
    return {
      text: "text-emerald-100",
      accent: "text-emerald-200",
      chip: "border-emerald-300/28 bg-emerald-300/14 text-emerald-100",
      bar: "from-emerald-300 via-emerald-200 to-white/90",
      glow: "shadow-[0_0_24px_rgba(110,231,183,0.22)]",
    };
  }
  if (score >= 50) {
    return {
      text: "text-amber-100",
      accent: "text-amber-200",
      chip: "border-amber-300/28 bg-amber-300/14 text-amber-50",
      bar: "from-amber-300 via-amber-200 to-white/85",
      glow: "shadow-[0_0_24px_rgba(252,211,77,0.2)]",
    };
  }
  return {
    text: "text-rose-100",
    accent: "text-rose-200",
    chip: "border-rose-300/28 bg-rose-300/14 text-rose-50",
    bar: "from-rose-300 via-rose-200 to-white/85",
    glow: "shadow-[0_0_24px_rgba(253,164,175,0.2)]",
  };
};

export function MachineAnalyzer() {
  const models = useStore((state) => state.models);
  const recipes = useStore((state) => state.recipes);
  const { info: runtimeInfo, error, loading, refresh } = useSystemRuntimeInfo();

  const [isOpen, setIsOpen] = useState(false);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [progress, setProgress] = useState(0);

  const presets = useMemo(() => {
    return [...ALL_PRESETS, ...recipesToPresets(Array.isArray(recipes) ? recipes : [])];
  }, [recipes]);

  const analysis = useMemo(() => {
    return buildMachineAnalysis(runtimeInfo, models as any, presets);
  }, [models, presets, runtimeInfo]);

  const closeAnalyzer = () => {
    setIsOpen(false);
    setAnalysisState("idle");
    setProgress(0);
  };

  const runAnalysis = async () => {
    setAnalysisState("scanning");
    setProgress(8);

    const interval = window.setInterval(() => {
      setProgress((value) => {
        if (value >= 92) return 92;
        return value + 4;
      });
    }, 120);

    try {
      await refresh(true);
      setProgress(100);
      window.setTimeout(() => {
        setAnalysisState("done");
      }, 160);
    } finally {
      window.clearInterval(interval);
    }
  };

  const scoreTone = toneClasses(analysis.overallScore);

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed right-5 top-5 z-30 rounded-[18px] border border-white/28 bg-white/18 p-3 text-white shadow-[0_18px_40px_rgba(0,0,0,0.18)] backdrop-blur-xl transition-all duration-300 hover:scale-[1.04] hover:border-white/42 hover:bg-white/24 active:scale-95"
        aria-label="Open machine analyzer"
      >
        <Monitor className="h-5 w-5" />
      </button>

      {isOpen && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-50 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.12),transparent_35%),rgba(17,21,33,0.38)] backdrop-blur-xl"
            onClick={closeAnalyzer}
            aria-label="Close machine analyzer overlay"
          />

          <div className="fixed left-1/2 top-1/2 z-[60] w-[min(860px,calc(100vw-2rem))] max-w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[2rem] border border-white/22 bg-[linear-gradient(145deg,rgba(255,255,255,0.2),rgba(255,255,255,0.08))] text-white shadow-[0_40px_120px_rgba(10,14,22,0.32)] backdrop-blur-2xl">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.2),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(123,166,255,0.16),transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))]" />
            <div className="pointer-events-none absolute inset-[1px] rounded-[calc(2rem-1px)] border border-white/10" />

            <div className="relative max-h-[85vh] overflow-y-auto px-6 pb-6 pt-6 sm:px-7">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-3">
                  <Badge className="border-white/18 bg-white/12 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.22em] text-white/62">
                    Machine Analysis
                  </Badge>
                  <div>
                    <h2 className="text-[28px] font-medium tracking-[-1px] text-white drop-shadow-[0_10px_28px_rgba(14,18,28,0.22)]">
                      Tune StemSep to this machine
                    </h2>
                    <p className="mt-2 max-w-[620px] text-[14px] leading-6 tracking-[-0.22px] text-white/62">
                      This view now scores the machine against StemSep&apos;s own
                      guide-derived registry, runtime checks and VRAM guardrails,
                      so the verdict reflects what the app can actually run well.
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={closeAnalyzer}
                  className="rounded-[16px] border border-white/10 bg-white/8 p-2.5 text-white/55 transition-all hover:border-white/18 hover:bg-white/14 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {analysisState === "idle" && (
                <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                  <div className="min-w-0 rounded-[1.8rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.06))] p-6 shadow-[0_24px_60px_rgba(10,14,22,0.18)]">
                    <div className="flex items-start gap-4">
                      <div className="relative mt-1 flex h-16 w-16 shrink-0 items-center justify-center rounded-[1.4rem] border border-white/18 bg-white/10 shadow-[0_18px_40px_rgba(10,14,22,0.22)]">
                        <div className="absolute inset-2 rounded-[1rem] bg-[radial-gradient(circle,rgba(255,255,255,0.22),transparent_70%)]" />
                        <Cpu className="relative h-7 w-7 text-white/88" />
                      </div>
                      <div className="min-w-0">
                        <p className="text-[22px] font-medium tracking-[-0.7px] text-white">
                          Analyze this setup
                        </p>
                        <p className="mt-2 max-w-[440px] text-[13px] leading-6 text-white/58">
                          We check actual runtime readiness, hardware tier, guided
                          workflow coverage and verified model fit. The result is
                          based on the same internal rules the app uses for
                          separation decisions.
                        </p>
                      </div>
                    </div>

                    <div className="mt-5 grid gap-3 md:grid-cols-3">
                      {[
                        {
                          label: "Performance",
                          value: "Headroom for larger chains and higher context settings",
                        },
                        {
                          label: "Runtime",
                          value: "Whether the current Torch path can really use the machine",
                        },
                        {
                          label: "Guidance",
                          value: "Preset and workflow advice grounded in the guide-backed registry",
                        },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="min-w-0 rounded-[1.2rem] border border-white/12 bg-white/8 p-3"
                        >
                          <div className="text-[10px] uppercase tracking-[0.16em] text-white/42">
                            {item.label}
                          </div>
                          <div className="mt-2 break-words text-[13px] leading-6 text-white/70">
                            {item.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        void runAnalysis();
                      }}
                      className="mt-6 inline-flex items-center justify-center gap-2 rounded-[999px] border border-white/26 bg-white/18 px-5 py-3 text-[14px] font-medium tracking-[-0.25px] text-white shadow-[0_18px_40px_rgba(10,14,22,0.18)] transition-all duration-300 hover:scale-[1.02] hover:border-white/38 hover:bg-white/24 active:scale-95"
                    >
                      <Sparkles className="h-4 w-4" />
                      Run Analysis
                    </button>
                  </div>

                  <div className="min-w-0 rounded-[1.8rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.04))] p-5">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                      What you get
                    </div>
                    <div className="mt-4 space-y-3">
                      {[
                        "A readiness score tied to current runtime health, not just raw hardware specs.",
                        "A view of which guided workflows are actually supported, limited or blocked.",
                        "Recommendations that follow the same 4 GB / 6 GB / 10 GB VRAM breakpoints used elsewhere in the app.",
                      ].map((copy) => (
                        <div
                          key={copy}
                          className="min-w-0 rounded-[1.1rem] border border-white/10 bg-white/7 p-3"
                        >
                          <p className="break-words text-[13px] leading-6 text-white/68">
                            {copy}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {analysisState === "scanning" && (
                <div className="mt-6 rounded-[1.9rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.05))] p-6 sm:p-7">
                  <div className="flex flex-col items-center text-center">
                    <div className="relative flex h-24 w-24 items-center justify-center">
                      <div className="absolute inset-0 rounded-full border border-white/12 bg-[radial-gradient(circle,rgba(255,255,255,0.16),transparent_65%)] blur-[2px]" />
                      <div className="absolute inset-3 rounded-full border border-white/16" />
                      <Loader2 className="relative h-9 w-9 animate-spin text-white/80" />
                    </div>

                    <div className="mt-4 text-[24px] font-medium tracking-[-0.8px] text-white">
                      Reading your system profile
                    </div>
                    <p className="mt-2 max-w-[460px] text-[14px] leading-6 text-white/60">
                      Refreshing runtime diagnostics, hardware info and guide-backed
                      compatibility data for the workflows used in StemSep.
                    </p>

                    <div className="mt-6 w-full max-w-[460px]">
                      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em] text-white/42">
                        <span>Progress</span>
                        <span>{progress}%</span>
                      </div>
                      <div className="mt-2 h-3 overflow-hidden rounded-full border border-white/10 bg-white/7 p-[2px]">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(255,255,255,0.92),rgba(255,223,191,0.92),rgba(160,194,255,0.9))] shadow-[0_0_18px_rgba(255,255,255,0.28)] transition-all duration-100"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {analysisState === "done" && (
                <div className="mt-6 space-y-5">
                  <div className="grid gap-5 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
                    <div className="min-w-0 rounded-[1.9rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.06))] p-6 shadow-[0_24px_60px_rgba(10,14,22,0.18)]">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="text-[11px] uppercase tracking-[0.18em] text-white/42">
                            Overall readiness
                          </div>
                          <div
                            className={`mt-3 text-[56px] font-medium leading-none tracking-[-3px] ${scoreTone.text}`}
                          >
                            {analysis.overallScore}
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <Badge className={`px-3 py-1 text-[11px] font-medium ${scoreTone.chip}`}>
                              {analysis.tierLabel}
                            </Badge>
                            <Badge className="border-white/16 bg-white/10 px-3 py-1 text-[11px] font-medium text-white/72">
                              {analysis.accelerationLabel}
                            </Badge>
                          </div>
                        </div>

                        <div
                          className={`relative flex h-24 w-24 shrink-0 items-center justify-center rounded-full border border-white/16 bg-white/7 ${scoreTone.glow}`}
                        >
                          <div
                            className="absolute inset-[7px] rounded-full border-[6px] border-white/12"
                            style={{
                              background: `conic-gradient(rgba(255,255,255,0) ${Math.max(
                                0,
                                Math.min(100, analysis.overallScore),
                              )}%, rgba(255,255,255,0.14) 0%)`,
                            }}
                          />
                          <Monitor className={`relative h-6 w-6 ${scoreTone.accent}`} />
                        </div>
                      </div>

                      <div className="mt-5 flex items-start gap-3 rounded-[1.3rem] border border-white/12 bg-white/7 p-4">
                        {analysis.overallScore >= 60 ? (
                          <CheckCircle className={`mt-0.5 h-5 w-5 shrink-0 ${scoreTone.accent}`} />
                        ) : (
                          <AlertTriangle className={`mt-0.5 h-5 w-5 shrink-0 ${scoreTone.accent}`} />
                        )}
                        <div className="min-w-0">
                          <div className="text-[15px] font-medium tracking-[-0.3px] text-white">
                            {analysis.verdictTitle}
                          </div>
                          <div className="mt-1 text-[13px] leading-6 text-white/64">
                            {analysis.verdictText}
                          </div>
                        </div>
                      </div>

                      {(analysis.issues.length > 0 || error) && (
                        <div className="mt-4 space-y-2">
                          {[...analysis.issues, ...(error ? [error] : [])].map((issue) => (
                            <div
                              key={issue}
                              className="rounded-[1.1rem] border border-amber-300/18 bg-amber-300/10 p-3 text-[12px] leading-5 text-amber-50/90"
                            >
                              {issue}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="min-w-0 rounded-[1.9rem] border border-white/14 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.04))] p-5">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-white/70" />
                        <div className="text-[13px] font-medium tracking-[-0.2px] text-white">
                          Recommended approach
                        </div>
                      </div>
                      <div className="mt-4 space-y-3">
                        {analysis.recommendations.map((item) => (
                          <div
                            key={item}
                            className="rounded-[1.15rem] border border-white/12 bg-white/8 p-3"
                          >
                            <p className="break-words text-[13px] leading-6 text-white/66">
                              {item}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    {analysis.metrics.map((item) => {
                      const tone = toneClasses(item.score);

                      return (
                        <div
                          key={item.label}
                          className="min-w-0 rounded-[1.35rem] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.05))] p-4"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="text-[14px] font-medium tracking-[-0.2px] text-white">
                                {item.label}
                              </div>
                              <div className="mt-0.5 break-words text-[12px] leading-5 text-white/58">
                                {item.value}
                              </div>
                            </div>

                            <Badge className={`px-3 py-1 text-[11px] font-medium ${tone.chip}`}>
                              {item.score}/100
                            </Badge>
                          </div>

                          <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/8">
                            <div
                              className={`h-full rounded-full bg-gradient-to-r ${tone.bar}`}
                              style={{ width: `${item.score}%` }}
                            />
                          </div>

                          <p className="mt-3 break-words text-[12px] leading-5 text-white/56">
                            {item.hint}
                          </p>
                        </div>
                      );
                    })}
                  </div>

                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    {analysis.workflows.map((workflow) => {
                      const tone =
                        workflow.status === "supported"
                          ? toneClasses(84)
                          : workflow.status === "limited"
                            ? toneClasses(62)
                            : toneClasses(34);

                      return (
                        <div
                          key={workflow.label}
                          className="min-w-0 rounded-[1.25rem] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.05))] p-4"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-[14px] font-medium tracking-[-0.2px] text-white">
                              {workflow.label}
                            </div>
                            <Badge className={`px-3 py-1 text-[11px] font-medium ${tone.chip}`}>
                              {workflow.status}
                            </Badge>
                          </div>
                          <div className="mt-3 text-[12px] text-white/76">
                            {workflow.presetName || "No fit found"}
                          </div>
                          <p className="mt-2 break-words text-[12px] leading-5 text-white/56">
                            {workflow.reason}
                          </p>
                        </div>
                      );
                    })}
                  </div>

                  <div className="flex flex-col gap-3 sm:flex-row">
                    <button
                      type="button"
                      onClick={() => {
                        void runAnalysis();
                      }}
                      className="inline-flex items-center justify-center gap-2 rounded-[999px] border border-white/24 bg-white/16 px-5 py-3 text-[14px] font-medium tracking-[-0.2px] text-white transition-all duration-300 hover:scale-[1.01] hover:border-white/34 hover:bg-white/22 active:scale-95"
                    >
                      <Sparkles className="h-4 w-4" />
                      Run Again
                    </button>
                    <button
                      type="button"
                      onClick={closeAnalyzer}
                      className="inline-flex items-center justify-center rounded-[999px] border border-white/14 bg-white/8 px-5 py-3 text-[14px] tracking-[-0.2px] text-white/72 transition-all hover:border-white/22 hover:bg-white/12 hover:text-white"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}

              {loading && analysisState === "idle" && (
                <div className="mt-4 text-[12px] text-white/50">
                  Runtime information is still loading.
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
