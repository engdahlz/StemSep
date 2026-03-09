import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Cpu,
  Gauge,
  HardDrive,
  Loader2,
  MemoryStick,
  Monitor,
  X,
} from "lucide-react";

import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";

type AnalysisState = "idle" | "scanning" | "done";

type SystemMetric = {
  label: string;
  value: string;
  score: number;
  icon: typeof Cpu;
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

export function MachineAnalyzer() {
  const { info: runtimeInfo } = useSystemRuntimeInfo();
  const [isOpen, setIsOpen] = useState(false);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [progress, setProgress] = useState(0);

  const systemMetrics = useMemo<SystemMetric[]>(() => {
    const gpuInfo = runtimeInfo?.gpu;
    const system = gpuInfo?.system_info;
    const activeGpu =
      gpuInfo?.gpus?.find((gpu: any) => gpu.recommended) || gpuInfo?.gpus?.[0];

    const cpuCount = Number(system?.cpu_count || 0);
    const ramGb = Number(system?.memory_total_gb || 0);
    const gpuVram = Number(activeGpu?.memory_gb || 0);

    const cpuScore = clamp(cpuCount * 9, 20, 100);
    const ramScore = clamp(ramGb * 6, 20, 100);
    const gpuScore = activeGpu
      ? clamp(gpuVram * 10 + 15, 25, 100)
      : clamp(ramGb * 3, 20, 48);
    const runtimeScore =
      gpuInfo?.has_cuda || activeGpu
        ? clamp(72 + Math.min(gpuVram, 8) * 3, 55, 100)
        : 42;
    const platformScore = system?.platform?.toLowerCase().includes("win")
      ? 82
      : 70;

    return [
      {
        label: "CPU Cores",
        value: cpuCount ? `${cpuCount} cores` : "Unknown",
        score: cpuScore,
        icon: Cpu,
      },
      {
        label: "Memory",
        value: ramGb ? `${ramGb} GB RAM` : "Unknown",
        score: ramScore,
        icon: MemoryStick,
      },
      {
        label: "GPU",
        value: activeGpu?.name || "CPU only",
        score: gpuScore,
        icon: Monitor,
      },
      {
        label: "Runtime",
        value: gpuInfo?.has_cuda || activeGpu ? "GPU-ready" : "CPU fallback",
        score: runtimeScore,
        icon: Gauge,
      },
      {
        label: "Platform",
        value: system?.platform || "Unknown",
        score: platformScore,
        icon: HardDrive,
      },
    ];
  }, [runtimeInfo]);

  const overallScore = useMemo(() => {
    if (!systemMetrics.length) return 0;
    const total = systemMetrics.reduce((sum, item) => sum + item.score, 0);
    return Math.round(total / systemMetrics.length);
  }, [systemMetrics]);

  const verdict = useMemo(() => {
    if (overallScore >= 80) {
      return {
        text: "Excellent — your machine should handle heavy separation chains comfortably.",
        icon: CheckCircle,
        color: "text-emerald-300",
      };
    }
    if (overallScore >= 60) {
      return {
        text: "Good — most models should run smoothly, with some limits on larger chains.",
        icon: CheckCircle,
        color: "text-amber-300",
      };
    }
    return {
      text: "Limited — prefer lighter models or CPU-safe workflows for stable results.",
      icon: AlertTriangle,
      color: "text-amber-300",
    };
  }, [overallScore]);

  const getScoreColor = (score: number) => {
    if (score >= 75) return "text-emerald-300";
    if (score >= 50) return "text-amber-300";
    return "text-rose-300";
  };

  const getBarColor = (score: number) => {
    if (score >= 75) return "bg-emerald-400/80";
    if (score >= 50) return "bg-amber-400/80";
    return "bg-rose-400/80";
  };

  const runAnalysis = () => {
    setAnalysisState("scanning");
    setProgress(0);

    const interval = window.setInterval(() => {
      setProgress((value) => {
        if (value >= 100) {
          window.clearInterval(interval);
          return 100;
        }
        return value + 2;
      });
    }, 40);

    window.setTimeout(() => {
      window.clearInterval(interval);
      setProgress(100);
      setAnalysisState("done");
    }, 2200);
  };

  const VerdictIcon = verdict.icon;

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed right-5 top-5 z-30 rounded-[14px] border border-white/30 bg-white/20 p-2.5 text-white shadow-lg shadow-black/10 backdrop-blur-md transition-all duration-300 hover:scale-105 hover:border-white/50 hover:bg-white/30 active:scale-95"
        aria-label="Open machine analyzer"
      >
        <Monitor className="h-5 w-5" />
      </button>

      {isOpen && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-50 bg-black/30 backdrop-blur-xl"
            onClick={() => {
              setIsOpen(false);
              setAnalysisState("idle");
              setProgress(0);
            }}
            aria-label="Close machine analyzer overlay"
          />
          <div className="fixed left-1/2 top-1/2 z-[60] max-h-[85vh] w-[460px] max-w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-3xl border border-white/30 bg-white/15 text-white shadow-2xl shadow-black/30 backdrop-blur-2xl">
            <div className="pointer-events-none absolute inset-0 rounded-3xl bg-gradient-to-br from-white/10 via-transparent to-white/5" />
            <div className="pointer-events-none absolute inset-0 rounded-3xl shadow-inner shadow-white/10" />
            <div className="relative">
              <div className="flex items-start justify-between gap-4 px-6 pb-4 pt-6">
                <div>
                  <h2 className="text-[18px] tracking-[-0.5px] text-white drop-shadow-lg">
                    Machine Analysis
                  </h2>
                  <p className="mt-0.5 text-[13px] tracking-[-0.2px] text-white/50 drop-shadow">
                    Check your system's readiness for stem separation
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setIsOpen(false);
                    setAnalysisState("idle");
                    setProgress(0);
                  }}
                  className="rounded-xl p-2 text-white/50 transition-all hover:bg-white/15 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {analysisState === "idle" && (
                <div className="flex flex-col items-center px-6 pb-6 pt-2">
                  <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/20 bg-white/10 shadow-lg shadow-black/10 backdrop-blur-md">
                    <Cpu className="h-7 w-7 text-white/50" />
                  </div>
                  <p className="mb-6 text-center text-[14px] leading-6 tracking-[-0.2px] text-white/60 drop-shadow">
                    Analyze your hardware to see which models
                    <br />
                    run best on your machine
                  </p>
                  <button
                    type="button"
                    onClick={runAnalysis}
                    className="rounded-[38px] border border-white/30 bg-white/20 px-6 py-2.5 text-[14px] tracking-[-0.3px] text-white shadow-lg shadow-black/20 backdrop-blur-md transition-all duration-300 hover:scale-105 hover:border-white/50 hover:bg-white/30 active:scale-95"
                  >
                    Run Analysis
                  </button>
                </div>
              )}

              {analysisState === "scanning" && (
                <div className="flex flex-col items-center px-6 pb-6 pt-2">
                  <Loader2 className="mb-5 h-10 w-10 animate-spin text-white/60 drop-shadow-lg" />
                  <p className="mb-4 text-[14px] tracking-[-0.2px] text-white/70 drop-shadow">
                    Scanning hardware…
                  </p>
                  <div className="h-1.5 w-full overflow-hidden rounded-full border border-white/5 bg-white/10">
                    <div
                      className="h-full rounded-full bg-white/80 shadow-[0_0_8px_rgba(255,255,255,0.4)] transition-all duration-100"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="mt-2 text-xs text-white/42">{progress}%</p>
                </div>
              )}

              {analysisState === "done" && (
                <div className="px-6 pb-6 pt-1">
                  <div className="mb-4 flex items-center justify-between rounded-2xl border border-white/20 bg-white/10 p-4 shadow-lg shadow-black/10 backdrop-blur-md">
                    <div>
                      <p className="mb-1 text-[12px] tracking-[-0.1px] text-white/50 drop-shadow">
                        Overall Score
                      </p>
                      <p
                        className={`text-[32px] tracking-[-1.5px] drop-shadow-lg ${getScoreColor(overallScore)}`}
                      >
                        {overallScore}
                      </p>
                    </div>
                    <div className="relative flex h-16 w-16 items-center justify-center rounded-full border-[3px] border-white/15">
                      <Gauge
                        className={`h-5 w-5 ${getScoreColor(overallScore)}`}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    {systemMetrics.map((item) => {
                      const Icon = item.icon;
                      return (
                        <div
                          key={item.label}
                          className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/8 px-4 py-3 backdrop-blur-sm"
                        >
                          <Icon className="h-4 w-4 shrink-0 text-white/40 drop-shadow" />
                          <div className="min-w-0 flex-1">
                            <div className="mb-1.5 flex items-center justify-between gap-3">
                              <span className="text-[12px] tracking-[-0.1px] text-white/55">
                                {item.label}
                              </span>
                              <span className="truncate text-[12px] tracking-[-0.1px] text-white/80 drop-shadow">
                                {item.value}
                              </span>
                            </div>
                            <div className="h-1 overflow-hidden rounded-full bg-white/10">
                              <div
                                className={`h-full rounded-full ${getBarColor(item.score)}`}
                                style={{ width: `${item.score}%` }}
                              />
                            </div>
                          </div>
                          <span className={`ml-1 text-[12px] ${getScoreColor(item.score)}`}>
                            {item.score}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-white/15 bg-white/8 px-4 py-3 backdrop-blur-sm">
                    <VerdictIcon
                      className={`mt-0.5 h-4 w-4 shrink-0 ${verdict.color}`}
                    />
                    <p className="text-[13px] leading-[1.5] tracking-[-0.2px] text-white/60 drop-shadow">
                      {verdict.text}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={runAnalysis}
                    className="mt-4 w-full rounded-[38px] border border-white/25 bg-white/15 px-4 py-2.5 text-[13px] tracking-[-0.2px] text-white/70 shadow-lg shadow-black/10 backdrop-blur-md transition-all duration-300 hover:scale-[1.02] hover:border-white/40 hover:bg-white/25 hover:text-white active:scale-95"
                  >
                    Run Again
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
