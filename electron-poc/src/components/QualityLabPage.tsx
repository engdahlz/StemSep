import { useEffect, useMemo, useState } from "react"
import { FlaskConical, GitCompareArrows, Wand2 } from "lucide-react"
import { toast } from "sonner"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { PageShell } from "./PageShell"
import { Badge } from "./ui/badge"

function parseJsonOrThrow<T = any>(raw: string, label: string): T {
  try {
    return JSON.parse(raw) as T
  } catch (e: any) {
    throw new Error(`${label} is not valid JSON: ${e?.message || String(e)}`)
  }
}

const EXAMPLE_SCENARIOS = [
  {
    label: "Vocals",
    baselineName: "vocals-baseline",
    modelIdsCsv: "unwa-big-beta-5e,gabox-voc-fv4",
    configJson: JSON.stringify({ device: "auto", recipe_id: "recipe_vocal_denoise" }, null, 2),
    outputFilesJson: JSON.stringify(
      {
        vocals: "C:/datasets/reference/vocals.wav",
        instrumental: "C:/datasets/reference/instrumental.wav",
      },
      null,
      2,
    ),
  },
  {
    label: "Instrumental",
    baselineName: "instrumental-baseline",
    modelIdsCsv: "unwa-inst-v1e-plus,becruily-vocal",
    configJson: JSON.stringify(
      { device: "auto", recipe_id: "workflow_phase_fix_instrumental" },
      null,
      2,
    ),
    outputFilesJson: JSON.stringify(
      {
        instrumental: "C:/datasets/reference/instrumental.wav",
        vocals: "C:/datasets/reference/vocals.wav",
      },
      null,
      2,
    ),
  },
  {
    label: "Karaoke",
    baselineName: "karaoke-baseline",
    modelIdsCsv: "bs-roformer-karaoke-becruily,anvuew-karaoke",
    configJson: JSON.stringify({ device: "auto", recipe_id: "recipe_karaoke_fusion" }, null, 2),
    outputFilesJson: JSON.stringify(
      {
        no_vocals: "C:/datasets/reference/no_vocals.wav",
        vocals: "C:/datasets/reference/vocals.wav",
      },
      null,
      2,
    ),
  },
  {
    label: "Restoration",
    baselineName: "restoration-baseline",
    modelIdsCsv: "anvuew-dereverb-room,aufr33-denoise-aggressive",
    configJson: JSON.stringify(
      { device: "auto", recipe_id: "workflow_live_restoration" },
      null,
      2,
    ),
    outputFilesJson: JSON.stringify(
      {
        vocals: "C:/datasets/reference/restored_vocals.wav",
      },
      null,
      2,
    ),
  },
] as const

export default function QualityLabPage() {
  const [baselineName, setBaselineName] = useState("quality-baseline")
  const [modelIdsCsv, setModelIdsCsv] = useState("")
  const [configJson, setConfigJson] = useState("{\n  \"device\": \"auto\"\n}")
  const [outputFilesJson, setOutputFilesJson] = useState(
    "{\n  \"vocals\": \"C:/path/to/vocals.wav\",\n  \"instrumental\": \"C:/path/to/instrumental.wav\"\n}",
  )
  const [manifestPath, setManifestPath] = useState("")

  const [baselineManifestPath, setBaselineManifestPath] = useState("")
  const [candidateManifestPath, setCandidateManifestPath] = useState("")
  const [candidateName, setCandidateName] = useState("quality-candidate")
  const [candidateModelIdsCsv, setCandidateModelIdsCsv] = useState("")
  const [candidateConfigJson, setCandidateConfigJson] =
    useState("{\n  \"device\": \"auto\"\n}")
  const [candidateOutputFilesJson, setCandidateOutputFilesJson] = useState(
    "{\n  \"vocals\": \"C:/path/to/candidate-vocals.wav\",\n  \"instrumental\": \"C:/path/to/candidate-instrumental.wav\"\n}",
  )

  const [busy, setBusy] = useState(false)
  const [qualityProgress, setQualityProgress] = useState<any>(null)
  const [lastResult, setLastResult] = useState<any>(null)

  useEffect(() => {
    const offProgress = window.electronAPI?.onQualityProgress?.((msg) => {
      setQualityProgress(msg)
    })
    const offComplete = window.electronAPI?.onQualityComplete?.((msg) => {
      setQualityProgress(msg)
    })
    return () => {
      offProgress?.()
      offComplete?.()
    }
  }, [])

  const progressText = useMemo(() => {
    if (!qualityProgress) return "Idle"
    const p = Number(qualityProgress?.progress)
    const pct = Number.isFinite(p) ? `${Math.round(p)}%` : ""
    return `${qualityProgress?.stage || "quality"} ${pct} ${qualityProgress?.message || ""}`.trim()
  }, [qualityProgress])

  const comparison = lastResult?.comparison
  const baseline = lastResult?.baseline || lastResult
  const candidate = lastResult?.candidate

  const qualitySummary = useMemo(() => {
    const outputsCount = Array.isArray(baseline?.outputs)
      ? baseline.outputs.length
      : baseline?.output_hashes
        ? Object.keys(baseline.output_hashes).length
        : 0

    return {
      compatible:
        typeof comparison?.compatible === "boolean" ? comparison.compatible : null,
      qualityScore:
        typeof comparison?.quality_score === "number"
          ? comparison.quality_score
          : null,
      diffCount:
        typeof comparison?.difference_count === "number"
          ? comparison.difference_count
          : Array.isArray(comparison?.differences)
            ? comparison.differences.length
            : 0,
      outputMismatchCount:
        typeof comparison?.output_mismatch_count === "number"
          ? comparison.output_mismatch_count
          : 0,
      metricsDeltaEntries: comparison?.metrics_delta
        ? Object.entries(comparison.metrics_delta)
        : [],
      baselineModels: Array.isArray(baseline?.models) ? baseline.models : [],
      candidateModels: Array.isArray(candidate?.models) ? candidate.models : [],
      outputsCount,
      baselineHash: comparison?.baseline_manifest_hash || baseline?.manifest_hash || null,
      candidateHash:
        comparison?.candidate_manifest_hash || candidate?.manifest_hash || null,
    }
  }, [baseline, candidate, comparison])

  const differenceRows = Array.isArray(comparison?.differences)
    ? comparison.differences
    : []

  const applyScenario = (scenario: (typeof EXAMPLE_SCENARIOS)[number]) => {
    setBaselineName(scenario.baselineName)
    setModelIdsCsv(scenario.modelIdsCsv)
    setConfigJson(scenario.configJson)
    setOutputFilesJson(scenario.outputFilesJson)
    setCandidateName(`${scenario.baselineName}-candidate`)
    setCandidateModelIdsCsv(scenario.modelIdsCsv)
    setCandidateConfigJson(scenario.configJson)
    setCandidateOutputFilesJson(scenario.outputFilesJson)
  }

  const createBaseline = async () => {
    if (!window.electronAPI?.qualityBaselineCreate) return
    setBusy(true)
    try {
      const modelIds = modelIdsCsv
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
      const payload: Record<string, any> = {
        name: baselineName || undefined,
        model_ids: modelIds,
        config: parseJsonOrThrow(configJson, "Config JSON"),
        output_files: parseJsonOrThrow(outputFilesJson, "Output files JSON"),
      }
      if (manifestPath.trim()) {
        payload.manifest_path = manifestPath.trim()
      }
      const result = await window.electronAPI.qualityBaselineCreate(payload)
      setLastResult(result)
      if (manifestPath.trim()) {
        setBaselineManifestPath(manifestPath.trim())
      }
      toast.success("Quality baseline created")
    } catch (e: any) {
      toast.error(e?.message || "Failed to create quality baseline")
    } finally {
      setBusy(false)
    }
  }

  const compareManifests = async () => {
    if (!window.electronAPI?.qualityCompare) return
    if (!baselineManifestPath.trim()) {
      toast.error("Provide a baseline manifest path")
      return
    }
    setBusy(true)
    try {
      const payload: Record<string, any> = {
        baseline_manifest_path: baselineManifestPath.trim(),
      }

      if (candidateManifestPath.trim()) {
        payload.candidate_manifest_path = candidateManifestPath.trim()
      } else {
        payload.candidate = {
          name: candidateName || undefined,
          model_ids: candidateModelIdsCsv
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          config: parseJsonOrThrow(candidateConfigJson, "Candidate config JSON"),
          output_files: parseJsonOrThrow(
            candidateOutputFilesJson,
            "Candidate output files JSON",
          ),
        }
      }

      const result = await window.electronAPI.qualityCompare(payload)
      setLastResult(result)
      const compatible = !!result?.comparison?.compatible
      if (compatible) toast.success("Quality compare passed")
      else toast.error("Quality compare found differences")
    } catch (e: any) {
      toast.error(e?.message || "Failed to compare quality manifests")
    } finally {
      setBusy(false)
    }
  }

  return (
    <PageShell>
      <div className="flex flex-col max-w-6xl mx-auto w-full p-6 gap-6">
        <div>
          <h1 className="text-3xl font-bold">Quality Lab</h1>
          <p className="text-muted-foreground mt-2">
            Internal QA workspace for reproducible baselines, candidate runs, and
            manifest diffs before promoting workflows into Simple mode.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Reference Scenarios</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {EXAMPLE_SCENARIOS.map((scenario) => (
              <Button
                key={scenario.label}
                variant="outline"
                size="sm"
                onClick={() => applyScenario(scenario)}
              >
                {scenario.label}
              </Button>
            ))}
          </CardContent>
        </Card>

        <div className="grid gap-6 xl:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Create Baseline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={baselineName}
                onChange={(e) => setBaselineName(e.target.value)}
                placeholder="Baseline name"
              />
              <Input
                value={modelIdsCsv}
                onChange={(e) => setModelIdsCsv(e.target.value)}
                placeholder="Model IDs (comma-separated)"
              />
              <Input
                value={manifestPath}
                onChange={(e) => setManifestPath(e.target.value)}
                placeholder="Optional manifest path"
              />
              <div className="grid grid-cols-1 gap-3">
                <textarea
                  className="min-h-40 rounded-md border border-border bg-background p-3 text-sm"
                  value={configJson}
                  onChange={(e) => setConfigJson(e.target.value)}
                  spellCheck={false}
                />
                <textarea
                  className="min-h-40 rounded-md border border-border bg-background p-3 text-sm"
                  value={outputFilesJson}
                  onChange={(e) => setOutputFilesJson(e.target.value)}
                  spellCheck={false}
                />
              </div>
              <Button disabled={busy} onClick={createBaseline}>
                <FlaskConical className="mr-2 h-4 w-4" />
                {busy ? "Working..." : "Create Baseline"}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Compare Candidate</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={baselineManifestPath}
                onChange={(e) => setBaselineManifestPath(e.target.value)}
                placeholder="Baseline manifest path"
              />
              <Input
                value={candidateManifestPath}
                onChange={(e) => setCandidateManifestPath(e.target.value)}
                placeholder="Candidate manifest path (optional)"
              />
              {!candidateManifestPath.trim() && (
                <div className="space-y-3 rounded-md border border-dashed p-3">
                  <div className="text-sm font-medium">
                    Inline candidate manifest
                  </div>
                  <Input
                    value={candidateName}
                    onChange={(e) => setCandidateName(e.target.value)}
                    placeholder="Candidate name"
                  />
                  <Input
                    value={candidateModelIdsCsv}
                    onChange={(e) => setCandidateModelIdsCsv(e.target.value)}
                    placeholder="Candidate model IDs (comma-separated)"
                  />
                  <textarea
                    className="min-h-28 rounded-md border border-border bg-background p-3 text-sm"
                    value={candidateConfigJson}
                    onChange={(e) => setCandidateConfigJson(e.target.value)}
                    spellCheck={false}
                  />
                  <textarea
                    className="min-h-28 rounded-md border border-border bg-background p-3 text-sm"
                    value={candidateOutputFilesJson}
                    onChange={(e) => setCandidateOutputFilesJson(e.target.value)}
                    spellCheck={false}
                  />
                </div>
              )}
              <Button disabled={busy} onClick={compareManifests}>
                <GitCompareArrows className="mr-2 h-4 w-4" />
                {busy ? "Working..." : "Compare"}
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Live Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground">{progressText}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quality Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-3">
              <div className="rounded-md border p-3 min-w-36">
                <div className="text-xs text-muted-foreground">Compatibility</div>
                <div className="mt-1">
                  {qualitySummary.compatible === null ? (
                    <Badge variant="secondary">N/A</Badge>
                  ) : qualitySummary.compatible ? (
                    <Badge>Pass</Badge>
                  ) : (
                    <Badge variant="destructive">Fail</Badge>
                  )}
                </div>
              </div>
              <div className="rounded-md border p-3 min-w-36">
                <div className="text-xs text-muted-foreground">Quality Score</div>
                <div className="mt-1 text-lg font-semibold">
                  {qualitySummary.qualityScore == null
                    ? "N/A"
                    : qualitySummary.qualityScore}
                </div>
              </div>
              <div className="rounded-md border p-3 min-w-36">
                <div className="text-xs text-muted-foreground">Differences</div>
                <div className="mt-1 text-lg font-semibold">
                  {qualitySummary.diffCount}
                </div>
              </div>
              <div className="rounded-md border p-3 min-w-36">
                <div className="text-xs text-muted-foreground">Output mismatches</div>
                <div className="mt-1 text-lg font-semibold">
                  {qualitySummary.outputMismatchCount}
                </div>
              </div>
              <div className="rounded-md border p-3 min-w-36">
                <div className="text-xs text-muted-foreground">Outputs</div>
                <div className="mt-1 text-lg font-semibold">
                  {qualitySummary.outputsCount}
                </div>
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-md border p-3 space-y-3">
                <div className="text-sm font-medium">Baseline</div>
                <div className="flex flex-wrap gap-2">
                  {qualitySummary.baselineModels.length > 0 ? (
                    qualitySummary.baselineModels.map((model: any) => (
                      <Badge key={model.id || model.name} variant="secondary">
                        {model.id || model.name}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">No baseline loaded</span>
                  )}
                </div>
                {qualitySummary.baselineHash && (
                  <div className="text-xs text-muted-foreground break-all">
                    Hash: {qualitySummary.baselineHash}
                  </div>
                )}
              </div>

              <div className="rounded-md border p-3 space-y-3">
                <div className="text-sm font-medium">Candidate</div>
                <div className="flex flex-wrap gap-2">
                  {qualitySummary.candidateModels.length > 0 ? (
                    qualitySummary.candidateModels.map((model: any) => (
                      <Badge key={model.id || model.name} variant="secondary">
                        {model.id || model.name}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">
                      Candidate shown after compare
                    </span>
                  )}
                </div>
                {qualitySummary.candidateHash && (
                  <div className="text-xs text-muted-foreground break-all">
                    Hash: {qualitySummary.candidateHash}
                  </div>
                )}
              </div>
            </div>

            {qualitySummary.metricsDeltaEntries.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Wand2 className="h-4 w-4 text-primary" />
                  Metric deltas
                </div>
                <div className="flex flex-wrap gap-2">
                  {qualitySummary.metricsDeltaEntries.map(([key, value]) => (
                    <Badge key={key} variant="secondary">
                      {key}: {Number(value).toFixed(3)}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Detailed Differences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {differenceRows.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                No structured differences yet.
              </div>
            ) : (
              differenceRows.map((difference: any, index: number) => (
                <div key={`${difference.type}-${index}`} className="rounded-md border p-3 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">{difference.type || "difference"}</Badge>
                    {difference.id && <Badge variant="secondary">{difference.id}</Badge>}
                    {difference.label && (
                      <Badge variant="secondary">{difference.label}</Badge>
                    )}
                  </div>
                  <div className="grid gap-2 md:grid-cols-2 text-xs">
                    <div className="rounded bg-muted/40 p-2 break-all">
                      <div className="text-muted-foreground mb-1">Baseline</div>
                      <div>{String(difference.baseline ?? "N/A")}</div>
                    </div>
                    <div className="rounded bg-muted/40 p-2 break-all">
                      <div className="text-muted-foreground mb-1">Candidate</div>
                      <div>{String(difference.candidate ?? "N/A")}</div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Last Result</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs rounded-md bg-muted p-3 overflow-auto max-h-[420px]">
              {JSON.stringify(lastResult, null, 2)}
            </pre>
          </CardContent>
        </Card>
      </div>
    </PageShell>
  )
}
