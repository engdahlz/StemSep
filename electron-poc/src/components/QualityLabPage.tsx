import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { PageShell } from "./PageShell";

function parseJsonOrThrow<T = any>(raw: string, label: string): T {
  try {
    return JSON.parse(raw) as T;
  } catch (e: any) {
    throw new Error(`${label} is not valid JSON: ${e?.message || String(e)}`);
  }
}

export default function QualityLabPage() {
  const [baselineName, setBaselineName] = useState("quality-baseline");
  const [modelIdsCsv, setModelIdsCsv] = useState("");
  const [configJson, setConfigJson] = useState("{\n  \"device\": \"auto\"\n}");
  const [outputFilesJson, setOutputFilesJson] = useState(
    "{\n  \"vocals\": \"C:/path/to/vocals.wav\",\n  \"instrumental\": \"C:/path/to/instrumental.wav\"\n}",
  );
  const [manifestPath, setManifestPath] = useState("");

  const [baselineManifestPath, setBaselineManifestPath] = useState("");
  const [candidateManifestPath, setCandidateManifestPath] = useState("");

  const [busy, setBusy] = useState(false);
  const [qualityProgress, setQualityProgress] = useState<any>(null);
  const [lastResult, setLastResult] = useState<any>(null);

  useEffect(() => {
    const offProgress = window.electronAPI?.onQualityProgress?.((msg) => {
      setQualityProgress(msg);
    });
    const offComplete = window.electronAPI?.onQualityComplete?.((msg) => {
      setQualityProgress(msg);
    });
    return () => {
      offProgress?.();
      offComplete?.();
    };
  }, []);

  const progressText = useMemo(() => {
    if (!qualityProgress) return "Idle";
    const p = Number(qualityProgress?.progress);
    const pct = Number.isFinite(p) ? `${Math.round(p)}%` : "";
    return `${qualityProgress?.stage || "quality"} ${pct} ${qualityProgress?.message || ""}`.trim();
  }, [qualityProgress]);

  const createBaseline = async () => {
    if (!window.electronAPI?.qualityBaselineCreate) return;
    setBusy(true);
    try {
      const modelIds = modelIdsCsv
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const payload: Record<string, any> = {
        name: baselineName || undefined,
        model_ids: modelIds,
        config: parseJsonOrThrow(configJson, "Config JSON"),
        output_files: parseJsonOrThrow(outputFilesJson, "Output files JSON"),
      };
      if (manifestPath.trim()) {
        payload.manifest_path = manifestPath.trim();
      }
      const result = await window.electronAPI.qualityBaselineCreate(payload);
      setLastResult(result);
      toast.success("Quality baseline created");
    } catch (e: any) {
      toast.error(e?.message || "Failed to create quality baseline");
    } finally {
      setBusy(false);
    }
  };

  const compareManifests = async () => {
    if (!window.electronAPI?.qualityCompare) return;
    if (!baselineManifestPath.trim() || !candidateManifestPath.trim()) {
      toast.error("Provide both baseline and candidate manifest paths");
      return;
    }
    setBusy(true);
    try {
      const result = await window.electronAPI.qualityCompare({
        baseline_manifest_path: baselineManifestPath.trim(),
        candidate_manifest_path: candidateManifestPath.trim(),
      });
      setLastResult(result);
      const compatible = !!result?.comparison?.compatible;
      if (compatible) toast.success("Quality compare passed");
      else toast.error("Quality compare found differences");
    } catch (e: any) {
      toast.error(e?.message || "Failed to compare quality manifests");
    } finally {
      setBusy(false);
    }
  };

  return (
    <PageShell>
      <div className="flex flex-col max-w-5xl mx-auto w-full p-6 gap-6">
        <div>
          <h1 className="text-3xl font-bold">Quality Lab</h1>
          <p className="text-muted-foreground mt-2">
            Developer workflow for reproducible quality baselines and manifest
            comparisons.
          </p>
        </div>

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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <textarea
                className="min-h-56 rounded-md border border-border bg-background p-3 text-sm"
                value={configJson}
                onChange={(e) => setConfigJson(e.target.value)}
                spellCheck={false}
              />
              <textarea
                className="min-h-56 rounded-md border border-border bg-background p-3 text-sm"
                value={outputFilesJson}
                onChange={(e) => setOutputFilesJson(e.target.value)}
                spellCheck={false}
              />
            </div>
            <Button disabled={busy} onClick={createBaseline}>
              {busy ? "Working..." : "Create Baseline"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Compare Manifests</CardTitle>
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
              placeholder="Candidate manifest path"
            />
            <Button disabled={busy} onClick={compareManifests}>
              {busy ? "Working..." : "Compare"}
            </Button>
          </CardContent>
        </Card>

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
  );
}

