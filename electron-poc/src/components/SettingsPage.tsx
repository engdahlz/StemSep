import { useEffect, useMemo, useState } from "react";
import { useStore } from "../stores/useStore";
import { useTheme } from "../contexts/ThemeContext";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { PageShell } from "./PageShell";
import {
  Monitor,
  Moon,
  Sun,
  FolderOpen,
  RefreshCw,
  Volume2,
  Settings2,
  HardDrive,
} from "lucide-react";
import { useSystemRuntimeInfo } from "../hooks/useSystemRuntimeInfo";
import { RuntimeDoctorCard } from "./RuntimeDoctorCard";
import { normalizeModel } from "../lib/models/normalizeModel";
import type { CatalogStatus } from "../types/modelCatalog";

type Tab = "general" | "audio" | "advanced";
type CudaDevice = { id: string; name?: string; index?: number };

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("general");
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [hasHfToken, setHasHfToken] = useState(false);
  const [catalogStatus, setCatalogStatus] = useState<CatalogStatus | null>(null);
  const { info: runtimeInfo } = useSystemRuntimeInfo();

  const { theme, setTheme } = useTheme();
  const {
    settings,
    setTheme: setStoreTheme,
    setDefaultOutputDir,
    setModelsDir,
    setModels,
    setDefaultModel,
    setAdvancedSettings,
    models,
  } = useStore();

  const gpuInfo = runtimeInfo?.gpu;
  const cudaDevices = useMemo<CudaDevice[]>(() => {
    const gpus = Array.isArray(gpuInfo?.gpus) ? gpuInfo.gpus : [];
    return gpus
      .filter(
        (g: any) => typeof g?.id === "string" && g.id.startsWith("cuda:"),
      )
      .map((g: any) => ({ id: g.id, name: g.name, index: g.index }));
  }, [gpuInfo?.gpus]);

  const hasCuda = useMemo(() => {
    return !!gpuInfo?.has_cuda || cudaDevices.length > 0;
  }, [cudaDevices.length, gpuInfo?.has_cuda]);

  useEffect(() => {
    const load = async () => {
      try {
        const cfg = await window.electronAPI.getAppConfig?.();
        const token = cfg?.hfToken;
        setHasHfToken(typeof token === "string" && token.trim().length > 0);
        const status = await window.electronAPI.getCatalogStatus?.();
        setCatalogStatus(status || null);
      } catch {
        setHasHfToken(false);
        setCatalogStatus(null);
      }
    };

    if (activeTab === "advanced") {
      load();
    }
  }, [activeTab]);

  const handleThemeChange = (newTheme: "light" | "dark" | "system") => {
    setTheme(newTheme);
    setStoreTheme(newTheme);
  };

  const handleBrowseOutputDir = async () => {
    try {
      const result = await window.electronAPI.selectOutputDirectory();
      if (result) {
        setDefaultOutputDir(result);
      }
    } catch (error) {
      console.error("Failed to open directory dialog:", error);
    }
  };

  const handleBrowseModelsDir = async () => {
    try {
      const result = await window.electronAPI.selectModelsDirectory?.();
      if (result) {
        const newPath = result;
        const applied = await window.electronAPI.setModelsDir?.(newPath);
        setModelsDir(newPath);
        if (Array.isArray(applied?.models)) {
          setModels(applied.models.map(normalizeModel));
        } else if (window.electronAPI?.getCatalog) {
          const refreshed = await window.electronAPI.getCatalog();
          if (Array.isArray(refreshed?.models)) {
            setModels(refreshed.models.map(normalizeModel));
          }
        }
      }
    } catch (error) {
      console.error("Failed to open directory dialog:", error);
      alert(`Failed to apply models directory: ${String(error)}`);
    }
  };

  return (
    <PageShell>
      <div className="flex flex-col max-w-4xl mx-auto w-full p-6 gap-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Settings2 className="w-8 h-8" />
            Settings
          </h1>
          <p className="text-muted-foreground mt-2">
            Configure application preferences and defaults.
          </p>
        </div>

        {/* Custom Tabs */}
        <div className="flex gap-2 border-b border-border">
          <button
            onClick={() => setActiveTab("general")}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "general"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            General
          </button>
          <button
            onClick={() => setActiveTab("audio")}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "audio"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            Audio & Models
          </button>
          <button
            onClick={() => setActiveTab("advanced")}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "advanced"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            Advanced
          </button>
        </div>

        <div className="space-y-6">
          {/* General Tab */}
          {activeTab === "general" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Monitor className="w-5 h-5" />
                    Appearance
                  </CardTitle>
                  <CardDescription>
                    Customize how the application looks.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <button
                      onClick={() => handleThemeChange("light")}
                      className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${
                        theme === "light"
                          ? "border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm"
                          : "border-border hover:border-primary/50 hover:bg-accent/50"
                      }`}
                    >
                      <Sun
                        className={`w-8 h-8 ${theme === "light" ? "text-primary fill-primary/20" : "text-muted-foreground"}`}
                      />
                      <span
                        className={`font-medium ${theme === "light" ? "text-primary" : "text-muted-foreground"}`}
                      >
                        Light
                      </span>
                    </button>
                    <button
                      onClick={() => handleThemeChange("dark")}
                      className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${
                        theme === "dark"
                          ? "border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm"
                          : "border-border hover:border-primary/50 hover:bg-accent/50"
                      }`}
                    >
                      <Moon
                        className={`w-8 h-8 ${theme === "dark" ? "text-primary fill-primary/20" : "text-muted-foreground"}`}
                      />
                      <span
                        className={`font-medium ${theme === "dark" ? "text-primary" : "text-muted-foreground"}`}
                      >
                        Dark
                      </span>
                    </button>
                    <button
                      onClick={() => handleThemeChange("system")}
                      className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${
                        theme === "system"
                          ? "border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm"
                          : "border-border hover:border-primary/50 hover:bg-accent/50"
                      }`}
                    >
                      <Monitor
                        className={`w-8 h-8 ${theme === "system" ? "text-primary" : "text-muted-foreground"}`}
                      />
                      <span
                        className={`font-medium ${theme === "system" ? "text-primary" : "text-muted-foreground"}`}
                      >
                        System
                      </span>
                    </button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FolderOpen className="w-5 h-5" />
                    Default Output
                  </CardTitle>
                  <CardDescription>
                    Set the default directory where separated stems will be
                    saved.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      value={settings.defaultOutputDir || ""}
                      readOnly
                      placeholder="Select a directory..."
                      className="font-mono text-sm"
                    />
                    <Button onClick={handleBrowseOutputDir} variant="outline">
                      Browse
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <HardDrive className="w-5 h-5" />
                    Models Directory
                  </CardTitle>
                  <CardDescription>
                    Where AI models are downloaded and stored.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      value={
                        settings.modelsDir || "~/.stemsep/models (default)"
                      }
                      readOnly
                      placeholder="Using default location..."
                      className="font-mono text-sm"
                    />
                    <Button onClick={handleBrowseModelsDir} variant="outline">
                      Browse
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Changes apply immediately. Existing models are not moved
                    automatically.
                  </p>
                </CardContent>
              </Card>
            </>
          )}

          {/* Audio Tab */}
          {activeTab === "audio" && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="w-5 h-5" />
                  Default Model
                </CardTitle>
                <CardDescription>
                  Choose which model is selected by default when starting the
                  app.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <select
                  value={settings.defaultModelId || ""}
                  onChange={(e) => setDefaultModel(e.target.value)}
                  className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                >
                  <option value="">-- Select a default model --</option>
                  {models
                    .filter((m) => m.installed)
                    .map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.category})
                      </option>
                    ))}
                </select>
                <p className="text-xs text-muted-foreground mt-2">
                  Only installed models are shown.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Advanced Tab */}
          {activeTab === "advanced" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <HardDrive className="w-5 h-5" />
                    System
                  </CardTitle>
                  <CardDescription>
                    Advanced system configurations.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <h4 className="font-medium">Reset Settings</h4>
                      <p className="text-sm text-muted-foreground">
                        Restore all settings to their default values.
                      </p>
                    </div>
                    <Button
                      variant="destructive"
                      onClick={() => {
                        if (
                          confirm(
                            "Are you sure you want to reset all settings?",
                          )
                        ) {
                          setTheme("system");
                          setStoreTheme("system");
                          setDefaultOutputDir("");
                          setDefaultModel("");
                        }
                      }}
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Reset
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCw className="w-5 h-5" />
                    Remote Catalog
                  </CardTitle>
                  <CardDescription>
                    StemSep refreshes the signed remote runtime catalog at startup and falls back to cache or the bundled bootstrap catalog if needed.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded-xl border border-border p-4">
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Status
                      </div>
                      <div className="mt-2 text-sm font-medium">
                        {catalogStatus?.fallback_kind || "Unavailable"}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {catalogStatus?.stale
                          ? "Using a fallback catalog snapshot."
                          : "Using the latest verified remote catalog."}
                      </div>
                    </div>
                    <div className="rounded-xl border border-border p-4">
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Signature
                      </div>
                      <div className="mt-2 text-sm font-medium">
                        {catalogStatus?.signature_valid ? "Verified" : "Not verified"}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground break-all">
                        {catalogStatus?.source_url || "No active catalog source"}
                      </div>
                    </div>
                  </div>
                  <div className="rounded-xl border border-border p-4 text-xs text-muted-foreground space-y-1">
                    <div>Revision: <span className="font-mono">{catalogStatus?.active_revision || "n/a"}</span></div>
                    <div>Fetched: <span className="font-mono">{catalogStatus?.fetched_at || "n/a"}</span></div>
                    <div>Active path: <span className="font-mono break-all">{catalogStatus?.active_path || "n/a"}</span></div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={async () => {
                        const status = await window.electronAPI.refreshCatalog?.();
                        setCatalogStatus(status || null);
                        const refreshed = await window.electronAPI.getCatalog?.();
                        if (Array.isArray(refreshed?.models)) {
                          setModels(refreshed.models.map(normalizeModel));
                        }
                      }}
                    >
                      Refresh Catalog
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <RuntimeDoctorCard />

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="w-5 h-5" />
                    Processing Device
                  </CardTitle>
                  <CardDescription>
                    Choose the default device for separation. Auto will use GPU
                    when available.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <label className="text-sm font-medium">Default Device</label>
                  <select
                    value={settings.advancedSettings?.device || "auto"}
                    onChange={(e) => {
                      const next = e.target.value as any;

                      // If user chooses legacy "cuda", normalize to a specific CUDA device when known.
                      if (next === "cuda") {
                        const preferred =
                          settings.advancedSettings?.preferredCudaDevice;
                        const fallback = cudaDevices[0]?.id || "cuda:0";
                        setAdvancedSettings({
                          device: preferred || fallback,
                          preferredCudaDevice: preferred || fallback,
                        });
                        return;
                      }

                      // If user chooses a specific CUDA index, persist it as preferredCudaDevice too.
                      if (
                        typeof next === "string" &&
                        next.startsWith("cuda:")
                      ) {
                        setAdvancedSettings({
                          device: next,
                          preferredCudaDevice: next,
                        });
                        return;
                      }

                      // CPU / auto: just save device. preferredCudaDevice remains unchanged.
                      setAdvancedSettings({ device: next });
                    }}
                    className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                  >
                    <option value="auto">Auto (Recommended)</option>
                    <option value="cpu">CPU</option>
                    <option value="cuda" disabled={!hasCuda}>
                      GPU (CUDA){!hasCuda ? " (Not detected)" : ""}
                    </option>
                    {cudaDevices.length > 0 && (
                      <optgroup label="Specific CUDA GPU">
                        {cudaDevices.map((g) => (
                          <option key={g.id} value={g.id}>
                            {g.name ? `GPU: ${g.name}` : g.id}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </select>
                  <p className="text-xs text-muted-foreground">
                    Preview stems are always saved as WAV. Use Export in Results
                    for MP3/FLAC.
                  </p>

                  {settings.advancedSettings?.device?.startsWith("cuda:") && (
                    <p className="text-xs text-muted-foreground">
                      Default GPU:{" "}
                      <span className="font-mono">
                        {settings.advancedSettings.device}
                      </span>
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Hugging Face Authorization (Optional)</CardTitle>
                  <CardDescription>
                    Some model links are gated on Hugging Face and require a
                    personal access token. This is optional — most public models
                    download without it.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <h4 className="font-medium">Status</h4>
                      <p className="text-sm text-muted-foreground">
                        {hasHfToken
                          ? "Token saved (downloads can access gated repos)."
                          : "No token saved."}
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => {
                        window.electronAPI.openExternalUrl?.(
                          "https://huggingface.co/settings/tokens",
                        );
                      }}
                    >
                      Open Token Page
                    </Button>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Token</label>
                    <Input
                      type="password"
                      value={hfTokenInput}
                      onChange={(e) => setHfTokenInput(e.target.value)}
                      placeholder="hf_..."
                      className="font-mono"
                    />
                    <p className="text-xs text-muted-foreground">
                      Stored in the app config. For security, treat it like a
                      password.
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={async () => {
                        const token = hfTokenInput.trim();
                        if (!token) {
                          alert("Paste a Hugging Face token first.");
                          return;
                        }
                        await window.electronAPI.saveAppConfig?.({
                          hfToken: token,
                        });
                        setHfTokenInput("");
                        const cfg = await window.electronAPI.getAppConfig?.();
                        const saved = cfg?.hfToken;
                        setHasHfToken(
                          typeof saved === "string" && saved.trim().length > 0,
                        );
                        alert(
                          "Hugging Face token saved. Restart the app for backend downloads to pick it up.",
                        );
                      }}
                    >
                      Save Token
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={async () => {
                        await window.electronAPI.saveAppConfig?.({
                          hfToken: "",
                        });
                        const cfg = await window.electronAPI.getAppConfig?.();
                        const saved = cfg?.hfToken;
                        setHasHfToken(
                          typeof saved === "string" && saved.trim().length > 0,
                        );
                        alert(
                          "Hugging Face token cleared. Restart the app for backend downloads to pick it up.",
                        );
                      }}
                    >
                      Clear Token
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}
