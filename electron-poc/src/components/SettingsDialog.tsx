import { useState, useEffect } from "react";
import {
  X,
  Cpu,
  Zap,
  Sliders,
  Settings as SettingsIcon,
  AlertTriangle,
  FolderInput,
} from "lucide-react";
import { useStore } from "../stores/useStore";
import { toast } from "sonner";
import { Button } from "./ui/button"; // Need Button component

interface GpuDevice {
  id: string;
  name: string;
  type: "cpu" | "cuda";
  memory_gb: number | null;
  index?: number;
  recommended: boolean;
}

interface SettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsDialog({
  isOpen,
  onClose,
}: SettingsDialogProps) {
  const [devices, setDevices] = useState<GpuDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("cpu");
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<
    "general" | "advanced" | "automation"
  >("general");
  const [systemInfo, setSystemInfo] = useState<{
    has_cuda: boolean;
    cuda_version: string | null;
    gpu_count: number;
    memory_total_gb?: number;
    recommended_profile?: {
      profile_name: string;
      settings: {
        batch_size?: number;
        use_amp?: boolean;
        chunk_size_modifier?: number;
        description?: string;
      };
      vram_gb: number;
    };
  } | null>(null);

  // Automation state
  const watchEnabled = useStore((state) => state.watchModeEnabled);
  const watchPath = useStore((state) => state.watchPath);
  const setWatchMode = useStore((state) => state.setWatchMode);
  const setWatchPath = useStore((state) => state.setWatchPath);

  // Advanced settings state
  const advancedSettings = useStore((state) => state.settings.advancedSettings);
  const setAdvancedSettings = useStore((state) => state.setAdvancedSettings);

  const normalizeOverlap = (value: unknown) => {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return 4;
    if (n < 1) {
      const denom = Math.max(1e-6, 1 - n);
      return Math.max(2, Math.min(50, Math.round(1 / denom)));
    }
    return Math.max(2, Math.min(50, Math.round(n)));
  };

  const [localAdvanced, setLocalAdvanced] = useState({
    device: advancedSettings?.device || "auto",
    shifts: advancedSettings?.shifts || 1,
    overlap: normalizeOverlap(advancedSettings?.overlap ?? 4),
    segmentSize: advancedSettings?.segmentSize ?? 0,
    outputFormat: advancedSettings?.outputFormat || "wav",
    bitrate: advancedSettings?.bitrate || "320k",
  });

  // Update local state when store changes
  useEffect(() => {
    if (advancedSettings) {
      setLocalAdvanced({
        device: advancedSettings.device || "auto",
        shifts: advancedSettings.shifts,
        overlap: normalizeOverlap(advancedSettings.overlap),
        segmentSize: advancedSettings.segmentSize,
        outputFormat: advancedSettings.outputFormat,
        bitrate: advancedSettings.bitrate,
      });
    }
  }, [advancedSettings]);

  const computeBestSegmentSize = () => {
    const vram =
      systemInfo?.recommended_profile?.vram_gb ??
      devices.find((d) => d.recommended)?.memory_gb ??
      0;

    const CHUNK_11S = 485100;
    const CHUNK_8S = 352800;
    const CHUNK_2_5S = 112455;

    if (!vram || !Number.isFinite(vram)) {
      return 0;
    }

    if (vram >= 8.0) return CHUNK_11S;
    if (vram >= 5.0) return CHUNK_8S;
    if (vram >= 4.0) return CHUNK_2_5S;
    return Math.floor(CHUNK_2_5S / 2);
  };

  useEffect(() => {
    const loadInfo = async () => {
      if (!window.electronAPI) return;

      try {
        const info = await window.electronAPI.getGpuDevices();
        // The API returns the info object directly ({ gpus: [...] })
        if (info && info.gpus) {
          const devicesList = info.gpus || [];

          setDevices(devicesList);

          // Auto-select recommended
          const recommended = devicesList.find((d: any) => d.recommended);
          if (recommended) {
            setSelectedDevice(recommended.id);
          }

          setSystemInfo({
            has_cuda: info.has_cuda,
            cuda_version: info.cuda_version,
            gpu_count: devicesList.length,
            memory_total_gb: info.system_info?.memory_total_gb,
            recommended_profile: info.recommended_profile || undefined,
          });
        } else {
          // toast.error('No GPU info returned') // Silent fail on init often better
        }
      } catch (error) {
        console.error("Failed to load GPU info:", error);
        // Fallback to CPU
        setDevices([
          {
            id: "cpu",
            name: "CPU (No GPU acceleration)",
            type: "cpu",
            memory_gb: null,
            recommended: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen) {
      loadInfo();
    }
  }, [isOpen]);

  const handleSave = async () => {
    localStorage.setItem("stemSepGpuDevice", selectedDevice);
    localStorage.setItem("stemSepDevice", localAdvanced.device);

    // Persist the specific CUDA index as part of defaults when relevant.
    // - If user selected a specific CUDA device (cuda:<idx>), store it.
    // - Otherwise, leave preferredCudaDevice unchanged.
    const preferredCudaDevice = selectedDevice.startsWith("cuda:")
      ? selectedDevice
      : undefined;

    setAdvancedSettings({
      ...localAdvanced,
      preferredCudaDevice,
    });

    // Handle Watch Mode
    if (watchEnabled && watchPath) {
      if (window.electronAPI?.startWatchMode) {
        await window.electronAPI.startWatchMode(watchPath);
        toast.success("Watch mode started");
      }
    } else {
      if (window.electronAPI?.stopWatchMode) {
        await window.electronAPI.stopWatchMode();
        if (watchEnabled && !watchPath)
          toast.error("Watch mode disabled: No folder selected");
      }
    }

    onClose();
  };

  const handleDeviceChange = (deviceId: string) => {
    setSelectedDevice(deviceId);

    // If user clicks a CUDA device, force GPU and persist the concrete index (cuda:<idx>).
    // Otherwise force CPU.
    if (deviceId.startsWith("cuda:")) {
      setLocalAdvanced((prev) => ({
        ...prev,
        device: deviceId,
      }));
    } else if (deviceId.startsWith("cuda")) {
      // Fallback: if backend ever returns "cuda" without an index, normalize to cuda:0.
      setLocalAdvanced((prev) => ({
        ...prev,
        device: "cuda:0",
      }));
    } else if (deviceId === "cpu") {
      setLocalAdvanced((prev) => ({ ...prev, device: "cpu" }));
    }
  };

  const handleBrowseWatch = async () => {
    if (window.electronAPI?.selectOutputDirectory) {
      const path = await window.electronAPI.selectOutputDirectory();
      if (path) setWatchPath(path);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-neutral-900 border border-neutral-700 rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-700">
          <h2 className="text-2xl font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-neutral-800 transition-colors"
            aria-label="Close settings"
          >
            <X className="w-5 h-5 text-neutral-400" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-neutral-700 px-6">
          <button
            onClick={() => setActiveTab("general")}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === "general"
                ? "border-primary text-primary"
                : "border-transparent text-neutral-400 hover:text-white"
            }`}
          >
            <SettingsIcon className="w-4 h-4" />
            General
          </button>
          <button
            onClick={() => setActiveTab("automation")}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === "automation"
                ? "border-primary text-primary"
                : "border-transparent text-neutral-400 hover:text-white"
            }`}
          >
            <FolderInput className="w-4 h-4" />
            Automation
          </button>
          <button
            onClick={() => setActiveTab("advanced")}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === "advanced"
                ? "border-primary text-primary"
                : "border-transparent text-neutral-400 hover:text-white"
            }`}
          >
            <Sliders className="w-4 h-4" />
            Advanced
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "general" ? (
            /* GPU Selection Section */
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                GPU Acceleration
              </h3>
              <p className="text-sm text-neutral-400 mb-4">
                Choose which device to use for audio separation. GPU
                acceleration significantly improves processing speed.
              </p>

              <div className="flex gap-2 flex-wrap mb-4">
                <Button
                  size="sm"
                  variant={
                    localAdvanced.device === "auto" ? "default" : "outline"
                  }
                  onClick={() =>
                    setLocalAdvanced((prev) => ({ ...prev, device: "auto" }))
                  }
                >
                  Auto
                </Button>
                {systemInfo?.has_cuda && (
                  <Button
                    size="sm"
                    variant={
                      localAdvanced.device.startsWith("cuda")
                        ? "default"
                        : "outline"
                    }
                    onClick={() =>
                      setLocalAdvanced((prev) => ({ ...prev, device: "cuda" }))
                    }
                  >
                    GPU (CUDA)
                  </Button>
                )}
                <Button
                  size="sm"
                  variant={
                    localAdvanced.device === "cpu" ? "default" : "outline"
                  }
                  onClick={() =>
                    setLocalAdvanced((prev) => ({ ...prev, device: "cpu" }))
                  }
                >
                  CPU
                </Button>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : (
                <div className="space-y-3">
                  {devices.map((device) => (
                    <button
                      key={device.id}
                      onClick={() => handleDeviceChange(device.id)}
                      className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                        selectedDevice === device.id
                          ? "border-primary bg-primary/10"
                          : "border-neutral-700 bg-neutral-800 hover:border-neutral-600"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          {device.type === "cpu" ? (
                            <Cpu className="w-5 h-5 mt-1 text-neutral-400" />
                          ) : (
                            <Zap className="w-5 h-5 mt-1 text-green-500" />
                          )}
                          <div className="flex-1">
                            <div className="font-medium text-white">
                              {device.name}
                            </div>
                            {device.memory_gb && (
                              <div className="text-sm text-neutral-400 mt-1">
                                {device.memory_gb}GB VRAM
                              </div>
                            )}
                            {device.type === "cpu" && (
                              <div className="text-sm text-neutral-400 mt-1">
                                Slower processing but works on any system
                              </div>
                            )}
                          </div>
                        </div>
                        {device.recommended && (
                          <span className="px-2 py-1 text-xs font-semibold bg-green-500/20 text-green-400 rounded">
                            Recommended
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* System Info */}
              {systemInfo && (
                <div className="mt-4 p-3 bg-neutral-800 rounded-lg border border-neutral-700">
                  <div className="text-sm text-neutral-400">
                    <div className="flex items-center justify-between mb-1">
                      <span>System RAM:</span>
                      <span className="text-white">
                        {systemInfo.memory_total_gb
                          ? `${systemInfo.memory_total_gb} GB`
                          : "Unknown"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mb-1">
                      <span>CUDA Available:</span>
                      <span
                        className={
                          systemInfo.has_cuda
                            ? "text-green-400"
                            : "text-neutral-500"
                        }
                      >
                        {systemInfo.has_cuda ? "Yes" : "No"}
                      </span>
                    </div>
                    {systemInfo.has_cuda && systemInfo.cuda_version && (
                      <div className="flex items-center justify-between mb-1">
                        <span>CUDA Version:</span>
                        <span className="text-white">
                          {systemInfo.cuda_version}
                        </span>
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <span>GPU Count:</span>
                      <span className="text-white">{systemInfo.gpu_count}</span>
                    </div>
                    {systemInfo.recommended_profile && (
                      <div className="mt-3 pt-3 border-t border-neutral-700">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-green-400" />
                          <span className="text-green-400 font-medium text-sm">
                            Optimized Profile:{" "}
                            {systemInfo.recommended_profile.profile_name}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                          {systemInfo.recommended_profile.settings
                            .batch_size && (
                            <>
                              <span className="text-neutral-500">
                                Batch Size:
                              </span>
                              <span className="text-neutral-300">
                                {
                                  systemInfo.recommended_profile.settings
                                    .batch_size
                                }
                              </span>
                            </>
                          )}
                          <span className="text-neutral-500">AMP:</span>
                          <span
                            className={
                              systemInfo.recommended_profile.settings.use_amp
                                ? "text-green-400"
                                : "text-neutral-500"
                            }
                          >
                            {systemInfo.recommended_profile.settings.use_amp
                              ? "Enabled"
                              : "Disabled"}
                          </span>
                          <span className="text-neutral-500">VRAM:</span>
                          <span className="text-neutral-300">
                            {systemInfo.recommended_profile.vram_gb.toFixed(1)}{" "}
                            GB
                          </span>
                        </div>
                        {systemInfo.recommended_profile.settings
                          .description && (
                          <p className="text-neutral-500 text-xs mt-2 italic">
                            {
                              systemInfo.recommended_profile.settings
                                .description
                            }
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : activeTab === "automation" ? (
            /* Automation Section */
            <div className="space-y-6">
              <div className="bg-neutral-800/50 rounded-lg p-6 border border-neutral-700">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-medium text-white">
                      Watch Folder Mode
                    </h3>
                    <p className="text-sm text-neutral-400">
                      Automatically separate files added to a folder.
                    </p>
                  </div>
                  <button
                    onClick={() => setWatchMode(!watchEnabled)}
                    className={`w-12 h-6 rounded-full transition-colors relative ${watchEnabled ? "bg-primary" : "bg-neutral-600"}`}
                  >
                    <div
                      className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-all ${watchEnabled ? "left-7" : "left-1"}`}
                    />
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-white mb-2">
                      Monitored Folder
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={watchPath}
                        readOnly
                        className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600"
                        placeholder="Select a folder to watch..."
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleBrowseWatch}
                      >
                        Browse
                      </Button>
                    </div>
                    <p className="text-xs text-neutral-500 mt-2">
                      Any audio file (mp3, wav, flac) added to this folder will
                      be automatically processed using the currently selected
                      default model.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Advanced Settings Section */
            <div className="space-y-6">
              <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-500 shrink-0 mt-0.5" />
                <div className="text-sm text-yellow-200/80">
                  <p className="font-semibold text-yellow-500 mb-1">
                    Performance Warning
                  </p>
                  Increasing shifts or overlap significantly increases
                  processing time. Use with caution, especially on CPU.
                </div>
              </div>

              {/* Shifts */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Shifts ({localAdvanced.shifts})
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={localAdvanced.shifts}
                  onChange={(e) =>
                    setLocalAdvanced((prev) => ({
                      ...prev,
                      shifts: parseInt(e.target.value),
                    }))
                  }
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-xs text-neutral-400 mt-1">
                  Performs multiple predictions with random shifts to reduce
                  artifacts. Higher = better quality, slower.
                </p>
              </div>

              {/* Overlap */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Overlap (x{localAdvanced.overlap})
                </label>
                <input
                  type="range"
                  min="2"
                  max="16"
                  step="1"
                  value={localAdvanced.overlap}
                  onChange={(e) =>
                    setLocalAdvanced((prev) => ({
                      ...prev,
                      overlap: parseInt(e.target.value),
                    }))
                  }
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-xs text-neutral-400 mt-1">
                  Guide-style overlap divisor. Higher values can improve blending but increase time/VRAM.
                </p>
              </div>

              {/* Segment Size */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Segment Size ({localAdvanced.segmentSize === 0 ? "Auto" : localAdvanced.segmentSize})
                </label>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const best = computeBestSegmentSize();
                      setLocalAdvanced((prev) => ({ ...prev, segmentSize: best }));
                      if (best === 0) {
                        toast.message(
                          "Could not detect a stable VRAM profile. Using Auto segment size (click Save Changes to persist).",
                        );
                      } else {
                        toast.success(
                          `Applied best segment size for this machine: ${best} (click Save Changes to persist)`,
                        );
                      }
                    }}
                  >
                    Apply best configuration for my machine
                  </Button>
                </div>
                <select
                  value={localAdvanced.segmentSize}
                  onChange={(e) =>
                    setLocalAdvanced((prev) => ({
                      ...prev,
                      segmentSize: parseInt(e.target.value),
                    }))
                  }
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                >
                  <option value="0">Auto (Recommended)</option>
                  <option value="56227">~1.3s (Very Low VRAM)</option>
                  <option value="112455">~2.5s (Low VRAM)</option>
                  <option value="352800">~8s (Balanced)</option>
                  <option value="485100">~11s (High VRAM / Best quality)</option>
                </select>
                <p className="text-xs text-neutral-400 mt-1">
                  Length of audio segments processed at once. Larger segments
                  require more VRAM.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {/* Default Export Format */}
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Default Export Format
                  </label>
                  <select
                    value={localAdvanced.outputFormat}
                    onChange={(e) =>
                      setLocalAdvanced((prev) => ({
                        ...prev,
                        outputFormat: e.target.value as any,
                      }))
                    }
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                  >
                    <option value="mp3">MP3</option>
                    <option value="wav">WAV</option>
                    <option value="flac">FLAC</option>
                  </select>
                </div>

                {/* Bitrate (MP3 only) */}
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Bitrate (MP3)
                  </label>
                  <select
                    value={localAdvanced.bitrate}
                    onChange={(e) =>
                      setLocalAdvanced((prev) => ({
                        ...prev,
                        bitrate: e.target.value,
                      }))
                    }
                    disabled={localAdvanced.outputFormat !== "mp3"}
                    className={`w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent ${
                      localAdvanced.outputFormat !== "mp3"
                        ? "opacity-50 cursor-not-allowed"
                        : ""
                    }`}
                  >
                    <option value="128k">128k</option>
                    <option value="192k">192k</option>
                    <option value="256k">256k</option>
                    <option value="320k">320k (High Quality)</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-neutral-700">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-neutral-600 text-white hover:bg-neutral-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-lg bg-primary text-white font-semibold hover:bg-primary/90 transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
