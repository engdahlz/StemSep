import { useCallback, useMemo } from "react";
import { useStore } from "../stores/useStore";
import type { SeparationConfig } from "../types/separation";
import { toast } from "sonner";
import { ALL_PRESETS, Preset } from "../presets";
import { recipesToPresets } from "@/lib/recipePresets";
import { resolveSeparationPlan } from "@/lib/separation/resolveSeparationPlan";

const MISSING_MODELS_EVENT = "stemsep:missing-models";

export const useSeparation = () => {
  const queue = useStore((state) => state.separation.queue);
  const isPaused = useStore((state) => state.separation.isPaused);
  const outputDirectory = useStore((state) => state.settings.defaultOutputDir);
  const setOutputDirectory = useStore((state) => state.setDefaultOutputDir);

  const setSeparationStatus = useStore((state) => state.startSeparation);
  const setSeparationProgress = useStore(
    (state) => state.setSeparationProgress,
  );
  const updateQueueItem = useStore((state) => state.updateQueueItem);
  const completeSeparation = useStore((state) => state.completeSeparation);
  const addLog = useStore((state) => state.addLog);
  const addToHistory = useStore((state) => state.addToHistory);
  const resumeQueue = useStore((state) => state.resumeQueue);
  const phaseParams = useStore((state) => state.settings.phaseParams);

  const recipes = useStore((state) => state.recipes);

  const combinedPresets: Preset[] = useMemo(() => {
    // Safety check for recipes
    if (!Array.isArray(recipes)) {
      return ALL_PRESETS;
    }

    return [...ALL_PRESETS, ...recipesToPresets(recipes)];
  }, [recipes]);

  const toBackendOverlap = useCallback((value: unknown) => {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return undefined;
    if (n <= 0) return undefined;

    // UI uses an integer divisor (2..50). Backend expects a 0..1 ratio.
    // Keep legacy ratio values (0..1) as-is.
    if (n >= 1) return 1 / n;
    return n;
  }, []);

  const handleSelectOutputDirectory = useCallback(async () => {
    if (!window.electronAPI) return;

    const selectedPath = await window.electronAPI.selectOutputDirectory();
    if (selectedPath) {
      setOutputDirectory(selectedPath);
    }
  }, [setOutputDirectory]);

  const startSeparation = useCallback(
    async (config: SeparationConfig) => {
      if (queue.length === 0) return;
      if (!outputDirectory) {
        toast.error("Please select an output folder first");
        handleSelectOutputDirectory();
        return;
      }

      // Ensure queue is running
      if (isPaused) {
        resumeQueue();
      }

      setSeparationStatus();
      addLog(`Starting batch separation of ${queue.length} files...`);

      let processedCount = 0;
      const totalFiles = queue.length;
      const allOutputs: Record<string, string[]> = {};

      // Process queue sequentially
      // Note: We access current queue from store inside loop? No, queue is from closure.
      // If queue changes during processing, this might be stale.
      // However, standard practice here usually involves blocking UI or locking queue.
      // Ideally we should ref the latest queue or use an index.
      // But for now we stick to the existing logic which iterates the queue captured at start.

      for (const item of queue) {
        // Check if item is already completed (e.g. from previous run)
        if (item.status === "completed") {
          processedCount++;
          continue;
        }

        updateQueueItem(item.id, {
          status: "processing",
          progress: 0,
          startTime: Date.now(),
          message: "Pre-flight...",
        });
        const fileName = item.file.split(/[\\/]/).pop() || item.file;
        setSeparationProgress(
          Math.round((processedCount / totalFiles) * 100),
          `Processing ${fileName}...`,
        );

        try {
          // DEBUG: Log full config to trace ensemble issue
          console.log(
            "[useSeparation] FULL CONFIG RECEIVED:",
            JSON.stringify(config, null, 2),
          );

          const plan = resolveSeparationPlan({
            config,
            presets: combinedPresets,
            models: useStore.getState().models,
            globalPhaseParams: phaseParams,
          });

          // Determine model and stems (canonical plan)
          const modelId = plan.effectiveModelId;
          const stems = plan.effectiveStems;

          const backendOverlap = toBackendOverlap(config.advancedParams?.overlap);

          // Block early if we already know models are missing
          if (!plan.canProceed) {
            const missingList = plan.missingModels
              .map((m) => {
                return `Missing: ${m.modelId}`;
              })
              .join(", ");

            const actionable =
              "Open the Model Browser to install the missing models.";

            // Update item status to failed with a clear, actionable error
            updateQueueItem(item.id, {
              status: "failed",
              error: `Missing models: ${missingList}`,
            });

            // Show actionable toasts (only once per batch to avoid spam)
            if (processedCount === 0) {
              toast.error("Cannot start: missing models.", {
                action: {
                  label: "Model Browser",
                  onClick: () => {
                    try {
                      window.dispatchEvent(
                        new CustomEvent(MISSING_MODELS_EVENT, {
                          detail: {
                            source: "batch",
                            file: item.file,
                            missingModels: plan.missingModels,
                          },
                        }),
                      );
                    } catch (e) {
                      console.log(
                        "[useSeparation] Failed to dispatch missing models event",
                        e,
                      );
                    }
                  },
                },
              });
              toast.message(actionable);
            } else {
              toast.error(`Skipped ${fileName}: missing models`);
            }

            addLog(`[useSeparation] Skipped "${fileName}": ${missingList}`);
            processedCount++;
            setSeparationProgress(
              Math.round((processedCount / totalFiles) * 100),
              `Processing next file...`,
            );
            continue;
          }

          if (!window.electronAPI)
            throw new Error("Electron API not available");

          const preflight = window.electronAPI.separationPreflight
            ? await window.electronAPI.separationPreflight(
                item.file,
                modelId,
                outputDirectory,
                stems,
                config.device && config.device !== "auto"
                  ? config.device
                  : undefined,
                backendOverlap,
                config.advancedParams?.segmentSize,
                config.advancedParams?.shifts,
                "wav",
                config.advancedParams?.bitrate,
                config.advancedParams?.tta,
                plan.effectiveEnsembleConfig,
                plan.effectiveEnsembleConfig?.algorithm,
                config.invert,
                config.splitFreq,
                plan.effectiveGlobalPhaseParams,
                plan.effectivePostProcessingSteps,
                config.volumeCompensation,
              )
            : null;

          const warnings = (preflight as any)?.warnings as string[] | undefined;
          if (warnings && warnings.length > 0 && processedCount === 0) {
            toast.warning(warnings[0]);
            addLog(`Pre-flight warning: ${warnings[0]}`);
          }

          const canProceed = (preflight as any)?.can_proceed;
          if (canProceed === false) {
            const errors = (preflight as any)?.errors as string[] | undefined;
            throw new Error(errors?.[0] || "Pre-flight failed");
          }

          // Call backend
          const result = await window.electronAPI.separateAudio(
            item.file,
            modelId,
            outputDirectory,
            stems,
            config.device && config.device !== "auto"
              ? config.device
              : undefined,
            backendOverlap,
            config.advancedParams?.segmentSize,
            config.advancedParams?.shifts,
            "wav",
            config.advancedParams?.bitrate,
            config.advancedParams?.tta,
            plan.effectiveEnsembleConfig,
            plan.effectiveEnsembleConfig?.algorithm,
            config.invert,
            config.splitFreq,
            plan.effectiveGlobalPhaseParams,
            plan.effectivePostProcessingSteps,
            config.volumeCompensation,
          );

          console.log("Separation Result (Frontend):", result);

          if (!result.success) {
            throw new Error(result.error || "Separation failed");
          }

          const outputs =
            (result.outputFiles as unknown as Record<string, string>) || {};
          allOutputs[item.id] = Object.values(outputs);

          // Update item status
          updateQueueItem(item.id, {
            status: "completed",
            progress: 100,
            outputFiles: outputs,
            backendJobId: result.jobId,
          });

          // Add to history
          // Determine display name.
          // Important: a stale presetId can hang around when switching to Advanced mode,
          // so only treat this as a preset run when mode === 'simple'.
          const isCustomEnsemble = !!(
            config.ensembleConfig?.models &&
            config.ensembleConfig.models.length > 0
          );
          const usedPreset =
            config.mode === "simple" && !!config.presetId && !isCustomEnsemble;
          const presetInfo = usedPreset
            ? combinedPresets.find((p) => p.id === config.presetId)
            : undefined;
          const displayName = isCustomEnsemble
            ? `Custom Ensemble (${config.ensembleConfig!.models.length} models)`
            : presetInfo?.name || modelId;

          addToHistory({
            inputFile: item.file,
            outputDir: outputDirectory,
            modelId,
            modelName: displayName || "Unknown Model",
            preset: presetInfo
              ? { id: presetInfo.id, name: presetInfo.name }
              : undefined,
            status: "completed",
            outputFiles: outputs,
            backendJobId: result.jobId,
            settings: {
              stems: stems || [],
              overlap: config.advancedParams?.overlap,
              segmentSize: config.advancedParams?.segmentSize,
            },
          });

          toast.success(`Separated: ${fileName}`);
        } catch (error) {
          const rawMessage =
            error instanceof Error
              ? error.message
              : String(error ?? "Unknown error");

          const prettyDeviceError = (() => {
            // Expect structured device errors to be embedded in the message as:
            // "STEMSEP_DEVICE_ERROR {json}"
            const prefix = "STEMSEP_DEVICE_ERROR ";
            if (!rawMessage.startsWith(prefix)) return null;

            const jsonPart = rawMessage.slice(prefix.length).trim();
            if (!jsonPart) return null;

            try {
              const payload = JSON.parse(jsonPart) as any;
              if (!payload || typeof payload !== "object") return null;

              const code =
                typeof payload.code === "string"
                  ? payload.code
                  : "DEVICE_ERROR";
              const msg =
                typeof payload.message === "string"
                  ? payload.message
                  : "Device error";
              const device =
                typeof payload.device === "string" ? payload.device : undefined;

              const title =
                code === "CUDA_OOM"
                  ? "GPU out of memory"
                  : code === "CUDA_NOT_AVAILABLE"
                    ? "GPU not available"
                    : code === "CUDA_DEVICE_INVALID"
                      ? "Invalid GPU device"
                      : code === "CUDA_INIT_FAILED"
                        ? "GPU initialization failed"
                        : "Device error";

              const description = device
                ? `${msg} (requested: ${device})`
                : msg;

              // Keep details out of the toast (can be long); we store a concise message on the item.
              const details =
                typeof payload.details === "string"
                  ? payload.details
                  : undefined;

              return { title, description, details, code, device };
            } catch {
              return null;
            }
          })();

          const errorMessage = prettyDeviceError
            ? prettyDeviceError.description
            : rawMessage;

          // Update item status to failed
          updateQueueItem(item.id, { status: "failed", error: errorMessage });

          if (prettyDeviceError) {
            toast.error(
              `Failed to separate ${fileName}: ${prettyDeviceError.title}`,
              {
                description: prettyDeviceError.description,
                duration: 12000,
              },
            );
            addLog(
              `[useSeparation] ${fileName} failed: ${prettyDeviceError.title}: ${prettyDeviceError.description}${
                prettyDeviceError.details
                  ? ` | details: ${prettyDeviceError.details}`
                  : ""
              }`,
            );
          } else {
            toast.error(`Failed to separate ${fileName}: ${errorMessage}`);
          }
        }
        processedCount++;
        // Note: progressMessage here is from outer scope (store value at render time).
        // Better to pass a fresh message or undefined if we want to keep current.
        // But setSeparationProgress updates the store message.
        setSeparationProgress(
          Math.round((processedCount / totalFiles) * 100),
          `Processing next file...`,
        );
      }

      // Finalize batch
      completeSeparation({});

      addLog("Batch processing complete!");
      toast.success("Batch processing complete!");
    },
    [
      queue,
      outputDirectory,
      isPaused,
      resumeQueue,
      setSeparationStatus,
      addLog,
      updateQueueItem,
      setSeparationProgress,
      combinedPresets,
      phaseParams,
      completeSeparation,
      addToHistory,
      handleSelectOutputDirectory,
    ],
  );

  return { startSeparation };
};
