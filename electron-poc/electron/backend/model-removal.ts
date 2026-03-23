import fs from "fs";
import path from "path";

type GetStoredModelsDir = () => string | null;

function urlBasename(url: unknown): string | null {
  if (typeof url !== "string") return null;
  const normalizedUrl = url.split("?")[0].replace(/\/+$/, "");
  const parts = normalizedUrl.split("/");
  const last = parts[parts.length - 1];
  return last || null;
}

function resolveAssetsDirForLocalOps() {
  const candidates: string[] = [
    path.join(process.resourcesPath, "StemSepApp", "assets"),
    path.join(__dirname, "../../StemSepApp/assets"),
    path.join(process.cwd(), "StemSepApp", "assets"),
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      // ignore
    }
  }
  return null;
}

function resolveCatalogRuntimePath(): string | null {
  const assetsDir = resolveAssetsDirForLocalOps();
  if (!assetsDir) return null;

  const runtimePath = path.join(assetsDir, "catalog.runtime.json");
  try {
    return fs.existsSync(runtimePath) ? runtimePath : null;
  } catch {
    return null;
  }
}

export function createModelRemovalController({
  getStoredModelsDir,
}: {
  getStoredModelsDir: GetStoredModelsDir;
}) {
  function removeModelLocal(
    modelId: string,
  ): { success: true; removedFiles: string[] } {
    const modelsDir = getStoredModelsDir();
    if (!modelsDir) {
      throw new Error("Models directory is not configured.");
    }

    const assetsDir = resolveAssetsDirForLocalOps();
    if (!assetsDir) {
      throw new Error(
        "Assets directory not found (cannot resolve model registry).",
      );
    }

    const registryPath = resolveCatalogRuntimePath();
    let model: any = null;
    try {
      if (!registryPath) {
        throw new Error("No catalog runtime manifest found.");
      }
      const raw = fs.readFileSync(registryPath, "utf-8");
      const json = JSON.parse(raw);
      const models = Array.isArray(json?.models) ? json.models : [];
      model = models.find((entry: any) => entry && entry.id === modelId) || null;
    } catch (error: any) {
      throw new Error(
        `Failed to read model registry at ${registryPath || "<missing>"}: ${
          error?.message || String(error)
        }`,
      );
    }

    const removed: string[] = [];
    const links =
      model?.links && typeof model.links === "object" ? model.links : null;
    const ckptBase = urlBasename(links?.checkpoint);
    const cfgBase = urlBasename(links?.config);

    const genericConfig = new Set(["config.yaml", "config.yml"]);
    const candidateNames = new Set<string>();

    if (ckptBase) candidateNames.add(ckptBase);
    if (cfgBase && !genericConfig.has(cfgBase.toLowerCase())) {
      candidateNames.add(cfgBase);
    }

    const addArtifactNames = (artifacts: any) => {
      for (const artifact of Array.isArray(artifacts) ? artifacts : []) {
        const filenames = [
          artifact?.filename,
          artifact?.relative_path,
          artifact?.relativePath,
          artifact?.canonical_path,
          artifact?.canonicalPath,
        ];
        for (const filename of filenames) {
          const basename = urlBasename(filename);
          if (!basename) continue;
          if (genericConfig.has(basename.toLowerCase())) continue;
          candidateNames.add(basename);
        }
      }
    };

    addArtifactNames(model?.download?.artifacts);
    addArtifactNames(model?.installation?.artifacts);
    addArtifactNames(model?.legacy_artifacts);

    for (const ext of [
      ".ckpt",
      ".chpt",
      ".pth",
      ".pt",
      ".safetensors",
      ".onnx",
      ".yaml",
      ".yml",
    ]) {
      candidateNames.add(`${modelId}${ext}`);
    }

    if (modelId === "mel-band-roformer-kim") {
      candidateNames.add("MelBandRoformer.ckpt");
      candidateNames.add("vocals_mel_band_roformer.ckpt");
      candidateNames.add("vocals_mel_band_roformer.yaml");
    }

    for (const name of Array.from(candidateNames)) {
      const filePath = path.join(modelsDir, name);
      try {
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          fs.unlinkSync(filePath);
          removed.push(filePath);
        }
      } catch {
        // ignore individual failures
      }
    }

    return { success: true, removedFiles: removed };
  }

  return {
    removeModelLocal,
  };
}
