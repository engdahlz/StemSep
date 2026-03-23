import { app } from "electron";
import fs from "fs";
import path from "path";

export function shouldUseRustBackend(): boolean {
  const value = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
  if (value === "python") return false;
  if (value === "rust") return true;
  return true;
}

function getAppPathCandidates(): string[] {
  const candidates = new Set<string>();

  try {
    const appPath = app.getAppPath();
    if (appPath) {
      candidates.add(appPath);
      candidates.add(path.dirname(appPath));
    }
  } catch {
    // ignore
  }

  try {
    candidates.add(process.cwd());
  } catch {
    // ignore
  }

  return Array.from(candidates);
}

export function resolveBundledPythonForRustBackend(): string | null {
  try {
    const venv = (process.env.VIRTUAL_ENV || "").trim();
    if (venv) {
      if (process.platform === "win32") {
        const candidate = path.join(venv, "Scripts", "python.exe");
        try {
          if (fs.existsSync(candidate)) return candidate;
        } catch {
          // ignore
        }
      } else {
        const candidate = path.join(venv, "bin", "python");
        try {
          if (fs.existsSync(candidate)) return candidate;
        } catch {
          // ignore
        }
      }
    }

    const appPaths = getAppPathCandidates();
    const candidates =
      process.platform === "win32"
        ? [
            path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
            path.join(process.cwd(), ".venv", "Scripts", "python.exe"),
            ...appPaths.flatMap((appPath) => [
              path.join(appPath, ".venv", "Scripts", "python.exe"),
              path.join(appPath, "..", ".venv", "Scripts", "python.exe"),
              path.join(appPath, "StemSepApp", ".venv", "Scripts", "python.exe"),
              path.join(
                appPath,
                "..",
                "StemSepApp",
                ".venv",
                "Scripts",
                "python.exe",
              ),
            ]),
          ]
        : [
            path.join(process.resourcesPath, ".venv", "bin", "python"),
            path.join(process.cwd(), ".venv", "bin", "python"),
            ...appPaths.flatMap((appPath) => [
              path.join(appPath, ".venv", "bin", "python"),
              path.join(appPath, "..", ".venv", "bin", "python"),
              path.join(appPath, "StemSepApp", ".venv", "bin", "python"),
              path.join(appPath, "..", "StemSepApp", ".venv", "bin", "python"),
            ]),
          ];

    for (const candidate of Array.from(new Set(candidates))) {
      try {
        if (fs.existsSync(candidate)) return candidate;
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }

  return null;
}
