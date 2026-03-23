import fs from "fs";
import path from "path";

export function shouldUseRustBackend(): boolean {
  const value = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
  if (value === "python") return false;
  if (value === "rust") return true;
  return true;
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

    const candidates =
      process.platform === "win32"
        ? [
            path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
            path.join(process.cwd(), ".venv", "Scripts", "python.exe"),
            path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
            path.join(
              process.cwd(),
              "StemSepApp",
              ".venv",
              "Scripts",
              "python.exe",
            ),
            path.join(__dirname, "../../.venv/Scripts/python.exe"),
          ]
        : [
            path.join(process.resourcesPath, ".venv", "bin", "python"),
            path.join(process.cwd(), ".venv", "bin", "python"),
            path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
            path.join(process.cwd(), "StemSepApp", ".venv", "bin", "python"),
            path.join(__dirname, "../../.venv/bin/python"),
          ];

    for (const candidate of candidates) {
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
