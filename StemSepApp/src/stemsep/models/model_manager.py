import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("StemSep")


@dataclass
class ModelInfo:
    id: str
    name: str
    architecture: str
    description: str = ""
    type: str = ""
    stems: Optional[List[str]] = None
    links: Optional[Dict[str, str]] = None
    vram_required: Optional[float] = None
    sdr: Optional[float] = None
    fullness: Optional[float] = None
    bleedless: Optional[float] = None
    recommended_overlap: Optional[int] = None
    recommended_settings: Optional[Dict[str, Any]] = None

    # === Strict-spec extensions (guide-as-spec) ===
    # These fields are additive/backwards compatible: older registries can omit them.
    runtime: Optional[Dict[str, Any]] = None
    compatibility: Optional[Dict[str, Any]] = None
    requirements: Optional[Dict[str, Any]] = None
    phase_fix: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    settings_semantics: Optional[Dict[str, Any]] = None

    installed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep compat with UI (some fields may be None)
        return d


class ModelManager:
    """Loads the model registry and tracks installed model files.

    This project historically used a richer ModelManager; this implementation
    is a compatibility layer for the Electron Python bridge and separation engine.
    """

    def __init__(
        self, models_dir: Optional[Path] = None, assets_dir: Optional[Path] = None
    ):
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            env_models = os.environ.get("STEMSEP_MODELS_DIR")
            if env_models and env_models.strip():
                self.models_dir = Path(env_models.strip())
            else:
                d_models = Path(r"D:\\StemSep Models")
                self.models_dir = d_models if d_models.exists() else (Path.home() / ".stemsep" / "models")
        # Repo layout: StemSepApp/assets (sibling to src/)
        # NOTE: This file lives at StemSepApp/src/stemsep/models/model_manager.py
        # so parents[2] == StemSepApp/src (WRONG for assets). We want parents[3] == StemSepApp.
        if assets_dir:
            self.assets_dir = Path(assets_dir)
        else:
            # Allow explicit override for packaged/dev environments.
            env_assets = os.environ.get("STEMSEP_ASSETS_DIR")
            candidates: List[Path] = []
            if env_assets:
                candidates.append(Path(env_assets))

            here = Path(__file__).resolve()
            candidates.extend(
                [
                    here.parents[3] / "assets",  # StemSepApp/assets (expected)
                    here.parents[2] / "assets",  # legacy/incorrect (keep as fallback)
                    here.parents[4] / "StemSepApp" / "assets",  # repo root layout fallback
                ]
            )

            picked = next((p for p in candidates if p.exists()), None)
            self.assets_dir = picked if picked is not None else (here.parents[3] / "assets")
        self._download_callbacks: List[Callable[[str, float], None]] = []

        self.models: Dict[str, ModelInfo] = {}
        self.installed_models: Dict[str, Path] = {}

        self._presets_cache: Optional[Dict[str, Any]] = None

        self._load_registry()
        self._refresh_install_flags()

    def load_presets(self) -> Optional[Dict[str, Any]]:
        """Load presets.json if available.

        SeparationManager still expects presets + ensembles to exist in some installations.
        This compatibility layer loads `assets/presets.json` when present.
        """
        if self._presets_cache is not None:
            return self._presets_cache

        try:
            path = self.assets_dir / "presets.json"
            if not path.exists():
                self._presets_cache = None
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._presets_cache = data if isinstance(data, dict) else None
            return self._presets_cache
        except Exception:
            self._presets_cache = None
            return None

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Return a preset dict by id (from presets.json), normalizing legacy keys."""
        if not preset_id or not isinstance(preset_id, str):
            return None

        data = self.load_presets()
        if not isinstance(data, dict):
            return None

        presets = data.get("presets") or []
        if not isinstance(presets, list):
            return None

        for p in presets:
            if not isinstance(p, dict):
                continue
            if p.get("id") != preset_id:
                continue

            # Normalize chunk_size -> segment_size for callers that expect segment_size.
            settings = p.get("settings")
            if (
                isinstance(settings, dict)
                and "chunk_size" in settings
                and "segment_size" not in settings
            ):
                try:
                    settings = dict(settings)
                    settings["segment_size"] = settings.get("chunk_size")
                    p = dict(p)
                    p["settings"] = settings
                except Exception:
                    pass

            return p

        return None

    def add_download_callback(self, cb: Callable[[str, float], None]):
        if cb:
            self._download_callbacks.append(cb)

    def _emit_download_progress(self, model_id: str, progress: float):
        for cb in list(self._download_callbacks):
            try:
                cb(model_id, progress)
            except Exception:
                pass

    def _registry_path(self) -> Path:
        # Primary registry in this repo is models.json.bak
        p1 = self.assets_dir / "models.json.bak"
        if p1.exists():
            return p1
        p2 = self.assets_dir / "models.json"
        return p2

    def _load_registry(self):
        path = self._registry_path()
        if not path.exists():
            raise FileNotFoundError(f"Model registry not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", [])
        out: Dict[str, ModelInfo] = {}

        for m in models:
            try:
                mid = str(m.get("id") or "").strip()
                if not mid:
                    continue
                out[mid] = ModelInfo(
                    id=mid,
                    name=str(m.get("name") or mid),
                    architecture=str(m.get("architecture") or ""),
                    description=str(m.get("description") or ""),
                    type=str(m.get("type") or ""),
                    stems=m.get("stems"),
                    links=m.get("links") or {},
                    vram_required=m.get("vram_required"),
                    sdr=m.get("sdr"),
                    fullness=m.get("fullness"),
                    bleedless=m.get("bleedless"),
                    recommended_overlap=m.get("recommended_overlap"),
                    recommended_settings=m.get("recommended_settings"),
                    # Strict-spec additions (optional in registry)
                    runtime=m.get("runtime"),
                    compatibility=m.get("compatibility"),
                    requirements=m.get("requirements"),
                    phase_fix=m.get("phase_fix"),
                    capabilities=m.get("capabilities"),
                    artifacts=m.get("artifacts"),
                    settings_semantics=m.get("settings_semantics"),
                    installed=False,
                )
            except Exception as e:
                log.warning(f"Failed to load model entry: {e}")

        self.models = out

    def _url_basename(self, url: str) -> Optional[str]:
        if not url or not isinstance(url, str):
            return None
        url = url.split("?")[0]
        parts = [p for p in url.split("/") if p]
        return parts[-1] if parts else None

    def _artifact_filename(self, model: Optional[ModelInfo], key: str) -> Optional[str]:
        if not model:
            return None
        artifacts = getattr(model, "artifacts", None)
        if not isinstance(artifacts, dict):
            return None
        obj = artifacts.get(key)
        if not isinstance(obj, dict):
            return None
        fn = obj.get("filename")
        if not isinstance(fn, str):
            return None
        fn = fn.strip()
        if not fn or ".MISSING" in fn:
            return None
        return fn

    def _is_installed(self, model_id: str, model: Optional[ModelInfo]) -> bool:
        if not model_id:
            return False

        ckpt_url = None
        cfg_url = None
        if model and isinstance(model.links, dict):
            ckpt_url = model.links.get("checkpoint")
            cfg_url = model.links.get("config")

        artifact_ckpt = self._artifact_filename(model, "primary")
        artifact_cfg = self._artifact_filename(model, "config")

        # Checkpoint requirement
        ok_ckpt = False
        if artifact_ckpt:
            ok_ckpt = (self.models_dir / artifact_ckpt).exists()
        elif isinstance(ckpt_url, str) and ckpt_url.strip():
            if ".onnx" in ckpt_url.lower():
                ok_ckpt = (self.models_dir / f"{model_id}.onnx").exists()
            else:
                base = self._url_basename(ckpt_url)
                ok_ckpt = bool(base and (self.models_dir / base).exists())

        # Config requirement (only if registry declares a config link)
        ok_cfg = True
        if isinstance(cfg_url, str) and cfg_url.strip():
            cfg_candidates: List[Path] = []
            if artifact_cfg:
                cfg_candidates.append(self.models_dir / artifact_cfg)

            cfg_candidates.append(self.models_dir / f"{model_id}.yaml")
            cfg_candidates.append(self.models_dir / f"{model_id}.yml")
            base = self._url_basename(cfg_url)
            if base:
                cfg_candidates.append(self.models_dir / base)

            ok_cfg = any(p.exists() for p in cfg_candidates)

        # If registry provides any explicit requirements (links or artifact filenames), be strict.
        if (
            artifact_ckpt
            or artifact_cfg
            or (isinstance(ckpt_url, str) and ckpt_url.strip())
            or (isinstance(cfg_url, str) and cfg_url.strip())
        ):
            return ok_ckpt and ok_cfg

        # Fallback heuristic for models without explicit links/artifacts.
        for ext in [".ckpt", ".pth", ".pt", ".onnx", ".safetensors", ".yaml", ".yml"]:
            if (self.models_dir / f"{model_id}{ext}").exists():
                return True

        return False

    def _candidate_files(self, model_id: str, model: Optional[ModelInfo]) -> List[Path]:
        candidates: List[Path] = []

        # Prefer explicit artifact filenames when present
        ckpt_fn = self._artifact_filename(model, "primary")
        if ckpt_fn:
            candidates.append(self.models_dir / ckpt_fn)
        cfg_fn = self._artifact_filename(model, "config")
        if cfg_fn:
            candidates.append(self.models_dir / cfg_fn)

        # Try explicit filename from links
        if model and model.links:
            ckpt = model.links.get("checkpoint")
            cfg = model.links.get("config")
            for url in [ckpt, cfg]:
                b = self._url_basename(url) if url else None
                if b:
                    candidates.append(self.models_dir / b)

        # Common naming
        for ext in [".ckpt", ".pth", ".pt", ".onnx", ".safetensors", ".yaml", ".yml"]:
            candidates.append(self.models_dir / f"{model_id}{ext}")

        return candidates

    def _refresh_install_flags(self):
        self.installed_models = {}
        for mid, model in self.models.items():
            installed = self._is_installed(mid, model)
            if installed:
                primary = None
                for p in self._candidate_files(mid, model):
                    if p.exists():
                        primary = p
                        break
                if primary:
                    self.installed_models[mid] = primary
            model.installed = installed

    def get_available_models(self) -> List[ModelInfo]:
        self._refresh_install_flags()
        return list(self.models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self.models.get(model_id)

    def get_expected_local_filename(self, model_id: str) -> Optional[str]:
        """Return the expected primary local filename for a registry model.

        This is used by the separation stack to translate a registry `model_id`
        into a concrete artifact filename stored under `models_dir`.

        Preference order:
        - `links.checkpoint` basename (most models)
        - `links.config` basename (config-only models, e.g. Demucs yaml)
        - fallback for well-known demucs ids: `{model_id}.yaml`
        """
        m = self.models.get(model_id)
        if not m:
            return None

        # Prefer explicit artifact filename when available
        artifact_ckpt = self._artifact_filename(m, "primary")
        if artifact_ckpt:
            return artifact_ckpt

        # Special-case: KimberleyJSN/melbandroformer provides a standalone checkpoint named
        # "MelBandRoformer.ckpt" without a config. Our runtime uses audio-separator, which
        # has a built-in Roformer entry expecting the canonical filename below.
        if model_id == "mel-band-roformer-kim":
            return "vocals_mel_band_roformer.ckpt"

        links = m.links or {}
        ckpt_url = links.get("checkpoint") if isinstance(links, dict) else None
        cfg_url = links.get("config") if isinstance(links, dict) else None

        ckpt_base = self._url_basename(ckpt_url) if ckpt_url else None
        if ckpt_base:
            # Keep historical convention for ONNX models stored as `<model_id>.onnx`.
            if ckpt_base.lower().endswith(".onnx"):
                return f"{model_id}.onnx"
            return ckpt_base

        cfg_base = self._url_basename(cfg_url) if cfg_url else None
        if cfg_base:
            return cfg_base

        # Demucs YAML models are often referenced by id.
        if (m.architecture or "").lower() == "demucs":
            return f"{model_id}.yaml"

        return None

    def is_model_installed(self, model_id: str) -> bool:
        self._refresh_install_flags()
        m = self.models.get(model_id)
        return bool(m and m.installed)

    def ensure_model_ready(
        self,
        model_id: str,
        auto_repair_filenames: bool = True,
        copy_instead_of_rename: bool = True,
    ) -> Dict[str, Any]:
        """Compatibility API used by python-bridge.

        This version does not perform complex repair logic. It reports missing and provides URLs.
        """
        self._refresh_install_flags()
        m = self.models.get(model_id)
        if not m:
            return {
                "ok": False,
                "model_id": model_id,
                "installed": False,
                "error": f"Unknown model_id: {model_id}",
            }

        expected_primary = self.get_expected_local_filename(model_id)

        # Best-effort filename repair:
        # If registry stores files under URL basenames (common), create a local alias
        # so tools expecting `model_id.yaml` + `model_id.<weights>` can work.
        if auto_repair_filenames and m.links and isinstance(m.links, dict):
            try:
                ckpt_url = m.links.get("checkpoint") or ""
                cfg_url = m.links.get("config") or ""

                ckpt_base = self._url_basename(str(ckpt_url)) if ckpt_url else None
                cfg_base = self._url_basename(str(cfg_url)) if cfg_url else None

                # Only attempt aliasing when both pieces exist locally.
                ckpt_src = (self.models_dir / ckpt_base) if ckpt_base else None
                cfg_src = (self.models_dir / cfg_base) if cfg_base else None

                if ckpt_src and cfg_src and ckpt_src.exists() and cfg_src.exists():
                    # Alias weights as `<model_id>.<ext>` (e.g. .ckpt)
                    weight_ext = ckpt_src.suffix
                    if weight_ext:
                        weight_alias = self.models_dir / f"{model_id}{weight_ext}"
                        if not weight_alias.exists():
                            try:
                                os.link(str(ckpt_src), str(weight_alias))
                            except Exception:
                                if copy_instead_of_rename:
                                    shutil.copy2(ckpt_src, weight_alias)

                    # Alias config as `<model_id>.yaml` (direct-load path expects .yaml)
                    yaml_alias = self.models_dir / f"{model_id}.yaml"
                    if not yaml_alias.exists():
                        try:
                            os.link(str(cfg_src), str(yaml_alias))
                        except Exception:
                            shutil.copy2(cfg_src, yaml_alias)
            except Exception:
                # Never fail readiness due to repair attempts.
                pass

        # Special-case: mel-band-roformer-kim
        # Registry points at KimberleyJSN/melbandroformer's MelBandRoformer.ckpt (no config).
        # audio-separator has an official built-in Roformer entry that expects:
        #   vocals_mel_band_roformer.ckpt + vocals_mel_band_roformer.yaml
        # Instead of inventing a YAML (which may mismatch weights), create a local alias
        # so the official audio-separator path can be used.
        if auto_repair_filenames and model_id == "mel-band-roformer-kim":
            try:
                ckpt_src = self.models_dir / "MelBandRoformer.ckpt"
                ckpt_alias = self.models_dir / "vocals_mel_band_roformer.ckpt"
                if ckpt_src.exists() and not ckpt_alias.exists():
                    try:
                        os.link(str(ckpt_src), str(ckpt_alias))
                    except Exception:
                        if copy_instead_of_rename:
                            shutil.copy2(ckpt_src, ckpt_alias)
            except Exception:
                pass

        # Refresh after any repair attempts.
        self._refresh_install_flags()
        m = self.models.get(model_id) or m

        report: Dict[str, Any] = {
            "ok": bool(m.installed),
            "model_id": model_id,
            "installed": bool(m.installed),
            "models_dir": str(self.models_dir),
            "architecture": m.architecture,
            "expected_filename": expected_primary,
            "expected_files": [str(p) for p in self._candidate_files(model_id, m)],
            "links": m.links or {},
        }

        if not m.installed:
            report["missing"] = True
            report["error"] = "Model not installed"

        return report

    def get_model_tech(self, model_id: str) -> Dict[str, Any]:
        m = self.models.get(model_id)
        if not m:
            return {"ok": False, "error": f"Unknown model_id: {model_id}"}

        # Include strict-spec registry fields so UI/backend can make deterministic decisions
        # (runtime routing, compatibility gating, requirements warnings, phase-fix reference selection).
        return {
            "ok": True,
            "model_id": model_id,
            "name": m.name,
            "architecture": m.architecture,
            "vram_required": m.vram_required,
            "sdr": m.sdr,
            "fullness": m.fullness,
            "bleedless": m.bleedless,
            "links": m.links or {},
            "runtime": getattr(m, "runtime", None),
            "compatibility": getattr(m, "compatibility", None),
            "requirements": getattr(m, "requirements", None),
            "phase_fix": getattr(m, "phase_fix", None),
            "capabilities": getattr(m, "capabilities", None),
            "artifacts": getattr(m, "artifacts", None),
            "settings_semantics": getattr(m, "settings_semantics", None),
            "recommended_settings": m.recommended_settings
            if isinstance(m.recommended_settings, dict)
            else None,
        }

    async def download_model(self, model_id: str) -> bool:
        """Best-effort download of the checkpoint/config declared in registry.

        NOTE: This is kept minimal; the full downloader lives in other parts of the project.
        """
        m = self.models.get(model_id)
        if not m or not m.links:
            return False

        try:
            import urllib.request

            self.models_dir.mkdir(parents=True, exist_ok=True)

            def download(url: str) -> bool:
                if not url:
                    return False
                name = self._url_basename(url)
                if not name:
                    return False
                dest = self.models_dir / name
                if dest.exists():
                    return True
                self._emit_download_progress(model_id, 0.0)
                urllib.request.urlretrieve(url, dest)
                self._emit_download_progress(model_id, 1.0)
                return dest.exists()

            ok_ckpt = download(m.links.get("checkpoint") or "")
            ok_cfg = True
            if m.links.get("config"):
                ok_cfg = download(m.links.get("config") or "")

            self._refresh_install_flags()
            return bool(ok_ckpt and ok_cfg and self.is_model_installed(model_id))
        except Exception as e:
            log.error(f"download_model failed for {model_id}: {e}")
            return False
