import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("StemSep")


def _merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested registry overrides without dropping base metadata."""

    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _coerce_path_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return str(value)


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
    guide_rank: Optional[int] = None
    guide_notes: Optional[List[str]] = None
    status: Optional[Dict[str, Any]] = None
    catalog_status: Optional[str] = None
    metrics_status: Optional[str] = None
    card_metrics: Optional[Dict[str, Any]] = None

    # === Strict-spec extensions (guide-as-spec) ===
    # These fields are additive/backwards compatible: older registries can omit them.
    runtime: Optional[Dict[str, Any]] = None
    compatibility: Optional[Dict[str, Any]] = None
    requirements: Optional[Dict[str, Any]] = None
    phase_fix: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    settings_semantics: Optional[Dict[str, Any]] = None
    install: Optional[Dict[str, Any]] = None
    quality_role: Optional[Any] = None
    best_for: Optional[List[str]] = None
    artifacts_risk: Optional[List[str]] = None
    vram_profile: Optional[str] = None
    chunk_overlap_policy: Optional[Dict[str, Any]] = None
    workflow_groups: Optional[List[str]] = None
    quality_axes: Optional[Dict[str, Any]] = None
    workflow_roles: Optional[List[str]] = None
    operating_profiles: Optional[Dict[str, Any]] = None
    content_fit: Optional[List[str]] = None
    download: Optional[Dict[str, Any]] = None
    availability: Optional[Dict[str, Any]] = None

    installed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.status, dict):
            d["readiness"] = self.status.get("readiness")
            d["simple_allowed"] = self.status.get("simple_allowed")
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
        return self.assets_dir / "catalog.runtime.json"

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
                    guide_rank=m.get("guide_rank"),
                    guide_notes=m.get("guide_notes"),
                    status=m.get("status"),
                    catalog_status=m.get("catalog_status"),
                    metrics_status=m.get("metrics_status"),
                    card_metrics=m.get("card_metrics"),
                    # Strict-spec additions (optional in registry)
                    runtime=m.get("runtime"),
                    compatibility=m.get("compatibility"),
                    requirements=m.get("requirements"),
                    phase_fix=m.get("phase_fix"),
                    capabilities=m.get("capabilities"),
                    artifacts=m.get("artifacts"),
                    settings_semantics=m.get("settings_semantics"),
                    install=m.get("install"),
                    quality_role=m.get("quality_role"),
                    best_for=m.get("best_for"),
                    artifacts_risk=m.get("artifacts_risk"),
                    vram_profile=m.get("vram_profile"),
                    chunk_overlap_policy=m.get("chunk_overlap_policy"),
                    workflow_groups=m.get("workflow_groups"),
                    quality_axes=m.get("quality_axes"),
                    workflow_roles=m.get("workflow_roles"),
                    operating_profiles=m.get("operating_profiles"),
                    content_fit=m.get("content_fit"),
                    download=m.get("download"),
                    availability=m.get("availability"),
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

    def _normalize_resolved_bundle(
        self, resolved_bundle: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(resolved_bundle, dict):
            return {}

        out = dict(resolved_bundle)

        for nested_key in (
            "resolved_bundle",
            "resolvedBundle",
            "bundle",
            "install_bundle",
            "installBundle",
        ):
            nested = out.get(nested_key)
            if isinstance(nested, dict):
                merged = dict(nested)
                merged.update(out)
                out = merged
                break

        paths = out.get("paths")
        if isinstance(paths, dict):
            for alias, canonical in (
                ("checkpoint", "checkpoint_path"),
                ("config", "config_path"),
                ("model", "checkpoint_path"),
                ("weights", "checkpoint_path"),
            ):
                if canonical not in out or not _coerce_path_string(out.get(canonical)):
                    candidate = _coerce_path_string(paths.get(alias))
                    if candidate:
                        out[canonical] = candidate

        alias_groups = {
            "checkpoint_path": (
                "checkpointPath",
                "checkpoint",
                "model_path",
                "modelPath",
                "weights_path",
                "weightsPath",
            ),
            "config_path": ("configPath", "config", "yaml_path", "yamlPath"),
            "download": ("downloadManifest", "manifest"),
            "installation": ("install",),
            "availability": ("status",),
            "artifacts": ("files",),
            "selection_type": ("selectionType",),
            "selection_id": ("selectionId",),
            "execution_plan": ("executionPlan",),
        }
        for canonical, aliases in alias_groups.items():
            if canonical in out and out.get(canonical) not in (None, "", []):
                continue
            for alias in aliases:
                candidate = out.get(alias)
                if candidate is not None and candidate != "":
                    out[canonical] = candidate
                    break

        return out

    def _model_family(self, model: Optional[ModelInfo]) -> str:
        if not model:
            return "other"
        runtime = model.runtime if isinstance(model.runtime, dict) else {}
        engine = str(runtime.get("engine") or "").strip().lower()
        model_type = str(runtime.get("model_type") or "").strip().lower()
        architecture = str(model.architecture or "").strip().lower()
        if engine == "demucs_native" or "demucs" in architecture or "demucs" in model_type:
            return "demucs"
        if architecture == "vr" or model_type == "vr":
            return "vr"
        if "mdx" in architecture or "mdx" in model_type:
            return "mdx"
        if (
            engine in {"msst_builtin", "custom_builtin_variant"}
            or "roformer" in architecture
            or "roformer" in model_type
            or "scnet" in architecture
            or "scnet" in model_type
            or "apollo" in architecture
            or "apollo" in model_type
            or "bandit" in architecture
            or "bandit" in model_type
        ):
            return "msst"
        return "other"

    def _artifact_relative_path(
        self, model_id: str, model: Optional[ModelInfo], filename: str
    ) -> str:
        if model and isinstance(model.download, dict):
            for item in model.download.get("artifacts") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("filename") or "").strip() != filename:
                    continue
                relative = str(item.get("relative_path") or "").strip()
                if relative:
                    return relative.replace("\\", "/")
        return f"{self._model_family(model)}/{model_id}/{filename}".replace("\\", "/")

    def _is_stale_legacy_mirror_url(self, url: Optional[str]) -> bool:
        if not isinstance(url, str):
            return False
        normalized = url.strip().lower()
        return (
            "github.com/engdahlz/stemsep-models/releases/download/models-v2-2026-01-01/"
            in normalized
        )

    def _legacy_artifact_paths(
        self, model_id: str, kind: str, filename: str
    ) -> List[Path]:
        candidates = [self.models_dir / filename]
        suffix = Path(filename).suffix
        if suffix:
            candidates.append(self.models_dir / f"{model_id}{suffix}")
        if kind == "config":
            candidates.append(self.models_dir / f"{model_id}.yaml")
            candidates.append(self.models_dir / f"{model_id}.yml")
        deduped: List[Path] = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    def resolve_download_manifest(self, model_id: str) -> Dict[str, Any]:
        model = self.models.get(model_id)
        if not model:
            return {
                "model_id": model_id,
                "family": "other",
                "mode": "unavailable",
                "install_mode": "unknown",
                "artifact_count": 0,
                "downloadable_artifact_count": 0,
                "strategy": "blocked_non_public",
                "source_policy": "unknown",
                "sources": [],
                "artifacts": [],
                "manual_instructions": [],
                "availability": {
                    "class": "blocked_non_public",
                    "reason": "Unknown model_id",
                },
            }

        runtime = model.runtime if isinstance(model.runtime, dict) else {}
        install = model.install if isinstance(model.install, dict) else {}
        links = model.links if isinstance(model.links, dict) else {}
        availability = model.availability if isinstance(model.availability, dict) else {}
        install_mode = (
            str(install.get("mode") or runtime.get("install_mode") or "direct")
            .strip()
            .lower()
        )
        availability_class = str(availability.get("class") or "").strip().lower()
        manual_registry = availability_class == "manual_import" or install_mode == "manual" or bool(
            runtime.get("requires_manual_assets")
        )

        sources: List[Dict[str, Any]] = []
        seen_sources = set()

        def push_source(role: str, source_obj: Any, manual: bool) -> None:
            if isinstance(source_obj, dict):
                url = source_obj.get("url")
                channel = source_obj.get("channel")
                priority = source_obj.get("priority")
                auth = source_obj.get("auth")
                verified = source_obj.get("verified")
                host = source_obj.get("host")
            else:
                url = source_obj
                channel = None
                priority = None
                auth = None
                verified = None
                host = None
            if not isinstance(url, str):
                return
            trimmed = url.strip()
            if not trimmed or trimmed in seen_sources:
                return
            seen_sources.add(trimmed)
            if not host:
                try:
                    host = trimmed.split("://", 1)[-1].split("/", 1)[0].strip()
                except Exception:
                    host = ""
            sources.append(
                {
                    "role": role,
                    "url": trimmed,
                    "host": host or "unknown",
                    "manual": manual,
                    "channel": channel or ("mirror" if "github.com/engdahlz/stemsep-models" in trimmed else "upstream"),
                    "priority": int(priority or (len(sources) + 1)),
                    "auth": auth or "none",
                    "verified": True if verified is None else bool(verified),
                }
            )

        push_source("checkpoint", links.get("checkpoint"), manual_registry)
        push_source("config", links.get("config"), manual_registry)
        push_source("homepage", links.get("homepage"), True)
        if isinstance(model.download, dict):
            for source in model.download.get("sources") or []:
                push_source(
                    str(source.get("role") or "source"),
                    source,
                    bool(source.get("manual", True)),
                )

        artifacts: List[Dict[str, Any]] = []
        seen_rel_paths = set()

        def push_artifact(
            kind: str,
            filename: Optional[str],
            selected_source: Optional[str],
            *,
            required: bool = True,
            manual: bool = False,
            sha256: Optional[str] = None,
            size_bytes: Optional[int] = None,
            relative_path: Optional[str] = None,
            artifact_sources: Optional[List[Dict[str, Any]]] = None,
        ) -> None:
            if not isinstance(filename, str):
                return
            trimmed = filename.strip()
            if not trimmed or ".MISSING" in trimmed:
                return
            rel = (
                relative_path.strip().replace("\\", "/")
                if isinstance(relative_path, str) and relative_path.strip()
                else self._artifact_relative_path(model_id, model, trimmed)
            )
            if rel in seen_rel_paths:
                return
            seen_rel_paths.add(rel)
            canonical = self.models_dir / Path(rel)
            legacy_candidates = self._legacy_artifact_paths(model_id, kind, trimmed)
            exists = canonical.exists() or any(path.exists() for path in legacy_candidates)
            chosen_source = selected_source.strip() if isinstance(selected_source, str) and selected_source.strip() else None
            stale_legacy_mirror = self._is_stale_legacy_mirror_url(chosen_source)
            artifacts.append(
                {
                    "kind": kind,
                    "filename": trimmed,
                    "relative_path": rel,
                    "required": required,
                    "manual": manual or stale_legacy_mirror,
                    "exists": exists,
                    "source": chosen_source,
                    "source_host": chosen_source.split("://", 1)[-1].split("/", 1)[0]
                    if chosen_source
                    else None,
                    "sources": artifact_sources or ([] if chosen_source is None else [{"url": chosen_source}]),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
            )

        if isinstance(model.download, dict) and isinstance(model.download.get("artifacts"), list):
            for item in model.download.get("artifacts") or []:
                if not isinstance(item, dict):
                    continue
                artifact_sources: List[Dict[str, Any]] = []
                for source in item.get("sources") or []:
                    if not isinstance(source, dict):
                        continue
                    source_url = str(source.get("url") or "").strip()
                    if not source_url:
                        continue
                    artifact_sources.append(
                        {
                            "url": source_url,
                            "channel": source.get("channel") or ("mirror" if "github.com/engdahlz/stemsep-models" in source_url else "upstream"),
                            "priority": int(source.get("priority") or (len(artifact_sources) + 1)),
                            "auth": source.get("auth") or "none",
                            "verified": True if source.get("verified") is None else bool(source.get("verified")),
                            "host": source.get("host")
                            or source_url.split("://", 1)[-1].split("/", 1)[0],
                        }
                    )
                legacy_source = str(item.get("source") or "").strip()
                if legacy_source and not artifact_sources:
                    artifact_sources.append(
                        {
                            "url": legacy_source,
                            "channel": item.get("channel") or ("mirror" if "github.com/engdahlz/stemsep-models" in legacy_source else "upstream"),
                            "priority": 1,
                            "auth": item.get("auth") or "none",
                            "verified": True if item.get("verified") is None else bool(item.get("verified")),
                            "host": legacy_source.split("://", 1)[-1].split("/", 1)[0],
                        }
                    )
                selected_source = None
                if artifact_sources:
                    selected_source = sorted(
                        artifact_sources,
                        key=lambda source: int(source.get("priority") or 0),
                    )[0].get("url")
                push_artifact(
                    str(item.get("kind") or "aux"),
                    item.get("filename"),
                    selected_source,
                    required=bool(item.get("required", True)),
                    manual=bool(item.get("manual", manual_registry or not artifact_sources)),
                    sha256=item.get("sha256"),
                    size_bytes=item.get("size_bytes"),
                    relative_path=item.get("relative_path"),
                    artifact_sources=artifact_sources,
                )
        else:
            artifact_ckpt = self._artifact_filename(model, "primary")
            artifact_cfg = self._artifact_filename(model, "config")
            runtime_ckpt = runtime.get("checkpoint_ref")
            runtime_cfg = runtime.get("config_ref")

            checkpoint_source = links.get("checkpoint")
            config_source = links.get("config")

            push_artifact(
                "checkpoint",
                artifact_ckpt or runtime_ckpt or self._url_basename(checkpoint_source or ""),
                checkpoint_source,
                manual=manual_registry and not checkpoint_source,
            )
            if artifact_cfg or runtime_cfg or config_source:
                push_artifact(
                    "config",
                    artifact_cfg or runtime_cfg or self._url_basename(config_source or ""),
                    config_source,
                    manual=manual_registry and not config_source,
                )

        for required_file in runtime.get("required_files") or []:
            filename = str(required_file or "").strip()
            if not filename:
                continue
            if any(artifact["filename"] == filename for artifact in artifacts):
                continue
            push_artifact("aux", filename, None, manual=True)

        manual_instructions = [
            str(note)
            for note in (install.get("notes") or [])
            if isinstance(note, str) and note.strip()
        ]
        needs_manual_guidance = manual_registry or any(
            self._is_stale_legacy_mirror_url(source.get("url")) for source in sources
        )
        if needs_manual_guidance and not any("Place the required files" in note for note in manual_instructions):
            manual_instructions.append(
                f"Place the required files under {self._model_family(model)}/{model_id}/ inside the configured models directory."
            )
        if any(self._is_stale_legacy_mirror_url(source.get("url")) for source in sources):
            manual_instructions.append(
                "Legacy StemSep mirror links for this model are stale. Use the listed source references or install the files manually until a verified direct upstream is curated."
            )
        if manual_registry and not sources:
            manual_instructions.append(
                "This model does not expose verified direct-download URLs yet. Use the listed required filenames and install notes for manual setup."
            )

        downloadable_artifact_count = sum(
            1 for artifact in artifacts if not artifact["manual"] and artifact.get("source")
        )
        if availability_class == "blocked_non_public":
            mode = "unavailable"
        elif not artifacts:
            mode = "unavailable"
        elif downloadable_artifact_count == 0:
            mode = "manual"
        elif len(artifacts) > 1:
            mode = "multi_artifact_direct"
        else:
            mode = "direct"
        source_policy = (
            "legacy_mirror_manual"
            if any(self._is_stale_legacy_mirror_url(source["url"]) for source in sources)
            else "mirror_fallback"
            if any("github.com/engdahlz/stemsep-models" in source["url"] for source in sources)
            else "manual_verified"
            if manual_registry
            else "upstream_direct"
        )

        return {
            "model_id": model_id,
            "family": self._model_family(model),
            "mode": mode,
            "strategy": availability_class or (
                "manual_import" if manual_registry else "direct"
            ),
            "install_mode": install_mode,
            "artifact_count": len(artifacts),
            "downloadable_artifact_count": downloadable_artifact_count,
            "source_policy": source_policy,
            "sources": sources,
            "artifacts": artifacts,
            "manual_instructions": manual_instructions,
            "availability": {
                "class": availability_class or ("manual_import" if manual_registry else "direct"),
                "reason": availability.get("reason"),
            },
        }

    def get_model_file_bundle(
        self,
        model_id: str,
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manifest = self.resolve_download_manifest(model_id)
        resolved_bundle = self._normalize_resolved_bundle(resolved_bundle)
        strict_resolved_bundle = bool(resolved_bundle)

        if resolved_bundle:
            manifest = dict(manifest)
            for key in ("download", "availability"):
                value = resolved_bundle.get(key)
                if isinstance(value, dict):
                    manifest[key] = _merge_dicts(
                        manifest.get(key) if isinstance(manifest.get(key), dict) else {},
                        value,
                    )
            if (
                _coerce_path_string(resolved_bundle.get("checkpoint_path"))
                or _coerce_path_string(resolved_bundle.get("config_path"))
                or isinstance(resolved_bundle.get("artifacts"), list)
            ):
                manifest["availability"] = (
                    resolved_bundle.get("availability")
                    if isinstance(resolved_bundle.get("availability"), dict)
                    else {"class": "direct"}
                )

        artifacts = (
            resolved_bundle.get("artifacts")
            if isinstance(resolved_bundle.get("artifacts"), list)
            else manifest.get("artifacts") or []
        )
        missing = []
        relative_paths = []
        checkpoint_path: Optional[Path] = None
        config_path: Optional[Path] = None

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            rel = str(artifact.get("relative_path") or "").strip()
            if rel:
                relative_paths.append(rel)
            canonical = self.models_dir / Path(rel) if rel else None
            legacy = (
                []
                if strict_resolved_bundle
                else self._legacy_artifact_paths(
                    model_id,
                    str(artifact.get("kind") or "aux"),
                    str(artifact.get("filename") or ""),
                )
            )
            resolved_existing = None
            if resolved_bundle:
                artifact_kind = str(artifact.get("kind") or "").strip().lower()
                if artifact_kind == "config":
                    candidate = _coerce_path_string(resolved_bundle.get("config_path"))
                    if candidate:
                        resolved_existing = Path(candidate)
                else:
                    candidate = _coerce_path_string(
                        resolved_bundle.get("checkpoint_path")
                        or resolved_bundle.get("model_path")
                        or resolved_bundle.get("weights_path")
                    )
                    if candidate:
                        resolved_existing = Path(candidate)

            existing = None
            if strict_resolved_bundle:
                if resolved_existing is not None and resolved_existing.exists():
                    existing = resolved_existing
                elif canonical and canonical.exists():
                    existing = canonical
            else:
                existing = (
                    canonical
                    if canonical and canonical.exists()
                    else resolved_existing
                    if resolved_existing is not None and resolved_existing.exists()
                    else next((path for path in legacy if path.exists()), None)
                )
            if bool(artifact.get("required", True)) and existing is None:
                missing.append(rel or str(artifact.get("filename") or ""))
            kind = str(artifact.get("kind") or "")
            if kind != "config" and checkpoint_path is None and existing is not None:
                checkpoint_path = existing
            if kind == "config" and config_path is None and existing is not None:
                config_path = existing

        if checkpoint_path is None:
            candidate = _coerce_path_string(resolved_bundle.get("checkpoint_path"))
            if candidate:
                checkpoint_path = Path(candidate)
        if config_path is None:
            candidate = _coerce_path_string(resolved_bundle.get("config_path"))
            if candidate:
                config_path = Path(candidate)

        installation = (
            resolved_bundle.get("installation")
            if isinstance(resolved_bundle.get("installation"), dict)
            else None
        )
        if installation and "installed" in installation:
            installed = bool(installation.get("installed"))
        elif checkpoint_path is not None or config_path is not None:
            paths_to_check = [p for p in (checkpoint_path, config_path) if p is not None]
            installed = bool(paths_to_check) and all(p.exists() for p in paths_to_check)
        else:
            installed = len(artifacts) > 0 and len(missing) == 0

        return {
            "model_id": model_id,
            "download": manifest,
            "availability": manifest.get("availability"),
            "installation": {
                "installed": installed,
                "missing_artifacts": missing,
                "relative_paths": relative_paths,
            },
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "artifacts": artifacts,
            "resolved_bundle": resolved_bundle or None,
        }

    def _is_installed(
        self,
        model_id: str,
        model: Optional[ModelInfo],
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not model_id:
            return False
        bundle = self.get_model_file_bundle(model_id, resolved_bundle=resolved_bundle)
        installation = bundle.get("installation") or {}
        return bool(installation.get("installed"))

    def _candidate_files(
        self,
        model_id: str,
        model: Optional[ModelInfo],
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        bundle = self.get_model_file_bundle(model_id, resolved_bundle=resolved_bundle)
        strict_resolved_bundle = bool(self._normalize_resolved_bundle(resolved_bundle))
        candidates: List[Path] = []
        seen = set()

        for direct_path in (
            bundle.get("checkpoint_path"),
            bundle.get("config_path"),
        ):
            if direct_path:
                path = Path(direct_path)
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)

        for artifact in bundle.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            rel = str(artifact.get("relative_path") or "").strip()
            if rel:
                path = self.models_dir / Path(rel)
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)
            filename = str(artifact.get("filename") or "").strip()
            kind = str(artifact.get("kind") or "aux")
            if not strict_resolved_bundle:
                for legacy in self._legacy_artifact_paths(model_id, kind, filename):
                    if legacy not in seen:
                        seen.add(legacy)
                        candidates.append(legacy)

        if not candidates and not strict_resolved_bundle:
            for ext in [".ckpt", ".pth", ".pt", ".onnx", ".safetensors", ".yaml", ".yml"]:
                path = self.models_dir / f"{model_id}{ext}"
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)

        return candidates

    def _refresh_install_flags(self):
        self.installed_models = {}
        for mid, model in self.models.items():
            bundle = self.get_model_file_bundle(mid)
            installation = bundle.get("installation") or {}
            installed = bool(installation.get("installed"))
            if installed:
                primary = bundle.get("checkpoint_path") or bundle.get("config_path")
                if primary is None:
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

    def get_expected_local_filename(
        self,
        model_id: str,
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the expected primary local filename for a registry model.

        This is used by the separation stack to translate a registry `model_id`
        into a concrete artifact filename stored under `models_dir`.

        Preference order:
        - `links.checkpoint` basename (most models)
        - `links.config` basename (config-only models, e.g. Demucs yaml)
        - fallback for well-known demucs ids: `{model_id}.yaml`
        """
        m = self.models.get(model_id)
        if not m and resolved_bundle is None:
            return None

        bundle = self.get_model_file_bundle(model_id, resolved_bundle=resolved_bundle)
        family = str((bundle.get("download") or {}).get("family") or "").lower()
        config_path = bundle.get("config_path")
        if family == "demucs" and config_path:
            return Path(config_path).name
        checkpoint_path = bundle.get("checkpoint_path")
        if checkpoint_path:
            return Path(checkpoint_path).name
        if config_path:
            return Path(config_path).name

        # Prefer explicit artifact filename when available
        artifact_ckpt = self._artifact_filename(m, "primary")
        if artifact_ckpt:
            return artifact_ckpt

        # Special-case: KimberleyJSN/melbandroformer provides a standalone checkpoint named
        # "MelBandRoformer.ckpt" without a config. Our runtime uses audio-separator, which
        # has a built-in Roformer entry expecting the canonical filename below.
        if model_id == "mel-band-roformer-kim":
            return "vocals_mel_band_roformer.ckpt"

        links = m.links if m and isinstance(m.links, dict) else {}
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
        if m and (m.architecture or "").lower() == "demucs":
            return f"{model_id}.yaml"

        return None

    def is_model_installed(
        self,
        model_id: str,
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._refresh_install_flags()
        m = self.models.get(model_id)
        if resolved_bundle is not None:
            bundle = self.get_model_file_bundle(model_id, resolved_bundle=resolved_bundle)
            installation = bundle.get("installation") or {}
            return bool(installation.get("installed"))
        return bool(m and m.installed)

    def ensure_model_ready(
        self,
        model_id: str,
        auto_repair_filenames: bool = True,
        copy_instead_of_rename: bool = True,
        resolved_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility API used by python-bridge.

        This version does not perform complex repair logic. It reports missing and provides URLs.
        """
        self._refresh_install_flags()
        m = self.models.get(model_id)
        if not m and resolved_bundle is None:
            return {
                "ok": False,
                "model_id": model_id,
                "installed": False,
                "error": f"Unknown model_id: {model_id}",
            }
        if not m:
            m = ModelInfo(id=model_id, name=model_id, architecture="")

        expected_primary = self.get_expected_local_filename(
            model_id, resolved_bundle=resolved_bundle
        )
        bundle = self.get_model_file_bundle(model_id, resolved_bundle=resolved_bundle)
        checkpoint_path = bundle.get("checkpoint_path")
        config_path = bundle.get("config_path")
        availability = bundle.get("availability") or {}
        availability_class = str(availability.get("class") or "").strip().lower()
        if availability_class == "blocked_non_public":
            return {
                "ok": False,
                "model_id": model_id,
                "installed": False,
                "error": availability.get("reason")
                or f"Model '{model_id}' is blocked and cannot be executed.",
                "download": bundle.get("download"),
                "installation": bundle.get("installation"),
            }

        # Best-effort filename repair:
        # If registry stores files under URL basenames (common), create a local alias
        # so tools expecting `model_id.yaml` + `model_id.<weights>` can work.
        allow_legacy_alias_repair = auto_repair_filenames and not bool(
            self._normalize_resolved_bundle(resolved_bundle)
        )
        if allow_legacy_alias_repair and checkpoint_path and config_path:
            try:
                ckpt_src = Path(checkpoint_path)
                cfg_src = Path(config_path)

                if ckpt_src.exists() and cfg_src.exists():
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
        if allow_legacy_alias_repair and model_id == "mel-band-roformer-kim":
            try:
                ckpt_src = Path(checkpoint_path) if checkpoint_path else None
                ckpt_alias = self.models_dir / "vocals_mel_band_roformer.ckpt"
                if ckpt_src and ckpt_src.exists() and not ckpt_alias.exists():
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
        installed = bool((bundle.get("installation") or {}).get("installed"))
        if not installed:
            installed = bool(getattr(m, "installed", False))

        report: Dict[str, Any] = {
            "ok": installed,
            "model_id": model_id,
            "installed": installed,
            "models_dir": str(self.models_dir),
            "architecture": m.architecture,
            "expected_filename": expected_primary,
            "expected_files": [
                str(p)
                for p in self._candidate_files(
                    model_id, m, resolved_bundle=resolved_bundle
                )
            ],
            "links": m.links or {},
            "download": bundle.get("download"),
            "availability": availability,
            "installation": bundle.get("installation"),
            "resolved_bundle": bundle.get("resolved_bundle"),
        }

        if availability_class == "manual_import" and not m.installed:
            report["missing"] = True
            report["error"] = availability.get("reason") or "Model requires manual import"
        elif not m.installed:
            report["missing"] = True
            report["error"] = "Model not installed"

        return report

    def get_model_tech(self, model_id: str) -> Dict[str, Any]:
        m = self.models.get(model_id)
        if not m:
            return {"ok": False, "error": f"Unknown model_id: {model_id}"}
        bundle = self.get_model_file_bundle(model_id)

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
            "install": getattr(m, "install", None),
            "quality_role": getattr(m, "quality_role", None),
            "best_for": getattr(m, "best_for", None),
            "artifacts_risk": getattr(m, "artifacts_risk", None),
            "vram_profile": getattr(m, "vram_profile", None),
            "chunk_overlap_policy": getattr(m, "chunk_overlap_policy", None),
            "workflow_groups": getattr(m, "workflow_groups", None),
            "quality_axes": getattr(m, "quality_axes", None),
            "workflow_roles": getattr(m, "workflow_roles", None),
            "operating_profiles": getattr(m, "operating_profiles", None),
            "content_fit": getattr(m, "content_fit", None),
            "availability": getattr(m, "availability", None),
            "recommended_settings": m.recommended_settings
            if isinstance(m.recommended_settings, dict)
            else None,
            "download": bundle.get("download"),
            "installation": bundle.get("installation"),
            "installed": bool(m.installed),
        }

    async def download_model(self, model_id: str) -> bool:
        """Best-effort download of the checkpoint/config declared in registry.

        NOTE: This is kept minimal; the full downloader lives in other parts of the project.
        """
        m = self.models.get(model_id)
        if not m:
            return False

        try:
            import urllib.request

            self.models_dir.mkdir(parents=True, exist_ok=True)

            manifest = self.resolve_download_manifest(model_id)
            direct_artifacts = [
                artifact
                for artifact in (manifest.get("artifacts") or [])
                if isinstance(artifact, dict)
                and not bool(artifact.get("manual"))
                and str(artifact.get("source") or "").strip()
            ]

            if not direct_artifacts:
                log.warning("download_model called for non-direct model %s", model_id)
                return False

            def download(artifact: Dict[str, Any]) -> bool:
                url = str(artifact.get("source") or "").strip()
                rel = str(artifact.get("relative_path") or "").strip()
                if not url or not rel:
                    return False
                dest = self.models_dir / Path(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    return True
                self._emit_download_progress(model_id, 0.0)
                urllib.request.urlretrieve(url, dest)
                self._emit_download_progress(model_id, 1.0)
                return dest.exists()

            ok = all(download(artifact) for artifact in direct_artifacts)

            self._refresh_install_flags()
            return bool(ok and self.is_model_installed(model_id))
        except Exception as e:
            log.error(f"download_model failed for {model_id}: {e}")
            return False
