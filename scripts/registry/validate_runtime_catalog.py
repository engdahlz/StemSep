#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from registry.catalog_v3_common import (
    DEFAULT_BOOTSTRAP_RUNTIME,
    DEFAULT_CATALOG_RUNTIME,
    DEFAULT_REMOTE_PUBLIC_KEY,
    DEFAULT_REMOTE_RUNTIME,
    DEFAULT_REMOTE_SIGNATURE,
    REPO_ROOT,
)


DEFAULT_SOURCE_REPORT = (
    REPO_ROOT / "StemSepApp" / "assets" / "registry" / "report_catalog_v3_sources.json"
)
DEFAULT_CATALOG_REPO_ROOT = REPO_ROOT.parent / "StemSep-catalog"


@dataclass
class Issue:
    level: str
    message: str
    path: Path | None = None

    def format(self) -> str:
        location = f" [{self.path}]" if self.path else ""
        return f"{self.level}: {self.message}{location}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def ed25519_public_key_pem(raw_public_key: bytes) -> bytes:
    spki_der = bytes.fromhex("302a300506032b6570032100") + raw_public_key
    encoded = base64.encodebytes(spki_der).replace(b"\n", b"")
    lines = [encoded[index : index + 64] for index in range(0, len(encoded), 64)]
    return (
        b"-----BEGIN PUBLIC KEY-----\n"
        + b"\n".join(lines)
        + b"\n-----END PUBLIC KEY-----\n"
    )


def validate_catalog_shape(name: str, path: Path, issues: list[Issue]) -> dict[str, Any] | None:
    if not path.exists():
        issues.append(Issue("ERROR", f"Missing {name} runtime catalog.", path))
        return None
    try:
        data = load_json(path)
    except Exception as exc:
        issues.append(Issue("ERROR", f"Failed to parse {name} runtime catalog: {exc}", path))
        return None

    if not isinstance(data, dict):
        issues.append(Issue("ERROR", f"{name} runtime catalog root must be an object.", path))
        return None

    if data.get("schema_version") != "catalog-runtime-v4":
        issues.append(
            Issue(
                "ERROR",
                f"{name} runtime catalog must declare schema_version='catalog-runtime-v4'.",
                path,
            )
        )

    for key in ("models", "recipes", "workflows", "sources"):
        if not isinstance(data.get(key), list):
            issues.append(
                Issue("ERROR", f"{name} runtime catalog is missing list field '{key}'.", path)
            )

    summary = data.get("summary")
    if not isinstance(summary, dict):
        issues.append(Issue("ERROR", f"{name} runtime catalog is missing summary.", path))
    else:
        expected_counts = {
            "models": len(data.get("models") or []),
            "recipes": len(data.get("recipes") or []),
            "workflows": len(data.get("workflows") or []),
            "sources": len(data.get("sources") or []),
        }
        for field, actual in expected_counts.items():
            recorded = summary.get(field)
            if isinstance(recorded, int) and recorded != actual:
                issues.append(
                    Issue(
                        "ERROR",
                        f"{name} summary.{field}={recorded} does not match actual count {actual}.",
                        path,
                    )
                )

    if not data.get("sources"):
        issues.append(Issue("ERROR", f"{name} runtime catalog contains no top-level sources.", path))

    if not isinstance(data.get("selection_index"), (dict, list)):
        issues.append(Issue("ERROR", f"{name} runtime catalog is missing selection_index.", path))

    return data


def validate_verified_models(runtime: dict[str, Any], path: Path, issues: list[Issue]) -> None:
    models = runtime.get("models") or []
    source_ids = {
        source.get("source_id")
        for source in runtime.get("sources") or []
        if isinstance(source, dict) and source.get("source_id")
    }

    for model in models:
        if not isinstance(model, dict) or model.get("catalog_status") != "verified":
            continue

        model_id = str(model.get("id") or "<missing-id>")
        if model.get("catalog_tier") != "verified":
            issues.append(
                Issue(
                    "ERROR",
                    f"{model_id}: verified catalog_status must also carry catalog_tier='verified'.",
                    path,
                )
            )

        availability_class = str(
            ((model.get("availability") or {}) if isinstance(model.get("availability"), dict) else {}).get("class")
            or ""
        ).strip()
        if availability_class not in {"direct", "mirror_fallback"}:
            issues.append(
                Issue(
                    "ERROR",
                    f"{model_id}: verified models must be installer-ready (availability.class direct/mirror_fallback).",
                    path,
                )
            )

        card_metrics = model.get("card_metrics")
        if not isinstance(card_metrics, dict):
            issues.append(Issue("ERROR", f"{model_id}: verified model is missing card_metrics.", path))
        else:
            labels = card_metrics.get("labels")
            values = card_metrics.get("values")
            if not isinstance(labels, list) or len(labels) != 3:
                issues.append(Issue("ERROR", f"{model_id}: card_metrics.labels must contain exactly 3 entries.", path))
            if not isinstance(values, list) or len(values) != 3:
                issues.append(Issue("ERROR", f"{model_id}: card_metrics.values must contain exactly 3 entries.", path))
            elif any(value in (None, "", "—") for value in values):
                issues.append(Issue("ERROR", f"{model_id}: card_metrics.values contains an empty slot.", path))

        verification = model.get("verification")
        if not isinstance(verification, dict) or verification.get("reachable") is not True:
            issues.append(
                Issue(
                    "ERROR",
                    f"{model_id}: verified model must carry verification.reachable=true.",
                    path,
                )
            )

        runtime_adapter = str(model.get("runtime_adapter") or "").strip()
        if not runtime_adapter:
            issues.append(Issue("ERROR", f"{model_id}: verified model is missing runtime_adapter.", path))

        artifacts = (
            (model.get("download") or {}).get("artifacts")
            if isinstance(model.get("download"), dict)
            else None
        )
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(
                Issue("ERROR", f"{model_id}: verified model is missing download.artifacts.", path)
            )
            continue

        for artifact in artifacts:
            if not isinstance(artifact, dict) or not artifact.get("required", True):
                continue
            canonical_path = str(artifact.get("canonical_path") or "").strip()
            if not canonical_path:
                issues.append(
                    Issue(
                        "ERROR",
                        f"{model_id}: required artifact is missing canonical_path.",
                        path,
                    )
                )
            artifact_source_ids = artifact.get("source_ids")
            if not isinstance(artifact_source_ids, list) or not artifact_source_ids:
                issues.append(
                    Issue(
                        "ERROR",
                        f"{model_id}: required artifact {artifact.get('filename') or canonical_path!r} is missing source_ids.",
                        path,
                    )
                )
                continue
            for source_id in artifact_source_ids:
                if source_id not in source_ids:
                    issues.append(
                        Issue(
                            "ERROR",
                            f"{model_id}: required artifact {artifact.get('filename') or canonical_path!r} references unknown source_id {source_id!r}.",
                            path,
                        )
                    )


def verify_remote_signature(
    payload_path: Path,
    signature_path: Path,
    public_key_path: Path,
    issues: list[Issue],
) -> None:
    for path in (payload_path, signature_path, public_key_path):
        if not path.exists():
            issues.append(Issue("ERROR", "Missing remote catalog signing artifact.", path))
            return

    try:
        payload = normalized_bytes(payload_path)
        signature = base64.b64decode(signature_path.read_text(encoding="utf-8").strip(), validate=True)
        public_key = base64.b64decode(public_key_path.read_text(encoding="utf-8").strip(), validate=True)
        openssl = shutil.which("openssl")
        if not openssl:
            issues.append(
                Issue(
                    "ERROR",
                    "OpenSSL is required to verify the remote catalog signature in this environment.",
                    signature_path,
                )
            )
            return

        with tempfile.TemporaryDirectory(prefix="stemsep_catalog_sig_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            payload_file = tmp_root / "catalog.runtime.remote.json"
            signature_file = tmp_root / "catalog.runtime.remote.json.sig"
            public_key_file = tmp_root / "catalog.public.pem"
            payload_file.write_bytes(payload)
            signature_file.write_bytes(signature)
            public_key_file.write_bytes(ed25519_public_key_pem(public_key))
            completed = subprocess.run(
                [
                    openssl,
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_key_file),
                    "-rawin",
                    "-in",
                    str(payload_file),
                    "-sigfile",
                    str(signature_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or completed.stdout or "").strip()
                issues.append(
                    Issue(
                        "ERROR",
                        f"Remote catalog signature verification failed: {stderr or 'unknown OpenSSL error'}",
                        signature_path,
                    )
                )
    except Exception as exc:
        issues.append(Issue("ERROR", f"Failed to verify remote catalog signature: {exc}", signature_path))


def validate_catalog_repo_sync(
    catalog_repo_root: Path,
    local_remote_runtime: Path,
    local_remote_sig: Path,
    local_remote_key: Path,
    issues: list[Issue],
) -> None:
    if not catalog_repo_root.exists():
        issues.append(Issue("WARN", "StemSep-catalog checkout not found; skipping repo sync validation.", catalog_repo_root))
        return

    repo_runtime = catalog_repo_root / "catalog.runtime.remote.json"
    repo_sig = catalog_repo_root / "catalog.runtime.remote.json.sig"
    repo_key = catalog_repo_root / "catalog.public.ed25519.txt"
    for lhs, rhs, label in (
        (local_remote_runtime, repo_runtime, "remote runtime payload"),
        (local_remote_sig, repo_sig, "remote runtime signature"),
        (local_remote_key, repo_key, "remote runtime public key"),
    ):
        if not rhs.exists():
            issues.append(Issue("ERROR", f"StemSep-catalog is missing {label}.", rhs))
            continue
        if normalized_bytes(lhs) != normalized_bytes(rhs):
            issues.append(
                Issue(
                    "ERROR",
                    f"App repo copy of {label} is out of sync with StemSep-catalog.",
                    rhs,
                )
            )


def validate_source_report(path: Path, issues: list[Issue]) -> None:
    if not path.exists():
        issues.append(Issue("ERROR", "Missing source verification report.", path))
        return
    try:
        report = load_json(path)
    except Exception as exc:
        issues.append(Issue("ERROR", f"Failed to parse source verification report: {exc}", path))
        return

    summary = report.get("summary")
    if not isinstance(summary, dict):
        issues.append(Issue("ERROR", "Source verification report is missing summary.", path))
        return
    if not summary.get("checked_at"):
        issues.append(Issue("ERROR", "Source verification report is missing checked_at.", path))
    failed = int(summary.get("failed") or 0)
    if failed:
        issues.append(
            Issue(
                "WARN",
                f"Source verification report still contains {failed} failing source checks. Review report_catalog_v3_sources.json for external blockers.",
                path,
            )
        )


def validate_provider_acceptance(path: Path, issues: list[Issue]) -> None:
    if not path.exists():
        issues.append(Issue("WARN", "Provider acceptance report not found; skipping acceptance validation.", path))
        return
    try:
        report = load_json(path)
    except Exception as exc:
        issues.append(Issue("ERROR", f"Failed to parse provider acceptance report: {exc}", path))
        return

    catalog_status = report.get("catalog_status")
    if not isinstance(catalog_status, dict):
        issues.append(Issue("ERROR", "Provider acceptance report is missing catalog_status.", path))
    else:
        fallback_kind = catalog_status.get("fallback_kind")
        if fallback_kind not in {"remote_current", "cached_fallback"}:
            issues.append(
                Issue(
                    "ERROR",
                    "Provider acceptance must run against a signed remote-derived catalog status.",
                    path,
                )
            )
        elif fallback_kind != "remote_current":
            issues.append(
                Issue(
                    "WARN",
                    f"Provider acceptance completed from {fallback_kind}; keep an eye on remote freshness if this persists.",
                    path,
                )
            )
        if catalog_status.get("signature_valid") is not True:
            issues.append(
                Issue(
                    "ERROR",
                    "Provider acceptance must record signature_valid=true.",
                    path,
                )
            )

    results = {
        str(item.get("provider")): item
        for item in report.get("results") or []
        if isinstance(item, dict) and item.get("provider")
    }
    expected_status = {
        "huggingface": {"installed"},
        "github_release": {"installed"},
        "github_raw": {"installed"},
        "google_drive": {"installed"},
        "proton_drive": {"installed", "blocked_external"},
    }
    for provider, allowed in expected_status.items():
        result = results.get(provider)
        if not result:
            issues.append(Issue("ERROR", f"Provider acceptance is missing result for {provider}.", path))
            continue
        if result.get("status") not in allowed:
            issues.append(
                Issue(
                    "ERROR",
                    f"Provider acceptance returned unexpected status for {provider}: {result.get('status')!r}.",
                    path,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate StemSep remote-first runtime catalog and signing artifacts."
    )
    parser.add_argument("--runtime", default=str(DEFAULT_CATALOG_RUNTIME))
    parser.add_argument("--bootstrap-runtime", default=str(DEFAULT_BOOTSTRAP_RUNTIME))
    parser.add_argument("--remote-runtime", default=str(DEFAULT_REMOTE_RUNTIME))
    parser.add_argument("--remote-signature", default=str(DEFAULT_REMOTE_SIGNATURE))
    parser.add_argument("--remote-public-key", default=str(DEFAULT_REMOTE_PUBLIC_KEY))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--catalog-repo-root", default=str(DEFAULT_CATALOG_REPO_ROOT))
    parser.add_argument("--acceptance-report")
    args = parser.parse_args()

    issues: list[Issue] = []

    runtime_path = Path(args.runtime)
    bootstrap_runtime_path = Path(args.bootstrap_runtime)
    remote_runtime_path = Path(args.remote_runtime)
    remote_signature_path = Path(args.remote_signature)
    remote_public_key_path = Path(args.remote_public_key)
    source_report_path = Path(args.source_report)
    catalog_repo_root = Path(args.catalog_repo_root)
    acceptance_report_path = (
        Path(args.acceptance_report)
        if args.acceptance_report
        else catalog_repo_root / "reports" / "provider_acceptance.json"
    )

    runtime_catalog = validate_catalog_shape("runtime", runtime_path, issues)
    bootstrap_catalog = validate_catalog_shape("bootstrap", bootstrap_runtime_path, issues)
    remote_catalog = validate_catalog_shape("remote", remote_runtime_path, issues)

    verify_remote_signature(
        remote_runtime_path,
        remote_signature_path,
        remote_public_key_path,
        issues,
    )

    if bootstrap_runtime_path.exists() and remote_runtime_path.exists():
        if normalized_bytes(bootstrap_runtime_path) != normalized_bytes(remote_runtime_path):
            issues.append(
                Issue(
                    "ERROR",
                    "Remote runtime payload must match the published bootstrap runtime payload.",
                    remote_runtime_path,
                )
            )

    validate_catalog_repo_sync(
        catalog_repo_root,
        remote_runtime_path,
        remote_signature_path,
        remote_public_key_path,
        issues,
    )
    validate_source_report(source_report_path, issues)
    validate_provider_acceptance(acceptance_report_path, issues)

    if runtime_catalog:
        validate_verified_models(runtime_catalog, runtime_path, issues)
    if bootstrap_catalog:
        validate_verified_models(bootstrap_catalog, bootstrap_runtime_path, issues)
    if remote_catalog:
        validate_verified_models(remote_catalog, remote_runtime_path, issues)

    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    issues_sorted = sorted(
        issues,
        key=lambda issue: (
            severity_order.get(issue.level, 99),
            str(issue.path or ""),
            issue.message,
        ),
    )

    for issue in issues_sorted:
        print(issue.format())

    errors = sum(1 for issue in issues_sorted if issue.level == "ERROR")
    warnings = sum(1 for issue in issues_sorted if issue.level == "WARN")
    print("")
    print(f"Runtime catalog validation complete. Errors: {errors}  Warnings: {warnings}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
