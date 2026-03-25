#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import shutil
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from registry.catalog_v3_common import (
    DEFAULT_BOOTSTRAP_RUNTIME,
    DEFAULT_CATALOG_FRAGMENTS_ROOT,
    DEFAULT_REMOTE_PUBLIC_KEY,
    DEFAULT_REMOTE_RUNTIME,
    DEFAULT_REMOTE_SIGNATURE,
)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key_data = path.read_bytes()
    key = serialization.load_pem_private_key(key_data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Signing key must be an Ed25519 private key.")
    return key


def resolve_output_paths(
    *,
    catalog_repo_root: Path | None,
    remote_out: Path,
    signature_out: Path,
    public_key_out: Path,
) -> tuple[Path, Path, Path]:
    if not catalog_repo_root:
        return remote_out, signature_out, public_key_out
    return (
        catalog_repo_root / "catalog.runtime.remote.json",
        catalog_repo_root / "catalog.runtime.remote.json.sig",
        catalog_repo_root / "catalog.public.ed25519.txt",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the compiled StemSep runtime catalog to the remote catalog path and sign it.",
    )
    parser.add_argument("--runtime", default=str(DEFAULT_BOOTSTRAP_RUNTIME))
    parser.add_argument("--remote-out", default=str(DEFAULT_REMOTE_RUNTIME))
    parser.add_argument("--signature-out", default=str(DEFAULT_REMOTE_SIGNATURE))
    parser.add_argument("--public-key-out", default=str(DEFAULT_REMOTE_PUBLIC_KEY))
    parser.add_argument("--private-key", required=True, help="Path to an Ed25519 PEM private key.")
    parser.add_argument("--catalog-repo-root", help="Optional checkout of the dedicated StemSep-catalog repo.")
    parser.add_argument("--verification-report")
    parser.add_argument("--fragments-root", default=str(DEFAULT_CATALOG_FRAGMENTS_ROOT))
    args = parser.parse_args()

    runtime_path = Path(args.runtime)
    catalog_repo_root = Path(args.catalog_repo_root) if args.catalog_repo_root else None
    remote_out, signature_out, public_key_out = resolve_output_paths(
        catalog_repo_root=catalog_repo_root,
        remote_out=Path(args.remote_out),
        signature_out=Path(args.signature_out),
        public_key_out=Path(args.public_key_out),
    )
    private_key_path = Path(args.private_key)
    fragments_root = Path(args.fragments_root)
    verification_report = Path(args.verification_report) if args.verification_report else None

    payload = runtime_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    private_key = load_private_key(private_key_path)
    signature = private_key.sign(payload)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    remote_out.parent.mkdir(parents=True, exist_ok=True)
    signature_out.parent.mkdir(parents=True, exist_ok=True)
    public_key_out.parent.mkdir(parents=True, exist_ok=True)

    remote_out.write_bytes(payload)
    signature_out.write_bytes(base64.b64encode(signature) + b"\n")
    public_key_out.write_bytes(base64.b64encode(public_key) + b"\n")

    if catalog_repo_root:
        reports_dir = catalog_repo_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        if verification_report and verification_report.exists():
            shutil.copy2(verification_report, reports_dir / verification_report.name)
        if fragments_root.exists():
            destination = catalog_repo_root / "catalog-fragments"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(fragments_root, destination)

    print(
        f"Published remote catalog -> {remote_out}\n"
        f"Signature -> {signature_out}\n"
        f"Public key -> {public_key_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
