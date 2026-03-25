#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the compiled StemSep runtime catalog to the remote catalog path and sign it.",
    )
    parser.add_argument("--runtime", default=str(DEFAULT_BOOTSTRAP_RUNTIME))
    parser.add_argument("--remote-out", default=str(DEFAULT_REMOTE_RUNTIME))
    parser.add_argument("--signature-out", default=str(DEFAULT_REMOTE_SIGNATURE))
    parser.add_argument("--public-key-out", default=str(DEFAULT_REMOTE_PUBLIC_KEY))
    parser.add_argument("--private-key", required=True, help="Path to an Ed25519 PEM private key.")
    args = parser.parse_args()

    runtime_path = Path(args.runtime)
    remote_out = Path(args.remote_out)
    signature_out = Path(args.signature_out)
    public_key_out = Path(args.public_key_out)
    private_key_path = Path(args.private_key)

    payload = runtime_path.read_bytes()
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
    signature_out.write_text(
        base64.b64encode(signature).decode("ascii") + "\n",
        encoding="utf-8",
    )
    public_key_out.write_text(
        base64.b64encode(public_key).decode("ascii") + "\n",
        encoding="utf-8",
    )

    print(
        f"Published remote catalog -> {remote_out}\n"
        f"Signature -> {signature_out}\n"
        f"Public key -> {public_key_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
