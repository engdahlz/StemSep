#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an Ed25519 keypair for StemSep remote catalog signing.")
    parser.add_argument("--private-key-out", required=True)
    args = parser.parse_args()

    private_key_path = Path(args.private_key_out)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"Generated Ed25519 private key -> {private_key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
