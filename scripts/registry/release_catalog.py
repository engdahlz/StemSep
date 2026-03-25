#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from compile_model_catalog_v3 import compile_catalog_v3
from registry.catalog_v3_common import (
    DEFAULT_BOOTSTRAP_RUNTIME,
    DEFAULT_CATALOG_FRAGMENTS_ROOT,
    DEFAULT_CATALOG_V3_SOURCE,
    DEFAULT_CATALOG_RUNTIME,
    DEFAULT_LEGACY_RUNTIME,
)
from verify_catalog_v3_sources import DEFAULT_REPORT, verify_catalog_sources


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot release entrypoint for the StemSep catalog repo."
    )
    parser.add_argument("--source", default=str(DEFAULT_CATALOG_V3_SOURCE))
    parser.add_argument("--fragments-root", default=str(DEFAULT_CATALOG_FRAGMENTS_ROOT))
    parser.add_argument("--runtime-out", default=str(DEFAULT_CATALOG_RUNTIME))
    parser.add_argument("--bootstrap-out", default=str(DEFAULT_BOOTSTRAP_RUNTIME))
    parser.add_argument("--legacy-out", default=str(DEFAULT_LEGACY_RUNTIME))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--catalog-repo-root", required=True)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--compute-sha256", action="store_true")
    parser.add_argument("--run-acceptance", action="store_true")
    parser.add_argument("--acceptance-report")
    parser.add_argument("--backend")
    args = parser.parse_args()

    source_path = Path(args.source)
    fragments_root = Path(args.fragments_root)
    report_path = Path(args.report)

    if not args.skip_verify:
        verify_catalog_sources(
            source_path,
            fragments_root,
            report_path,
            update_source=False,
            update_source_fragments=True,
            compute_sha256=args.compute_sha256,
            source_ids=None,
        )

    compile_catalog_v3(
        source_path=source_path,
        fragments_root=fragments_root,
        runtime_path=Path(args.runtime_out),
        bootstrap_runtime_path=Path(args.bootstrap_out),
        legacy_runtime_path=Path(args.legacy_out),
    )

    publish_script = Path(__file__).resolve().parent / "publish_remote_catalog.py"
    subprocess.run(
        [
            sys.executable,
            str(publish_script),
            "--runtime",
            args.bootstrap_out,
            "--private-key",
            args.private_key,
            "--catalog-repo-root",
            args.catalog_repo_root,
            "--verification-report",
            args.report,
            "--fragments-root",
            args.fragments_root,
        ],
        check=True,
    )
    if args.run_acceptance:
        acceptance_script = SCRIPT_DIR / "run_provider_acceptance.py"
        acceptance_cmd = [sys.executable, str(acceptance_script)]
        if args.backend:
            acceptance_cmd.extend(["--backend", args.backend])
        acceptance_cmd.extend(
            [
                "--catalog-repo-root",
                args.catalog_repo_root,
                "--use-local-catalog-server",
            ]
        )
        if args.acceptance_report:
            acceptance_cmd.extend(["--report", args.acceptance_report])
        subprocess.run(acceptance_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
