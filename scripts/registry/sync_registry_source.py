#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
EXTRA_MODELS_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "extra_models.json"
GUIDE_OVERRIDES_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "guide_model_overrides.json"
LINK_FIXES_PATH = REPO_ROOT / "scripts" / "registry" / "registry_link_fixes.json"

DIRECT_PATCHES: Dict[str, Dict[str, Any]] = {
    "htdemucs": {
        "links": {
            "checkpoint": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "config": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
            "homepage": "https://github.com/facebookresearch/demucs",
        },
        "runtime": {
            "install_mode": "direct",
            "requires_manual_assets": False,
            "required_files": ["955717e8-8726e21a.th", "config_musdb18_htdemucs.yaml"],
        },
        "install": {
            "mode": "direct",
            "notes": [
                "StemSep downloads the Demucs config bag and official upstream checkpoint directly.",
                "Files are stored under demucs/htdemucs/ inside the configured model root.",
            ],
        },
        "download": {
            "sources": [
                {
                    "role": "official_release",
                    "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
                    "manual": False,
                },
                {
                    "role": "yaml_config",
                    "url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
                    "manual": False,
                },
            ],
            "artifacts": [
                {
                    "kind": "checkpoint",
                    "filename": "955717e8-8726e21a.th",
                    "relative_path": "demucs/htdemucs/955717e8-8726e21a.th",
                    "required": True,
                    "manual": False,
                    "source": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
                },
                {
                    "kind": "config",
                    "filename": "config_musdb18_htdemucs.yaml",
                    "relative_path": "demucs/htdemucs/config_musdb18_htdemucs.yaml",
                    "required": True,
                    "manual": False,
                    "source": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
                },
            ],
        },
    },
    "htdemucs-6s": {
        "links": {
            "checkpoint": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
            "config": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
            "homepage": "https://github.com/facebookresearch/demucs",
        },
        "runtime": {
            "install_mode": "direct",
            "requires_manual_assets": False,
            "required_files": ["5c90dfd2-34c22ccb.th", "config_htdemucs_6stems.yaml"],
        },
        "install": {
            "mode": "direct",
            "notes": [
                "StemSep downloads the 6-stem Demucs config bag and official upstream checkpoint directly.",
                "Files are stored under demucs/htdemucs-6s/ inside the configured model root.",
            ],
        },
        "download": {
            "sources": [
                {
                    "role": "official_release",
                    "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
                    "manual": False,
                },
                {
                    "role": "yaml_config",
                    "url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
                    "manual": False,
                },
            ],
            "artifacts": [
                {
                    "kind": "checkpoint",
                    "filename": "5c90dfd2-34c22ccb.th",
                    "relative_path": "demucs/htdemucs-6s/5c90dfd2-34c22ccb.th",
                    "required": True,
                    "manual": False,
                    "source": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
                },
                {
                    "kind": "config",
                    "filename": "config_htdemucs_6stems.yaml",
                    "relative_path": "demucs/htdemucs-6s/config_htdemucs_6stems.yaml",
                    "required": True,
                    "manual": False,
                    "source": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
                },
            ],
        },
    },
    "scnet-masked-xl": {
        "links": {
            "checkpoint": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
            "config": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
            "homepage": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.17",
        },
        "runtime": {
            "install_mode": "direct",
            "requires_manual_assets": False,
            "required_files": ["model_scnet_masked_ep_111_sdr_9.8286.ckpt", "config_musdb18_scnet_xl_ihf.yaml"],
        },
        "install": {
            "mode": "direct",
            "notes": [
                "StemSep downloads the published SCNet Masked XL IHF checkpoint and config directly.",
                "Files are stored under msst/scnet-masked-xl/ inside the configured model root.",
            ],
        },
        "download": {
            "sources": [
                {
                    "role": "checkpoint",
                    "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
                    "manual": False,
                },
                {
                    "role": "config",
                    "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
                    "manual": False,
                },
            ],
            "artifacts": [
                {
                    "kind": "checkpoint",
                    "filename": "model_scnet_masked_ep_111_sdr_9.8286.ckpt",
                    "relative_path": "msst/scnet-masked-xl/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
                    "required": True,
                    "manual": False,
                    "source": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
                },
                {
                    "kind": "config",
                    "filename": "config_musdb18_scnet_xl_ihf.yaml",
                    "relative_path": "msst/scnet-masked-xl/config_musdb18_scnet_xl_ihf.yaml",
                    "required": True,
                    "manual": False,
                    "source": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
                },
            ],
        },
    },
    "apollo-restoration": {
        "links": {
            "checkpoint": "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin",
            "config": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_apollo.yaml",
            "homepage": "https://huggingface.co/JusperLee/Apollo",
        },
        "runtime": {
            "install_mode": "direct",
            "requires_manual_assets": False,
            "required_files": ["pytorch_model.bin", "config_apollo.yaml"],
        },
        "install": {
            "mode": "direct",
            "notes": [
                "StemSep downloads the published Apollo restoration weight and config directly.",
                "Files are stored under msst/apollo-restoration/ inside the configured model root.",
            ],
        },
        "download": {
            "sources": [
                {
                    "role": "checkpoint",
                    "url": "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin",
                    "manual": False,
                },
                {
                    "role": "config",
                    "url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_apollo.yaml",
                    "manual": False,
                },
            ],
            "artifacts": [
                {
                    "kind": "checkpoint",
                    "filename": "pytorch_model.bin",
                    "relative_path": "msst/apollo-restoration/pytorch_model.bin",
                    "required": True,
                    "manual": False,
                    "source": "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin",
                },
                {
                    "kind": "config",
                    "filename": "config_apollo.yaml",
                    "relative_path": "msst/apollo-restoration/config_apollo.yaml",
                    "required": True,
                    "manual": False,
                    "source": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_apollo.yaml",
                },
            ],
        },
    },
    "bandit-dnr": {
        "links": {
            "homepage": "https://github.com/kwatcharasupat/bandit-v2"
        },
        "install": {
            "mode": "manual",
            "notes": [
                "BandIt v2 is documented in the research report, but the current registry does not yet have a stable public weights URL for automated download.",
                "Import the original checkpoint and YAML manually if you have access to the OG release or UVR download-center bundle.",
                "Place the files under msst/bandit-dnr/ inside the configured model root.",
            ],
        },
    },
    "drypd-side-extraction": {
        "links": {
            "homepage": "https://drive.google.com/drive/folders/1ZSUw6ZuhJusv7HE5eMa-MORKA0XbSEht?usp=sharing"
        },
        "install": {
            "mode": "manual",
            "notes": [
                "Side/difference extraction is distributed as a Drive bundle in the source report.",
                "Import the original checkpoint manually if you have access to the Drive folder.",
                "Place the file under scnet/drypd-side-extraction/ inside the configured model root.",
            ],
        },
    },
}


def make_manual_placeholder(
    *,
    model_id: str,
    name: str,
    architecture: str,
    homepage: str,
    description: str,
    tags: List[str] | None = None,
    stems: List[str] | None = None,
    required_files: List[str] | None = None,
) -> Dict[str, Any]:
    files = required_files or []
    return {
        "id": model_id,
        "name": name,
        "type": description,
        "architecture": architecture,
        "description": description,
        "tags": tags or [],
        "stems": stems or [],
        "links": {
            "checkpoint": None,
            "config": None,
            "homepage": homepage,
        },
        "runtime": {
            "install_mode": "manual",
            "requires_manual_assets": True,
            "required_files": files,
        },
        "install": {
            "mode": "manual",
            "notes": [
                f"Model source: {homepage}",
                "This model is tracked in the report coverage set but does not yet have a curated, verified direct artifact URL in StemSep.",
                "Import the original published artifact(s) manually if you have access to them.",
            ]
            + (
                [f"Expected filenames: {', '.join(files)}"]
                if files
                else []
            ),
        },
        "status": {
            "readiness": "manual",
            "simple_allowed": False,
            "blocking_reason": "Manual import required until curated direct artifacts are verified.",
        },
        "catalog_status": "manual_only",
    }


def make_direct_placeholder(
    *,
    model_id: str,
    name: str,
    architecture: str,
    checkpoint_url: str,
    description: str,
    checkpoint_filename: str | None = None,
    homepage: str | None = None,
    tags: List[str] | None = None,
    stems: List[str] | None = None,
    kind: str = "checkpoint",
) -> Dict[str, Any]:
    filename = checkpoint_filename or checkpoint_url.split("?", 1)[0].rsplit("/", 1)[-1]
    return {
        "id": model_id,
        "name": name,
        "type": description,
        "architecture": architecture,
        "description": description,
        "tags": tags or [],
        "stems": stems or [],
        "links": {
            "checkpoint": checkpoint_url,
            "config": None,
            "homepage": homepage or checkpoint_url,
        },
        "artifacts": {
            "primary": {
                "kind": kind,
                "filename": filename,
                "sha256": None,
                "size_bytes": None,
            },
            "config": None,
            "additional": [],
        },
        "runtime": {
            "install_mode": "direct",
            "requires_manual_assets": False,
            "required_files": [filename],
        },
        "install": {
            "mode": "direct",
            "notes": [
                "StemSep downloads this published artifact directly.",
            ],
        },
        "status": {
            "readiness": "experimental",
            "simple_allowed": True,
        },
        "catalog_status": "candidate",
    }


REPORT_PLACEHOLDERS: List[Dict[str, Any]] = [
    make_manual_placeholder(
        model_id="bs-roformer-inst-exp-value-residual",
        name="BS-Roformer Inst-EXP-Value-Residual",
        architecture="BS-Roformer",
        homepage="https://huggingface.co/pcunwa/BS-Roformer-Inst-EXP-Value-Residual",
        description="Community BS-Roformer instrumental residual variant from the report.",
        tags=["instrumental"],
        stems=["instrumental"],
    ),
    make_manual_placeholder(
        model_id="xlancelab-msr-9stems",
        name="xlancelab MSR Models (9 stems)",
        architecture="Other",
        homepage="https://github.com/ModistAndrew/xlance-msr",
        description="9-stem music source removal research bundle from xlancelab.",
        tags=["multi-stem"],
    ),
    make_manual_placeholder(
        model_id="bs-mamba2",
        name="BS Mamba2 (ZFTurbo)",
        architecture="Other",
        homepage="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.19",
        description="Experimental BS Mamba2 separator from the report.",
    ),
    make_manual_placeholder(
        model_id="bs-conformer-medium",
        name="BS Conformer Medium (ZFTurbo)",
        architecture="Other",
        homepage="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.18",
        description="Experimental BS Conformer Medium separator from the report.",
    ),
    make_manual_placeholder(
        model_id="gabox-inst-v6",
        name="Gabox INSTV6",
        architecture="Mel-Roformer",
        homepage="https://huggingface.co/GaboxR67",
        description="Gabox instrumental variant v6 from the report.",
        tags=["instrumental"],
        stems=["instrumental"],
    ),
    make_manual_placeholder(
        model_id="gabox-inst-v7n",
        name="Gabox INSTV7N",
        architecture="Mel-Roformer",
        homepage="https://huggingface.co/GaboxR67",
        description="Gabox instrumental variant v7n from the report.",
        tags=["instrumental"],
        stems=["instrumental"],
    ),
    make_manual_placeholder(
        model_id="becruily-deux",
        name="Deux (Becruily)",
        architecture="Mel-Roformer",
        homepage="https://huggingface.co/becruily/mel-band-roformer-deux",
        description="Becruily Deux community Roformer model.",
        tags=["vocals"],
        stems=["vocals"],
    ),
    make_direct_placeholder(
        model_id="amane-kapm-vocal",
        name="FullnessVocalModel / kapm (Aname-Tommy)",
        architecture="Mel-Roformer",
        checkpoint_url="https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/FullnessVocalModel.ckpt",
        description="Aname-Tommy FullnessVocalModel checkpoint from the report.",
        tags=["vocals"],
        stems=["vocals"],
    ),
    make_direct_placeholder(
        model_id="melband-roformer-syhftv3epsilon",
        name="MelBandRoformerSYHFTV3Epsilon (SYH99999)",
        architecture="Mel-Roformer",
        checkpoint_url="https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt",
        description="SYH99999 MelBand Roformer epsilon variant.",
        tags=["vocals"],
        stems=["vocals"],
    ),
    make_manual_placeholder(
        model_id="scnet-tran",
        name="SCNet Tran (ZFTurbo)",
        architecture="SCNet",
        homepage="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.14",
        description="SCNet Tran release tracked from the report.",
    ),
    make_direct_placeholder(
        model_id="uvr-mdxnet-main",
        name="UVR_MDXNET_Main",
        architecture="MDX-Net",
        checkpoint_url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/UVR_MDXNET_Main.onnx",
        description="Classic UVR MDX-Net Main ONNX model.",
        tags=["instrumental"],
        stems=["instrumental"],
        kind="onnx",
    ),
    make_direct_placeholder(
        model_id="mdxnet-inst-hq-2",
        name="UVR-MDX-NET-Inst_HQ_2",
        architecture="MDX-Net",
        checkpoint_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_2.onnx",
        description="Classic UVR MDX-Net Inst HQ 2 ONNX model.",
        tags=["instrumental"],
        stems=["instrumental"],
        kind="onnx",
    ),
    make_direct_placeholder(
        model_id="mdxnet-inst-hq-3",
        name="UVR-MDX-NET-Inst_HQ_3",
        architecture="MDX-Net",
        checkpoint_url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/UVR-MDX-NET-Inst_HQ_3.onnx",
        description="Classic UVR MDX-Net Inst HQ 3 ONNX model.",
        tags=["instrumental"],
        stems=["instrumental"],
        kind="onnx",
    ),
    make_direct_placeholder(
        model_id="uvr-denoise",
        name="UVR-DeNoise",
        architecture="VR",
        checkpoint_url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/VR_Arch/UVR-DeNoise.pth",
        description="Classic UVR DeNoise VR model.",
        tags=["denoise"],
        stems=["restored"],
    ),
    make_manual_placeholder(
        model_id="uvr-denoise-lite",
        name="UVR-DeNoise Lite",
        architecture="VR",
        homepage="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
        description="Classic UVR DeNoise Lite VR model.",
        tags=["denoise"],
        stems=["restored"],
        required_files=["UVR-DeNoise-Lite.pth"],
    ),
    make_direct_placeholder(
        model_id="mdx23c-instvoc-hq-2",
        name="MDX23C-8KFFT-InstVoc_HQ_2",
        architecture="MDX23C",
        checkpoint_url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
        description="MDX23C InstVoc HQ 2 checkpoint from the report.",
        tags=["instrumental", "vocals"],
        stems=["instrumental", "vocals"],
    ),
    make_manual_placeholder(
        model_id="mdx23c-similarity",
        name="MDX23C Similarity (wesleyr36)",
        architecture="MDX23C",
        homepage="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.10",
        description="MDX23C Similarity checkpoint family from the report.",
    ),
    make_direct_placeholder(
        model_id="demucs-v3-mdx",
        name="Demucs v3 mdx",
        architecture="Demucs",
        checkpoint_url="https://dl.fbaipublicfiles.com/demucs/mdx_final/6b9c2ca1-3fd82607.th",
        description="Demucs v3 mdx artifact from the report.",
    ),
    make_manual_placeholder(
        model_id="demucs-v3-mdx-extra",
        name="Demucs v3 mdx_extra",
        architecture="Demucs",
        homepage="https://dl.fbaipublicfiles.com/demucs/mdx_final/",
        description="Demucs v3 mdx_extra artifact tracked from the report.",
        required_files=["mdx_extra.th"],
    ),
    make_manual_placeholder(
        model_id="vr-arch-5-hp",
        name="VR Arch 5_HP (Karaoke-UVR)",
        architecture="VR",
        homepage="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
        description="VR Arch 5_HP karaoke model from the report.",
        required_files=["5_HP-Karaoke-UVR.pth"],
    ),
    make_manual_placeholder(
        model_id="vr-arch-6-hp",
        name="VR Arch 6_HP (Karaoke-UVR)",
        architecture="VR",
        homepage="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
        description="VR Arch 6_HP karaoke model from the report.",
        required_files=["6_HP-Karaoke-UVR.pth"],
    ),
    make_manual_placeholder(
        model_id="vr-arch-9-hp2-uvr",
        name="VR Arch 9_HP2-UVR (500m_1)",
        architecture="VR",
        homepage="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
        description="VR Arch 9_HP2-UVR model from the report.",
        required_files=["9_HP2-UVR.pth"],
    ),
    make_manual_placeholder(
        model_id="medleyvox",
        name="MedleyVox",
        architecture="Other",
        homepage="https://github.com/jeonchangbin49/medleyvox",
        description="MedleyVox singing voice separation project from the report.",
        tags=["vocals"],
    ),
    make_manual_placeholder(
        model_id="audiosep",
        name="AudioSep",
        architecture="Other",
        homepage="https://github.com/Audio-AGI/AudioSep",
        description="AudioSep zero-shot audio separation project from the report.",
    ),
    make_manual_placeholder(
        model_id="natanworkspace-melband-roformer",
        name="natanworkspace/melband_roformer",
        architecture="Mel-Roformer",
        homepage="https://huggingface.co/natanworkspace/melband_roformer",
        description="natanworkspace community MelBand Roformer repository from the report.",
        tags=["vocals"],
        stems=["vocals"],
    ),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_runtime(runtime: Dict[str, Any], architecture: str) -> Dict[str, Any]:
    normalized = deepcopy(runtime)
    allowed = normalized.get("allowed")
    preferred = normalized.get("preferred")
    engine = str(normalized.get("engine") or "").strip().lower()
    model_type = str(normalized.get("model_type") or "").strip().lower()
    architecture_lower = str(architecture or "").strip().lower()
    variant = str(normalized.get("variant") or "").strip().lower()

    if not isinstance(allowed, list) or not allowed:
        if engine == "demucs_native" or "demucs" in architecture_lower or "demucs" in model_type:
            allowed = ["stemsep-legacy"]
        elif engine in {"msst_builtin", "custom_builtin_variant"}:
            allowed = ["msst"]
        elif architecture_lower == "vr" or model_type == "vr":
            allowed = ["stemsep-legacy"]
        elif "mdx" in architecture_lower or "mdx" in model_type:
            allowed = ["stemsep-legacy"]
        elif variant == "demucs":
            allowed = ["stemsep-legacy"]
        elif (
            "roformer" in architecture_lower
            or "roformer" in model_type
            or "scnet" in architecture_lower
            or "scnet" in model_type
            or "apollo" in architecture_lower
            or "apollo" in model_type
            or "bandit" in architecture_lower
            or "bandit" in model_type
        ):
            allowed = ["msst"]
        else:
            allowed = ["stemsep-legacy"]

    if not preferred or preferred not in allowed:
        preferred = allowed[0]

    normalized["allowed"] = allowed
    normalized["preferred"] = preferred
    return normalized


def normalize_status(status: Dict[str, Any], install: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(status)
    readiness = str(normalized.get("readiness") or "").strip().lower()
    readiness_map = {
        "downloadable": "experimental",
        "ready": "verified",
    }
    if readiness in readiness_map:
        normalized["readiness"] = readiness_map[readiness]
    elif readiness not in {"verified", "experimental", "manual", "blocked", ""}:
        normalized["readiness"] = "experimental"

    if not normalized.get("readiness"):
        if str(install.get("mode") or runtime.get("install_mode") or "").strip().lower() == "manual":
            normalized["readiness"] = "manual"
        else:
            normalized["readiness"] = "experimental"
    return normalized


def normalize_extra_model(extra: Dict[str, Any]) -> Dict[str, Any]:
    architecture = str(extra.get("architecture") or "Other")
    runtime = normalize_runtime(
        extra.get("runtime") if isinstance(extra.get("runtime"), dict) else {},
        architecture,
    )
    install = extra.get("install") if isinstance(extra.get("install"), dict) else {}
    status = normalize_status(
        extra.get("status") if isinstance(extra.get("status"), dict) else {},
        install,
        runtime,
    )

    metrics = {
        "sdr": extra.get("sdr"),
        "fullness": extra.get("fullness"),
        "bleedless": extra.get("bleedless"),
        "aura_stft": extra.get("metrics", {}).get("aura_stft") if isinstance(extra.get("metrics"), dict) else None,
        "aura_mrstft": extra.get("metrics", {}).get("aura_mrstft") if isinstance(extra.get("metrics"), dict) else None,
        "log_wmse": extra.get("metrics", {}).get("log_wmse") if isinstance(extra.get("metrics"), dict) else None,
    }

    links = extra.get("links") if isinstance(extra.get("links"), dict) else {}
    download = extra.get("download") if isinstance(extra.get("download"), dict) else None
    if not links and download and isinstance(download.get("sources"), list) and download["sources"]:
        first_url = next(
            (
                str(source.get("url") or "").strip()
                for source in download["sources"]
                if isinstance(source, dict) and str(source.get("url") or "").strip()
            ),
            "",
        )
        links = {"homepage": first_url} if first_url else {}

    install_mode = str(install.get("mode") or runtime.get("install_mode") or "").strip().lower()
    catalog_status = extra.get("catalog_status")
    if not isinstance(catalog_status, str):
        catalog_status = "manual_only" if install_mode == "manual" else "candidate"

    normalized = {
        "id": extra.get("id"),
        "name": extra.get("name"),
        "type": extra.get("type") or str(extra.get("category") or ""),
        "architecture": architecture,
        "description": str(extra.get("description") or ""),
        "tags": extra.get("tags") if isinstance(extra.get("tags"), list) else [],
        "stems": extra.get("stems") if isinstance(extra.get("stems"), list) else [],
        "links": links,
        "artifacts": extra.get("artifacts") if isinstance(extra.get("artifacts"), dict) else None,
        "metrics": metrics,
        "vram_required": extra.get("vram_required"),
        "recommended_settings": extra.get("recommended_settings") if isinstance(extra.get("recommended_settings"), dict) else {},
        "settings_semantics": extra.get("settings_semantics") if isinstance(extra.get("settings_semantics"), dict) else None,
        "runtime": runtime,
        "status": status,
        "compatibility": extra.get("compatibility") if isinstance(extra.get("compatibility"), dict) else None,
        "requirements": extra.get("requirements") if isinstance(extra.get("requirements"), dict) else None,
        "phase_fix": extra.get("phase_fix") if isinstance(extra.get("phase_fix"), dict) else None,
        "capabilities": extra.get("capabilities") if isinstance(extra.get("capabilities"), dict) else None,
        "notes": extra.get("notes") if isinstance(extra.get("notes"), dict) else None,
        "guide_revision": extra.get("guide_revision"),
        "quality_profile": extra.get("quality_profile") if isinstance(extra.get("quality_profile"), dict) else None,
        "hardware_tiers": extra.get("hardware_tiers") if isinstance(extra.get("hardware_tiers"), list) else None,
        "stability_notes": extra.get("stability_notes") if isinstance(extra.get("stability_notes"), list) else None,
        "card_metrics": extra.get("card_metrics") if isinstance(extra.get("card_metrics"), dict) else None,
        "catalog_status": catalog_status,
        "metrics_status": extra.get("metrics_status"),
        "metrics_evidence": extra.get("metrics_evidence") if isinstance(extra.get("metrics_evidence"), list) else None,
        "install": install or None,
        "quality_role": extra.get("quality_role"),
        "best_for": extra.get("best_for") if isinstance(extra.get("best_for"), list) else None,
        "artifacts_risk": extra.get("artifacts_risk") if isinstance(extra.get("artifacts_risk"), list) else None,
        "vram_profile": extra.get("vram_profile"),
        "chunk_overlap_policy": extra.get("chunk_overlap_policy") if isinstance(extra.get("chunk_overlap_policy"), dict) else None,
        "workflow_groups": extra.get("workflow_groups") if isinstance(extra.get("workflow_groups"), list) else None,
        "quality_axes": extra.get("quality_axes") if isinstance(extra.get("quality_axes"), dict) else None,
        "workflow_roles": extra.get("workflow_roles") if isinstance(extra.get("workflow_roles"), list) else None,
        "operating_profiles": extra.get("operating_profiles") if isinstance(extra.get("operating_profiles"), dict) else None,
        "content_fit": extra.get("content_fit") if isinstance(extra.get("content_fit"), list) else None,
        "download": download,
    }
    return {key: value for key, value in normalized.items() if value is not None}


def sync_registry_source(write: bool) -> Dict[str, Any]:
    source_payload = load_json(SOURCE_PATH)
    models = source_payload.get("models")
    if not isinstance(models, list):
        raise ValueError("models.v2.source.json must contain a models list")

    by_id: Dict[str, Dict[str, Any]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        normalized = deepcopy(model)
        normalized["runtime"] = normalize_runtime(
            normalized.get("runtime") if isinstance(normalized.get("runtime"), dict) else {},
            str(normalized.get("architecture") or ""),
        )
        by_id[model_id] = normalized

    overrides = load_json(GUIDE_OVERRIDES_PATH).get("models") or {}
    if isinstance(overrides, dict):
        for model_id, overlay in overrides.items():
            if not isinstance(overlay, dict) or model_id not in by_id:
                continue
            by_id[model_id] = deep_merge(by_id[model_id], overlay)

    extra_models = load_json(EXTRA_MODELS_PATH).get("models") or []
    if isinstance(extra_models, list):
        for raw_model in extra_models:
            if not isinstance(raw_model, dict):
                continue
            normalized = normalize_extra_model(raw_model)
            model_id = str(normalized.get("id") or "").strip()
            if not model_id:
                continue
            if model_id in by_id:
                by_id[model_id] = deep_merge(by_id[model_id], normalized)
            else:
                by_id[model_id] = normalized

    link_fixes = load_json(LINK_FIXES_PATH)
    if isinstance(link_fixes, dict):
        for model_id, overlay in link_fixes.items():
            if not isinstance(overlay, dict) or model_id not in by_id:
                continue
            by_id[model_id] = deep_merge(by_id[model_id], overlay)

    for model_id, patch in DIRECT_PATCHES.items():
        if model_id not in by_id:
            continue
        by_id[model_id] = deep_merge(by_id[model_id], patch)

    for raw_model in REPORT_PLACEHOLDERS:
        normalized = normalize_extra_model(raw_model)
        model_id = str(normalized.get("id") or "").strip()
        if not model_id:
            continue
        if model_id in by_id:
            by_id[model_id] = deep_merge(by_id[model_id], normalized)
        else:
            by_id[model_id] = normalized

    normalized_models: List[Dict[str, Any]] = []
    for model_id in sorted(by_id.keys()):
        model = deepcopy(by_id[model_id])
        model["runtime"] = normalize_runtime(
            model.get("runtime") if isinstance(model.get("runtime"), dict) else {},
            str(model.get("architecture") or ""),
        )
        if "status" in model and isinstance(model["status"], dict):
            model["status"] = normalize_status(
                model["status"],
                model.get("install") if isinstance(model.get("install"), dict) else {},
                model["runtime"],
            )
        normalized_models.append(model)

    output = deepcopy(source_payload)
    output["models"] = normalized_models
    output.setdefault("source", {})
    if isinstance(output["source"], dict):
        notes = str(output["source"].get("notes") or "").strip()
        suffix = " | Synced from extra_models.json, guide_model_overrides.json, and registry_link_fixes.json"
        output["source"]["notes"] = notes + suffix if notes and suffix not in notes else notes or suffix.strip(" |")

    if write:
        write_json(SOURCE_PATH, output)
    return {
        "models_total": len(normalized_models),
        "write": write,
        "source_path": str(SOURCE_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync runtime overlays into models.v2.source.json")
    parser.add_argument("--write", action="store_true", help="Write changes back to models.v2.source.json")
    args = parser.parse_args()

    result = sync_registry_source(write=args.write)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
