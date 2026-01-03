"""ModelFactory: minimal architecture-to-model constructor.

This project runs multiple separation model families via different backends.
Some families (SCNet, BandIt, Apollo) are shipped as PyTorch checkpoints + YAML
configs from ZFTurbo's Music-Source-Separation-Training (MSST) ecosystem.

The separation stack expects a `ModelFactory.create_model(architecture, config)`
entrypoint for these families.

Design goals:
- Keep dependencies local to StemSepApp (vendored MSST model code).
- Be conservative: only implement architectures we can actually instantiate.
- Provide actionable errors when an architecture isn't available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ModelFactory:
    @staticmethod
    def get_model_config(
        model_id: str, architecture: Optional[str] = None
    ) -> Dict[str, Any]:
        """Best-effort default config fallback.

        Most modern models ship YAML configs and should not rely on this.
        """
        _ = model_id
        _ = architecture
        return {}

    @staticmethod
    def create_model(architecture: str, config: Dict[str, Any]):
        if not architecture:
            raise ValueError("Missing architecture")

        arch = str(architecture).strip().lower()

        # --- SCNet ---
        if "scnet" in arch:
            from stemsep.models.architectures.zfturbo_vendor.scnet.scnet import SCNet

            if not isinstance(config, dict):
                raise TypeError("SCNet config must be a dict")
            return SCNet(**config)

        # --- BandIt ---
        if "bandit" in arch:
            # MSST BandIt reference: MultiMaskMultiSourceBandSplitRNNSimple
            from stemsep.models.architectures.zfturbo_vendor.bandit.core.model import (
                MultiMaskMultiSourceBandSplitRNNSimple,
            )

            if not isinstance(config, dict):
                raise TypeError("BandIt config must be a dict")

            # The RNNSimple wrapper already exposes forward(x: Tensor)->Tensor
            # returning (B, stems, C, T) after ISTFT.
            return MultiMaskMultiSourceBandSplitRNNSimple(**config)

        # --- Apollo ---
        if "apollo" in arch:
            from stemsep.models.architectures.zfturbo_vendor.look2hear.models.base_model import (
                BaseModel,
            )

            if not isinstance(config, dict):
                raise TypeError("Apollo config must be a dict")
            return BaseModel.apollo(**config)

        # --- Roformer family (used by some internal tools) ---
        if "roformer" in arch:
            # Prefer our local Roformer implementations (kept compatible with audio-separator).
            from stemsep.audio.roformer import BSRoformer, MelBandRoformer

            variant = None
            if isinstance(config, dict):
                variant = config.get("__stemsep_variant")

            if not isinstance(config, dict):
                raise TypeError("Roformer config must be a dict")

            if variant in ("hyperace", "hyperace_v1"):
                from stemsep.audio.roformer.variants import get_hyperace_v1_bs_roformer_cls

                cfg = dict(config)
                cfg.pop("__stemsep_variant", None)
                if isinstance(cfg.get("freqs_per_bands"), list):
                    cfg["freqs_per_bands"] = tuple(cfg["freqs_per_bands"])
                return get_hyperace_v1_bs_roformer_cls()(**cfg)

            if variant in ("hyperace_v2",):
                from stemsep.audio.roformer.variants import get_hyperace_v2_bs_roformer_cls

                cfg = dict(config)
                cfg.pop("__stemsep_variant", None)
                if isinstance(cfg.get("freqs_per_bands"), list):
                    cfg["freqs_per_bands"] = tuple(cfg["freqs_per_bands"])
                return get_hyperace_v2_bs_roformer_cls()(**cfg)

            if variant == "fno":
                from stemsep.audio.roformer.variants import get_fno_bs_roformer_cls

                cfg = dict(config)
                cfg.pop("__stemsep_variant", None)
                if isinstance(cfg.get("freqs_per_bands"), list):
                    cfg["freqs_per_bands"] = tuple(cfg["freqs_per_bands"])
                return get_fno_bs_roformer_cls()(**cfg)

            # Heuristic like MSST: freqs_per_bands => BSRoformer, num_bands => MelBandRoformer.
            if "freqs_per_bands" in config:
                cfg = dict(config)
                # Ensure tuple (some configs store list)
                if isinstance(cfg.get("freqs_per_bands"), list):
                    cfg["freqs_per_bands"] = tuple(cfg["freqs_per_bands"])
                return BSRoformer(**cfg)

            if "num_bands" in config or "num_subbands" in config:
                cfg = dict(config)
                if "num_bands" not in cfg and "num_subbands" in cfg:
                    cfg["num_bands"] = cfg["num_subbands"]
                return MelBandRoformer(**cfg)

            raise ValueError(
                "Roformer config missing 'freqs_per_bands' or 'num_bands' (cannot detect subtype)."
            )

        raise ValueError(f"Unsupported architecture for ModelFactory: {architecture}")
