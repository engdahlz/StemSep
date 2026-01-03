def get_hyperace_v1_bs_roformer_cls():
    from .hyperace_v1_bs_roformer import BSRoformer

    return BSRoformer


def get_hyperace_v2_bs_roformer_cls():
    from .hyperace_v2_bs_roformer import BSRoformer

    return BSRoformer


def get_fno_bs_roformer_cls():
    from .fno_bs_roformer import BSRoformer

    return BSRoformer


__all__ = [
    "get_hyperace_v1_bs_roformer_cls",
    "get_hyperace_v2_bs_roformer_cls",
    "get_fno_bs_roformer_cls",
]
