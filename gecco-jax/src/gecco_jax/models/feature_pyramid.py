from typing import List, Literal, NamedTuple, Optional, Sequence
from einops import rearrange

import jax
import equinox as eqx
import eqxvision as eqv
import jax.numpy as jnp
from torch_dimcheck import dimchecked, A
from gecco_jax.types import BatchIndexHelper, Context3d


# @dimchecked
class FeaturePyramidContext(NamedTuple):
    features: List[A["h w 3"]]
    K: A["3 3"]
    wmat: Optional[jnp.array] = None

    @property
    def index(self):
        return BatchIndexHelper(self)


class ClassificationBackboneConditioner(eqx.Module):
    backbone: eqx.Module

    levels_to_return: int

    def __init__(
        self,
        backbone: eqx.Module,
        levels_to_return=1,
    ):
        self.backbone = backbone
        self.levels_to_return = levels_to_return

    @dimchecked
    def __call__(
        self,
        ctx_raw: Context3d,
        *,
        key,
    ) -> FeaturePyramidContext:
        image_chw = rearrange(ctx_raw.image, "h w c -> c h w")
        features_chw = self.backbone(image_chw, key)
        features_chw = features_chw[-self.levels_to_return :]
        features_hwc = [rearrange(chw, "c h w -> h w c") for chw in features_chw]

        return FeaturePyramidContext(
            features=features_hwc,
            K=ctx_raw.K,
            wmat=ctx_raw.wmat,
        )


CONVNEXT_SIZES = ("tiny", "small", "base", "large")


class Extractor(eqx.Module):
    layers: Sequence[eqx.Module]
    returned_feature_maps: Sequence[int]

    def __call__(self, input, key):
        y = input
        keys = jax.random.split(key, len(self.layers))
        ys = []
        for layer, key in zip(self.layers, keys):
            y = layer(y, key=key)
            ys.append(y)

        return [ys[i] for i in self.returned_feature_maps]

    @classmethod
    def convnext_global(cls, model: str):
        assert model in CONVNEXT_SIZES

        source = getattr(eqv.models, f"convnext_{model}")(
            torch_weights=eqv.utils.CLASSIFICATION_URLS[f"convnext_{model}"],
        )
        return cls(
            source.features.layers[:-2],  # clip the lowest resolution part
            returned_feature_maps=(-1,),
        )

    @classmethod
    def convnext_local(cls, model: str):
        assert model in CONVNEXT_SIZES

        source = getattr(eqv.models, f"convnext_{model}")(
            torch_weights=eqv.utils.CLASSIFICATION_URLS[f"convnext_{model}"],
        )
        return cls(
            source.features.layers[:-2],  # clip the lowest resolution part
            returned_feature_maps=(1, 3, 5),
        )
