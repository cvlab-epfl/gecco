from functools import partial
from typing import List, Literal, Any

import jax
import jax.numpy as jnp
import equinox as eqx
from torch_dimcheck import dimchecked, A

from gecco_jax.models.feature_pyramid import FeaturePyramidContext
from gecco_jax.models.normalization import MoveChannels
from gecco_jax.models.reparam import Reparam
from gecco_jax.types import PyTree
from gecco_jax.models.util import splitter
from gecco_jax.models.embed import LinearSpaceEmbedding


@dimchecked
def interpolate_2d(
    image: A["H W C"],
    coords: A["2 N"],
) -> A["N C"]:
    """
    Looks up locations given by `coords` in an array `image`
    returning the features at queries location. Assumes coords
    are (h, w), that is the 1st coordinate in `coords` corresponds
    to the 1st coordinate of `image`.
    """
    h, w, _ = image.shape
    coords_scaled = coords * jnp.array([h, w]).reshape(2, 1)

    return jax.vmap(
        partial(
            jax.scipy.ndimage.map_coordinates,
            order=1,
        ),
        in_axes=(-1, None),
        out_axes=-1,
    )(image, coords_scaled)


class PointNetwork(eqx.Module):
    backbone: eqx.Module
    xyz_embed: LinearSpaceEmbedding
    reparam: Reparam
    output_norm: eqx.Module
    output_proj: eqx.nn.Linear

    backbone_geometry_space: Literal["diffusion", "data"]

    def __init__(
        self,
        backbone: eqx.Module,
        reparam: Reparam,
        feature_dim: int,
        geometry_dim: int = 3,
        backbone_geometry_space: Literal["diffusion", "data"] = "diffusion",
        *,
        key,
    ):
        self.reparam = reparam
        keys = splitter(key)

        self.xyz_embed = LinearSpaceEmbedding(
            in_features=geometry_dim,
            out_features=feature_dim,
            key=next(keys),
        )

        self.backbone = backbone

        self.output_norm = MoveChannels(
            eqx.nn.GroupNorm(
                groups=32,
                channels=feature_dim,
                channelwise_affine=False,
            ),
        )

        self.output_proj = eqx.nn.Linear(
            feature_dim,
            geometry_dim,
            key=next(keys),
        )

        self.backbone_geometry_space = backbone_geometry_space

    @dimchecked
    def _geometry_for_backbone(
        self,
        geometry: A["N F"],
        ctx: Any,
    ) -> A["N F"]:
        if self.backbone_geometry_space == "data":
            return self.reparam.diffusion_to_data(geometry, ctx)
        return geometry


class RayNetwork(PointNetwork):
    ctx_dim_reductor: eqx.nn.Linear
    normalize_uvl: bool

    def __init__(
        self,
        backbone: eqx.Module,
        reparam: eqx.Module,
        feature_dim: int,
        input_ctx_dim: int,
        geometry_dim: int = 3,
        normalize_uvl: bool = False,
        backbone_geometry_space: Literal["diffusion", "data"] = "diffusion",
        *,
        key,
    ):
        keys = splitter(key)
        super(RayNetwork, self).__init__(
            backbone=backbone,
            reparam=reparam,
            feature_dim=feature_dim,
            geometry_dim=geometry_dim,
            backbone_geometry_space=backbone_geometry_space,
            key=next(keys),
        )

        self.ctx_dim_reductor = eqx.nn.Linear(
            in_features=input_ctx_dim,
            out_features=feature_dim,
            key=next(keys),
        )

        if normalize_uvl:
            # FIXME
            raise AssertionError(
                "normalize_uvl was left only to not break backwards compatibility"
            )
        self.normalize_uvl = normalize_uvl

    @dimchecked
    def _extract_ctx_features(
        self,
        x_diffusion: A["N 3"],
        feature_pyramid: List[jnp.array],
        K: A["3 3"],
    ) -> A["N C"]:
        # look up features in all levels of the pyramid
        features = []
        for level in feature_pyramid:
            features.append(self.lookup_2d(x_diffusion, level, K))

        # concatenate and map (down) to feature_dim we'll be working with
        features = jnp.concatenate(features, axis=-1)  # N C_high
        if features.shape[1] != self.ctx_dim_reductor.in_features:
            input_shapes = [level.shape for level in feature_pyramid]
            raise ValueError(
                f"Expected a total of {self.ctx_dim_reductor.in_features} "
                f"features, got {features.shape[1]} (from feature_pyramid of "
                f"shape {input_shapes})."
            )

        return jax.vmap(self.ctx_dim_reductor)(features)

    @dimchecked
    def lookup_2d(
        self,
        x_diffusion: A["N 3"],
        features: A["H W C"],
        K: A["3 3"],
    ) -> A["N C"]:
        hw_01 = self.reparam.diffusion_to_hw(x_diffusion, K)
        return interpolate_2d(features, hw_01.T)

    @dimchecked
    def __call__(
        self,
        t: A[""],
        x: A["N F"],
        ctx: FeaturePyramidContext,
        *,
        key,
    ) -> A["N F"]:
        xyz_features = self.xyz_embed(x)  # N C
        t_features = t.reshape(
            1,
        )
        img_features = self._extract_ctx_features(
            x,
            ctx.features,
            K=ctx.K,
        )  # N C

        point_features = xyz_features + img_features

        processed = self.backbone(
            features=point_features,
            geometry=self._geometry_for_backbone(x, ctx),
            embed=t_features,
            key=key,
        )

        processed_normalized = self.output_norm(processed)
        return jax.vmap(self.output_proj)(processed_normalized)


class UnconditionalPointNetwork(PointNetwork):
    @dimchecked
    def __call__(
        self,
        t: A[""],
        x: A["N F"],
        ctx: PyTree,
        *,
        key,
    ) -> A["N F"]:
        del ctx

        xyz_features = self.xyz_embed(x)  # N C
        t_features = t.reshape(
            1,
        )

        processed = self.backbone(
            features=xyz_features,
            geometry=self._geometry_for_backbone(x, ctx=None),
            embed=t_features,
            key=key,
        )

        processed_normalized = self.output_norm(processed)
        return jax.vmap(self.output_proj)(processed_normalized)


class GlobalConditioningNetwork(PointNetwork):
    @dimchecked
    def _extract_ctx_features(
        self,
        feature_pyramid: List[jnp.array],
    ) -> A["C"]:
        assert len(feature_pyramid) == 1
        (global_features,) = feature_pyramid
        assert global_features.ndim == 3, global_features.shape
        return global_features.mean(axis=(0, 1))

    @dimchecked
    def __call__(
        self,
        t: A[""],
        x: A["N F"],
        ctx: FeaturePyramidContext,
        *,
        key,
    ) -> A["N F"]:
        xyz_features = self.xyz_embed(x)  # N C
        img_features = self._extract_ctx_features(ctx.features)  # [C]

        t_and_img_features = jnp.concatenate(
            [
                t.reshape(
                    1,
                ),
                img_features,
            ]
        )

        processed = self.backbone(
            features=xyz_features,
            geometry=self._geometry_for_backbone(x, ctx),
            embed=t_and_img_features,
            key=key,
        )

        processed_normalized = self.output_norm(processed)
        return jax.vmap(self.output_proj)(processed_normalized)
