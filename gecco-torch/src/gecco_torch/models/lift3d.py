from typing import Any
from gecco.models.set_transformer import SetTransformer
from torch import nn, Tensor

class LinearLift(nn.Module):
    def __init__(
        self,
        inner: SetTransformer,
        feature_dim: int,
        geometry_dim: int = 3,
        do_norm: bool = True,
    ):
        super().__init__()
        self.lift = nn.Linear(geometry_dim, feature_dim)
        self.inner = inner
        
        if do_norm:
            self.lower = nn.Sequential(
                nn.LayerNorm(feature_dim, elementwise_affine=False),
                nn.Linear(feature_dim, geometry_dim),
            )
        else:
            self.lower = nn.Linear(feature_dim, geometry_dim)
    
    def forward(
        self,
        geometry: Tensor,
        embed: Tensor,
        raw_context: Any,
        post_context: Any,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        del raw_context, post_context

        features = self.lift(geometry)
        features, out_cache = self.inner(features, embed, geometry, do_cache, cache)
        return self.lower(features), out_cache

