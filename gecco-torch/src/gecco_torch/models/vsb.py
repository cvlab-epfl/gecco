from functools import partial
from typing import Callable
from torch import nn

from .mlp import MLP
from .activation import GaussianActivation
from .normalization import AdaGN

class VSBLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        blowup: int = 2,
        norm_cls: Callable = AdaGN,
    ):
        super().__init__()
        self.norm = norm_cls(feature_dim, embed_dim)
        self.mlp = MLP(
            feature_dim,
            feature_dim,
            width_size=blowup * feature_dim,
            activation=GaussianActivation,
        )

    def forward(self, nodes, embed):
        y = self.norm(nodes, embed)
        y = self.mlp(y)

        return nodes + y

class VSB(nn.ModuleList):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        n_layers: int,
        mlp_blowup: int = 2,
        norm_cls: Callable = AdaGN,
    ):
        make_layer = partial(VSBLayer, feature_dim, embed_dim, blowup=mlp_blowup, norm_cls=norm_cls)
        super().__init__([make_layer() for _ in range(n_layers)])

    def forward(self, features, embed, geometry):
        del geometry

        for layer in self:
            features = layer(features, embed)
        
        return features