"""
Wraps a pretrained feature extractor to produce a feature pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, Tensor
import torchvision.models as tvm

from gecco_torch.structs import Context3d


@dataclass
class FeaturePyramidContext:
    features: list[Tensor]
    K: Tensor


class FeaturePyramidExtractor(nn.Module):
    def forward(self, ctx_raw: Context3d) -> FeaturePyramidContext:
        raise NotImplementedError()


class ConvNeXtExtractor(FeaturePyramidExtractor):
    def __init__(
        self,
        n_stages: int = 3,
        model: Literal["tiny", "small"] = "tiny",
        pretrained: bool = True,
    ):
        super().__init__()

        if model == "tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = tvm.convnext_tiny(weights=weights)
        elif model == "small":
            weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            convnext = tvm.convnext_small(weights=weights)
        else:
            raise ValueError(f"Unknown model {model}")

        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(
                nn.Sequential(convnext.features[i], convnext.features[i + 1])
            )

        self.stages = self.stages[:n_stages]
        self._remove_stochastic_depth()

    def _remove_stochastic_depth(self):
        """We found SD to harm generative performance"""
        for submodule in self.modules():
            if isinstance(submodule, tvm.convnext.CNBlock):
                submodule.stochastic_depth = torch.nn.Identity()

    def forward(self, raw_ctx: Context3d) -> FeaturePyramidContext:
        images = raw_ctx.image

        features = []
        for stage in self.stages:
            images = stage(images)
            features.append(images)

        return FeaturePyramidContext(
            features=features,
            K=raw_ctx.K,
        )
