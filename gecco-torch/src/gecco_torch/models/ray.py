from einops import rearrange
import torch
from torch import nn, Tensor
from kornia.geometry.camera.perspective import project_points

from gecco_torch.reparam import Reparam
from gecco_torch.structs import Context3d
from gecco_torch.models.feature_pyramid import FeaturePyramidContext
from gecco_torch.models.set_transformer import SetTransformer

class GroupNormBNC(nn.GroupNorm):
    def forward(self, tensor_bnc: Tensor) -> Tensor:
        assert tensor_bnc.ndim == 3

        tensor_bcn = rearrange(tensor_bnc, 'b n c -> b c n')
        result_bcn = super().forward(tensor_bcn)
        return rearrange(result_bcn, 'b c n -> b n c')

class RayNetwork(nn.Module):
    def __init__(
        self,
        backbone: SetTransformer,
        reparam: Reparam,
        context_dims: list[int],
    ):
        super().__init__()
        self.backbone = backbone
        self.reparam = reparam
        self.context_dims = context_dims

        self.xyz_embed = nn.Linear(reparam.dim, backbone.feature_dim)
        self.img_feature_proj = nn.Sequential(
            GroupNormBNC(16, sum(context_dims), affine=False),
            nn.Linear(sum(context_dims), backbone.feature_dim),

        )
        self.output_proj = nn.Sequential(
            GroupNormBNC(16, backbone.feature_dim, affine=False),
            nn.Linear(backbone.feature_dim, reparam.dim),
        )
    
    def extra_repr(self) -> str:
        return f'context_dims={self.context_dims}'
    
    def extract_image_features(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: Context3d,
    ) -> Tensor:
        geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)
        hw_01 = project_points(geometry_data, ctx.K.unsqueeze(1))[..., :2]
        hw_flat = rearrange(hw_01, 'b n t -> b n 1 t')

        lookups = []
        for feature in features:
            lookup = torch.nn.functional.grid_sample(feature, hw_flat * 2 - 1, align_corners=False)
            lookup = rearrange(lookup, 'b c n 1 -> b n c')
            lookups.append(lookup)
        
        return torch.cat(lookups, dim=-1)
    
    def forward(
        self,
        geometry: Tensor,
        t: Tensor,
        raw_ctx: Context3d,
        post_context: FeaturePyramidContext,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor | list[Tensor] | None]:
        xyz_features = self.xyz_embed(geometry)
        t_features = t
        with torch.autocast(device_type='cuda', enabled=False): # extract_image_features is suspected of causing divergences in fp16
            geometry_f32 = geometry.to(dtype=torch.float32)
            features_f32 = [f.to(dtype=torch.float32) for f in post_context.features]
            raw_ctx_f32 = raw_ctx.apply_to_tensors(lambda t: t.to(dtype=torch.float32))
            img_features_raw = self.extract_image_features(geometry_f32, features_f32, raw_ctx_f32)

        img_features = self.img_feature_proj(img_features_raw)

        point_features = xyz_features + img_features

        processed, out_cache = self.backbone(point_features, t_features, geometry, do_cache, cache)

        return self.output_proj(processed), out_cache