from functools import partial
from typing import Callable, Sequence, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.lax import stop_gradient
from torch_dimcheck import dimchecked, A

from gecco_jax.geometry import project_points, unproject_points


class Reparam(eqx.Module):
    def data_to_diffusion(self, data, ctx):
        return data

    def diffusion_to_data(self, diff, ctx):
        return diff

    def ladj_diffusion_to_data(self, diff, ctx):
        raise NotImplementedError()

    def ladj_data_to_diffusion(self, diff, ctx):
        raise NotImplementedError()


class ReparamDiagonalBlockJacrev(Reparam):
    @dimchecked
    def ladj_data_to_diffusion(self, data: A["N 3"], ctx) -> A[""]:
        @dimchecked
        def apply_1(data: A["3"]) -> A[""]:
            fn = lambda x: self.data_to_diffusion(x, ctx)
            jac = jax.jacrev(fn)(data)
            _, logdet = jnp.linalg.slogdet(jac)
            return logdet

        return jax.vmap(apply_1)(data).sum()

    @dimchecked
    def ladj_diffusion_to_data(self, diff: A["N 3"], ctx) -> A[""]:
        @dimchecked
        def apply_1(diff: A["3"]) -> A[""]:
            fn = lambda x: self.diffusion_to_data(x, ctx)
            jac = jax.jacrev(fn)(diff)
            _, logdet = jnp.linalg.slogdet(jac)
            return logdet

        return jax.vmap(apply_1)(diff).sum()


class ReparamDiagonal2d(Reparam):
    """
    Given an implementation of the two required methods,
    implements log-abs-det-jacobian for the transformation
    """

    @dimchecked
    def _diffusion_to_data(self, dim: int, diff: A[""], ctx) -> A[""]:
        raise NotImplementedError()

    @dimchecked
    def _data_to_diffusion(self, dim: int, data: A[""], ctx) -> A[""]:
        raise NotImplementedError()

    def _check_dims(self, data_shape: Sequence[int]):
        # by default no checking, but it's good to override
        # to avoid silent bugs
        pass

    @dimchecked
    def diffusion_to_data(self, data: A["N D"], ctx) -> A["N D"]:
        self._check_dims(data.shape)

        dimensions = []
        for dim_ix, val in enumerate(data.T):
            fn = lambda val: self._diffusion_to_data(dim=dim_ix, diff=val, ctx=ctx)
            dimensions.append(jax.vmap(fn)(val))
        return jnp.stack(dimensions, axis=-1)

    @dimchecked
    def data_to_diffusion(self, diff: A["N D"], ctx) -> A["N D"]:
        self._check_dims(diff.shape)

        dimensions = []
        for dim_ix, val in enumerate(diff.T):
            fn = lambda val: self._data_to_diffusion(dim=dim_ix, data=val, ctx=ctx)
            dimensions.append(jax.vmap(fn)(val))
        return jnp.stack(dimensions, axis=-1)

    @dimchecked
    def ladj_data_to_diffusion(self, data: A["N D"], ctx) -> A[""]:
        log_abs_det = 0.0

        for dim_ix, val in enumerate(data.T):
            fn = lambda val: self._data_to_diffusion(dim=dim_ix, data=val, ctx=ctx)
            vg_fn = jax.vmap(jax.grad(fn))
            delta = jnp.log(jnp.abs(vg_fn(val))).sum()
            log_abs_det = log_abs_det + delta

        return log_abs_det

    @dimchecked
    def ladj_diffusion_to_data(self, diff: A["N D"], ctx) -> A[""]:
        log_abs_det = 0.0

        for dim_ix, val in enumerate(diff.T):
            fn = lambda val: self._diffusion_to_data(dim=dim_ix, diff=val, ctx=ctx)
            vg_fn = jax.vmap(jax.grad(fn))
            delta = jnp.log(jnp.abs(vg_fn(val))).sum()
            log_abs_det = log_abs_det + delta

        return log_abs_det


class _PseudoCtx(NamedTuple):
    # FIXME
    """An ugly hack to use diffusion_to_data without having the entire Context3d"""
    K: A["3 3"]


class GaussianReparam(ReparamDiagonal2d):
    mean: A["D"] = jnp.zeros(3)
    std: A["D"] = jnp.ones(3)

    def _check_dims(self, data_shape: Sequence[int]):
        if self.mean.shape != self.std.shape:
            raise ValueError(f"Inconsistent {self.mean.shape=} and {self.std.shape=}.")

        if data_shape[-1] != self.mean.shape[0]:
            raise ValueError(f"Inconsistent {data_shape[-1]=} and {self.mean.shape=}.")

    @dimchecked
    def _data_to_diffusion(self, dim: int, data: A[""], ctx) -> A[""]:
        del ctx

        mean = stop_gradient(self.mean)[dim]
        std = stop_gradient(self.std)[dim]

        return (data - mean) / std

    @dimchecked
    def _diffusion_to_data(self, dim: int, diff: A[""], ctx) -> A[""]:
        del ctx

        mean = stop_gradient(self.mean)[dim]
        std = stop_gradient(self.std)[dim]

        return diff * std + mean

    @partial(jnp.vectorize, signature="(a),(b,c)->(d)", excluded=(0,))
    @dimchecked
    def diffusion_to_hw(self, diff: A["3"], K: A["3 3"]) -> A["2"]:
        # hack around data_to_diffusion expecing a Context3d struct and
        # [N, 3] input shape
        data = self.diffusion_to_data(diff[None, :], _PseudoCtx(K)).squeeze(0)
        wh = project_points(data, K)
        return wh[::-1]

    @dimchecked
    def data_to_diffusion_normals(self, data: A["B* D"], ctx) -> A["B* D"]:
        del ctx
        std = stop_gradient(self.std)[(None,) * (data.ndim - 1)]
        return data / std

    @dimchecked
    def diffusion_to_data_normals(self, diff: A["B* D"], ctx) -> A["B* D"]:
        del ctx
        std = stop_gradient(self.std)[(None,) * (diff.ndim - 1)]
        return diff * std


class UVLReparam(ReparamDiagonalBlockJacrev):
    """
    There are three parametrizations which we use
    1) xyz, R^3, z is the axis facing away from the camera
    2) hwd, [0, 1]^2 x [0, inf] where hw indicate image coordinates and d is depth
    3) uvl, R^3 which correspond to hwd by logit on the uv dimensions and logarithm on d
    """

    logit_scale: float = 1.1
    depth_to_real: Callable[[float], float] = jnp.log
    real_to_depth: Callable[[float], float] = jnp.exp

    uvl_mean: A["3"] = jnp.array([1.1159e-03, -3.6975e-03, 1.3792e00])
    uvl_std: A["3"] = jnp.array([0.5989, 0.6476, 1.0569])

    @dimchecked
    def _real_to_01(self, r: A[""]) -> A[""]:
        s = jnp.tanh(r)
        s = s * self.logit_scale
        s = s + 1.0
        s = s / 2
        return s

    @dimchecked
    def _01_to_real(self, s: A[""]) -> A[""]:
        s = 2 * s
        s = s - 1.0
        s = s / self.logit_scale
        r = jnp.arctanh(s)
        return r

    @partial(jnp.vectorize, signature="(a),(b,c)->(a)", excluded=(0,))
    @dimchecked
    def xyz_to_hwd(
        self,
        xyz: A["3"],
        K: A["3 3"],
    ) -> A["3"]:
        wh = project_points(xyz, K)
        hw = wh[::-1]
        d = jnp.linalg.norm(xyz).reshape(1)

        return jnp.concatenate([hw, d])

    @partial(jnp.vectorize, signature="(a),(b,c)->(a)", excluded=(0,))
    @dimchecked
    def hwd_to_xyz(
        self,
        hwd: A["3"],
        K: A["3 3"],
    ) -> A["3"]:
        hw = hwd[:2]
        wh = hw[::-1]
        d = hwd[-1]

        return unproject_points(wh, d, K)

    @partial(jnp.vectorize, signature="(a)->(a)", excluded=(0,))
    @dimchecked
    def hwd_to_uvl(self, hwd: A["3"]) -> A["3"]:
        mean = stop_gradient(self.uvl_mean)
        std = stop_gradient(self.uvl_std)

        h, w, d = hwd

        uvl_denorm = jnp.stack(
            [
                self._01_to_real(h),
                self._01_to_real(w),
                self.depth_to_real(d),
            ],
            axis=-1,
        )

        uvl_norm = (uvl_denorm - mean) / std

        return uvl_norm

    @partial(jnp.vectorize, signature="(a)->(a)", excluded=(0,))
    @dimchecked
    def uvl_to_hwd(self, uvl_norm: A["3"]) -> A["3"]:
        mean = stop_gradient(self.uvl_mean)
        std = stop_gradient(self.uvl_std)

        uvl_denorm = uvl_norm * std + mean
        u, v, l = uvl_denorm

        return jnp.stack(
            [
                self._real_to_01(u),
                self._real_to_01(v),
                self.real_to_depth(l),
            ]
        )

    @partial(jnp.vectorize, signature="(a),(b,c)->(a)", excluded=(0,))
    @dimchecked
    def xyz_to_uvl(
        self,
        xyz: A["3"],
        K: A["3 3"],
    ) -> A["3"]:
        return self.hwd_to_uvl(self.xyz_to_hwd(xyz, K))

    @partial(jnp.vectorize, signature="(a),(b,c)->(a)", excluded=(0,))
    @dimchecked
    def uvl_to_xyz(
        self,
        uvl: A["3"],
        K: A["3 3"],
    ) -> A["3"]:
        return self.hwd_to_xyz(self.uvl_to_hwd(uvl), K)

    def data_to_diffusion(self, data, ctx):
        return self.xyz_to_uvl(data, ctx.K)

    def diffusion_to_data(self, diff, ctx):
        return self.uvl_to_xyz(diff, ctx.K)

    @partial(jnp.vectorize, signature="(a),(b,c)->(d)", excluded=(0,))
    @dimchecked
    def diffusion_to_hw(self, uvl: A["3"], K: A["3 3"]) -> A["2"]:
        del K
        return self.uvl_to_hwd(uvl)[..., :-1]


def softplus(x, beta: float = 1.0, threshold: float = 20.0):
    direct = (1 / beta) * jax.nn.softplus(beta * x)
    linear = x
    return jax.lax.select(beta * x > threshold, linear, direct)


def inv_softplus(x, beta: float = 1.0, threshold: float = 20.0):
    direct = (1 / beta) * jnp.log(jnp.expm1(beta * x))
    linear = x
    return jax.lax.select(beta * x > threshold, linear, direct)
