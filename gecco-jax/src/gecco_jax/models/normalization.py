import equinox as eqx
import jax
import jax.numpy as jnp
from torch_dimcheck import A, dimchecked

from gecco_jax.models.util import splitter


def _manual_init(module, attr: str, value: float):
    return eqx.tree_at(
        lambda module_: getattr(module_, attr),
        module,
        replace_fn=lambda param: jnp.ones_like(param) * value,
    )


class AdaNorm(eqx.Module):
    scale_linear: eqx.nn.Linear
    bias_linear: eqx.nn.Linear

    num_features: int
    embed_dim: int

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        *,
        key,
    ):
        keys = splitter(key)

        scale_linear = eqx.nn.Linear(
            embed_dim,
            num_features,
            key=next(keys),
        )
        scale_linear = _manual_init(scale_linear, "weight", 0.0)
        scale_linear = _manual_init(scale_linear, "bias", 1.0)
        self.scale_linear = scale_linear

        bias_linear = eqx.nn.Linear(
            embed_dim,
            num_features,
            key=next(keys),
        )
        bias_linear = _manual_init(bias_linear, "weight", 0.0)
        bias_linear = _manual_init(bias_linear, "bias", 0.0)
        self.bias_linear = bias_linear

        self.num_features = num_features
        self.embed_dim = embed_dim

    @dimchecked
    def _raw_norm(self, x: A["C N"]) -> A["C N"]:
        raise NotImplementedError()

    @dimchecked
    def __call__(
        self,
        x: A["C D*"],
        embed: A["E"],
    ) -> A["C D*"]:
        assert x.shape[0] == self.num_features, (x.shape, self.num_features)
        assert embed.shape[0] == self.embed_dim, (embed.shape, self.embed_dim)

        scale = self.scale_linear(embed)
        bias = self.bias_linear(embed)

        y_normed = self._raw_norm(x)
        shape = (x.shape[0], *((1,) * (x.ndim - 1)))
        return scale.reshape(shape) * y_normed + bias.reshape(shape)


class AdaGN(AdaNorm):
    group_norm: eqx.nn.GroupNorm

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        groups: int = 32,
        *,
        key,
    ):
        super(AdaGN, self).__init__(
            num_features=num_features,
            embed_dim=embed_dim,
            key=key,
        )

        self.group_norm = eqx.nn.GroupNorm(
            groups=groups,
            channels=num_features,
            channelwise_affine=False,
        )

    @dimchecked
    def _raw_norm(self, x: A["C D*"]) -> A["C D*"]:
        return self.group_norm(x)


class AdaLN(AdaNorm):
    layer_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        *,
        key,
    ):
        super(AdaLN, self).__init__(
            num_features=num_features,
            embed_dim=embed_dim,
            key=key,
        )

        self.layer_norm = eqx.nn.LayerNorm(
            shape=None,
            elementwise_affine=False,
        )

    @dimchecked
    def _raw_norm(self, x: A["C D*"]) -> A["C D*"]:
        return jax.vmap(self.layer_norm, in_axes=-1, out_axes=-1)(x)


class MoveChannels(eqx.Module):
    inner: eqx.Module

    def __init__(
        self,
        inner: eqx.Module,
    ):
        self.inner = inner

    @dimchecked
    def __call__(
        self,
        x_c_last: A["D* C"],
        *args,
        **kwargs,
    ) -> A["D* C"]:
        x_c_first = jnp.swapaxes(x_c_last, 0, -1)
        y_c_first = self.inner(x_c_first, *args, **kwargs)
        return jnp.swapaxes(y_c_first, 0, -1)


def _default_norm_ctx(
    channels: int,
    embed_dim: int,
    *,
    key,
):
    return MoveChannels(
        AdaGN(
            groups=32,
            num_features=channels,
            embed_dim=embed_dim,
            key=key,
        ),
    )
