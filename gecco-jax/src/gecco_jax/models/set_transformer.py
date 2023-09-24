from typing import Callable, List, Optional
from functools import partial

import jax
import equinox as eqx
from torch_dimcheck import dimchecked, A
from jaxtyping import Array, Float
from equinox.nn import Linear, Dropout
from equinox._module import static_field
from equinox.nn._attention import dot_product_attention

from gecco_jax.models.mlp import MLP
from gecco_jax.models.normalization import _default_norm_ctx
from gecco_jax.models.activation import GaussianActivation


class AttentionPool(eqx.Module):
    """
    A cross-attention layer between a learnable set of queries and input keys/values.
    Compared to using standard MultiHeadedAttention plus a learnable set of queries,
    this avoids the unnecessary query projection weight matrix.
    """

    inducers: A["I H hC"]
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout

    num_heads: int = static_field()
    feature_dim: int = static_field()
    use_key_bias: bool = static_field()
    use_value_bias: bool = static_field()
    use_output_bias: bool = static_field()

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_inducers: int,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        *,
        key: "jax.random.PRNGKey",
    ):
        super().__init__()
        ikey, kkey, vkey, okey = jax.random.split(key, 4)

        assert feature_dim % num_heads == 0, (feature_dim, num_heads)
        dims_per_head = feature_dim // num_heads

        self.inducers = jax.random.normal(
            ikey, (num_inducers, num_heads, dims_per_head)
        )
        self.key_proj = Linear(
            feature_dim, feature_dim, use_bias=use_key_bias, key=kkey
        )
        self.value_proj = Linear(
            feature_dim, feature_dim, use_bias=use_value_bias, key=vkey
        )
        self.output_proj = Linear(
            feature_dim, feature_dim, use_bias=use_output_bias, key=okey
        )
        self.dropout = Dropout(dropout_p)

        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    def __call__(
        self,
        kv: Float[Array, "N C"],
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Float[Array, "I C"]:
        if kv.shape[-1] != self.feature_dim:
            raise ValueError(f"{kv.shape=} with {self.feature_dim=}.")

        query_heads = self.inducers
        key_heads = self._project(self.key_proj, kv)
        value_heads = self._project(self.value_proj, kv)

        attn_fn = partial(
            dot_product_attention,
            dropout=self.dropout,
        )
        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
            query_heads, key_heads, value_heads, key=keys
        )
        attn = attn.reshape(self.inducers.shape[0], -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)


class Broadcast(eqx.Module):
    pool: AttentionPool
    norm_1: eqx.Module
    mlp: MLP
    norm_2: eqx.Module
    unpool: eqx.nn.MultiheadAttention

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        embed_dim: int,
        num_heads: int = 8,
        mlp_blowup: int = 2,
        norm_cls: Callable = _default_norm_ctx,
        activation_cls: Callable = GaussianActivation,
        *,
        key,
    ):
        pool_key, norm_1_key, mlp_key, norm_2_key, unpool_key = jax.random.split(key, 5)
        self.pool = AttentionPool(feature_dim, num_heads, num_inducers, key=pool_key)
        self.norm_1 = norm_cls(
            feature_dim,
            embed_dim,
            key=norm_1_key,
        )
        self.mlp = MLP(
            feature_dim,
            feature_dim,
            width_size=mlp_blowup * feature_dim,
            depth=1,
            activation=activation_cls(),
            key=mlp_key,
        )
        self.norm_2 = norm_cls(
            feature_dim,
            embed_dim,
            key=norm_2_key,
        )
        self.unpool = eqx.nn.MultiheadAttention(
            num_heads,
            feature_dim,
            key=unpool_key,
        )

    @dimchecked
    def __call__(self, x: A["N C"], embed: A["E"], *, key) -> A["N C"]:
        pool_key, mlp_key, unpool_key = jax.random.split(key, 3)
        h = self.pool(x, key=pool_key)
        h = self.norm_1(h, embed)
        h = self.mlp.vmap_with_key(h, key=mlp_key)
        h = self.norm_2(h, embed)
        return self.unpool(x, h, h, key=unpool_key)


class BroadcastingLayer(eqx.Module):
    broadcast_norm: eqx.Module
    broadcast: Broadcast
    mlp_norm: eqx.Module
    mlp: MLP

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        embed_dim: int,
        mlp_blowup: int = 2,
        norm_cls: Callable = _default_norm_ctx,
        activation_cls: Callable = GaussianActivation,
        *,
        key,
        **kwargs,
    ):
        bc_norm_key, bc_key, mlp_norm_key, mlp_key = jax.random.split(key, 4)

        self.broadcast_norm = norm_cls(
            feature_dim,
            embed_dim,
            key=bc_norm_key,
        )
        self.broadcast = Broadcast(
            feature_dim,
            num_inducers,
            embed_dim,
            mlp_blowup=mlp_blowup,
            norm_cls=norm_cls,
            activation_cls=activation_cls,
            **kwargs,
            key=bc_key,
        )
        self.mlp_norm = norm_cls(
            feature_dim,
            embed_dim,
            key=mlp_norm_key,
        )
        self.mlp = MLP(
            feature_dim,
            feature_dim,
            width_size=mlp_blowup * feature_dim,
            depth=1,
            activation=activation_cls(),
            key=mlp_key,
        )

    @dimchecked
    def __call__(self, x: A["N C"], embed: A["E"], *, key) -> A["N C"]:
        broadcast_key, mlp_key = jax.random.split(key, 2)

        y = self.broadcast_norm(x, embed)
        x = x + self.broadcast(y, embed, key=broadcast_key)

        y = self.mlp_norm(x, embed)
        return x + self.mlp.vmap_with_key(x, key=mlp_key)


class BroadcastingSetTransformer(eqx.Module):
    layers: List[BroadcastingLayer]

    def __init__(
        self,
        n_layers: int,
        *args,
        key,
        **kwargs,
    ):
        keys = jax.random.split(key, n_layers)
        self.layers = [BroadcastingLayer(*args, **kwargs, key=key) for key in keys]

    @dimchecked
    def __call__(
        self,
        features: A["N C"],
        geometry: A["N D"],
        embed: A["E"],
        *,
        key: A["2"],
    ) -> A["N C"]:
        del geometry

        keys = jax.random.split(key, len(self.layers))

        for layer, key in zip(self.layers, keys):
            features = layer(features, embed, key=key)

        return features
