import math
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from gecco_jax.models.mlp import MLP
from gecco_jax.models.set_transformer import AttentionPool

def _replace_weights(pytree, filter_fn, attr_getter, replace_fn):
    def get_leaves(pytree):
        is_leaf = lambda x: filter_fn(x) and x is not pytree

        out = [attr_getter(pytree)] if filter_fn(pytree) else []

        leaves = [x for x in jtu.tree_leaves(pytree, is_leaf=is_leaf) if is_leaf(x)]

        for x in leaves:
            out.extend(get_leaves(x))
        return out

    return eqx.tree_at(get_leaves, pytree, replace_fn=replace_fn, is_leaf=lambda x: x is None)

def _bias_init(bias):
    if bias is None or (bias == 1.).all():
        return bias
    return jnp.zeros_like(bias)

def gpt_init(model):
    bb = model.network.backbone
    n_layers = len(bb.layers)

    # init linear bias with 0
    bb = _replace_weights(
        bb,
        lambda mod: isinstance(mod, eqx.nn.Linear),
        lambda linear: linear.bias,
        _bias_init,
    )

    # scale down mlp output projections by sqrt(2 * n_layers)
    bb = _replace_weights(
        bb,
        lambda mod: isinstance(mod, MLP),
        lambda mlp: mlp.layers[-1].weight,
        lambda mlp_weight: mlp_weight / math.sqrt(2 * n_layers),
    )
    
    # scale down attention output projections by sqrt(2 * n_layers)
    bb = _replace_weights(
        bb,
        lambda mod: isinstance(mod, (AttentionPool, eqx.nn.MultiheadAttention)),
        lambda attn: attn.output_proj.weight,
        lambda out_weight: out_weight / math.sqrt(2 * n_layers),
    )
    
    return eqx.tree_at(lambda m: m.network.backbone, model, replace_fn=lambda _: bb)