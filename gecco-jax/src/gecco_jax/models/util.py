from typing import NamedTuple
import jax
import equinox as eqx

from gecco_jax.types import PyTree

def splitter(key):
    while True:
        key, subkey = jax.random.split(key, 2)
        yield subkey

class Frozen(NamedTuple): # NamedTuple to autoregister as PyTree
    value: PyTree

    def __get__(self, obj, objtype=None):
        return jax.lax.stop_gradient(self.value)

def count_parameters(model):
    return jax.tree_util.tree_reduce(
        lambda carry, param: carry + param.size if eqx.is_inexact_array(param) else carry,
        model,
        0,
    )