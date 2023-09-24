from typing import Optional
from functools import partial
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from torch_dimcheck import dimchecked, A

from gecco_jax.models.util import splitter

__all__ = ['LinearSpaceEmbedding', 'LinearTimeEmbedding']

class LinearSpaceEmbedding(eqx.nn.Linear):
    @dimchecked
    def __call__(self, xyzs: A['N D']) -> A['N F']:
        return jax.vmap(super().__call__)(xyzs)

class LinearTimeEmbedding(eqx.Module):
    weights: A['E']

    def __init__(self, dim: int, *, key):
        self.weights = 0.1 * jax.random.normal(key, (dim, ))
    
    @dimchecked
    def __call__(self, t: A['']) -> A['E']:
        return t * self.weights