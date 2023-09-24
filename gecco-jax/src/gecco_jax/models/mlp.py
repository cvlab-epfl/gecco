from typing import Callable, List, Optional

import jax
import jax.nn as jnn
import jax.random as jrandom

import equinox as eqx
from equinox import nn
from equinox._module import static_field

def _identity(x):
    return x

class MLP(eqx.Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    layers: List[nn.Linear]
    dropout: nn.Dropout
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        dropout: float = 0.0,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key: "jax.random.PRNGKey",
        **kwargs
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(nn.Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, x: jax.Array, key: Optional["jax.random.PRNGKey"],
    ) -> jax.Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: For dropout

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        keys = jrandom.split(key, len(self.layers) - 1)
        for layer, key in zip(self.layers[:-1], keys):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x, key=key)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
    
    def vmap_with_key(self, x, key):
        if key is not None:
            key = jax.random.split(key, x.shape[0])
        
        return jax.vmap(self)(x, key)