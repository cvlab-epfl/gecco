import jax.numpy as jnp
import equinox as eqx

from torch_dimcheck import A, dimchecked

class GaussianActivation(eqx.Module):
    alpha: A[''] = jnp.array(1.)
    normalized: bool = False

    @dimchecked
    def __call__(self, x: A['D*']) -> A['D*']:
        y = jnp.exp(-x**2 / (2 * self.alpha**2))
        if self.normalized:
            # normalize by activation mean and std assuming
            # `x ~ N(0, 1)`
            y = (y - 0.7) / 0.28

        return y