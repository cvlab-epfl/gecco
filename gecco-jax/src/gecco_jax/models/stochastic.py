import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from torch_dimcheck import A, dimchecked

from gecco_jax.types import PyTree


@eqx.filter_jit
@dimchecked
def _sample_stochastic(
    self,
    shape: Tuple[int, int],
    ctx: PyTree,
    s_churn: float = 0.0,
    s_noise: float = 1.0,
    *,
    key,
) -> A["N 3"]:
    init_key, loop_key = jax.random.split(key)
    n_steps = self.schedule.n_solver_steps

    i2s = lambda i: self.schedule.sigma(self.schedule.t_i(i))

    def loop_body(i, x_and_rng):
        x_cur, rng = x_and_rng
        key, churn_key, net_key_1, net_key_2 = jax.random.split(rng, 4)

        s_cur = i2s(i)
        s_next = i2s(i + 1)

        gamma = min(s_churn / n_steps, math.sqrt(2) - 1)
        s_hat = s_cur * (1 + gamma)
        churn_std = jnp.sqrt(s_hat**2 - s_cur**2) * s_noise
        x_hat = x_cur + churn_std * jax.random.normal(churn_key, shape)

        denoised = self.denoise(s_hat, x_hat, ctx, key=net_key_1)
        d_cur = (x_hat - denoised) / s_hat
        x_next = x_hat + (s_next - s_hat) * d_cur

        def second_order(x):
            denoised = self.denoise(s_next, x, ctx, key=net_key_2)
            d_prime = (x - denoised) / s_next
            return x_hat + (s_next - s_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_next = jax.lax.cond(
            i < n_steps - 1,
            second_order,
            lambda x: x,
            x_next,
        )

        return x_next, key

    s_0 = i2s(jnp.array(0))
    x_init = jax.random.normal(init_key, shape) * s_0

    samples_diff, _key = jax.lax.fori_loop(
        0,
        n_steps,
        loop_body,
        (x_init, loop_key),
    )

    return self.reparam.diffusion_to_data(samples_diff, ctx)


@eqx.filter_jit
def sample_stochastic(
    self,
    shape: Tuple[int, int],
    raw_ctx: PyTree,
    n: int = 1,
    s_churn: float = 0.0,
    s_noise: float = 1.0,
    *,
    key,
):
    keys = jax.random.split(key, n + 1)
    cond_key = keys[0]
    keys = keys[1:]

    ctx = self.cond(raw_ctx, key=cond_key)

    return jax.vmap(
        partial(
            _sample_stochastic,
            self=self,
            shape=shape,
            ctx=ctx,
            s_churn=s_churn,
            s_noise=s_noise,
        )
    )(key=keys)


@dimchecked
def _sample_inpaint(
    self,
    known: A["N 3"],
    m_to_inpaint: int,
    ctx: PyTree,
    s_churn: float = 0.0,
    s_noise: float = 1.0,
    n_substeps: int = 1,
    *,
    key,
) -> A["NM 3"]:
    init_key, loop_key = jax.random.split(key, 2)
    n_steps = self.schedule.n_solver_steps
    known_diff = self.reparam.data_to_diffusion(known, ctx)

    i2s = lambda i: self.schedule.sigma(self.schedule.t_i(i))
    identity = lambda x: x

    def outer_loop_body(i, x_and_rng):
        def inner_loop_body(j, x_and_rng):
            x_cur, rng = x_and_rng

            (
                key,
                churn_key,
                known_key,
                redo_key,
                net_key_1,
                net_key_2,
            ) = jax.random.split(rng, 6)

            s_cur = i2s(i)
            s_next = i2s(i + 1)

            x_cur = jnp.concatenate(
                [
                    x_cur[:m_to_inpaint],
                    known_diff + jax.random.normal(known_key, known.shape) * s_cur,
                ],
                axis=0,
            )

            gamma = min(s_churn / n_steps, math.sqrt(2) - 1)
            s_hat = s_cur * (1 + gamma)
            churn_std = jnp.sqrt(s_hat**2 - s_cur**2) * s_noise
            x_hat = x_cur + churn_std * jax.random.normal(churn_key, x_cur.shape)

            denoised = self.denoise(s_hat, x_hat, ctx, key=net_key_1)
            d_cur = (x_hat - denoised) / s_hat
            x_next = x_hat + (s_next - s_hat) * d_cur

            def second_order(x):
                denoised = self.denoise(s_next, x, ctx, key=net_key_2)
                d_prime = (x - denoised) / s_next
                return x_hat + (s_next - s_hat) * (0.5 * d_cur + 0.5 * d_prime)

            x_next = jax.lax.cond(
                i < n_steps - 1,
                second_order,
                identity,
                x_next,
            )

            def redo_noise(x):
                std = jnp.sqrt(s_cur**2 - s_next**2)
                return x + std * jax.random.normal(redo_key, x.shape)

            x_next = jax.lax.cond(
                j < n_substeps - 1,
                redo_noise,
                identity,
                x_next,
            )

            return x_next, key

        return jax.lax.fori_loop(
            0,
            n_substeps,
            inner_loop_body,
            x_and_rng,
        )

    s_0 = i2s(jnp.array(0))
    x_init = jnp.concatenate(
        [
            jnp.zeros((m_to_inpaint, 3)),
            known_diff,
        ],
        axis=0,
    )
    x_init = x_init + jax.random.normal(init_key, x_init.shape) * s_0

    samples_diff, _key = jax.lax.fori_loop(
        0,
        n_steps,
        outer_loop_body,
        (x_init, loop_key),
    )

    return self.reparam.diffusion_to_data(samples_diff, ctx)[:m_to_inpaint]


@eqx.filter_jit
def sample_inpaint(
    self,
    known: A["N 3"],
    m_to_inpaint: int,
    raw_ctx: PyTree,
    n_completions=1,
    *,
    key,
    **kwargs,
):
    keys = jax.random.split(key, n_completions + 1)
    cond_key = keys[0]
    keys = keys[1:]

    ctx = self.cond(raw_ctx, key=cond_key)

    return jax.vmap(
        partial(
            _sample_inpaint,
            self=self,
            known=known,
            ctx=ctx,
            m_to_inpaint=m_to_inpaint,
            **kwargs,
        )
    )(key=keys)
