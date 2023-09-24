from functools import partial
from typing import (
    Sequence,
    Optional,
    Callable,
    Tuple,
    Union,
)
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
from torch_dimcheck import dimchecked, A
from jax.profiler import annotate_function

from gecco_jax.types import PyTree, SampleDetails, LogpDetails
from gecco_jax.models.reparam import Reparam
from gecco_jax.models.stochastic import sample_stochastic, sample_inpaint

@dimchecked
def mse(xs: A['N* D'], ys: A['N* D']) -> A['']:
    return ((xs - ys) ** 2).mean()

def ema_update(old, new, alpha):
    def _single_ema_update(old, new):
        if eqx.is_inexact_array(new):
            return alpha * old + (1.0 - alpha) * new
        else:
            return new
    return jax.tree_map(_single_ema_update, old, new)

class NoCond(eqx.Module):
    '''
    A conditioning module for unconditional diffusion processes.
    '''
    def __call__(self, raw_ctx, *, key):
        del key

        return raw_ctx # this is likely a None anyway

class NoCondEqWrapper(eqx.Module):
    '''
    A wrapper which adapts equations of signature
    eq(t, x) -> dx
    to a conditional signature of
    eq(t, x, ctx) -> dx
    '''
    inner: eqx.Module

    def __call__(self, t, x, ctx, *, key):
        del ctx

        return self.inner(t, x, key=key)

class Schedule(eqx.Module):
    sigma_max: float = 25.
    sigma_data: float = 1.0
    n_solver_steps: int = 16

    sigma_min: float = 0.002
    rho: float = 7.0

    @dimchecked
    def sigma(self, t: A['']) -> A['']:
        return t

    @dimchecked
    def scale(self, t: A['']) -> A['']:
        return jnp.array(1.0)

    @dimchecked
    def c_skip(self, sigma: A['']) -> A['']:
        s_d = self.sigma_data
        return (s_d ** 2) / (sigma ** 2 + s_d ** 2)

    @dimchecked
    def c_out(self, sigma: A['']) -> A['']:
        s_d = self.sigma_data
        return sigma * s_d / jnp.sqrt(s_d ** 2 + sigma ** 2)

    @dimchecked
    def c_in(self, sigma: A['']) -> A['']:
        s_d = self.sigma_data
        return 1.0 / jnp.sqrt(sigma ** 2 + s_d ** 2)

    @dimchecked
    def c_noise(self, sigma: A['']) -> A['']:
        return sigma

    @dimchecked
    def sample_sigma(
        self,
        n: int,
        key,
    ) -> A['N']:
        raise NotImplementedError()

    @dimchecked
    def sample_latent(
        self,
        shape: Tuple[int],
        *,
        key: A['2']
    ) -> A['X*']:
        return self.sigma_max * jax.random.normal(key, shape)

    @dimchecked
    def loss_weight(
        self,
        sigma: A[''],
    ) -> A['']:
        s_d = self.sigma_data
        return (sigma ** 2 + s_d ** 2) / ((sigma * s_d) ** 2)
    
    @dimchecked
    def t_i(self, i: A['']) -> A['']:
        rho = self.rho
        N = self.n_solver_steps
        rho_inv = 1.0 / rho
        a = self.sigma_max ** rho_inv
        b = self.sigma_min ** rho_inv

        return (a + i / (N - 1) * (b - a)) ** rho

@dimchecked
def low_discrepancy_uniform(
    key: A['2'],
    n: int,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> A['N']:
    u = jax.random.uniform(key, (n,), minval=0, maxval=1 / n)
    u = u + (1 / n) * jnp.arange(n)
    
    return u * (maxval - minval) + minval

class LogUniformSchedule(Schedule):
    @dimchecked
    def sample_sigma(
        self,
        n: int,
        key,
    ) -> A['N']:
        log_sigma = low_discrepancy_uniform(
            key,
            n,
            minval=jnp.log(self.sigma_min),
            maxval=jnp.log(self.sigma_max),
        )

        return jnp.exp(log_sigma)
    
class LogNormalSchedule(Schedule):
    sigma_log_mean: float = 0.5
    sigma_log_std: float = 1.0

    @dimchecked
    def sample_sigma(
        self,
        n: int,
        key,
    ) -> A['N']:
        normal = jax.random.normal(key, shape=(n, ))
        log_sigma = self.sigma_log_std * normal + self.sigma_log_mean
        return jnp.exp(log_sigma)

@dimchecked
def trace_jac_estimator(
    fn: Callable,
    x: A['X*'],
    key: A['2'],
    n_samples: int = 1,
) -> A['']:
    @dimchecked
    def dot_all(a: A['X*'], b: A['X*']) -> A['']:
        return jnp.dot(a.flatten(), b.flatten())
    
    @dimchecked
    def single_estimator(noise: A['X*']) -> A['']:
        fn_e = lambda x: dot_all(fn(x), noise)
        grad_fn_e = jax.grad(fn_e)(x)
        return dot_all(grad_fn_e, noise)

    noise = jax.random.rademacher(key, (n_samples, *x.shape))
    return jax.vmap(single_estimator)(noise).mean(axis=0)

class Diffusion(eqx.Module):
    network: eqx.Module # [t, x, ctx, key] -> shape_like(x)
    cond: eqx.Module # raw_ctx -> ctx
    reparam: Reparam
    schedule: Schedule

    divergence_fn: Callable

    def __init__(
        self,
        network: eqx.Module,
        cond: Optional[eqx.Module],
        reparam: Reparam,
        schedule: Schedule = Schedule(),
        divergence_fn: Callable = mse,
    ):
        if cond is None:
            cond = NoCond()

        self.network = network
        self.cond = cond
        self.reparam = reparam
        self.schedule = schedule
        self.divergence_fn = divergence_fn

    @dimchecked
    def _network_forward(
        self,
        sigma: A[''],
        x: A['X*'], 
        ctx: PyTree,
        *,
        key,
    ) -> A['X*']:
        c_in = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)
        return self.network(
            t=c_noise,
            x=c_in * x,
            ctx=ctx,
            key=key,
        )

    @dimchecked
    def denoise(
        self,
        sigma: A[''],
        x: A['X*'], 
        ctx: PyTree,
        *,
        key,
    ) -> A['X*']:
        c_out = self.schedule.c_out(sigma)
        c_skip = self.schedule.c_skip(sigma)

        f = self._network_forward(sigma, x, ctx, key=key)
        return c_skip * x + c_out * f
    
    @dimchecked
    def score(
        self,
        sigma: A[''],
        x: A['X*'],
        ctx: PyTree,
        *, 
        key,
    ) -> A['X*']:
        return x - self.denoise(sigma, x, ctx, key=key)

    @dimchecked
    def _perturb_data(
        self,
        sigma: A[''],
        x: A['X*'],
        key,
    ) -> A['X*']:
        eps = jax.random.normal(key, x.shape)
        return x + sigma * eps

    @dimchecked
    def single_loss_fn(
        self,
        sigma: A[''],
        x: A['X*'],
        raw_ctx: PyTree,
        key,
    ) -> A['']:
        cond_key, data_key, net_key = jax.random.split(key, 3)

        x = self.reparam.data_to_diffusion(x, raw_ctx)
        ctx = self.cond(raw_ctx, key=cond_key)

        perturbed_x = self._perturb_data(sigma, x, data_key)
        x_hat = self.denoise(sigma, perturbed_x, ctx, key=net_key)
        weight = self.schedule.loss_weight(sigma)
        divergence = self.divergence_fn(x_hat, x)

        return weight * divergence
    
    @eqx.filter_jit
    @dimchecked
    def batch_loss_fn(
        self,
        x: A['B X*'],
        raw_ctx: PyTree,
        key,
        loss_scale: float = 1.0,
    ) -> A['']:
        batch_size = x.shape[0]
        sigma_key, noise_key = jax.random.split(key)
        noise_keys = jax.random.split(noise_key, batch_size)
        sigma = self.schedule.sample_sigma(batch_size, sigma_key)
        loss_fn = jax.vmap(self.single_loss_fn)
        return loss_scale * loss_fn(sigma, x, raw_ctx, noise_keys).mean()

    @dimchecked
    def _dx_dt(
        self,
        t: A[''],
        y: A['X*'],
        args: Tuple, # (ctx, network_prng_key)
    ) -> A['X*']:
        ctx, network_prng_key = args

        sigma, sigma_dot = jax.value_and_grad(self.schedule.sigma)(t)
        scale, scale_dot = jax.value_and_grad(self.schedule.scale)(t)

        denoised = self.denoise(
            sigma=sigma,
            x=y / scale,
            ctx=ctx,
            key=network_prng_key,
        )

        return (
            (sigma_dot / sigma + scale_dot / scale) * y
            - ((sigma_dot * scale) / sigma) * denoised
        )

    @dimchecked
    def solve_sample_ode(
        self,
        latent: A['X*'],
        raw_ctx: Optional[PyTree],
        ctx: Optional[PyTree] = None,
        return_full_trajectory: bool = False,
        *,
        key,
    ) -> A['T X*']:
        if (ctx is not None) and (raw_ctx is not None):
            raise ValueError('Both `ctx` and `raw_ctx` were provided.')

        cond_key, net_key = jax.random.split(key, 2)

        if ctx is None:
            ctx = self.cond(raw_ctx, key=cond_key)

        term = dfx.ODETerm(self._dx_dt)
        solver = dfx.Heun()
        ts = jax.vmap(self.schedule.t_i)(jnp.arange(self.schedule.n_solver_steps))
        t0 = ts[0]
        t1 = ts[-1]
        
        if return_full_trajectory:
            kw = dict(saveat=dfx.SaveAt(ts=ts))
        else:
            kw = dict()

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0=None, # using explicit step size controller
            y0=latent,
            stepsize_controller=dfx.StepTo(ts=ts),
            args=(ctx, net_key),
            **kw,
        )
        
        return sol.ys

    def _sample(
        self,
        x_shape: Sequence[int],
        ctx: Optional[PyTree] = None,
        return_details: bool = False,
        temperature: float = 1.0,
        *,
        key,
    ) -> Union[A['X*'], SampleDetails]:
        ''' 
        Sample a single example, assuming the context has already been pre-processed.
        This is intended to be vmapped inside `sample` to generate multiple samples per
        a single evaluation of the conditioning network.
        '''
        ode_key, latent_key = jax.random.split(key, 2)
        latent = self.schedule.sample_latent(x_shape, key=latent_key)
        latent = temperature * latent
        ys = self.solve_sample_ode(
            latent=latent,
            raw_ctx=None,
            ctx=ctx,
            key=ode_key,
            return_full_trajectory=return_details,
        )
        sample_diff = ys[-1]
        
        reparam = lambda diff: self.reparam.diffusion_to_data(diff, ctx)
        
        if not return_details:
            return reparam(sample_diff)
        else:
            return SampleDetails(
                latent=latent,
                sample_diff=sample_diff,
                sample_data=reparam(sample_diff),
                trajectory_diff=ys,
                trajectory_data=jax.vmap(reparam)(ys),
            )

    @eqx.filter_jit
    def sample(
        self,
        x_shape: Sequence[int],
        raw_ctx: PyTree,
        n: int,
        return_details: bool = False,
        temperature: float = 1.0,
        *,
        key,
    ) -> Union[A['N X*'], SampleDetails]:
        keys = jax.random.split(key, n+1)
        cond_key = keys[0]
        sample_keys = keys[1:]

        ctx = self.cond(raw_ctx, key=cond_key)
        sample_fn = lambda key: self._sample(
            x_shape,
            ctx=ctx,
            key=key,
            return_details=return_details,
            temperature=temperature,
        )
        
        return jax.vmap(sample_fn)(sample_keys)
    
    sample_stochastic = sample_stochastic
    sample_inpaint = sample_inpaint

    @dimchecked
    @eqx.filter_jit
    def evaluate_logp(
        self,
        data: A['X*'],
        raw_ctx: Optional[PyTree],
        ctx: Optional[PyTree],
        return_details: bool = False,
        n_log_det_jac_samples: int = 1,
        *,
        key: A['2'],
    ): # -> Union[A[''], LogpDetails]:
        @dimchecked
        def dx_dt(
            t: A[''],
            y: PyTree, # (data, logp)
            args: PyTree, # (ctx, net_key, noise)
        ) -> Tuple[A['X*'], A['']]:
            data, _logp = y
            ctx, net_key, noise_key = args
            
            dx_dt = lambda y: self._dx_dt(t=t, y=y, args=(ctx, net_key))

            ddata_dt = dx_dt(data)
            ddiv_dt = trace_jac_estimator(
                dx_dt,
                x=data,
                key=noise_key,
                n_samples=n_log_det_jac_samples,
            )
            
            return ddata_dt, ddiv_dt
        
        if (ctx is not None) and (raw_ctx is not None):
            raise ValueError('Both `ctx` and `raw_ctx` were provided.')

        cond_key, net_key, noise_key = jax.random.split(key, 3)

        if ctx is None:
            ctx = self.cond(raw_ctx, key=cond_key)

        data_diff = self.reparam.data_to_diffusion(data, ctx)

        y0 = (data_diff, jnp.array(0.0))
        term = dfx.ODETerm(dx_dt)
        solver = dfx.Heun()
        ii = jnp.arange(self.schedule.n_solver_steps)[::-1]
        ts = jax.vmap(self.schedule.t_i)(ii)
        t0 = ts[0]
        t1 = ts[-1]

        if return_details:
            kw = dict(saveat=dfx.SaveAt(ts=ts))
        else:
            kw = dict()

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0=None, # using explicit step size controller
            y0=y0,
            stepsize_controller=dfx.StepTo(ts=ts),
            args=(ctx, net_key, noise_key),
            **kw,
        )
        
        ys, delta_divs = sol.ys
        latent = ys[-1]
        delta_div = delta_divs[-1]
        
        prior_logp = jax.scipy.stats.norm.logpdf(
            latent,
            loc=0.0,
            scale=self.schedule.sigma_max,
        ).sum()

        delta_reparam = self.reparam.ladj_data_to_diffusion(data, ctx)

        logp = prior_logp + delta_div + delta_reparam

        if not return_details:
            return logp
        else:
            trajectory_data = jax.vmap(self.reparam.diffusion_to_data, in_axes=(0, None))(ys, ctx)

            return LogpDetails(
                logp=logp,
                prior_logp=prior_logp,
                delta_reparam=delta_reparam,
                delta_jacobian=delta_div,
                trajectory_diff=ys,
                trajectory_data=trajectory_data,
                latent=latent,
            )
    
    @classmethod
    def make_step(
        cls,
        model: 'Diffusion',
        x: jnp.ndarray,
        raw_ctx: PyTree,
        key,
        opt_state: PyTree,
        ema_state: 'Diffusion',
        opt_update: Callable,
        loss_scale: float = 1.0,
        is_distributed: bool = True,
        ema_alpha: float = 0.999,
    ):
        '''
        An associated function defining the entire training step
        (loss, gradient computation, optimizer update).
        '''
        loss_fn = cls.batch_loss_fn # get unbound method
        grad_fn = eqx.filter_value_and_grad(
            partial(
                loss_fn,
                loss_scale=loss_scale,
            ),
        )
        grad_fn = annotate_function(grad_fn, name='value_and_grad')
        loss, grads = grad_fn(model, x, raw_ctx, key)
        
        if is_distributed:
            loss = jax.lax.pmean(loss, axis_name='device')
            grads = jax.lax.pmean(grads, axis_name='device')

        opt_update = annotate_function(opt_update, name='opt_update')
        updates, opt_state = opt_update(grads, opt_state, model)
        apply_updates = annotate_function(eqx.apply_updates, name='apply_updates')
        model = apply_updates(model, updates)
        ema_state = ema_update(ema_state, model, alpha=ema_alpha)
        return loss, model, opt_state, ema_state
