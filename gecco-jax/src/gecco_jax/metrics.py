from typing import Callable, Dict, Sequence, Tuple
from functools import partial
import jax

import jax.numpy as jnp
import equinox as eqx
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_dimcheck import dimchecked, A

from gecco_jax.models.diffusion import Diffusion, LogpDetails
from gecco_jax.geometry import distance_matrix
from gecco_jax.types import PyTree, PRNGKey

class Metric:
    name: str

    def __call__(
        self,
        model: Diffusion,
        data: PyTree,
        raw_ctx: PyTree,
        key: PRNGKey,
    ):
        raise NotImplementedError()

class LossMetric(Metric, eqx.Module):
    loss_scale: float
    name: str = 'loss'

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model: Diffusion,
        data: A['B X*'],
        raw_ctx: PyTree,
        key: A['2'],
    ) -> Dict[str, A['']]:
        loss = type(model).batch_loss_fn(
            model,
            data,
            raw_ctx,
            key=key,
            loss_scale=self.loss_scale,
        )

        return {'loss': loss}

class LogpMetric(Metric, eqx.Module):
    name: str = 'logp'
    n_log_det_jac_samples: int = 1

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model: Diffusion,
        data: A['B X*'],
        raw_ctx: PyTree,
        key: A['2'],
    ) -> Dict[str, A['']]:
        @eqx.filter_vmap
        def v_sample_fn(data, raw_ctx, key):
            return model.evaluate_logp(
                data=data,
                raw_ctx=raw_ctx,
                ctx=None,
                return_details=True,
                n_log_det_jac_samples=self.n_log_det_jac_samples,
                key=key,
            )

        keys = jax.random.split(key, data.shape[0])
        details: LogpDetails = v_sample_fn(
            data,
            raw_ctx,
            keys,
        )

        return {
            'total': details.logp,
            'prior': details.prior_logp,
            'det-jac': details.delta_jacobian,
            'reparam': details.delta_reparam,
        }

@dimchecked
def chamfer_distance(
    a: A['N D'],
    b: A['N D'],
    squared: bool = False,
) -> A['']:
    dist_m = distance_matrix(a, b, squared=squared)
    min_a = dist_m.min(axis=0).mean()
    min_b = dist_m.min(axis=1).mean()

    return (min_a + min_b) / 2

@dimchecked
def chamfer_distance_squared(
    a: A['N D'],
    b: A['N D'],
) -> A['']:
    return chamfer_distance(a, b, squared=True)

@dimchecked
def _scipy_lsa(cost_matrix: A['N N']) -> Tuple[A['N'], A['N']]:
    shape = jnp.zeros(cost_matrix.shape[0], dtype=np.int32)
    
    def inner(cost_matrix):
        rows, cols = linear_sum_assignment(cost_matrix)
        return rows.astype(np.int32), cols.astype(np.int32)
    
    return jax.pure_callback(
        inner,
        (shape, shape),
        jax.lax.stop_gradient(cost_matrix),
        vectorized=False,
    )

@dimchecked
def scipy_emd(p1: A['N D'], p2: A['N D'], match='l1', average='l1') -> A['']:
    match_squared = {'l1': False, 'l2': True}[match]
    match_dist = distance_matrix(p1, p2, squared=match_squared)
    rows, cols = _scipy_lsa(match_dist)
    
    average_squared = {'l1': False, 'l2': True}[average]
    if average_squared == match_squared:
        average_dist = match_dist
    else:
        average_dist = distance_matrix(p1, p2, squared=average_squared)
        
    return average_dist[rows, cols].mean()

@dimchecked
def sinkhorn_emd(
    p1: A['N D'],
    p2: A['N D'],
    epsilon: float = 0.01,
) -> A['']:
    import ott

    cloud = ott.geometry.pointcloud.PointCloud(p1, p2, epsilon=epsilon)
    ot_prob = ott.problems.linear.linear_problem.LinearProblem(cloud)
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    solution = solver(ot_prob)
    return jnp.einsum('ab,ab->', solution.matrix, cloud.cost_matrix)

class _SinkhornEMDMetric:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.__name__ = f'sinkhorn_emd_epsilon_{epsilon}'
    
    def __call__(self, p1, p2):
        return sinkhorn_emd(p1, p2, epsilon=self.epsilon)

class SupervisedMetric(Metric, eqx.Module):
    name: str = 'supervised'
    metrics: Sequence[Callable] = (
        chamfer_distance,
        #_SinkhornEMDMetric(epsilon=0.01),
        #_SinkhornEMDMetric(epsilon=1.0),
    )

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model: Diffusion,
        data: A['B X*'],
        raw_ctx: PyTree,
        key: A['2'],
    ) -> Dict[str, A['']]:
        @eqx.filter_vmap
        def v_sample_fn(raw_ctx, key) -> jnp.ndarray:
            return model.sample(
                x_shape=data.shape[-2:],
                raw_ctx=raw_ctx,
                n=1,
                key=key
            )
        #v_sample_fn = eqx.filter_vmap(partial(
        #    model.sample,
        #    x_shape=data.shape[-2:],
        #    raw_ctx=raw_ctx,
        #    n=1,
        #))
        keys = jax.random.split(key, data.shape[0])
        samples = v_sample_fn(raw_ctx, keys)
        samples = samples.squeeze(1) # n_samples dimension
        
        results = {}
        for metric in self.metrics:
            results[metric.__name__] = jax.vmap(metric)(samples, data)
        
        return results

class MetricPmapWrapper(Metric):
    def __init__(self, inner):
        self.inner = inner
    
    @property
    def name(self):
        return self.inner.name

    def __call__(self, model, xyz, raw_ctx, key):
        keys = jax.random.split(key, jax.device_count())
        keys = jax.device_put_sharded(list(keys), jax.devices())
        values = eqx.filter_pmap(self.inner)(model, xyz, raw_ctx, keys)
        return jax.tree_map(
            lambda array: array.mean(axis=0),
            values,
        )