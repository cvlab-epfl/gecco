import os
import math
from typing import Iterable, Optional, Callable, Union
from functools import partial

import jax
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from gecco_jax.metrics import (
    chamfer_distance,
    sinkhorn_emd,
    chamfer_distance_squared,
)
from gecco_jax.models.diffusion import Diffusion


def batched_pairwise_distance(a, b, distance_fn, block_size):
    """
    Returns an N-by-N array of the distance between a[i] and b[j]
    """
    assert a.shape == b.shape, (a.shape, b.shape)

    dist = jax.vmap(jax.vmap(distance_fn, in_axes=(None, 0)), in_axes=(0, None))
    dist = jax.jit(dist)

    n_batches = int(math.ceil(a.shape[0] / block_size))
    distances = []
    for a_block in tqdm(
        np.array_split(a, n_batches), desc="Computing pairwise distances"
    ):
        row = []
        for b_block in np.array_split(b, n_batches):
            row.append(np.asarray(dist(a_block, b_block)))
        distances.append(np.concatenate(row, axis=1))
    return np.concatenate(distances, axis=0)


def extract_data(loader: Iterable, n_examples: Optional[int]):
    if n_examples is not None:
        assert len(loader.dataset) >= n_examples, len(loader.dataset)

    collected = []
    for batch in loader:
        collected.append(batch.points.numpy())
        if n_examples is not None and sum(c.shape[0] for c in collected) >= n_examples:
            break
    data = np.concatenate(collected, axis=0)[:n_examples]
    return data


class BenchmarkCallback:
    def __init__(
        self,
        data: np.array,
        batch_size: int = 64,
        tag_prefix: str = "benchmark",
        rng_seed: int = 42,
        block_size: int = 16,
        distance_fn: Union[str, Callable] = chamfer_distance,
        save_path: Optional[str] = None,
    ):
        self.data = data
        self.n_points = self.data.shape[1]
        self.batch_size = batch_size
        self.tag_prefix = tag_prefix
        self.n_batches = int(math.ceil(self.data.shape[0] / self.batch_size))
        self.rng_seed = rng_seed
        self.block_size = block_size

        if isinstance(distance_fn, str):
            distance_fn = {
                "chamfer": chamfer_distance,
                "chamfer_squared": chamfer_distance_squared,
                "emd": partial(sinkhorn_emd, epsilon=0.1),
            }[distance_fn]

        if hasattr(distance_fn, "func"):
            self.distance_fn_name = distance_fn.func.__name__
        else:
            self.distance_fn_name = distance_fn.__name__

        self.distance_fn = partial(
            batched_pairwise_distance,
            distance_fn=distance_fn,
            block_size=self.block_size,
        )

        self.dd_dist = self.distance_fn(self.data, self.data)

        if save_path is not None:
            save_path = os.path.join(
                save_path, "benchmark-checkpoints", self.distance_fn_name
            )
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.lowest_1nn = float("inf")

    @classmethod
    def from_loader(
        cls,
        loader,
        n_examples: Optional[int] = None,
        **kwargs,
    ) -> "BenchmarkCallback":
        data = extract_data(loader, n_examples)
        return cls(data, batch_size=loader.batch_size, **kwargs)

    def sample_from_model(self, model):
        key = jax.random.PRNGKey(self.rng_seed)

        samples = []
        for key in tqdm(
            jax.random.split(key, self.n_batches), desc="Sampling for benchmark"
        ):
            sample = model.sample(
                (self.n_points, 3),
                n=self.batch_size,
                raw_ctx=None,
                key=key,
            )
            samples.append(np.asarray(sample))
        return np.concatenate(samples, axis=0)[: self.data.shape[0]]

    def _assemble_dist_m(self, ss_dist, sd_dist):
        dd_dist = self.dd_dist
        ds_dist = sd_dist.T

        return np.concatenate(
            [
                np.concatenate([ss_dist, sd_dist], axis=1),
                np.concatenate([ds_dist, dd_dist], axis=1),
            ],
            axis=0,
        )

    def _one_nn_acc(self, ss_dist, sd_dist):
        dist_m = self._assemble_dist_m(ss_dist, sd_dist)

        n = ss_dist.shape[0]
        np.fill_diagonal(dist_m, float("inf"))

        amin = dist_m.argmin(axis=0)
        one_nn_1 = amin[:n] <= n
        one_nn_2 = amin[n:] > n

        return np.concatenate([one_nn_1, one_nn_2]).mean()

    def _mmd(self, sd_dist):
        return sd_dist.min(axis=0).min()

    def _cov(self, sd_dist):
        return np.unique(sd_dist.argmin(axis=1)).size / sd_dist.shape[1]

    def _distance_hist(self, ss_dist, sd_dist):
        dd_dist = self.dd_dist

        fig, ax = plt.subplots(tight_layout=True)
        kw = dict(
            histtype="step",
            bins=np.linspace(0, dd_dist.max() * 1.3, 20),
        )
        ax.hist(dd_dist.flatten(), color="r", label="data-data", **kw)
        ax.hist(ss_dist.flatten(), color="b", label="sample-sample", **kw)
        ax.hist(sd_dist.flatten(), color="g", label="sample-data", **kw)
        fig.legend()

        return fig

    def _plot_dist_m(self, ss_dist, sd_dist):
        dist_m = self._assemble_dist_m(ss_dist, sd_dist)
        dist_inf = dist_m + np.diag(np.ones(dist_m.shape[0]) * float("inf"))

        fig, ax = plt.subplots(tight_layout=True, figsize=(6, 6))
        ax.imshow(dist_inf, vmax=self.dd_dist.max())
        ax.set_xticks([ss_dist.shape[0]])
        ax.set_yticks([ss_dist.shape[0]])
        return fig

    def call_without_logging(self, samples):
        ss_dist = self.distance_fn(samples, samples)
        sd_dist = self.distance_fn(samples, self.data)

        one_nn_acc = self._one_nn_acc(ss_dist, sd_dist)
        mmd = self._mmd(sd_dist)
        cov = self._cov(sd_dist)

        histogram = self._distance_hist(ss_dist, sd_dist)
        dist_m_fig = self._plot_dist_m(ss_dist, sd_dist)

        scalars = {
            f"{self.tag_prefix}/1-nn-acc/{self.distance_fn_name}": one_nn_acc,
            f"{self.tag_prefix}/mmd/{self.distance_fn_name}": mmd,
            f"{self.tag_prefix}/cov/{self.distance_fn_name}": cov,
        }

        plots = {
            f"{self.tag_prefix}/histograms/{self.distance_fn_name}": histogram,
            f"{self.tag_prefix}/dist-mat/{self.distance_fn_name}": dist_m_fig,
        }

        return scalars, plots

    def __call__(
        self,
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        samples = self.sample_from_model(model)
        scalars, plots = self.call_without_logging(samples)

        for key, value in scalars.items():
            logger.add_scalar(key, scalar_value=value, global_step=epoch)

        for key, value in plots.items():
            logger.add_figure(key, figure=value, global_step=epoch)

        if self.save_path is None:
            return
        _1nn_tag = f"{self.tag_prefix}/1-nn-acc/{self.distance_fn_name}"
        _1nn_score = scalars[_1nn_tag]
        if not _1nn_score < self.lowest_1nn:
            return
        print(f"{_1nn_score} improves over {self.lowest_1nn} at {_1nn_tag}.")
        self.lowest_1nn = _1nn_score
        eqx.tree_serialise_leaves(f"{self.save_path}/{epoch}.eqx", model)
