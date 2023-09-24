from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from matplotlib.collections import LineCollection
from einops import rearrange
from torch_dimcheck import dimchecked, A
from torch.utils.tensorboard import SummaryWriter

from gecco_jax.models.diffusion import Diffusion
from gecco_jax.types import PyTree
from gecco_jax.data.pc_mnist import Example

FIG_SIZE = (8, 8)
DPI = 100


def _plot_diffusion_trajectories(
    trajectories: A["B T N 2"],
    layout: Tuple[int, int] = (4, 4),
    plot_range: Optional[Tuple[float, float]] = None,
):
    assert layout[0] * layout[1] == trajectories.shape[0]

    fig, axes = plt.subplots(
        *layout,
        figsize=FIG_SIZE,
        tight_layout=True,
        dpi=DPI,
    )

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, trajectories.shape[1]))

    @dimchecked
    def pad_trajectory(traj: A["T N 2"]) -> A["t n 2"]:
        traj = np.stack([traj[:-1], traj[1:]], axis=0)  # 2 T N 2
        traj = np.pad(
            traj, ((1, 0), (0, 0), (0, 0), (0, 0)), constant_values=float("nan")
        )
        traj = rearrange(traj, "p t n d -> t (n p) d")
        return traj

    if plot_range is None:
        xrange, yrange = np.abs(trajectories).max(axis=(0, 1, 2))
    else:
        xrange, yrange = plot_range

    for ax, trajectory in zip(axes.flat, trajectories):
        ax.add_collection(
            LineCollection(
                pad_trajectory(trajectory),
                colors=colors,
                linewidths=2.5,
            ),
        )

        ax.set_xlim([-xrange, xrange])
        ax.set_ylim([yrange, -yrange])

    return fig


@dimchecked
def _plot_diffusion_sequence(
    sequence: A["B T N 2"],
    plot_range: Optional[Tuple[float, float]] = None,
):
    fig, axes = plt.subplots(
        *sequence.shape[:2],
        figsize=(16, 9),
        facecolor="gray",
        dpi=DPI,
    )

    if plot_range is None:
        xrange, yrange = np.abs(sequence).max(axis=(0, 1, 2))
    else:
        xrange, yrange = plot_range

    for (
        ax_row,
        sample_row,
    ) in zip(axes, sequence):
        for ax, sample in zip(ax_row, sample_row):
            ax.scatter(*sample.T, marker=".", s=5)
            ax.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
            ax.set_xlim([-xrange, xrange])
            ax.set_ylim([yrange, -yrange])
    fig.tight_layout(pad=0.25)

    return fig


def _logp_figure(
    model: Diffusion,
    data: PyTree,
    layout: Tuple[int, int],
    *,
    key,
):
    n_figures = layout[0] * layout[1]
    assert data.points.shape[0] >= n_figures
    data = jax.tree_map(
        lambda array: array[:n_figures],
        data,
    )

    ii = jnp.arange(model.schedule.n_solver_steps)
    sigmas = jax.vmap(lambda i: model.schedule.sigma(model.schedule.t_i(i)))(ii)
    diff_sigmas = np.sqrt(1 + sigmas[::-1] ** 2)

    logp_detail = jax.vmap(
        lambda data, key: model.evaluate_logp(
            data=data,
            raw_ctx=None,
            ctx=None,
            return_details=True,
            key=key,
        ),
    )(data.points, jax.random.split(key, n_figures))

    normalized_trajectories = (
        logp_detail.trajectory_diff / diff_sigmas[None, :, None, None]
    )

    overlay_fig = _plot_diffusion_trajectories(
        normalized_trajectories[:n_figures],
        layout=layout,
        plot_range=(3, 3),
    )

    sequence_fig = _plot_diffusion_sequence(
        normalized_trajectories[:n_figures],
        plot_range=(3, 3),
    )

    return overlay_fig, sequence_fig


def make_logp_callback(
    data: PyTree,
    layout: Tuple[int, int] = (3, 3),
    key=jax.random.PRNGKey(42),
):
    n_figures = layout[0] * layout[1]
    assert data.points.shape[0] >= n_figures
    data = jax.tree_map(
        lambda tensor: jnp.asarray(tensor[:n_figures]),
        data,
    )

    def callback(
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        trajectory_fig, sequence_fig = _logp_figure(
            model,
            data,
            layout,
            key=key,
        )

        logger.add_figure(
            "logp-trajectories",
            figure=trajectory_fig,
            global_step=epoch,
        )

        logger.add_figure(
            "logp-sequence",
            figure=sequence_fig,
            global_step=epoch,
        )

    return callback


def _sample_figures(
    model: Diffusion,
    layout: Tuple[int, int],
    n_points: int,
    *,
    key,
):
    n_figures = layout[0] * layout[1]

    ii = jnp.arange(model.schedule.n_solver_steps)
    sigmas = jax.vmap(lambda i: model.schedule.sigma(model.schedule.t_i(i)))(ii)
    diff_sigmas = np.sqrt(1 + sigmas**2)

    sample_detail = model.sample(
        (n_points, 2),
        n=n_figures,
        raw_ctx=None,
        return_details=True,
        key=key,
    )

    sample_fig, axes = plt.subplots(
        *layout,
        figsize=FIG_SIZE,
        tight_layout=True,
        dpi=DPI,
    )

    for ax, sample in zip(axes.flat, sample_detail.sample_data):
        ax.scatter(*sample.T, marker=".")
        ax.set_xlim([0, 28])
        ax.set_ylim([28, 0])

    normalized_trajectory = (
        sample_detail.trajectory_diff / diff_sigmas[None, :, None, None]
    )
    trajectory_fig = _plot_diffusion_trajectories(
        normalized_trajectory,
        layout=layout,
        plot_range=(3, 3),
    )

    sequence_fig = _plot_diffusion_sequence(
        normalized_trajectory,
        plot_range=(3, 3),
    )

    return sample_fig, trajectory_fig, sequence_fig


def make_sample_callback(
    layout: Tuple[int, int] = (3, 3),
    n_points: int = 128,
    key=jax.random.PRNGKey(42),
):
    def callback(
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        sample_fig, trajectory_fig, sequence_fig = _sample_figures(
            model, layout, n_points, key=key
        )
        logger.add_figure(
            "samples",
            figure=sample_fig,
            global_step=epoch,
        )
        logger.add_figure(
            "sample-trajectories",
            figure=trajectory_fig,
            global_step=epoch,
        )
        logger.add_figure(
            "sample-sequences",
            figure=sequence_fig,
            global_step=epoch,
        )

    return callback


@eqx.filter_jit
def _evaluate_denoising(
    model: Diffusion,
    data: PyTree,
    n_steps: int = 6,
    key: int = 42,
):
    s = model.schedule
    steps = jnp.linspace(0, s.n_solver_steps - 1, n_steps)
    sigmas = jax.vmap(lambda step: s.sigma(s.t_i(step)))(steps)
    noise_key, model_key = jax.random.split(jax.random.PRNGKey(key), 2)

    @dimchecked
    def denoise(
        x: PyTree,
        sigma: A[""],
    ):
        data_diff = model.reparam.data_to_diffusion(x.points, x.ctx)
        perturbed = model._perturb_data(sigma, data_diff, noise_key)
        denoised = model.denoise(sigma, data_diff, x.ctx, key=model_key)

        # rescale the noisy images to have the same std as raw data, to help
        # visualization by overlaying in the same plot
        noise_scale = jnp.sqrt(1 + sigma**2)
        perturbed_scaled = perturbed / noise_scale

        return data_diff, perturbed_scaled, denoised

    return jax.vmap(
        jax.vmap(
            denoise,
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )(data, sigmas)


def _denoising_figure(
    model: Diffusion,
    data: Example,
    key: int = 42,
    n_steps: int = 6,
    n_examples: int = 6,
):
    data = jax.tree_map(
        lambda tensor: jnp.asarray(tensor)[:n_examples],
        data,
    )

    data_diff, pert, denoised = [
        np.asarray(arr)
        for arr in _evaluate_denoising(model, data, key=key, n_steps=n_steps)
    ]
    fig, axes = plt.subplots(
        n_steps,
        n_examples,
        figsize=FIG_SIZE,
        facecolor="gray",
        tight_layout=True,
        dpi=DPI,
    )

    kw = dict(marker=".", s=5)
    for i, (ax_row, data_row, pert_row, denoise_row) in enumerate(
        zip(axes, data_diff, pert, denoised)
    ):
        for ax, digit, data_ex, pert_ex, denoised_ex in zip(
            ax_row, data.ctx.digit, data_row, pert_row, denoise_row
        ):
            ax.scatter(*pert_ex.T, color="blue", **kw)
            ax.scatter(*data_ex.T, color="black", **kw)
            ax.scatter(*denoised_ex.T, color="red", **kw)
            ax.set_xlim([-3, 3])
            ax.set_ylim([3, -3])
            ax.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )

            if i == 0:
                ax.set_title(str(int(digit)))

    return fig, axes


def make_denoise_callback(
    data: PyTree,
    key: int = 42,
    n_steps: int = 6,
    n_examples: int = 6,
):
    def callback(
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        fig, _axes = _denoising_figure(
            model,
            data,
            n_steps=n_steps,
            key=key,
            n_examples=n_examples,
        )

        logger.add_figure(
            "denoising",
            figure=fig,
            global_step=epoch,
        )

    return callback
