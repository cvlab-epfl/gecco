import jax
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from gecco_jax.models.diffusion import Diffusion, SampleDetails

CMAP = plt.get_cmap("viridis")


def make_unconditional_sample_callback(
    geom_dim: int = 3,
    n_samples: int = 8,
    n_points: int = 128,
    point_size: float = 0.1,
    key=jax.random.PRNGKey(42),
):
    def callback(
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        samples: SampleDetails = model.sample(
            (n_points, geom_dim),
            n=n_samples,
            raw_ctx=None,
            return_details=True,
            key=key,
        )

        points = np.asarray(samples.sample_data)
        latent = np.asarray(samples.latent)

        latent_r = np.linalg.norm(latent, axis=-1)
        r_normalized = 1.0 - np.clip(
            latent_r / (2 * model.schedule.sigma_max), 0.0, 1.0
        )
        colors = CMAP(r_normalized, bytes=True)[..., :3]  # clip away the alpha channel

        logger.add_mesh(
            tag="samples",
            vertices=points,
            colors=colors,
            global_step=epoch,
            config_dict={
                "material": {
                    "cls": "PointsMaterial",
                    "size": point_size,
                },
            },
        )

    return callback
