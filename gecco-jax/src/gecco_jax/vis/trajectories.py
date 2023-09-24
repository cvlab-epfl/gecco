from typing import Optional
import numpy as np
import k3d
from torch_dimcheck import dimchecked, A
from einops import rearrange, repeat


@dimchecked
def k3d_trajectories(
    trajectories: A["T N 3"],
    endpoint_size: float = 0.1,
    knot_size: Optional[float] = None,
):
    @dimchecked
    def add_nans(trajectories: A["T N C"]) -> A["t N C"]:
        """
        k3d doesn't normally support adding multiple line plots with a single API call
        but if we separate the lines with NaNs, they become disconnected
        """
        traj = np.asarray(trajectories).astype(np.float32)

        # any point which has a NaN as a neighbour will disappear from the plot
        # so we need to replicate the edges to keep them
        traj = np.pad(
            traj,
            ((1, 1), (0, 0), (0, 0)),
            mode="edge",
        )
        # and then add the NaNs themselves
        traj = np.pad(
            traj,
            ((0, 1), (0, 0), (0, 0)),
            mode="constant",
            constant_values=float("nan"),
        )
        return traj

    t = np.linspace(0, 1, trajectories.shape[0])
    t = repeat(t, "t -> t n 1", n=trajectories.shape[1])

    plot = k3d.plot()
    plot += k3d.line(
        rearrange(add_nans(trajectories), "t n d -> (n t) d"),
        attribute=rearrange(add_nans(t), "t n 1 -> (n t)"),
        color_range=(0.0, 1.0),
        shader="thick",
    )

    # mark starts with green
    plot += k3d.points(
        trajectories[0],
        color=0x00FF00,
        point_size=endpoint_size,
    )
    # mark ends with red
    plot += k3d.points(
        trajectories[-1],
        color=0xFF0000,
        point_size=endpoint_size,
    )

    if knot_size is not None:
        # mark intermediate evaluation points with smaller blue spheres
        plot += k3d.points(
            trajectories[1:-1].reshape(-1, 3),
            color=0x0000FF,
            point_size=knot_size,
        )

    return plot
