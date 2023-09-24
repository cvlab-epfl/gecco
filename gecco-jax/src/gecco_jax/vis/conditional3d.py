from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import jax
import numpy as np

import mitsuba as mi

if mi.variant() is None:
    try:
        mi.set_variant("llvm_ad_rgb")
    except AttributeError:
        mi.set_variant("scalar_rgb")

from mitsuba import ScalarTransform4f as T

from torch.utils.tensorboard import SummaryWriter
from torch_dimcheck import dimchecked, A

from gecco_jax.models import Diffusion
from gecco_jax.types import Example
from gecco_jax.types import PRNGKey


def to_srgb(image):
    return image ** (1.0 / 2.2)


def to_uint8(image):
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


@dataclass
class Camera:
    to_world: np.ndarray
    fov: float
    image_shape: Tuple[int, int]
    sample_count: int = 16

    def create_sensor(self):
        return mi.load_dict(
            {
                "type": "perspective",
                "fov": self.fov,
                "to_world": self.to_world,
                "sampler": {
                    "type": "independent",
                    "sample_count": self.sample_count,
                },
                "film": {
                    "type": "hdrfilm",
                    "width": self.image_shape[0],
                    "height": self.image_shape[1],
                    "rfilter": {
                        "type": "tent",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    @classmethod
    @dimchecked
    def look_at(
        cls,
        origin: A["3"],
        target: A["3"],
        up: A["3"],
        **kwargs,
    ):
        return cls(
            to_world=T.look_at(
                origin=origin,
                target=target,
                up=up,
            ),
            **kwargs,
        )


def _default_povs() -> Sequence[Camera]:
    at = np.array([0, 0, 1])

    origins = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
        ]
    ) + at.reshape(1, 3)

    ups = np.array(
        [
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    return tuple(
        Camera.look_at(origin, at, up, fov=55.0, image_shape=(256, 256))
        for origin, up in zip(origins, ups)
    )


@dataclass
class Renderer:
    cameras: Sequence[Camera] = _default_povs()
    colors: Sequence[np.ndarray] = (
        np.array([0.8, 0.1, 0.1]),
        np.array([0.1, 0.1, 0.8]),
    )
    point_size: float = 0.01
    floor: Optional[np.ndarray] = None

    bsdfs: Sequence[object] = ()

    def __post_init__(self):
        bsdfs = []
        for color in self.colors:
            bsdfs.append(
                mi.load_dict(
                    {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": color},
                    }
                )
            )
        self.bsdfs = tuple(bsdfs)

    def render(
        self,
        points: Sequence[np.ndarray],
        srgb: bool = True,
        uint8: bool = True,
    ) -> Sequence[np.ndarray]:
        points = [np.asarray(p) for p in points]
        scene = self._create_scene(points)
        renders = [
            mi.render(
                scene,
                sensor=c.create_sensor(),
            )
            for c in self.cameras
        ]
        if srgb:
            renders = list(map(to_srgb, renders))
        if uint8:
            renders = list(map(to_uint8, renders))
        return renders

    def _create_scene(
        self,
        points: Sequence[np.ndarray],
    ):
        spheres = {}

        for i, (cloud, bsdf) in enumerate(zip(points, self.bsdfs)):
            for j, p in enumerate(cloud):
                spheres[f"sphere_{i}_{j}"] = {
                    "type": "sphere",
                    "center": p,
                    "radius": self.point_size,
                    "bsdf": bsdf,
                }

        if self.floor is not None:
            floor = {
                "floor": {
                    "type": "rectangle",
                    "to_world": self.floor,
                    "material": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "rgb",
                            "value": np.array([1, 1, 1]),
                        },
                    },
                }
            }
        else:
            floor = {}

        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path"},
                "light": {"type": "constant"},
                **spheres,
                **floor,
            }
        )


@dataclass
class ConditionalSampleCallback3d:
    batch: Example
    key: PRNGKey = jax.random.PRNGKey(42)
    name: str = "conditional_samples"

    n_examples: int = 8
    point_radius: float = 0.01
    pov_radius: float = 1.75
    pred_color: np.ndarray = np.array([0.8, 0.1, 0.1])
    gt_color: np.ndarray = np.array([0.1, 0.1, 0.8])
    fov: Union[float, Sequence[float], None] = None

    def __post_init__(self):
        self.batch = jax.tree_map(
            lambda array: np.asarray(array)[: self.n_examples],
            self.batch,
        )

        if self.fov is None:
            # calculate per-example FOVs
            fx = self.batch.ctx.K[:, 0, 0]  # in relative units thus no need for width
            fov = np.rad2deg(2 * np.arctan(1 / (2 * fx)))
            self.fov = fov.tolist()
        elif isinstance(self.fov, float):
            # replicate per-example
            self.fov = [self.fov] * self.batch.points.shape[0]

    def __call__(
        self,
        model: Diffusion,
        logger: SummaryWriter,
        epoch: int,
    ):
        for index, img_grid in enumerate(self._render_all(model)):
            logger.add_image(
                f"{self.name}/sample_{index}",
                img_tensor=img_grid,
                global_step=epoch,
                dataformats="HWC",
            )

    def _render_all(self, model: Diffusion) -> List[A["h w 3"]]:
        n_points = self.batch.points.shape[1]

        sample_one = lambda ctx, key: model.sample(
            (n_points, 3),
            n=1,
            raw_ctx=ctx,
            return_details=False,
            key=key,
        )

        samples = jax.vmap(sample_one, in_axes=(0, None))(
            self.batch.ctx,
            self.key,
        )
        samples = np.asarray(samples).squeeze(1)

        job_args = list(
            zip(
                self.batch.points,
                samples,
                self.batch.ctx.image,
                self.fov,
            )
        )

        return list(map(self._job, job_args))

    def _job(self, args):
        gt_pts, pred_pts, image, fov = args

        renders = self._render(
            gt_points=gt_pts,
            pred_points=pred_pts,
            shape=image.shape[:-1][::-1],
            fov=fov,
        )
        renders = [to_srgb(render) for render in renders]

        grid = np.concatenate(
            [
                np.concatenate(
                    [
                        image,
                        renders[0],
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        renders[1],
                        renders[2],
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )

        return grid

    @dimchecked
    def _create_scene(
        self,
        gt_points: A["N 3"],
        pred_points: A["N 3"],
    ):
        pred_ball_bsdf = mi.load_dict(
            {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": self.pred_color},
            }
        )

        gt_ball_bsdf = mi.load_dict(
            {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": self.gt_color},
            }
        )

        spheres = {}
        for i, p in enumerate(gt_points):
            spheres[f"gt_sphere_{i}"] = {
                "type": "sphere",
                "center": p,
                "radius": self.point_radius,
                "bsdf": gt_ball_bsdf,
            }

        for i, p in enumerate(pred_points):
            spheres[f"pred_sphere_{i}"] = {
                "type": "sphere",
                "center": p,
                "radius": self.point_radius,
                "bsdf": pred_ball_bsdf,
            }

        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path"},
                "light": {"type": "constant"},
                **spheres,
            }
        )

    @dimchecked
    def _create_sensor(
        self,
        origin: A["3"],
        target: A["3"],
        up: A["3"],
        shape: Tuple[int, int],
        fov: float,
    ):
        return mi.load_dict(
            {
                "type": "perspective",
                "fov": fov,
                "fov_axis": "x",
                "to_world": T.look_at(
                    origin=origin,
                    target=target,
                    up=up,
                ),
                "sampler": {"type": "independent", "sample_count": 16},
                "film": {
                    "type": "hdrfilm",
                    "width": shape[0],
                    "height": shape[1],
                    "rfilter": {
                        "type": "tent",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    @dimchecked
    def _render(
        self,
        gt_points: A["N 3"],
        pred_points: A["N 3"],
        shape: Tuple[int, int],
        fov: float,
    ):
        gt_center = gt_points.mean(axis=0)

        origins = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
            ]
        )
        radius = self.pov_radius * np.linalg.norm(gt_points - gt_center, axis=-1).max()
        povs = radius * origins + gt_center.reshape(1, 3)
        povs = np.concatenate(
            [
                np.array([[0, 0, 0]]),
                origins,
            ],
            axis=0,
        )

        ups = np.array(
            [
                [0, -1, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]
        )

        sensors = []
        for pov, up in zip(povs, ups):
            sensors.append(self._create_sensor(pov, gt_center, up, shape, fov))

        scene = self._create_scene(
            gt_points=gt_points,
            pred_points=pred_points,
        )

        return [mi.render(scene, sensor=sensor) for sensor in sensors]
