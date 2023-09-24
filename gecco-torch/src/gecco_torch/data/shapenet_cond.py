import os
import re
from typing import Callable, NamedTuple, Optional, Union, List, Any
from functools import partial

import torch
import numpy as np
import imageio as iio
import multiprocess as mp
import lightning.pytorch as pl
from tqdm.auto import tqdm

from gecco_torch.structs import Example, Context3d

IM_SIZE = 137  # 137 x 137 pixels
WORLD_MAT_RE = re.compile(r"world_mat_(\d+)")
CAMERA_MAT_RE = re.compile(r"camera_mat_(\d+)")
FIX_MASK_RE = re.compile(r"mask_(\d+)")


class TestData(NamedTuple):
    points_raw: np.ndarray
    scale: np.ndarray
    loc: np.ndarray
    wmat: np.ndarray
    category: List[str]
    object_id: List[str]


class ShapeNetVolModel:
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
    ):
        self.root = root
        self.n_points = n_points

        self.wmats, self.cmats = None, None

    def get_camera_params(self, index: int):
        if self.wmats is None:
            npz = np.load(os.path.join(self.root, "img_choy2016", "cameras.npz"))

            world_mat_ids = set()
            camera_mat_ids = set()

            for key in npz.keys():
                if (m := WORLD_MAT_RE.match(key)) is not None:
                    world_mat_ids.add(int(m.group(1)))
                    continue
                if (m := CAMERA_MAT_RE.match(key)) is not None:
                    camera_mat_ids.add(int(m.group(1)))
                    continue

            assert world_mat_ids == camera_mat_ids

            indices = np.array(sorted(list(world_mat_ids)))
            if (indices != np.arange(24)).all():
                raise AssertionError("Bad shapenet model")

            world_mats = np.stack([npz[f"world_mat_{i}"] for i in indices])
            camera_mats = np.stack([npz[f"camera_mat_{i}"] for i in indices])

            # normalize camera matrices
            camera_mats /= np.array([IM_SIZE + 1, IM_SIZE + 1, 1]).reshape(3, 1)

            self.wmats = world_mats.astype(np.float32)
            self.cmats = camera_mats.astype(np.float32)

        return self.wmats[index], self.cmats[index]

    @property
    def pointcloud_npz_path(self):
        return os.path.join(self.root, "pointcloud.npz")

    def points_scale_loc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.pointcloud_npz_path) as pc:
            points = pc["points"].astype(np.float32)
            scale = pc["scale"].astype(np.float32)
            loc = pc["loc"].astype(np.float32)

        return points, scale, loc

    def points_world(self):
        points, scale, loc = self.points_scale_loc()

        if self.n_points is not None:
            subset = np.random.permutation(points.shape[0])[: self.n_points]
            points = points[subset]
        return points * scale + loc[None, :]

    def __len__(self):
        return 24

    def __getitem__(self, index) -> Example:
        wmat, cmat = self.get_camera_params(index)
        points_world = self.points_world()
        points_view = np.einsum("ab,nb->na", wmat[:, :3], points_world) + wmat[:, -1]

        image_index = index
        image_path = os.path.join(
            self.root,
            "img_choy2016",
            f"{image_index:03d}.jpg",
        )
        image = iio.imread(image_path).astype(np.float32) / 255
        image = np.asarray(image)
        if image.ndim == 2:  # grayscale to rgb
            image = image[..., None].repeat(3, 2)
        image = image.transpose(2, 0, 1)

        return Example(
            data=points_view,
            ctx=Context3d(
                image=image,
                K=cmat.copy(),  # to avoid accidental mutation of self.cmats
                # wmat=wmat.copy(),
            ),
        )


class ShapeNetVolClass(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: str,
        **kw,
    ):
        with open(os.path.join(root, f"{split}.lst")) as split_file:
            split_ids = [line.strip() for line in split_file]
        paths = [os.path.join(root, id) for id in split_ids]
        make_model = partial(ShapeNetVolModel, **kw)

        if kw.get("posed", False) or kw.get("skip_fixed", False):
            # takes a while to load npzs to it's good to parallelize
            with mp.Pool() as pool:
                subsets = list(pool.imap(make_model, paths))
        else:
            # faster to not pay multiprocess overhead
            subsets = list(map(make_model, paths))

        super().__init__(subsets)
        self.root = root
        self.split = split


identity = lambda e: e


class ShapeNetVol(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: Union[str, List[str]],
        **kw,
    ):
        if isinstance(split, str):
            subroots = []
            for maybe_dir in os.listdir(root):
                maybe_dir_path = os.path.join(root, maybe_dir)
                if not os.path.isdir(maybe_dir_path):
                    continue
                subroots.append(maybe_dir_path)

            models = [
                ShapeNetVolClass(subroot, split, **kw) for subroot in tqdm(subroots)
            ]
        else:
            assert isinstance(split, (list, tuple))
            assert all(isinstance(path, str) for path in split)

            models = [ShapeNetVolModel(path, **kw) for path in tqdm(split)]

        super().__init__(models)


class ShapenetCondDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
    ):
        super().__init__()
        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size

    def setup(self, stage: Optional[str] = None):
        self.train = ShapeNetVol(self.root, "train")
        self.val = ShapeNetVol(self.root, "val")

    def train_dataloader(self):
        if self.epoch_size is None:
            return torch.utils.data.DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
            )

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=torch.utils.data.RandomSampler(
                self.train, replacement=True, num_samples=self.epoch_size
            ),
            pin_memory=True,
            shuffle=False,
        )

    def val_dataloader(self):
        if self.val_size is None:
            sampler = None
        else:
            sampler = torch.utils.data.RandomSampler(
                self.val, replacement=True, num_samples=self.val_size
            )

        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=False,
        )
