import os
import re
from importlib import resources
from typing import Tuple, List
import torch
import numpy as np
import h5py
from tqdm.auto import tqdm
import imageio as iio

from gecco_jax.types import Context3d, Example


class Building:
    def __init__(
        self,
        name: str,
        h5_path: str,
        rgb_path: str,
        n_points: int = 2048,
    ):
        self.name = name
        self.h5_path = os.path.join(h5_path, f"{name}.h5")
        self.rgb_path = os.path.join(rgb_path, name)
        self.n_points = n_points
        self.return_image_path = False

        with h5py.File(self.h5_path, "r") as h5_file:
            points = h5_file["point"][()]
            views = h5_file["view"][()]

        self.points_and_views = list(zip(points.tolist(), views.tolist()))
        missing_points_and_views = self.missing_points_and_views()
        is_available = ~np.array(
            [(pv in missing_points_and_views) for pv in self.points_and_views]
        )
        indices = np.arange(len(self.points_and_views))
        self.reindex = indices[is_available]

    def return_image_path_(self, value: bool) -> None:
        self.return_image_path = value

    def rgb_file_path(self, index, name_only: bool = False) -> str:
        point, view = self.points_and_views[index]
        fname = f"{self.name}_{point}_{view}.jpg"
        if name_only:
            return fname
        return os.path.join(self.rgb_path, fname)

    def missing_points_and_views(self) -> List[Tuple[int, int]]:
        existing_files = frozenset(os.listdir(self.rgb_path))
        requested_files = frozenset(
            self.rgb_file_path(i, name_only=True)
            for i in range(len(self.points_and_views))
        )

        missing_files = requested_files - existing_files

        fname_re = re.compile(r"\w+_(\d+)_(\d+)\.jpg")
        missing_points_and_views = set()
        for missing_file in missing_files:
            if (m := fname_re.match(missing_file)) is None:
                raise RuntimeError(f"{missing_file=} doesn't match format.")
            point = int(m.group(1))
            view = int(m.group(2))
            missing_points_and_views.add((point, view))

        return missing_points_and_views

    def __len__(self):
        return len(self.reindex)

    def __getitem__(self, index):
        index = self.reindex[index]

        with h5py.File(self.h5_path, "r") as h5_file:
            pc = h5_file["pc"][index]
            K = h5_file["k"][index]

        image_path = self.rgb_file_path(index)
        image = np.asarray(iio.imread(image_path))
        image = image.astype(np.float32) / 255

        perm = np.random.permutation(pc.shape[0])[: self.n_points]
        pc = pc[perm]

        if self.return_image_path:
            extras = (image_path,)
        else:
            extras = ()

        return Example(
            points=pc.astype(np.float32),
            ctx=Context3d(
                image=image,
                K=K,
            ),
            extras=extras,
        )


def parse_split_file(split_file):
    splits = {}
    for line in list(split_file)[1:]:  # skip header
        name, is_train, is_val, is_test = line.split(",")
        if int(is_train):
            splits[name] = "train"
        if int(is_val):
            splits[name] = "val"
        if int(is_test):
            splits[name] = "test"
    return splits


class Taskonomy(torch.utils.data.ConcatDataset):
    def __init__(self, path: str, split: str = "all", n_points: int = 2048):
        self.h5_path = os.path.join(path, "point_clouds")
        self.rgb_path = os.path.join(path, "rgb")
        self.split = split

        with open(os.path.join(path, "taskonomy_split.csv")) as split_file:
            splits = parse_split_file(split_file)

        if split == "all":
            belongs_in_split = lambda _name: True
        else:
            belongs_in_split = lambda name: splits[name] == split

        buildings = []
        for file in tqdm(os.listdir(self.h5_path)):
            name = file[: -len(".h5")]

            if not belongs_in_split(name):
                continue

            buildings.append(
                Building(name, self.h5_path, self.rgb_path, n_points=n_points)
            )

        super().__init__(buildings)

    def __repr__(self):
        return f"Taskonomy(split={self.split}, n_buildings={len(self.datasets)}, len={len(self)})"

    def return_image_path_(self, value: bool) -> None:
        for dataset in self.datasets:
            dataset.return_image_path_(value)
