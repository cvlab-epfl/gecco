import os
import numpy as np
import torch

from gecco_jax.types import Example


class TorchShapenet:
    def __init__(self, root: str, category: str, split: str, n_points: int = 2048):
        self.path = os.path.join(root, category, split)
        self.npys = [f for f in os.listdir(self.path) if f.endswith(".npy")]
        self.n_points = n_points

    def __len__(self):
        return len(self.npys)

    def __getitem__(self, index):
        points = np.load(os.path.join(self.path, self.npys[index]))
        points = torch.from_numpy(points)
        perm = torch.randperm(points.shape[0])
        selected = points[perm[: self.n_points]]
        return Example(selected, [])
